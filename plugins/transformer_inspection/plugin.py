"""
主变自主巡视插件 - 完整实现

功能范围:
- 外观缺陷识别: 破损、锈蚀、渗漏油、异物悬挂
- 状态识别: 呼吸器硅胶变色、阀门开闭状态
- 热成像集成: 红外图像温度提取
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import copy
import importlib.util
import sys
import numpy as np

from platform_core.plugin_manager.base import (
    BasePlugin,
    HealthStatus,
    PluginContext,
    PluginManifest,
)
from platform_core.schema.models import (
    Alarm,
    AlarmLevel,
    AlarmRule,
    RecognitionResult,
    ROI,
    BoundingBox,
)


def _load_detector_class():
    """动态加载检测器类，解决相对导入问题"""
    detector_path = Path(__file__).parent / "detector.py"
    spec = importlib.util.spec_from_file_location("transformer_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformer_detector"] = module
    spec.loader.exec_module(module)
    return module.TransformerDetector


TransformerDetector = _load_detector_class()


class TransformerInspectionPlugin(BasePlugin):
    """
    主变自主巡视插件

    实现主变压器的缺陷检测和状态识别功能
    """

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector: Any = None
        self._thermal_enabled = False
        self._initialized = False
        self.confidence_threshold = 0.5

    def init(self, config: dict[str, Any]) -> bool:
        """
        初始化插件

        Args:
            config: 配置字典

        Returns:
            初始化是否成功
        """
        try:
            self._config = config
            self._thermal_enabled = config.get("thermal", {}).get("enabled", False)
            self.confidence_threshold = config.get("inference", {}).get("confidence_threshold", 0.5)

            # 创建检测器实例
            self._detector = TransformerDetector(config)

            # 标记初始化完成
            self._initialized = True

            print(f"[{self.id}] 插件初始化成功")
            return True

        except Exception as e:
            print(f"[{self.id}] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _merge_config(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """合并运行时配置，避免覆盖默认配置"""
        if not overrides:
            return base
        merged = copy.deepcopy(base)

        def _merge(target: dict[str, Any], incoming: dict[str, Any]) -> None:
            for key, value in incoming.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    _merge(target[key], value)
                else:
                    target[key] = value

        _merge(merged, overrides)
        return merged

    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """
        执行推理

        Args:
            frame: 输入图像帧 (BGR格式)
            rois: 识别区域列表
            context: 运行上下文

        Returns:
            识别结果列表
        """
        if not self._initialized or self._detector is None:
            print(f"[{self.id}] 警告: 插件未初始化")
            return []

        runtime_config = self._merge_config(self._config, context.config)
        self._thermal_enabled = runtime_config.get("thermal", {}).get("enabled", False)
        self._detector.update_config(runtime_config)

        results: list[RecognitionResult] = []
        h, w = frame.shape[:2]

        for roi in rois:
            try:
                # 提取ROI区域
                roi_bbox = roi.bbox
                x1 = int(roi_bbox.x * w)
                y1 = int(roi_bbox.y * h)
                x2 = int((roi_bbox.x + roi_bbox.width) * w)
                y2 = int((roi_bbox.y + roi_bbox.height) * h)

                # 确保坐标在范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                roi_image = frame[y1:y2, x1:x2]

                if roi_image.size == 0:
                    print(f"[{self.id}] 警告: ROI {roi.id} 区域为空")
                    continue

                # 获取ROI类型
                roi_type = roi.roi_type
                if hasattr(roi_type, 'value'):
                    roi_type = roi_type.value
                roi_type = str(roi_type)

                # 根据 README 定义的 ROI 类型映射检测任务
                # bushing (套管) -> 破损、污损、渗漏油
                # radiator (散热器) -> 渗漏油、锈蚀
                # oil_level (油位计) -> 刻度读数、破损
                # breather (呼吸器) -> 硅胶颜色变化
                # terminal_box (端子箱) -> 外观异常、破损
                roi_defect_map = {
                    "bushing": ["damage", "oil_leak"],
                    "radiator": ["oil_leak", "rust"],
                    "oil_level": ["damage"],
                    "terminal_box": ["damage", "foreign_object"],
                    "defect": ["damage", "rust", "oil_leak", "foreign_object"],
                }
                roi_state_map = {
                    "breather": ["silica_gel"],
                    "valve": ["valve"],
                    "state": ["silica_gel", "valve"],
                }
                roi_reading_map = {
                    "oil_level": True,  # 需要读取刻度
                }

                run_defects = roi_type in roi_defect_map
                run_state = roi_type in roi_state_map
                run_reading = roi_type in roi_reading_map

                # 执行缺陷检测
                if run_defects:
                    defect_filter = roi_defect_map.get(roi_type, [])
                    defects = self._detector.detect_defects(roi_image, roi_type, defect_filter)

                    # 转换检测结果为RecognitionResult
                    for defect in defects:
                        # 将相对坐标转换为绝对坐标
                        def_bbox = defect["bbox"]
                        abs_x = roi_bbox.x + def_bbox["x"] * roi_bbox.width
                        abs_y = roi_bbox.y + def_bbox["y"] * roi_bbox.height
                        abs_width = def_bbox["width"] * roi_bbox.width
                        abs_height = def_bbox["height"] * roi_bbox.height

                        result = RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=getattr(context, 'component_id', ''),
                            roi_id=roi.id,
                            bbox=BoundingBox(
                                x=abs_x,
                                y=abs_y,
                                width=abs_width,
                                height=abs_height
                            ),
                            label=defect["label"],
                            value=None,
                            confidence=defect["confidence"],
                            model_version=self.version,
                            code_version=self.code_hash,
                        )
                        results.append(result)

                # 执行状态检测
                if run_state:
                    state_filter = roi_state_map.get(roi_type, [])
                    states = self._detector.detect_state(roi_image, roi_type, state_filter)
                    for state in states:
                        result = RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=getattr(context, 'component_id', ''),
                            roi_id=roi.id,
                            bbox=roi.bbox,
                            label=state["label"],
                            value=state.get("value"),
                            confidence=state["confidence"],
                            model_version=self.version,
                            code_version=self.code_hash,
                        )
                        results.append(result)

                # 执行油位计刻度读数
                if run_reading:
                    reading = self._detector.read_oil_level(roi_image)
                    if reading and reading["confidence"] >= self.confidence_threshold:
                        result = RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=getattr(context, 'component_id', ''),
                            roi_id=roi.id,
                            bbox=roi.bbox,
                            label="oil_level_reading",
                            value=reading["value"],
                            confidence=reading["confidence"],
                            model_version=self.version,
                            code_version=self.code_hash,
                        )
                        results.append(result)

            except Exception as e:
                print(f"[{self.id}] 处理ROI {roi.id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[{self.id}] 检测完成，共 {len(results)} 个结果")
        return results

    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """
        后处理和告警生成

        Args:
            results: 推理结果列表
            rules: 告警规则列表

        Returns:
            告警列表
        """
        alarms: list[Alarm] = []

        # 遍历结果，根据规则生成告警
        for result in results:
            # 缺陷类告警 - 严重
            if result.label in ["oil_leak", "damage"]:
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.ERROR,
                    title=f"检测到{self._get_label_name(result.label)}",
                    message=f"在 {result.roi_id} 区域检测到{self._get_label_name(result.label)}，置信度: {result.confidence:.2f}",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)

            # 缺陷类告警 - 警告
            elif result.label in ["rust", "foreign_object"]:
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=f"检测到{self._get_label_name(result.label)}",
                    message=f"在 {result.roi_id} 区域检测到{self._get_label_name(result.label)}，置信度: {result.confidence:.2f}",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)

            # 状态类告警
            elif result.label == "silica_gel_abnormal":
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title="呼吸器硅胶变色",
                    message=f"呼吸器硅胶已变色，需要更换，置信度: {result.confidence:.2f}",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)

        print(f"[{self.id}] 生成 {len(alarms)} 个告警")
        return alarms

    def analyze_thermal(self, frame: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """热成像温度提取"""
        if self._detector is None:
            return None

        runtime_config = self._merge_config(self._config, config or {})
        thermal_cfg = runtime_config.get("thermal", {})
        if not thermal_cfg.get("enabled", False):
            return None

        self._detector.update_config(runtime_config)
        return self._detector.extract_thermal_metrics(frame, thermal_cfg)

    def healthcheck(self) -> HealthStatus:
        """健康检查"""
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
            )

        if self._detector is None:
            return HealthStatus(
                healthy=False,
                message="检测器未加载",
            )

        return HealthStatus(
            healthy=True,
            message="插件运行正常",
            details={
                "initialized": self._initialized,
                "thermal_enabled": self._thermal_enabled,
                "detector_ready": self._detector is not None,
            }
        )

    def _get_label_name(self, label: str) -> str:
        """获取标签的中文名称"""
        label_map = {
            "damage": "破损",
            "rust": "锈蚀",
            "oil_leak": "渗漏油",
            "foreign_object": "异物悬挂",
            "silica_gel_normal": "硅胶正常",
            "silica_gel_abnormal": "硅胶变色",
            "valve_open": "阀门开启",
            "valve_closed": "阀门关闭",
            "oil_level_reading": "油位读数",
        }
        return label_map.get(label, label)


# 为了方便测试，提供一个简单的Plugin别名
Plugin = TransformerInspectionPlugin
