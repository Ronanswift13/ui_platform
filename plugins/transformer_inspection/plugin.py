"""
主变自主巡视插件 - 完整实现
输变电激光星芒破夜绘明监测平台 (A组)

功能范围:
- 外观缺陷识别: 破损、锈蚀、渗漏油、异物悬挂
- 状态识别: 呼吸器硅胶变色、阀门开闭状态
- 热成像集成: 红外图像温度提取

性能指标:
- 置信度阈值: >= 0.5
- 最大处理时间: < 500ms/帧
- 支持分辨率: 640x480 ~ 3840x2160
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import importlib.util
import sys
import numpy as np

from platform_core.plugin_manager.base import (
    BasePlugin,
    HealthStatus,
    PluginContext,
    PluginManifest,
    PluginStatus,
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
    # 优先加载增强版检测器
    detector_path = Path(__file__).parent / "detector_enhanced.py"
    if not detector_path.exists():
        detector_path = Path(__file__).parent / "detector.py"

    spec = importlib.util.spec_from_file_location("transformer_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformer_detector"] = module
    spec.loader.exec_module(module)

    # 优先返回增强版检测器类
    if hasattr(module, 'TransformerDetectorEnhanced'):
        return module.TransformerDetectorEnhanced
    return module.TransformerDetector


# 延迟加载检测器类
_TransformerDetector = None

def get_detector_class():
    global _TransformerDetector
    if _TransformerDetector is None:
        _TransformerDetector = _load_detector_class()
    return _TransformerDetector


class TransformerInspectionPlugin(BasePlugin):
    """
    主变自主巡视插件
    
    实现主变压器的缺陷检测和状态识别功能
    
    性能指标(按验收标准):
    - 置信度阈值: 0.5
    - NMS阈值: 0.4
    - 最大检测数: 100
    - 热成像温度阈值: 80℃
    """
    
    # 标签名称映射
    LABEL_NAMES = {
        "oil_leak": "渗漏油",
        "damage": "破损",
        "rust": "锈蚀",
        "foreign_object": "异物悬挂",
        "silica_gel_normal": "硅胶正常",
        "silica_gel_abnormal": "硅胶变色",
        "silica_gel_unknown": "硅胶状态未知",
        "valve_open": "阀门开启",
        "valve_closed": "阀门关闭",
        "valve_intermediate": "阀门中间态",
        "valve_unknown": "阀门状态未知",
        "oil_level_reading": "油位读数",
        "normal": "正常",
        "unknown": "未知",
    }
    
    # 告警级别映射
    ALARM_LEVELS = {
        "oil_leak": AlarmLevel.ERROR,
        "damage": AlarmLevel.ERROR,
        "rust": AlarmLevel.WARNING,
        "foreign_object": AlarmLevel.WARNING,
        "silica_gel_abnormal": AlarmLevel.WARNING,
        "valve_intermediate": AlarmLevel.INFO,
    }
    
    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector: Any = None
        self._thermal_enabled = False
        self._initialized = False
        self._last_inference_time: Optional[datetime] = None
        self._inference_count = 0
        self._error_count = 0
        
        # 性能指标参数
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.max_detections = 100
    
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
            
            # 读取推理配置
            inference_config = config.get("inference", {})
            self.confidence_threshold = inference_config.get("confidence_threshold", 0.5)
            self.nms_threshold = inference_config.get("nms_threshold", 0.4)
            self.max_detections = inference_config.get("max_detections", 100)
            
            # 读取热成像配置
            thermal_config = config.get("thermal", {})
            self._thermal_enabled = thermal_config.get("enabled", False)
            
            # 创建检测器实例
            TransformerDetector = get_detector_class()
            self._detector = TransformerDetector(config)
            
            # 更新状态
            self.status = PluginStatus.READY
            self._initialized = True
            
            print(f"[{self.id}] 插件初始化成功")
            print(f"[{self.id}] 置信度阈值: {self.confidence_threshold}")
            print(f"[{self.id}] 热成像: {'启用' if self._thermal_enabled else '禁用'}")
            
            return True
            
        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            print(f"[{self.id}] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
            print(f"[{self.id}] 插件未初始化")
            return []
        
        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1
        
        results: list[RecognitionResult] = []
        h, w = frame.shape[:2]
        
        for roi in rois:
            try:
                # 提取ROI区域
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    continue
                
                roi_type = getattr(roi, 'roi_type', None)
                if roi_type is not None:
                    roi_type = roi_type.value if hasattr(roi_type, 'value') else str(roi_type)
                else:
                    roi_type = "unknown"
                
                # 缺陷检测
                defects = self._detector.detect_defects(roi_image, roi_type)
                
                for defect in defects[:self.max_detections]:
                    if defect["confidence"] >= self.confidence_threshold:
                        # 转换bbox坐标(ROI相对 -> 全图相对)
                        abs_bbox = self._convert_bbox_to_absolute(
                            defect["bbox"], roi.bbox
                        )
                        
                        result = RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=context.component_id,
                            roi_id=roi.id,
                            bbox=abs_bbox,
                            label=defect["label"],
                            confidence=defect["confidence"],
                            model_version=self.version,
                            code_version=self.code_hash,
                            metadata=defect.get("metadata", {}),
                        )
                        results.append(result)
                
                # 状态识别
                state = self._detector.recognize_state(roi_image, roi_type)
                if state and state.get("confidence", 0) >= self.confidence_threshold:
                    result = RecognitionResult(
                        task_id=context.task_id,
                        site_id=context.site_id,
                        device_id=context.device_id,
                        component_id=context.component_id,
                        roi_id=roi.id,
                        bbox=roi.bbox,
                        label=state["label"],
                        value=state.get("value"),
                        confidence=state["confidence"],
                        model_version=self.version,
                        code_version=self.code_hash,
                        metadata=state.get("metadata", {}),
                    )
                    results.append(result)
                    
            except Exception as e:
                self._error_count += 1
                print(f"[{self.id}] 处理ROI {roi.id} 时出错: {e}")
                continue
        
        # 热成像分析(如果启用)
        thermal_frame = getattr(context, "thermal_frame", None)
        if self._thermal_enabled and thermal_frame is not None:
            thermal_result = self._detector.analyze_thermal(thermal_frame)
            if thermal_result.get("is_overtemp", False):
                # 添加超温告警结果
                for hotspot in thermal_result.get("hotspots", []):
                    result = RecognitionResult(
                        task_id=context.task_id,
                        site_id=context.site_id,
                        device_id=context.device_id,
                        component_id=context.component_id,
                        roi_id="thermal",
                        bbox=BoundingBox(
                            x=hotspot["x"],
                            y=hotspot["y"],
                            width=hotspot["width"],
                            height=hotspot["height"]
                        ),
                        label="overtemp",
                        value=hotspot["max_temp"],
                        confidence=0.9,
                        model_version=self.version,
                        code_version=self.code_hash,
                        metadata={"thermal_analysis": thermal_result},
                    )
                    results.append(result)
        
        self.status = PluginStatus.READY
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
        
        for result in results:
            # 获取告警级别
            alarm_level = self.ALARM_LEVELS.get(result.label)
            
            if alarm_level is not None:
                label_name = self.LABEL_NAMES.get(result.label, result.label)
                
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=alarm_level,
                    title=f"检测到{label_name}",
                    message=f"在 {result.roi_id} 区域检测到{label_name}，置信度: {result.confidence:.2f}",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
            
            # 处理超温告警
            if result.label == "overtemp" and result.value is not None:
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.ERROR,
                    title="检测到设备超温",
                    message=f"设备温度 {result.value:.1f}℃ 超过阈值",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
        
        return alarms
    
    def healthcheck(self) -> HealthStatus:
        """
        健康检查
        
        Returns:
            健康状态
        """
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
                details={"status": self.status.value}
            )
        
        if self._detector is None:
            return HealthStatus(
                healthy=False,
                message="检测器未就绪",
                details={"status": self.status.value}
            )
        
        return HealthStatus(
            healthy=True,
            message="插件运行正常",
            details={
                "status": self.status.value,
                "inference_count": self._inference_count,
                "error_count": self._error_count,
                "last_inference": self._last_inference_time.isoformat() if self._last_inference_time else None,
                "thermal_enabled": self._thermal_enabled,
                "config": {
                    "confidence_threshold": self.confidence_threshold,
                    "nms_threshold": self.nms_threshold,
                    "max_detections": self.max_detections,
                }
            }
        )
    
    # ==================== 辅助方法 ====================
    
    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """
        从帧中提取ROI区域
        
        Args:
            frame: 完整帧
            bbox: ROI边界框
            
        Returns:
            ROI图像
        """
        h, w = frame.shape[:2]
        
        x1 = int(bbox.x * w)
        y1 = int(bbox.y * h)
        x2 = int((bbox.x + bbox.width) * w)
        y2 = int((bbox.y + bbox.height) * h)
        
        # 边界检查
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2].copy()
    
    def _convert_bbox_to_absolute(
        self,
        rel_bbox: dict,
        roi_bbox: BoundingBox
    ) -> BoundingBox:
        """
        将ROI相对坐标转换为全图相对坐标
        
        Args:
            rel_bbox: ROI内相对坐标 {"x", "y", "width", "height"}
            roi_bbox: ROI在全图中的坐标
            
        Returns:
            全图相对坐标
        """
        abs_x = roi_bbox.x + rel_bbox["x"] * roi_bbox.width
        abs_y = roi_bbox.y + rel_bbox["y"] * roi_bbox.height
        abs_w = rel_bbox["width"] * roi_bbox.width
        abs_h = rel_bbox["height"] * roi_bbox.height
        
        return BoundingBox(x=abs_x, y=abs_y, width=abs_w, height=abs_h)
    
    def _get_label_name(self, label: str) -> str:
        """获取标签的中文名称"""
        return self.LABEL_NAMES.get(label, label)
