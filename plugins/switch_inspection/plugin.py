"""
开关间隔自主巡视插件 - 完整实现
输变电激光星芒破夜绘明监测平台 (B组)

功能范围:
- 分合位状态识别: 断路器/隔离开关/接地开关
- 互锁/逻辑校验: 五防规则验证
- 清晰度评价: 用于自动调焦触发
- SF6表读数(可选)

性能指标:
- 单帧单ROI CPU: < 300ms
- 状态识别准确率: >= 95%
- 逻辑校验误报率: < 2%
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
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
    """动态加载检测器类"""
    # 优先加载增强版检测器
    detector_path = Path(__file__).parent / "detector_enhanced.py"
    if not detector_path.exists():
        detector_path = Path(__file__).parent / "detector.py"

    spec = importlib.util.spec_from_file_location("switch_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["switch_detector"] = module
    spec.loader.exec_module(module)

    # 优先返回增强版检测器类
    if hasattr(module, 'SwitchDetectorEnhanced'):
        return module.SwitchDetectorEnhanced
    return module.SwitchDetector


_SwitchDetector = None

def get_detector_class():
    global _SwitchDetector
    if _SwitchDetector is None:
        _SwitchDetector = _load_detector_class()
    return _SwitchDetector


@dataclass
class BayState:
    """间隔状态缓存"""
    breaker: str = "unknown"
    isolator: str = "unknown"
    grounding: str = "unknown"
    timestamp: Optional[datetime] = None


class SwitchInspectionPlugin(BasePlugin):
    """
    开关间隔自主巡视插件
    
    实现分合位状态识别、逻辑校验和清晰度评价
    
    性能指标(按验收标准):
    - 置信度阈值: 0.6
    - 状态识别准确率: >= 95%
    - 单帧处理时间: < 300ms
    """
    
    # 标签名称映射
    LABEL_NAMES = {
        "breaker_open": "断路器分闸",
        "breaker_closed": "断路器合闸",
        "breaker_intermediate": "断路器中间态",
        "isolator_open": "隔离开关分闸",
        "isolator_closed": "隔离开关合闸",
        "grounding_open": "接地开关分闸",
        "grounding_closed": "接地开关合闸",
        "clarity_low": "清晰度过低",
        "sf6_pressure": "SF6压力读数",
        "sf6_density": "SF6密度读数",
    }
    
    # 失败原因码
    REASON_CODES = {
        1001: "清晰度过低",
        1002: "OCR失败或无文本",
        1003: "未检测到有效角度/线段",
        1004: "表盘/指针检测失败",
        2001: "互锁逻辑异常",
        2002: "异常工况提示",
        9000: "未知错误",
    }
    
    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector: Any = None
        self._initialized = False
        self._last_inference_time: Optional[datetime] = None
        self._inference_count = 0
        self._error_count = 0
        
        # 间隔状态缓存(用于逻辑校验)
        self._bay_states: Dict[str, BayState] = {}
        
        # 性能指标参数
        self.confidence_threshold = 0.6
        self.min_clarity_score = 0.70
    
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
            
            # 读取配置
            inference_config = config.get("inference", {})
            self.confidence_threshold = inference_config.get("confidence_threshold", 0.6)
            
            quality_config = config.get("image_quality", {})
            self.min_clarity_score = quality_config.get("min_clarity_score", 0.70)
            
            # 创建检测器实例
            SwitchDetector = get_detector_class()
            self._detector = SwitchDetector(config)
            
            # 更新状态
            self.status = PluginStatus.READY
            self._initialized = True
            
            print(f"[{self.id}] 插件初始化成功")
            print(f"[{self.id}] 置信度阈值: {self.confidence_threshold}")
            print(f"[{self.id}] 最小清晰度: {self.min_clarity_score}")
            
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
        
        # 获取或创建间隔状态
        bay_id = f"{context.site_id}_{context.device_id}"
        if bay_id not in self._bay_states:
            self._bay_states[bay_id] = BayState()
        bay_state = self._bay_states[bay_id]
        bay_state.timestamp = datetime.now()
        
        for roi in rois:
            try:
                # 提取ROI区域
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    continue
                
                roi_type = self._get_roi_type(roi)
                
                # 1. 清晰度评价(对清晰度锚点)
                if roi_type == "clarity_anchor" or "indicator" in roi_type:
                    is_clear, clarity_score = self._detector.evaluate_clarity(roi_image)

                    if not is_clear:
                        result = RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=context.component_id,
                            roi_id=roi.id,
                            bbox=roi.bbox,
                            label="clarity_low",
                            confidence=max(0.0, 1.0 - clarity_score),
                            model_version=self.version,
                            code_version=self.code_hash,
                            failure_reason=str(1001),
                            metadata={
                                "clarity_score": clarity_score,
                                "suggested_action": "REFOCUS_OR_RECAPTURE"
                            },
                        )
                        results.append(result)
                        continue  # 清晰度不足,跳过后续识别
                
                # 2. 状态识别(对指示器和连杆)
                if "indicator" in roi_type or "linkage" in roi_type or "handle" in roi_type:
                    device_type = self._get_device_type(roi_type)

                    if "indicator" in roi_type:
                        state_result = self._detector.recognize_indicator_state(
                            roi_image, device_type
                        )
                    else:
                        state_result = self._detector.recognize_linkage_state(
                            roi_image, device_type
                        )

                    if state_result.confidence >= self.confidence_threshold:
                        label = self._state_to_label(device_type, state_result.state)
                        evidence = state_result.evidence

                        result = RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=context.component_id,
                            roi_id=roi.id,
                            bbox=roi.bbox,
                            label=label,
                            confidence=state_result.confidence,
                            model_version=self.version,
                            code_version=self.code_hash,
                            failure_reason=state_result.reason_code,
                            metadata={
                                "state": state_result.state,
                                "device_type": device_type,
                                "evidence": {
                                    "ocr_text": evidence.ocr_text,
                                    "ocr_confidence": evidence.ocr_confidence,
                                    "red_ratio": evidence.red_ratio,
                                    "green_ratio": evidence.green_ratio,
                                    "angle_deg": evidence.angle_deg,
                                    "clarity_score": evidence.clarity_score,
                                },
                                "debug": state_result.extra,
                            },
                        )
                        results.append(result)

                        # 更新间隔状态缓存
                        state = state_result.state
                        if device_type == "breaker":
                            bay_state.breaker = state
                        elif device_type == "isolator":
                            bay_state.isolator = state
                        elif device_type == "grounding":
                            bay_state.grounding = state
                
                # 3. SF6表计读数
                if "gauge" in roi_type:
                    gauge_result = self._detector.read_gauge(roi_image)
                    gauge_reader = getattr(self._detector, "gauge_reader", None)
                    gauge_enabled = getattr(gauge_reader, "enabled", False)

                    if gauge_result.value is not None:
                        label = "sf6_pressure" if "pressure" in roi_type else "sf6_density"
                        result = RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=context.component_id,
                            roi_id=roi.id,
                            bbox=roi.bbox,
                            label=label,
                            value=gauge_result.value,
                            confidence=gauge_result.confidence,
                            model_version=self.version,
                            code_version=self.code_hash,
                            metadata={
                                "unit": gauge_result.unit
                            },
                        )
                        results.append(result)
                    elif gauge_enabled:
                        # 读数失败
                        result = RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=context.component_id,
                            roi_id=roi.id,
                            bbox=roi.bbox,
                            label="gauge_reading_failed",
                            confidence=0.0,
                            model_version=self.version,
                            code_version=self.code_hash,
                            failure_reason=str(gauge_result.reason_code or 1004),
                        )
                        results.append(result)
                        
            except Exception as e:
                self._error_count += 1
                print(f"[{self.id}] 处理ROI {roi.id} 时出错: {e}")
                continue
        
        # 4. 逻辑校验
        logic_alarms = self._detector.validate_interlock({
            "breaker": bay_state.breaker,
            "isolator": bay_state.isolator,
            "grounding": bay_state.grounding,
        })
        
        for alarm_info in logic_alarms:
            rule_name = alarm_info.get("rule_name") or alarm_info.get("rule") or "互锁逻辑异常"
            description = alarm_info.get("description") or alarm_info.get("message") or "检测到逻辑异常"
            result = RecognitionResult(
                task_id=context.task_id,
                site_id=context.site_id,
                device_id=context.device_id,
                component_id=context.component_id,
                roi_id="logic_validation",
                bbox=BoundingBox(x=0, y=0, width=1, height=1),
                label=f"logic_{alarm_info['severity']}",
                confidence=1.0,
                model_version=self.version,
                code_version=self.code_hash,
                failure_reason=alarm_info.get("reason_code"),
                metadata={
                    "rule": rule_name,
                    "rule_id": alarm_info.get("rule_id"),
                    "states": alarm_info.get("states", {}),
                    "message": description
                },
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
            # 清晰度告警
            if result.label == "clarity_low":
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title="图像清晰度不足",
                    message=f"ROI {result.roi_id} 清晰度评分过低，建议调焦或重新抓拍",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
            
            # 逻辑校验告警
            elif result.label.startswith("logic_"):
                severity = result.label.replace("logic_", "")
                level = AlarmLevel.ERROR if severity == "error" else AlarmLevel.WARNING
                
                metadata = result.metadata or {}
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=level,
                    title=f"五防逻辑校验: {metadata.get('rule', '规则异常')}",
                    message=metadata.get("message", "检测到逻辑异常"),
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
            
            # 状态异常告警(中间态)
            elif "intermediate" in result.label:
                label_name = self.LABEL_NAMES.get(result.label, result.label)
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.INFO,
                    title=f"检测到{label_name}",
                    message="设备处于中间态，请确认状态",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
        
        return alarms
    
    def healthcheck(self) -> HealthStatus:
        """健康检查"""
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
                "bay_states_count": len(self._bay_states),
            }
        )

    def get_ui_config(self) -> dict[str, Any]:
        """获取UI配置"""
        return {
            "detection_types": [
                {
                    "id": "state",
                    "name": "状态识别",
                    "icon": "eye",
                    "description": "断路器、隔离开关、接地开关分合位状态识别和五防逻辑校验",
                    "enabled": True,
                    "capabilities": [
                        {"label": "断路器状态", "tags": ["breaker_open", "breaker_closed", "breaker_intermediate"]},
                        {"label": "隔离开关状态", "tags": ["isolator_open", "isolator_closed"]},
                        {"label": "接地开关状态", "tags": ["grounding_open", "grounding_closed"]},
                        {"label": "五防逻辑校验", "tags": ["interlock_validation"]},
                    ]
                },
                {
                    "id": "clarity",
                    "name": "清晰度评价",
                    "icon": "image",
                    "description": "图像清晰度评估，用于自动调焦触发",
                    "enabled": True,
                    "capabilities": [
                        {"label": "清晰度评分", "tags": ["clarity_score"]},
                        {"label": "自动调焦建议", "tags": ["focus_suggestion"]},
                    ]
                },
                {
                    "id": "gauge_reading",
                    "name": "SF6表计读数",
                    "icon": "speedometer2",
                    "description": "SF6压力表和密度表读数（可选）",
                    "enabled": False,
                    "capabilities": [
                        {"label": "SF6压力读数", "tags": ["sf6_pressure"]},
                        {"label": "SF6密度读数", "tags": ["sf6_density"]},
                    ]
                }
            ],
            "parameters": [
                {
                    "name": "confidence_threshold",
                    "label": "置信度阈值",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": self.confidence_threshold,
                    "description": "状态识别的最小置信度"
                },
                {
                    "name": "min_clarity_score",
                    "label": "最小清晰度",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": self.min_clarity_score,
                    "description": "清晰度评价的最小阈值"
                }
            ]
        }
    
    # ==================== 辅助方法 ====================
    
    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """从帧中提取ROI区域"""
        h, w = frame.shape[:2]
        
        x1 = int(bbox.x * w)
        y1 = int(bbox.y * h)
        x2 = int((bbox.x + bbox.width) * w)
        y2 = int((bbox.y + bbox.height) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2].copy()
    
    def _get_roi_type(self, roi: ROI) -> str:
        """获取ROI类型字符串"""
        roi_type = getattr(roi, 'roi_type', None)
        if roi_type is not None:
            return roi_type.value if hasattr(roi_type, 'value') else str(roi_type)
        return getattr(roi, 'name', 'unknown').lower()
    
    def _get_device_type(self, roi_type: str) -> str:
        """从ROI类型推断设备类型"""
        if "breaker" in roi_type:
            return "breaker"
        elif "isolator" in roi_type:
            return "isolator"
        elif "grounding" in roi_type:
            return "grounding"
        return "breaker"  # 默认

    def _state_to_label(self, device_type: str, state: str) -> str:
        """将状态映射到标准标签"""
        prefix = device_type if device_type in {"breaker", "isolator", "grounding"} else "breaker"
        if state == "open":
            return f"{prefix}_open"
        if state == "closed":
            return f"{prefix}_closed"
        if state == "intermediate":
            return f"{prefix}_intermediate"
        return "unknown"
