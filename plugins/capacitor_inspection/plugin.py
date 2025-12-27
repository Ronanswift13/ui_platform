"""
电容器自主巡视插件 - 完整实现
输变电激光星芒破夜绘明监测平台 (D组)

功能范围:
- 结构完整性检测: 倾斜/倒塌/部件缺失
- 区域入侵检测: 人员/车辆/动物入侵告警

性能指标:
- 置信度阈值: 0.55
- 倾斜检测阈值: 5.0°
- 入侵告警延迟: 2.0s
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import importlib.util
import sys
import numpy as np

from platform_core.plugin_manager.base import (
    BasePlugin, HealthStatus, PluginContext, PluginManifest, PluginStatus,
)
from platform_core.schema.models import (
    Alarm, AlarmLevel, AlarmRule, RecognitionResult, ROI, BoundingBox,
)


def _load_detector_class():
    detector_path = Path(__file__).parent / "detector.py"
    spec = importlib.util.spec_from_file_location("capacitor_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["capacitor_detector"] = module
    spec.loader.exec_module(module)
    return module.CapacitorDetector


_CapacitorDetector = None

def get_detector_class():
    global _CapacitorDetector
    if _CapacitorDetector is None:
        _CapacitorDetector = _load_detector_class()
    return _CapacitorDetector


class CapacitorInspectionPlugin(BasePlugin):
    """电容器自主巡视插件"""
    
    LABEL_NAMES = {
        "tilt_warning": "电容器倾斜(警告)", "tilt_error": "电容器倾斜(严重)",
        "collapse": "电容器倒塌", "missing_unit": "电容器单元缺失",
        "intrusion_person": "人员入侵", "intrusion_vehicle": "车辆入侵",
        "intrusion_animal": "动物入侵", "intrusion_unknown": "未知入侵",
    }
    
    ALARM_LEVELS = {
        "tilt_warning": AlarmLevel.WARNING, "tilt_error": AlarmLevel.ERROR,
        "collapse": AlarmLevel.ERROR, "missing_unit": AlarmLevel.ERROR,
        "intrusion_person": AlarmLevel.ERROR, "intrusion_vehicle": AlarmLevel.ERROR,
        "intrusion_animal": AlarmLevel.WARNING, "intrusion_unknown": AlarmLevel.INFO,
    }
    
    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector = None
        self._initialized = False
        self._last_inference_time = None
        self._inference_count = 0
        self._error_count = 0
        self.confidence_threshold = 0.55
    
    def init(self, config: dict[str, Any]) -> bool:
        try:
            self._config = config
            self.confidence_threshold = config.get("inference", {}).get("confidence_threshold", 0.55)
            self._detector = get_detector_class()(config)
            self.status = PluginStatus.READY
            self._initialized = True
            print(f"[{self.id}] 插件初始化成功")
            return True
        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            return False
    
    def infer(self, frame: np.ndarray, rois: list[ROI], context: PluginContext) -> list[RecognitionResult]:
        if not self._initialized:
            return []
        
        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1
        results = []
        
        for roi in rois:
            try:
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None:
                    continue
                
                roi_type = self._get_roi_type(roi)
                
                if roi_type in ["capacitor_bank", "capacitor_unit", "fuse", "connecting_bar"]:
                    for defect in self._detector.detect_structural_defects(roi_image, roi_type):
                        results.append(self._create_result(context, roi, defect))
                
                if roi_type in ["fence", "warning_zone", "restricted_zone"]:
                    zone_type = "restricted" if "restricted" in roi_type else "warning"
                    for intrusion in self._detector.detect_intrusion(roi_image, zone_type):
                        results.append(self._create_result(context, roi, intrusion))
            except Exception as e:
                self._error_count += 1
        
        self.status = PluginStatus.READY
        return results
    
    def postprocess(self, results: list[RecognitionResult], rules: list[AlarmRule]) -> list[Alarm]:
        alarms = []
        for result in results:
            level = self.ALARM_LEVELS.get(result.label)
            if level:
                alarms.append(Alarm(
                    task_id=result.task_id, result_id=None, level=level,
                    title=f"检测到{self.LABEL_NAMES.get(result.label, result.label)}",
                    message=f"在 {result.roi_id} 区域检测到异常",
                    site_id=result.site_id, device_id=result.device_id, component_id=result.component_id,
                ))
        return alarms
    
    def healthcheck(self) -> HealthStatus:
        if not self._initialized:
            return HealthStatus(healthy=False, message="插件未初始化")
        return HealthStatus(healthy=True, message="插件运行正常", details={
            "inference_count": self._inference_count, "error_count": self._error_count
        })
    
    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox):
        h, w = frame.shape[:2]
        x1, y1 = int(bbox.x * w), int(bbox.y * h)
        x2, y2 = int((bbox.x + bbox.width) * w), int((bbox.y + bbox.height) * h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)].copy()
    
    def _get_roi_type(self, roi: ROI) -> str:
        roi_type = getattr(roi, 'roi_type', None)
        return (roi_type.value if hasattr(roi_type, 'value') else str(roi_type)) if roi_type else "unknown"
    
    def _create_result(self, context, roi, detection) -> RecognitionResult:
        abs_bbox = BoundingBox(
            x=roi.bbox.x + detection["bbox"]["x"] * roi.bbox.width,
            y=roi.bbox.y + detection["bbox"]["y"] * roi.bbox.height,
            width=detection["bbox"]["width"] * roi.bbox.width,
            height=detection["bbox"]["height"] * roi.bbox.height
        )
        return RecognitionResult(
            task_id=context.task_id, site_id=context.site_id, device_id=context.device_id,
            component_id=context.component_id, roi_id=roi.id, bbox=abs_bbox,
            label=detection["label"], confidence=detection["confidence"],
            model_version=self.version, code_version=self.code_hash, metadata=detection.get("metadata", {})
        )
