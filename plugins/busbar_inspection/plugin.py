"""
母线自主巡视插件 - 完整实现
输变电激光星芒破夜绘明监测平台 (C组)

功能范围:
- 远距小目标检测: 销钉缺失、裂纹、异物
- 多目标并发处理: 批量ROI高效推理
- 环境过滤: 逆光/雨雾/遮挡场景识别
- 变焦建议: 目标过小时输出zoom建议

性能指标:
- pin_missing: Recall >= 0.85, Precision >= 0.85
- crack: Recall >= 0.70, Precision >= 0.80
- GPU P95 <= 800ms, CPU P95 <= 5s
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, List
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
    """动态加载检测器类"""
    detector_path = Path(__file__).parent / "detector.py"
    spec = importlib.util.spec_from_file_location("busbar_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["busbar_detector"] = module
    spec.loader.exec_module(module)
    return module.BusbarDetector


_BusbarDetector = None

def get_detector_class():
    global _BusbarDetector
    if _BusbarDetector is None:
        _BusbarDetector = _load_detector_class()
    return _BusbarDetector


class BusbarInspectionPlugin(BasePlugin):
    """
    母线自主巡视插件
    
    实现远距小目标检测、环境过滤和变焦建议
    
    性能指标(按验收标准):
    - pin_missing: Recall >= 0.85, Precision >= 0.85
    - crack: Recall >= 0.70, Precision >= 0.80
    - GPU P95 <= 800ms
    """
    
    # 标签名称映射
    LABEL_NAMES = {
        "pin_missing": "销钉缺失",
        "crack": "裂纹",
        "foreign_object": "异物",
        "quality_failed": "质量门禁未通过",
    }
    
    # 告警级别映射
    ALARM_LEVELS = {
        "pin_missing": AlarmLevel.ERROR,
        "crack": AlarmLevel.WARNING,
        "foreign_object": AlarmLevel.WARNING,
    }
    
    # 原因码说明
    REASON_DESCRIPTIONS = {
        101: "逆光/过曝/低对比",
        102: "遮挡/不可见",
        103: "模糊/失焦",
        104: "雨雾/霾导致低能见度",
        105: "运动干扰",
        201: "目标过小,需要变焦",
        202: "检测不稳定",
        301: "结果不可信",
    }
    
    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector: Any = None
        self._initialized = False
        self._last_inference_time: Optional[datetime] = None
        self._inference_count = 0
        self._error_count = 0
        self._quality_fail_count = 0
        
        # 配置参数
        self.confidence_threshold = 0.5
    
    def init(self, config: dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self._config = config
            
            inference_config = config.get("inference", {})
            self.confidence_threshold = inference_config.get("confidence_threshold", 0.5)
            
            # 创建检测器
            BusbarDetector = get_detector_class()
            self._detector = BusbarDetector(config)
            
            self.status = PluginStatus.READY
            self._initialized = True
            
            print(f"[{self.id}] 插件初始化成功")
            print(f"[{self.id}] 置信度阈值: {self.confidence_threshold}")
            
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
        
        支持4K大视场切片检测,输出质量门禁结果
        """
        if not self._initialized or self._detector is None:
            print(f"[{self.id}] 插件未初始化")
            return []
        
        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1
        
        results: list[RecognitionResult] = []
        
        for roi in rois:
            try:
                # 提取ROI区域
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    continue
                
                roi_type = self._get_roi_type(roi)
                
                # 执行检测(包含质量门禁)
                defects, quality_gate = self._detector.detect_defects(roi_image, roi_type)
                
                # 处理质量门禁失败
                if quality_gate is not None and not quality_gate.passed:
                    self._quality_fail_count += 1
                    
                    result = RecognitionResult(
                        task_id=context.task_id,
                        site_id=context.site_id,
                        device_id=context.device_id,
                        component_id=context.component_id,
                        roi_id=roi.id,
                        bbox=roi.bbox,
                        label="quality_failed",
                        confidence=1.0,
                        model_version=self.version,
                        code_version=self.code_hash,
                        failure_reason=quality_gate.reason_code,
                        metadata={
                            "reason": quality_gate.reason,
                            "suggested_action": quality_gate.suggested_action,
                            "details": quality_gate.details
                        },
                    )
                    results.append(result)
                    continue
                
                # 处理检测结果
                for defect in defects:
                    # 转换坐标
                    abs_bbox = self._convert_bbox_to_absolute(defect["bbox"], roi.bbox)
                    
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
                        failure_reason=defect.get("reason_code"),
                        metadata={
                            **defect.get("metadata", {}),
                            "suggested_zoom": defect.get("suggested_zoom"),
                            "suggested_action": defect.get("suggested_action"),
                        },
                    )
                    results.append(result)
                    
            except Exception as e:
                self._error_count += 1
                print(f"[{self.id}] 处理ROI {roi.id} 时出错: {e}")
                continue
        
        self.status = PluginStatus.READY
        print(f"[{self.id}] 检测完成，共 {len(results)} 个结果")
        
        return results
    
    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """后处理和告警生成"""
        alarms: list[Alarm] = []
        
        for result in results:
            # 质量门禁告警
            if result.label == "quality_failed":
                reason_code = result.failure_reason
                reason_desc = self.REASON_DESCRIPTIONS.get(reason_code, "未知原因")
                
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=f"质量门禁未通过: {reason_desc}",
                    message=f"ROI {result.roi_id} 图像质量不满足检测条件，建议: {result.metadata.get('suggested_action', '重新抓拍')}",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
                continue
            
            # 缺陷告警
            alarm_level = self.ALARM_LEVELS.get(result.label)
            if alarm_level is not None:
                label_name = self.LABEL_NAMES.get(result.label, result.label)
                
                message = f"在 {result.roi_id} 区域检测到{label_name}，置信度: {result.confidence:.2f}"
                
                # 添加变焦建议
                if result.metadata and result.metadata.get("suggested_zoom"):
                    message += f"。建议变焦 {result.metadata['suggested_zoom']}x 进行复核"
                
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=alarm_level,
                    title=f"检测到{label_name}",
                    message=message,
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
                "quality_fail_count": self._quality_fail_count,
                "last_inference": self._last_inference_time.isoformat() if self._last_inference_time else None,
            }
        )
    
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
        """获取ROI类型"""
        roi_type = getattr(roi, 'roi_type', None)
        if roi_type is not None:
            return roi_type.value if hasattr(roi_type, 'value') else str(roi_type)
        return getattr(roi, 'name', 'unknown').lower()
    
    def _convert_bbox_to_absolute(
        self,
        rel_bbox: dict,
        roi_bbox: BoundingBox
    ) -> BoundingBox:
        """转换坐标"""
        abs_x = roi_bbox.x + rel_bbox["x"] * roi_bbox.width
        abs_y = roi_bbox.y + rel_bbox["y"] * roi_bbox.height
        abs_w = rel_bbox["width"] * roi_bbox.width
        abs_h = rel_bbox["height"] * roi_bbox.height
        
        return BoundingBox(x=abs_x, y=abs_y, width=abs_w, height=abs_h)
