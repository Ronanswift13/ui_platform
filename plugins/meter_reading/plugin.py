"""
表计无建模增强读数插件 - 完整实现
输变电激光星芒破夜绘明监测平台 (E组)

功能范围:
- 任意角度读数: 关键点检测+透视矫正
- 自动量程识别: 刻度检测
- 失败兜底: 重试+历史参考+人工复核标记

性能指标:
- 置信度阈值: 0.6
- 最大旋转角度: 45°
- 重试次数: 3
- 人工复核阈值: 0.5
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
    spec = importlib.util.spec_from_file_location("meter_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["meter_detector"] = module
    spec.loader.exec_module(module)
    return module.MeterReadingDetector


_MeterReadingDetector = None

def get_detector_class():
    global _MeterReadingDetector
    if _MeterReadingDetector is None:
        _MeterReadingDetector = _load_detector_class()
    return _MeterReadingDetector


class MeterReadingPlugin(BasePlugin):
    """
    表计读数插件
    
    实现任意角度读数、自动量程识别和失败兜底
    """
    
    # 表计类型名称映射
    METER_NAMES = {
        "pressure_gauge": "压强表",
        "temperature_gauge": "温度表",
        "oil_level_gauge": "油位表",
        "sf6_density_gauge": "SF6密度表",
        "digital_display": "数字显示屏",
        "LED_indicator": "LED指示灯",
        "ammeter": "电流表",
        "voltmeter": "电压表",
    }
    
    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector = None
        self._initialized = False
        self._last_inference_time = None
        self._inference_count = 0
        self._success_count = 0
        self._manual_review_count = 0
        
        self.confidence_threshold = 0.6
        self.retry_count = 3
    
    def init(self, config: dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self._config = config
            
            inference_config = config.get("inference", {})
            self.confidence_threshold = inference_config.get("confidence_threshold", 0.6)
            
            fallback_config = config.get("fallback", {})
            self.retry_count = fallback_config.get("retry_count", 3)
            
            MeterReadingDetector = get_detector_class()
            self._detector = MeterReadingDetector(config)
            
            self.status = PluginStatus.READY
            self._initialized = True
            
            print(f"[{self.id}] 插件初始化成功")
            return True
            
        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            print(f"[{self.id}] 初始化失败: {e}")
            return False
    
    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """执行推理"""
        if not self._initialized or self._detector is None:
            return []
        
        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1
        
        results: list[RecognitionResult] = []
        
        for roi in rois:
            try:
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    continue
                
                meter_type = self._get_meter_type(roi)
                roi_id = f"{context.device_id}_{roi.id}"
                
                # 读取表计(带重试)
                reading = None
                for attempt in range(self.retry_count):
                    reading = self._detector.read_meter(roi_image, meter_type, roi_id)
                    if reading.value is not None and reading.confidence >= self.confidence_threshold:
                        break
                
                if reading is None:
                    continue
                
                # 统计
                if reading.value is not None:
                    self._success_count += 1
                if reading.need_manual_review:
                    self._manual_review_count += 1
                
                # 构建标签
                if reading.value is not None:
                    label = f"{meter_type}_reading"
                    value = reading.value
                else:
                    label = f"{meter_type}_failed"
                    value = None
                
                result = RecognitionResult(
                    task_id=context.task_id,
                    site_id=context.site_id,
                    device_id=context.device_id,
                    component_id=context.component_id,
                    roi_id=roi.id,
                    bbox=roi.bbox,
                    label=label,
                    value=value,
                    confidence=reading.confidence,
                    model_version=self.version,
                    code_version=self.code_hash,
                    metadata={
                        "unit": reading.unit,
                        "meter_type": reading.meter_type.value,
                        "need_manual_review": reading.need_manual_review,
                        "keypoints": reading.keypoints,
                    },
                )
                results.append(result)
                
            except Exception as e:
                print(f"[{self.id}] 处理ROI {roi.id} 时出错: {e}")
                continue
        
        self.status = PluginStatus.READY
        print(f"[{self.id}] 读数完成，共 {len(results)} 个结果")
        
        return results
    
    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """后处理和告警生成"""
        alarms: list[Alarm] = []
        
        for result in results:
            metadata = result.metadata or {}
            
            # 读数失败告警
            if "_failed" in result.label:
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.INFO,
                    title="表计读数失败",
                    message=f"无法读取 {result.roi_id} 的表计读数",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
            
            # 需要人工复核告警
            elif metadata.get("need_manual_review", False):
                meter_name = self._get_meter_name(result.label)
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.INFO,
                    title="表计读数需人工复核",
                    message=f"{meter_name} 读数 {result.value} {metadata.get('unit', '')}，置信度较低，请人工确认",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
            
            # 读数超限告警(根据规则)
            for rule in rules:
                if self._check_rule(result, rule):
                    alarm = Alarm(
                        task_id=result.task_id,
                        result_id=None,
                        level=AlarmLevel.WARNING,
                        title="表计读数超限",
                        message=f"{self._get_meter_name(result.label)} 读数 {result.value} 超出正常范围",
                        site_id=result.site_id,
                        device_id=result.device_id,
                        component_id=result.component_id,
                    )
                    alarms.append(alarm)
        
        return alarms
    
    def healthcheck(self) -> HealthStatus:
        """健康检查"""
        if not self._initialized:
            return HealthStatus(healthy=False, message="插件未初始化")
        
        if self._detector is None:
            return HealthStatus(healthy=False, message="检测器未就绪")
        
        success_rate = self._success_count / self._inference_count if self._inference_count > 0 else 0
        
        return HealthStatus(
            healthy=True,
            message="插件运行正常",
            details={
                "status": self.status.value,
                "inference_count": self._inference_count,
                "success_count": self._success_count,
                "success_rate": round(success_rate, 3),
                "manual_review_count": self._manual_review_count,
                "last_inference": self._last_inference_time.isoformat() if self._last_inference_time else None,
            }
        )
    
    # ==================== 辅助方法 ====================
    
    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """从帧中提取ROI区域"""
        h, w = frame.shape[:2]
        x1, y1 = int(bbox.x * w), int(bbox.y * h)
        x2, y2 = int((bbox.x + bbox.width) * w), int((bbox.y + bbox.height) * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()
    
    def _get_meter_type(self, roi: ROI) -> str:
        """获取表计类型"""
        roi_type = getattr(roi, 'roi_type', None)
        if roi_type is not None:
            return roi_type.value if hasattr(roi_type, 'value') else str(roi_type)
        
        name = getattr(roi, 'name', '').lower()
        for mtype in self.METER_NAMES.keys():
            if mtype in name:
                return mtype
        
        return "pressure_gauge"  # 默认
    
    def _get_meter_name(self, label: str) -> str:
        """获取表计中文名称"""
        for key, name in self.METER_NAMES.items():
            if key in label:
                return name
        return "表计"
    
    def _check_rule(self, result: RecognitionResult, rule: AlarmRule) -> bool:
        """检查告警规则"""
        if result.value is None:
            return False
        
        # 简化的规则检查
        if hasattr(rule, 'min_value') and result.value < rule.min_value:
            return True
        if hasattr(rule, 'max_value') and result.value > rule.max_value:
            return True
        
        return False
