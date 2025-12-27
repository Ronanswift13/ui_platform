"""
自动ROI检测器
输变电激光监测平台 - 全自动AI巡检增强

实现功能:
- 自动检测设备部件并生成ROI
- 设备拓扑管理
- ROI跟踪和更新
- 与任务引擎的无缝集成
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import threading

try:
    import cv2
except ImportError:
    cv2 = None


class DeviceType(Enum):
    """设备类型"""
    TRANSFORMER = "transformer"           # 变压器
    BREAKER = "breaker"                   # 断路器
    ISOLATOR = "isolator"                 # 隔离开关
    GROUNDING = "grounding"               # 接地开关
    BUSBAR = "busbar"                     # 母线
    INSULATOR = "insulator"               # 绝缘子
    CAPACITOR = "capacitor"               # 电容器
    METER_ANALOG = "meter_analog"         # 模拟表
    METER_DIGITAL = "meter_digital"       # 数字表
    INDICATOR = "indicator"               # 指示灯
    OIL_LEVEL = "oil_level"               # 油位计
    SILICA_GEL = "silica_gel"             # 硅胶罐
    VALVE = "valve"                       # 阀门
    SF6_GAUGE = "sf6_gauge"               # SF6密度表
    FENCE = "fence"                       # 围栏


class ROIType(Enum):
    """ROI类型"""
    DETECTION = "detection"               # 缺陷检测区域
    STATE = "state"                       # 状态识别区域
    READING = "reading"                   # 读数区域
    THERMAL = "thermal"                   # 热成像区域
    INTRUSION = "intrusion"               # 入侵检测区域


@dataclass
class AutoROI:
    """自动生成的ROI"""
    roi_id: str
    device_type: DeviceType
    roi_type: ROIType
    bbox: Dict[str, float]                # 归一化坐标 {x, y, width, height}
    confidence: float
    class_name: str
    parent_device_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tracking_id: Optional[str] = None     # 跟踪ID


@dataclass
class DeviceTopology:
    """设备拓扑信息"""
    device_id: str
    device_type: DeviceType
    name: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    expected_rois: List[ROIType] = field(default_factory=list)
    template_features: Optional[np.ndarray] = None  # 模板特征用于匹配


class AutoROIDetector:
    """
    自动ROI检测器
    
    使用深度学习模型自动检测设备部件并生成ROI
    """
    
    # 设备类型到检测类别的映射
    DEVICE_CLASS_MAP = {
        "transformer": DeviceType.TRANSFORMER,
        "breaker": DeviceType.BREAKER,
        "isolator": DeviceType.ISOLATOR,
        "grounding_switch": DeviceType.GROUNDING,
        "busbar": DeviceType.BUSBAR,
        "insulator": DeviceType.INSULATOR,
        "capacitor": DeviceType.CAPACITOR,
        "pressure_gauge": DeviceType.METER_ANALOG,
        "temperature_gauge": DeviceType.METER_ANALOG,
        "digital_display": DeviceType.METER_DIGITAL,
        "indicator_light": DeviceType.INDICATOR,
        "oil_level_gauge": DeviceType.OIL_LEVEL,
        "silica_gel": DeviceType.SILICA_GEL,
        "valve": DeviceType.VALVE,
        "sf6_gauge": DeviceType.SF6_GAUGE,
        "fence": DeviceType.FENCE,
    }
    
    # 设备类型到ROI类型的映射
    DEVICE_ROI_MAP = {
        DeviceType.TRANSFORMER: [ROIType.DETECTION, ROIType.THERMAL],
        DeviceType.BREAKER: [ROIType.STATE, ROIType.DETECTION],
        DeviceType.ISOLATOR: [ROIType.STATE],
        DeviceType.GROUNDING: [ROIType.STATE],
        DeviceType.BUSBAR: [ROIType.DETECTION],
        DeviceType.INSULATOR: [ROIType.DETECTION],
        DeviceType.CAPACITOR: [ROIType.DETECTION, ROIType.INTRUSION],
        DeviceType.METER_ANALOG: [ROIType.READING],
        DeviceType.METER_DIGITAL: [ROIType.READING],
        DeviceType.INDICATOR: [ROIType.STATE],
        DeviceType.OIL_LEVEL: [ROIType.READING],
        DeviceType.SILICA_GEL: [ROIType.STATE],
        DeviceType.VALVE: [ROIType.STATE],
        DeviceType.SF6_GAUGE: [ROIType.READING],
        DeviceType.FENCE: [ROIType.INTRUSION],
    }
    
    def __init__(self, model_registry=None):
        """
        初始化检测器
        
        Args:
            model_registry: 模型注册表实例
        """
        self._model_registry = model_registry
        self._device_detector_model = "device_detector_yolov8"
        self._topology: Dict[str, DeviceTopology] = {}
        self._roi_cache: Dict[str, List[AutoROI]] = {}
        self._tracker = ROITracker()
        self._lock = threading.Lock()
    
    def detect_rois(
        self,
        image: np.ndarray,
        site_id: str,
        position_id: str,
        use_cache: bool = True,
        track: bool = True,
    ) -> List[AutoROI]:
        """
        检测图像中的设备并生成ROI
        
        Args:
            image: BGR图像
            site_id: 站点ID
            position_id: 点位ID
            use_cache: 是否使用缓存
            track: 是否启用跟踪
            
        Returns:
            自动生成的ROI列表
        """
        cache_key = f"{site_id}_{position_id}"
        
        # 检查缓存
        if use_cache and cache_key in self._roi_cache:
            cached_rois = self._roi_cache[cache_key]
            if track:
                return self._tracker.update(cached_rois, image)
            return cached_rois
        
        # 执行检测
        rois = self._detect_devices(image)
        
        # 关联设备拓扑
        rois = self._associate_topology(rois, site_id)
        
        # 更新缓存
        with self._lock:
            self._roi_cache[cache_key] = rois
        
        # 跟踪
        if track:
            rois = self._tracker.update(rois, image)
        
        return rois
    
    def _detect_devices(self, image: np.ndarray) -> List[AutoROI]:
        """使用模型检测设备"""
        rois = []
        
        # 如果有模型注册表，使用深度学习模型
        if self._model_registry is not None:
            try:
                # 修复: 使用正确的导入路径
                from platform_core.inference_engine import infer
                result = infer(self._device_detector_model, image)
                
                for i, det in enumerate(result.detections):
                    device_type = self.DEVICE_CLASS_MAP.get(
                        det["class_name"], 
                        DeviceType.TRANSFORMER
                    )
                    roi_types = self.DEVICE_ROI_MAP.get(device_type, [ROIType.DETECTION])
                    
                    for roi_type in roi_types:
                        roi = AutoROI(
                            roi_id=f"auto_{i}_{roi_type.value}",
                            device_type=device_type,
                            roi_type=roi_type,
                            bbox=self._normalize_bbox(det["bbox"], image.shape[:2]),
                            confidence=det["confidence"],
                            class_name=det["class_name"],
                        )
                        rois.append(roi)
                
                return rois
            except Exception as e:
                print(f"[AutoROI] 深度学习检测失败，回退到传统方法: {e}")
        
        # 回退：使用传统方法检测
        return self._detect_devices_traditional(image)
    
    def _detect_devices_traditional(self, image: np.ndarray) -> List[AutoROI]:
        """使用传统方法检测设备（回退方案）"""
        if cv2 is None:
            return []
        
        rois = []
        h, w = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤和分类轮廓
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 1000 or area > h * w * 0.5:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 1
            
            # 简单分类
            if 0.8 < aspect_ratio < 1.2:
                device_type = DeviceType.METER_ANALOG
                roi_type = ROIType.READING
            elif aspect_ratio > 2:
                device_type = DeviceType.BUSBAR
                roi_type = ROIType.DETECTION
            else:
                device_type = DeviceType.TRANSFORMER
                roi_type = ROIType.DETECTION
            
            roi = AutoROI(
                roi_id=f"trad_{i}",
                device_type=device_type,
                roi_type=roi_type,
                bbox={"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                confidence=0.6,
                class_name=device_type.value,
            )
            rois.append(roi)
        
        return rois
    
    def _normalize_bbox(self, bbox: Dict, image_shape: Tuple[int, ...]) -> Dict[str, float]:
        """归一化边界框坐标"""
        h, w = image_shape[0], image_shape[1]
        return {
            "x": bbox["x"] / w,
            "y": bbox["y"] / h,
            "width": bbox["width"] / w,
            "height": bbox["height"] / h,
        }
    
    def _associate_topology(self, rois: List[AutoROI], site_id: str) -> List[AutoROI]:
        """关联设备拓扑信息"""
        for roi in rois:
            # 查找匹配的设备拓扑
            for device_id, topology in self._topology.items():
                if topology.device_type == roi.device_type:
                    roi.parent_device_id = device_id
                    roi.metadata["device_name"] = topology.name
                    break
        return rois
    
    def register_topology(self, topology: DeviceTopology) -> None:
        """注册设备拓扑"""
        with self._lock:
            self._topology[topology.device_id] = topology
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """清除缓存"""
        with self._lock:
            if cache_key:
                self._roi_cache.pop(cache_key, None)
            else:
                self._roi_cache.clear()
    
    def convert_to_schema_roi(self, auto_roi: AutoROI) -> Dict[str, Any]:
        """转换为平台Schema的ROI格式"""
        return {
            "id": auto_roi.roi_id,
            "name": f"{auto_roi.class_name}_{auto_roi.roi_type.value}",
            "roi_type": auto_roi.roi_type.value,
            "bbox": {
                "x": auto_roi.bbox["x"],
                "y": auto_roi.bbox["y"],
                "width": auto_roi.bbox["width"],
                "height": auto_roi.bbox["height"],
            },
            "metadata": {
                "auto_detected": True,
                "confidence": auto_roi.confidence,
                "device_type": auto_roi.device_type.value,
                "parent_device_id": auto_roi.parent_device_id,
                **auto_roi.metadata,
            }
        }


class ROITracker:
    """
    ROI跟踪器
    
    在视频流中跟踪ROI，保持ID一致性
    """
    
    def __init__(self, iou_threshold: float = 0.5, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: Dict[str, Dict] = {}
        self._next_id = 0
    
    def update(self, rois: List[AutoROI], image: np.ndarray) -> List[AutoROI]:
        """更新跟踪"""
        if not self._tracks:
            # 初始化跟踪
            for roi in rois:
                track_id = f"track_{self._next_id}"
                self._next_id += 1
                roi.tracking_id = track_id
                self._tracks[track_id] = {
                    "roi": roi,
                    "age": 0,
                    "hits": 1,
                }
            return rois
        
        # 匹配现有跟踪
        matched_tracks = set()
        matched_rois = set()
        
        for i, roi in enumerate(rois):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self._tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._compute_iou(roi.bbox, track["roi"].bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id:
                matched_tracks.add(best_track_id)
                matched_rois.add(i)
                roi.tracking_id = best_track_id
                self._tracks[best_track_id]["roi"] = roi
                self._tracks[best_track_id]["age"] = 0
                self._tracks[best_track_id]["hits"] += 1
        
        # 创建新跟踪
        for i, roi in enumerate(rois):
            if i not in matched_rois:
                track_id = f"track_{self._next_id}"
                self._next_id += 1
                roi.tracking_id = track_id
                self._tracks[track_id] = {
                    "roi": roi,
                    "age": 0,
                    "hits": 1,
                }
        
        # 更新未匹配的跟踪
        for track_id in list(self._tracks.keys()):
            if track_id not in matched_tracks:
                self._tracks[track_id]["age"] += 1
                if self._tracks[track_id]["age"] > self.max_age:
                    del self._tracks[track_id]
        
        return rois
    
    def _compute_iou(self, box1: Dict, box2: Dict) -> float:
        """计算IoU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


# 便捷函数
_detector_instance: Optional[AutoROIDetector] = None

def get_auto_roi_detector() -> AutoROIDetector:
    """获取自动ROI检测器实例"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AutoROIDetector()
    return _detector_instance


def detect_rois(image: np.ndarray, site_id: str, position_id: str) -> List[AutoROI]:
    """检测ROI的便捷函数"""
    return get_auto_roi_detector().detect_rois(image, site_id, position_id)
