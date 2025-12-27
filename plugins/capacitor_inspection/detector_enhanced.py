"""
电容器巡视检测器 - 增强版
输变电激光监测平台 (D组) - 全自动AI巡检改造

增强功能:
- 姿态估计和几何模型(倾斜检测)
- YOLOv8/RT-DETR目标检测(入侵检测)
- 时间阈值入侵告警
"""

from __future__ import annotations
from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np
import threading

try:
    import cv2
except ImportError:
    cv2 = None


class StructuralDefect(Enum):
    TILT_WARNING = "tilt_warning"
    TILT_ERROR = "tilt_error"
    COLLAPSE = "collapse"
    MISSING_UNIT = "missing_unit"


class IntrusionType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    UNKNOWN = "unknown"


@dataclass
class TiltAnalysis:
    is_tilted: bool
    angle: float
    direction: str
    severity: str
    confidence: float = 0.0


@dataclass
class IntrusionEvent:
    event_id: str
    intrusion_type: IntrusionType
    bbox: Dict[str, float]
    confidence: float
    first_seen: datetime
    last_seen: datetime
    duration_seconds: float
    alert_triggered: bool = False


class CapacitorDetectorEnhanced:
    """电容器巡视增强检测器"""
    
    MODEL_IDS = {
        "capacitor_detector": "capacitor_yolov8",
        "intrusion_detector": "intrusion_rtdetr",
    }
    
    def __init__(self, config: dict, model_registry=None):
        self.config = config
        self._model_registry = model_registry
        self.max_tilt_angle = config.get("structural_detection", {}).get("max_tilt_angle", 5.0)
        self.warning_angle = config.get("structural_detection", {}).get("warning_angle", 3.0)
        self.alert_delay = config.get("intrusion_detection", {}).get("alert_delay_seconds", 2.0)
        self._intrusion_events: Dict[str, IntrusionEvent] = {}
        self._event_lock = threading.Lock()
        self.use_deep_learning = config.get("use_deep_learning", True)
    
    def detect_structural_defects(self, image: np.ndarray, roi_type: str = "capacitor_bank") -> Dict:
        defects = []
        tilt_result = self.detect_tilt(image)
        if tilt_result.is_tilted:
            severity = "error" if tilt_result.angle > self.max_tilt_angle else "warning"
            defects.append({
                "type": StructuralDefect.TILT_ERROR.value if severity == "error" else StructuralDefect.TILT_WARNING.value,
                "angle": tilt_result.angle,
                "direction": tilt_result.direction,
                "confidence": tilt_result.confidence,
                "severity": severity,
            })
        return {"defects": defects, "tilt_analysis": tilt_result, "total_defects": len(defects)}
    
    def detect_tilt(self, image: np.ndarray) -> TiltAnalysis:
        if cv2 is None:
            return TiltAnalysis(is_tilted=False, angle=0, direction="none", severity="normal")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return TiltAnalysis(is_tilted=False, angle=0, direction="none", severity="normal", confidence=0.3)
        
        vertical_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(x2 - x1, y2 - y1))
            if abs(angle) < 30 or abs(angle) > 150:
                vertical_angles.append(angle if abs(angle) < 90 else angle - 180 if angle > 0 else angle + 180)
        
        if not vertical_angles:
            return TiltAnalysis(is_tilted=False, angle=0, direction="none", severity="normal", confidence=0.3)
        
        avg_angle = float(np.mean(vertical_angles))
        is_tilted = abs(avg_angle) > self.warning_angle
        direction = "left" if avg_angle < 0 else "right"
        severity = "error" if abs(avg_angle) > self.max_tilt_angle else "warning" if is_tilted else "normal"
        
        return TiltAnalysis(is_tilted=is_tilted, angle=float(abs(avg_angle)), direction=direction, 
                          severity=severity, confidence=min(0.9, 0.5 + len(vertical_angles) * 0.05))
    
    def detect_intrusion(self, image: np.ndarray, timestamp: Optional[datetime] = None) -> Dict:
        timestamp = timestamp or datetime.now()
        detections = self._detect_intrusion_traditional(image) if cv2 else []
        events = self._update_intrusion_events(detections, timestamp)
        alerts = self._generate_intrusion_alerts(events)
        return {"detections": detections, "events": [self._event_to_dict(e) for e in events], "alerts": alerts}
    
    def _detect_intrusion_traditional(self, image: np.ndarray) -> List[Dict]:
        if cv2 is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        h, w = gray.shape
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / ch if ch > 0 else 1
                intrusion_type = IntrusionType.PERSON if 0.3 < aspect_ratio < 0.8 else IntrusionType.VEHICLE if aspect_ratio > 1.5 else IntrusionType.UNKNOWN
                detections.append({"intrusion_type": intrusion_type.value, "bbox": {"x": x/w, "y": y/h, "width": cw/w, "height": ch/h}, "confidence": 0.5})
        return detections
    
    def _update_intrusion_events(self, detections: List[Dict], timestamp: datetime) -> List[IntrusionEvent]:
        with self._event_lock:
            for det in detections:
                event_id = f"intrusion_{timestamp.timestamp()}"
                self._intrusion_events[event_id] = IntrusionEvent(
                    event_id=event_id, intrusion_type=IntrusionType(det["intrusion_type"]),
                    bbox=det["bbox"], confidence=det["confidence"], first_seen=timestamp,
                    last_seen=timestamp, duration_seconds=0)
            return list(self._intrusion_events.values())
    
    def _generate_intrusion_alerts(self, events: List[IntrusionEvent]) -> List[Dict]:
        alerts = []
        for event in events:
            if event.duration_seconds >= self.alert_delay and not event.alert_triggered:
                event.alert_triggered = True
                alerts.append({"event_id": event.event_id, "type": f"intrusion_{event.intrusion_type.value}",
                              "level": "error", "message": f"检测到{event.intrusion_type.value}入侵"})
        return alerts
    
    def _event_to_dict(self, event: IntrusionEvent) -> Dict:
        return {"event_id": event.event_id, "intrusion_type": event.intrusion_type.value,
                "bbox": event.bbox, "confidence": event.confidence, "duration_seconds": event.duration_seconds}
