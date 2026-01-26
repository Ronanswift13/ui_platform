"""
电容器巡视增强检测器 V3.6
=========================

实现电容器巡视的AI检测功能:
- 电容器单元识别
- 倾斜/倒塌检测
- 缺失位置检测
- 熔丝状态检测
- 人员/动物入侵检测
- 围栏完整性检测

支持功能:
- 实时视频流检测
- 多目标跟踪
- 入侵事件确认
- 告警联动

版本: 3.6.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import time
import hashlib
import logging
import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


# =============================================================================
# 检测类型定义
# =============================================================================
class CapacitorDefectType(Enum):
    """电容器缺陷类型"""
    CAPACITOR_UNIT = "capacitor_unit"          # 电容器单元(正常)
    CAPACITOR_TILTED = "capacitor_tilted"      # 电容器倾斜
    CAPACITOR_FALLEN = "capacitor_fallen"      # 电容器倒塌
    CAPACITOR_MISSING = "capacitor_missing"    # 电容器缺失
    FUSE_BLOWN = "fuse_blown"                  # 熔丝熔断
    FUSE_NORMAL = "fuse_normal"                # 熔丝正常
    CONNECTION_LOOSE = "connection_loose"      # 连接松动
    FENCE_DAMAGE = "fence_damage"              # 围栏破损
    PERSON_INTRUSION = "person_intrusion"      # 人员入侵
    ANIMAL_INTRUSION = "animal_intrusion"      # 动物入侵
    VEHICLE_INTRUSION = "vehicle_intrusion"    # 车辆入侵
    UNKNOWN_OBJECT = "unknown_object"          # 未知物体


# 检测类别标签
DETECTION_LABELS = {
    CapacitorDefectType.CAPACITOR_UNIT: "电容器单元",
    CapacitorDefectType.CAPACITOR_TILTED: "电容器倾斜",
    CapacitorDefectType.CAPACITOR_FALLEN: "电容器倒塌",
    CapacitorDefectType.CAPACITOR_MISSING: "电容器缺失",
    CapacitorDefectType.FUSE_BLOWN: "熔丝熔断",
    CapacitorDefectType.FUSE_NORMAL: "熔丝正常",
    CapacitorDefectType.CONNECTION_LOOSE: "连接松动",
    CapacitorDefectType.FENCE_DAMAGE: "围栏破损",
    CapacitorDefectType.PERSON_INTRUSION: "人员入侵",
    CapacitorDefectType.ANIMAL_INTRUSION: "动物入侵",
    CapacitorDefectType.VEHICLE_INTRUSION: "车辆入侵",
    CapacitorDefectType.UNKNOWN_OBJECT: "未知物体",
}

# 告警级别映射
ALARM_LEVELS = {
    CapacitorDefectType.CAPACITOR_UNIT: "info",
    CapacitorDefectType.CAPACITOR_TILTED: "warning",
    CapacitorDefectType.CAPACITOR_FALLEN: "critical",
    CapacitorDefectType.CAPACITOR_MISSING: "error",
    CapacitorDefectType.FUSE_BLOWN: "error",
    CapacitorDefectType.FUSE_NORMAL: "info",
    CapacitorDefectType.CONNECTION_LOOSE: "warning",
    CapacitorDefectType.FENCE_DAMAGE: "warning",
    CapacitorDefectType.PERSON_INTRUSION: "critical",
    CapacitorDefectType.ANIMAL_INTRUSION: "warning",
    CapacitorDefectType.VEHICLE_INTRUSION: "critical",
    CapacitorDefectType.UNKNOWN_OBJECT: "warning",
}


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class CapacitorDetection:
    """电容器检测结果"""
    detection_type: CapacitorDefectType
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) normalized
    label: str = ""
    alarm_level: str = "info"
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.label:
            self.label = DETECTION_LABELS.get(self.detection_type, "未知")
        if self.alarm_level == "info":
            self.alarm_level = ALARM_LEVELS.get(self.detection_type, "info")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.detection_type.value,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "alarm_level": self.alarm_level,
            "track_id": self.track_id,
            "metadata": self.metadata
        }


@dataclass
class TrackedObject:
    """跟踪对象"""
    track_id: int
    detection_type: CapacitorDefectType
    bbox: Tuple[float, float, float, float]
    confidence: float
    first_seen: float
    last_seen: float
    frame_count: int = 1
    confirmed: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    def update(self, bbox: Tuple[float, float, float, float], 
               confidence: float, timestamp: float):
        """更新跟踪状态"""
        old_center = (self.bbox[0] + self.bbox[2]/2, self.bbox[1] + self.bbox[3]/2)
        new_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
        
        dt = timestamp - self.last_seen
        if dt > 0:
            self.velocity = (
                (new_center[0] - old_center[0]) / dt,
                (new_center[1] - old_center[1]) / dt
            )
        
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.frame_count += 1
        
        # 确认逻辑：连续检测超过N帧
        if self.frame_count >= 3:
            self.confirmed = True


# =============================================================================
# 简单跟踪器
# =============================================================================
class SimpleTracker:
    """简单的多目标跟踪器"""
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_disappeared: int = 10,
                 confirm_frames: int = 3):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.confirm_frames = confirm_frames
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.disappeared: Dict[int, int] = {}
    
    def update(self, detections: List[CapacitorDetection], 
               timestamp: float) -> List[CapacitorDetection]:
        """更新跟踪状态"""
        if not detections:
            # 标记所有轨迹为消失
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return []
        
        # 匹配检测与轨迹
        if not self.tracks:
            # 初始化新轨迹
            for det in detections:
                self._create_track(det, timestamp)
        else:
            # 计算IoU矩阵
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(detections), len(track_ids)))
            
            for i, det in enumerate(detections):
                for j, tid in enumerate(track_ids):
                    iou_matrix[i, j] = self._calculate_iou(
                        det.bbox, self.tracks[tid].bbox
                    )
            
            # 贪心匹配
            matched_dets = set()
            matched_tracks = set()
            
            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                
                if max_iou < self.iou_threshold:
                    break
                
                det_idx, track_idx = max_idx
                track_id = track_ids[track_idx]
                
                # 更新轨迹
                self.tracks[track_id].update(
                    detections[det_idx].bbox,
                    detections[det_idx].confidence,
                    timestamp
                )
                detections[det_idx].track_id = track_id
                self.disappeared[track_id] = 0
                
                matched_dets.add(det_idx)
                matched_tracks.add(track_idx)
                
                # 清除已匹配的行列
                iou_matrix[det_idx, :] = -1
                iou_matrix[:, track_idx] = -1
            
            # 处理未匹配的检测（创建新轨迹）
            for i, det in enumerate(detections):
                if i not in matched_dets:
                    self._create_track(det, timestamp)
            
            # 处理未匹配的轨迹（标记消失）
            for j, tid in enumerate(track_ids):
                if j not in matched_tracks:
                    self.disappeared[tid] = self.disappeared.get(tid, 0) + 1
                    if self.disappeared[tid] > self.max_disappeared:
                        del self.tracks[tid]
                        del self.disappeared[tid]
        
        # 返回带有track_id的检测
        return detections
    
    def _create_track(self, detection: CapacitorDetection, timestamp: float):
        """创建新轨迹"""
        track = TrackedObject(
            track_id=self.next_id,
            detection_type=detection.detection_type,
            bbox=detection.bbox,
            confidence=detection.confidence,
            first_seen=timestamp,
            last_seen=timestamp
        )
        self.tracks[self.next_id] = track
        self.disappeared[self.next_id] = 0
        detection.track_id = self.next_id
        self.next_id += 1
    
    def _calculate_iou(self, 
                       box1: Tuple[float, float, float, float],
                       box2: Tuple[float, float, float, float]) -> float:
        """计算IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def get_confirmed_intrusions(self) -> List[TrackedObject]:
        """获取已确认的入侵事件"""
        intrusion_types = {
            CapacitorDefectType.PERSON_INTRUSION,
            CapacitorDefectType.ANIMAL_INTRUSION,
            CapacitorDefectType.VEHICLE_INTRUSION,
            CapacitorDefectType.UNKNOWN_OBJECT
        }
        
        return [
            track for track in self.tracks.values()
            if track.confirmed and track.detection_type in intrusion_types
        ]


# =============================================================================
# 倾斜检测器
# =============================================================================
class TiltDetector:
    """倾斜检测器"""
    
    def __init__(self, 
                 tilt_threshold_degrees: float = 5.0,
                 fallen_threshold_degrees: float = 45.0):
        self.tilt_threshold = tilt_threshold_degrees
        self.fallen_threshold = fallen_threshold_degrees
    
    def analyze_tilt(self, image: np.ndarray, 
                     capacitor_bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """分析电容器倾斜角度"""
        if cv2 is None:
            return {"tilt_angle": 0, "status": "normal"}
        
        h, w = image.shape[:2]
        x, y, bw, bh = capacitor_bbox
        
        # 提取ROI
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + bw) * w)
        y2 = int((y + bh) * h)
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {"tilt_angle": 0, "status": "normal"}
        
        # 边缘检测
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return {"tilt_angle": 0, "status": "normal"}
        
        # 计算主要垂直线的角度
        vertical_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            
            # 筛选接近垂直的线
            if abs(abs(angle) - 90) < 30:
                vertical_angles.append(90 - abs(angle))
        
        if not vertical_angles:
            return {"tilt_angle": 0, "status": "normal"}
        
        # 平均倾斜角度
        avg_tilt = np.mean(vertical_angles)
        
        # 判断状态
        if abs(avg_tilt) >= self.fallen_threshold:
            status = "fallen"
        elif abs(avg_tilt) >= self.tilt_threshold:
            status = "tilted"
        else:
            status = "normal"
        
        return {
            "tilt_angle": round(avg_tilt, 2),
            "status": status,
            "confidence": min(0.5 + abs(avg_tilt) / 90, 0.95)
        }


# =============================================================================
# 电容器增强检测器
# =============================================================================
class CapacitorEnhancedDetector:
    """电容器增强检测器
    
    集成功能:
    - 电容器单元检测
    - 倾斜/倒塌分析
    - 入侵检测与跟踪
    - 多目标跟踪
    """
    
    def __init__(self,
                 structure_model_path: Optional[str] = None,
                 intrusion_model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 enable_tracking: bool = True):
        
        self.confidence_threshold = confidence_threshold
        self.enable_tracking = enable_tracking
        
        # 组件初始化
        self.tilt_detector = TiltDetector()
        
        if enable_tracking:
            self.tracker = SimpleTracker()
        else:
            self.tracker = None
        
        # ONNX模型
        self.structure_model = None
        self.intrusion_model = None
        
        if structure_model_path and ort is not None:
            try:
                self.structure_model = ort.InferenceSession(structure_model_path)
            except Exception as e:
                logger.warning(f"结构检测模型加载失败: {e}")
        
        if intrusion_model_path and ort is not None:
            try:
                self.intrusion_model = ort.InferenceSession(intrusion_model_path)
            except Exception as e:
                logger.warning(f"入侵检测模型加载失败: {e}")
        
        self._initialized = True
        logger.info("[CapacitorDetector] 增强检测器初始化完成")
    
    def detect(self,
               image: np.ndarray,
               detection_types: Optional[List[str]] = None,
               confidence_threshold: Optional[float] = None,
               enable_intrusion: bool = True) -> Dict[str, Any]:
        """执行检测"""
        start_time = time.time()
        conf_thresh = confidence_threshold or self.confidence_threshold
        timestamp = time.time()
        
        all_detections: List[CapacitorDetection] = []
        
        # 1. 电容器结构检测
        structure_types = ["capacitor_unit", "capacitor_tilted", 
                         "capacitor_fallen", "capacitor_missing"]
        if not detection_types or any(t in detection_types for t in structure_types):
            structure_detections = self._detect_structure(image, conf_thresh)
            all_detections.extend(structure_detections)
            
            # 对检测到的电容器进行倾斜分析
            for det in structure_detections:
                if det.detection_type == CapacitorDefectType.CAPACITOR_UNIT:
                    tilt_result = self.tilt_detector.analyze_tilt(image, det.bbox)
                    
                    if tilt_result["status"] == "fallen":
                        det.detection_type = CapacitorDefectType.CAPACITOR_FALLEN
                        det.label = DETECTION_LABELS[CapacitorDefectType.CAPACITOR_FALLEN]
                        det.alarm_level = "critical"
                        det.metadata["tilt_angle"] = tilt_result["tilt_angle"]
                    elif tilt_result["status"] == "tilted":
                        det.detection_type = CapacitorDefectType.CAPACITOR_TILTED
                        det.label = DETECTION_LABELS[CapacitorDefectType.CAPACITOR_TILTED]
                        det.alarm_level = "warning"
                        det.metadata["tilt_angle"] = tilt_result["tilt_angle"]
        
        # 2. 入侵检测
        intrusion_types = ["person_intrusion", "animal_intrusion", 
                          "vehicle_intrusion", "unknown_object"]
        if enable_intrusion and (not detection_types or 
                                  any(t in detection_types for t in intrusion_types)):
            intrusion_detections = self._detect_intrusion(image, conf_thresh)
            all_detections.extend(intrusion_detections)
        
        # 3. 多目标跟踪
        if self.tracker and self.enable_tracking:
            all_detections = self.tracker.update(all_detections, timestamp)
        
        # 4. 生成告警
        alarms = self._generate_alarms(all_detections)
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "detections": [d.to_dict() for d in all_detections],
            "alarms": alarms,
            "inference_time_ms": round(inference_time, 2),
            "metadata": {
                "image_size": [image.shape[1], image.shape[0]],
                "detection_count": len(all_detections),
                "tracking_enabled": self.enable_tracking,
                "confirmed_intrusions": len(self.tracker.get_confirmed_intrusions()) if self.tracker else 0
            }
        }
    
    def _detect_structure(self, image: np.ndarray,
                         confidence_threshold: float) -> List[CapacitorDetection]:
        """检测电容器结构"""
        # 如果有ONNX模型，使用模型推理
        if self.structure_model:
            return self._model_inference(self.structure_model, image, 
                                        confidence_threshold, "structure")
        
        # 模拟检测
        return self._simulate_structure_detection(image, confidence_threshold)
    
    def _detect_intrusion(self, image: np.ndarray,
                         confidence_threshold: float) -> List[CapacitorDetection]:
        """检测入侵"""
        # 如果有ONNX模型，使用模型推理
        if self.intrusion_model:
            return self._model_inference(self.intrusion_model, image,
                                        confidence_threshold, "intrusion")
        
        # 模拟检测
        return self._simulate_intrusion_detection(image, confidence_threshold)
    
    def _model_inference(self, session, image: np.ndarray,
                        confidence_threshold: float,
                        detection_mode: str) -> List[CapacitorDetection]:
        """ONNX模型推理"""
        # 预处理
        input_size = (640, 640)
        resized = cv2.resize(image, input_size) if cv2 else np.zeros((*input_size, 3))
        
        if cv2:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = resized
        
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        # 推理
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: batched})
        
        # 后处理
        detections = []
        if outputs and len(outputs) > 0:
            output = outputs[0]
            if len(output.shape) == 3:
                output = output[0]
            
            for det in output:
                if len(det) < 5:
                    continue
                
                x, y, w, h = det[:4]
                class_scores = det[4:]
                
                if len(class_scores) > 0:
                    class_id = int(np.argmax(class_scores))
                    confidence = float(class_scores[class_id])
                    
                    if confidence >= confidence_threshold:
                        det_type = self._map_class_to_type(class_id, detection_mode)
                        
                        detections.append(CapacitorDetection(
                            detection_type=det_type,
                            confidence=confidence,
                            bbox=(float(x)/input_size[0], float(y)/input_size[1],
                                  float(w)/input_size[0], float(h)/input_size[1]),
                            metadata={"class_id": class_id, "source": "onnx"}
                        ))
        
        return detections
    
    def _map_class_to_type(self, class_id: int, mode: str) -> CapacitorDefectType:
        """映射类别ID到检测类型"""
        if mode == "structure":
            mapping = {
                0: CapacitorDefectType.CAPACITOR_UNIT,
                1: CapacitorDefectType.CAPACITOR_TILTED,
                2: CapacitorDefectType.CAPACITOR_FALLEN,
                3: CapacitorDefectType.CAPACITOR_MISSING,
                4: CapacitorDefectType.FUSE_BLOWN,
                5: CapacitorDefectType.FUSE_NORMAL,
            }
        else:  # intrusion
            mapping = {
                0: CapacitorDefectType.PERSON_INTRUSION,
                1: CapacitorDefectType.ANIMAL_INTRUSION,
                2: CapacitorDefectType.VEHICLE_INTRUSION,
                3: CapacitorDefectType.UNKNOWN_OBJECT,
            }
        
        return mapping.get(class_id, CapacitorDefectType.UNKNOWN_OBJECT)
    
    def _simulate_structure_detection(self, image: np.ndarray,
                                      confidence_threshold: float) -> List[CapacitorDetection]:
        """模拟结构检测"""
        import random
        
        detections = []
        
        # 随机生成电容器单元检测
        num_units = random.randint(3, 8)
        for i in range(num_units):
            x = 0.1 + (i % 4) * 0.2
            y = 0.2 + (i // 4) * 0.3
            
            det_type = CapacitorDefectType.CAPACITOR_UNIT
            confidence = random.uniform(0.7, 0.95)
            
            # 随机生成异常
            if random.random() < 0.1:  # 10%概率倾斜
                det_type = CapacitorDefectType.CAPACITOR_TILTED
                confidence = random.uniform(0.6, 0.85)
            elif random.random() < 0.05:  # 5%概率缺失
                det_type = CapacitorDefectType.CAPACITOR_MISSING
                confidence = random.uniform(0.5, 0.8)
            
            if confidence >= confidence_threshold:
                detections.append(CapacitorDetection(
                    detection_type=det_type,
                    confidence=confidence,
                    bbox=(x, y, 0.15, 0.25),
                    metadata={"unit_id": i, "source": "simulation"}
                ))
        
        return detections
    
    def _simulate_intrusion_detection(self, image: np.ndarray,
                                      confidence_threshold: float) -> List[CapacitorDetection]:
        """模拟入侵检测"""
        import random
        
        detections = []
        
        # 随机生成入侵检测
        if random.random() < 0.2:  # 20%概率有入侵
            det_types = [
                (CapacitorDefectType.PERSON_INTRUSION, 0.6),
                (CapacitorDefectType.ANIMAL_INTRUSION, 0.3),
                (CapacitorDefectType.UNKNOWN_OBJECT, 0.1),
            ]
            
            det_type, _ = random.choices(det_types, weights=[p for _, p in det_types])[0]
            confidence = random.uniform(0.5, 0.9)
            
            if confidence >= confidence_threshold:
                detections.append(CapacitorDetection(
                    detection_type=det_type,
                    confidence=confidence,
                    bbox=(random.uniform(0.1, 0.7), random.uniform(0.1, 0.7),
                          random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)),
                    metadata={"source": "simulation"}
                ))
        
        return detections
    
    def _generate_alarms(self, detections: List[CapacitorDetection]) -> List[Dict[str, Any]]:
        """生成告警"""
        alarms = []
        
        for det in detections:
            if det.alarm_level in ["warning", "error", "critical"]:
                alarms.append({
                    "type": det.detection_type.value,
                    "level": det.alarm_level,
                    "message": f"检测到{det.label}，置信度{det.confidence:.1%}",
                    "timestamp": datetime.now().isoformat(),
                    "track_id": det.track_id,
                    "detection_id": hashlib.md5(
                        f"{det.detection_type.value}{det.confidence}{time.time()}".encode()
                    ).hexdigest()[:8]
                })
        
        return alarms


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    "CapacitorDefectType",
    "CapacitorDetection",
    "TrackedObject",
    "SimpleTracker",
    "TiltDetector",
    "CapacitorEnhancedDetector",
    "DETECTION_LABELS",
    "ALARM_LEVELS",
]
