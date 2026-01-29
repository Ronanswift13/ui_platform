#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消防监测插件 V1.0
==================

基于迭代方案实现室内消防监测:
- 深度学习火焰/烟雾检测 (YOLOv8)
- 多模态融合 (视觉+温度+烟雾传感器)
- DeepSORT火势跟踪
- 分级报警系统

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from collections import deque
import threading
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 类型定义
# =============================================================================
class FireType(Enum):
    """火情类型"""
    FLAME = "flame"             # 明火
    SMOKE = "smoke"             # 烟雾
    SPARK = "spark"             # 火花
    SMOLDER = "smolder"         # 阴燃
    NONE = "none"               # 无


class AlarmLevel(Enum):
    """报警级别"""
    NORMAL = "normal"           # 正常
    ATTENTION = "attention"     # 关注
    WARNING = "warning"         # 预警
    ALARM = "alarm"            # 报警
    CRITICAL = "critical"       # 紧急


class SensorType(Enum):
    """传感器类型"""
    VISUAL = "visual"           # 视觉
    THERMAL = "thermal"         # 热成像
    SMOKE_SENSOR = "smoke_sensor"  # 烟雾传感器
    TEMP_SENSOR = "temp_sensor"    # 温度传感器


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class FireDetection:
    """火情检测结果"""
    detection_id: str
    fire_type: FireType
    confidence: float
    bbox: Dict[str, float]              # {x, y, w, h}
    position: Optional[Tuple[float, float]] = None  # 地面位置
    
    # 温度信息
    temperature: Optional[float] = None  # 检测区域温度
    temp_delta: Optional[float] = None   # 温度变化率 (°C/s)
    
    # 烟雾信息
    smoke_density: Optional[float] = None  # 烟雾浓度
    
    # 跟踪
    track_id: Optional[str] = None
    velocity: Tuple[float, float] = (0, 0)  # 扩散速度
    spread_rate: float = 0.0              # 扩散速率 (m²/s)
    
    # 评估
    alarm_level: AlarmLevel = AlarmLevel.NORMAL
    risk_score: float = 0.0
    
    # 时间
    timestamp: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    duration: float = 0.0
    
    # 元数据
    camera_id: str = ""
    zone_id: Optional[str] = None
    sources: List[SensorType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "detection_id": self.detection_id,
            "fire_type": self.fire_type.value,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "position": {"x": self.position[0], "y": self.position[1]} if self.position else None,
            "temperature": self.temperature,
            "temp_delta": self.temp_delta,
            "smoke_density": self.smoke_density,
            "track_id": self.track_id,
            "spread_rate": self.spread_rate,
            "alarm_level": self.alarm_level.value,
            "risk_score": self.risk_score,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "sources": [s.value for s in self.sources],
        }


@dataclass
class SensorReading:
    """传感器读数"""
    sensor_id: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    location: Optional[Dict[str, float]] = None  # {x, y, z}
    is_alarm: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "location": self.location,
            "is_alarm": self.is_alarm,
        }


@dataclass
class FireZoneStatus:
    """区域火情状态"""
    zone_id: str
    zone_name: str
    alarm_level: AlarmLevel
    fire_detected: bool = False
    smoke_detected: bool = False
    temperature: Optional[float] = None
    smoke_density: Optional[float] = None
    last_update: float = field(default_factory=time.time)
    active_detections: List[str] = field(default_factory=list)


@dataclass
class FireEvent:
    """火情事件"""
    event_id: str
    event_type: str         # fire_detected, smoke_detected, alarm_triggered, alarm_cleared
    fire_type: FireType
    alarm_level: AlarmLevel
    location: Dict[str, Any]
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 火焰/烟雾检测器
# =============================================================================
class FireSmokeDetector:
    """
    火焰/烟雾检测器
    
    基于YOLOv8进行火焰和烟雾检测
    """
    
    # 类别映射
    CLASS_MAPPING = {
        0: FireType.FLAME,
        1: FireType.SMOKE,
        2: FireType.SPARK,
        3: FireType.SMOLDER,
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        use_gpu: bool = False,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_gpu = use_gpu
        
        self._model = None
        self._initialized = False
        self._detection_count = 0
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            if self.model_path and Path(self.model_path).exists():
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if self.use_gpu else ['CPUExecutionProvider']
                self._model = ort.InferenceSession(self.model_path, providers=providers)
                self._initialized = True
                logger.info(f"[FireDetector] 模型加载成功: {self.model_path}")
            else:
                logger.info("[FireDetector] 未指定模型，使用模拟检测")
                self._initialized = True
        except Exception as e:
            logger.warning(f"[FireDetector] 模型加载失败: {e}")
            self._initialized = True
    
    def detect(self, image: np.ndarray, camera_id: str = "") -> List[FireDetection]:
        """检测火焰和烟雾"""
        if not self._initialized:
            return []
        
        try:
            if self._model is not None:
                return self._detect_with_model(image, camera_id)
            else:
                return self._detect_simulated(image, camera_id)
        except Exception as e:
            logger.error(f"[FireDetector] 检测失败: {e}")
            return []
    
    def _detect_with_model(self, image: np.ndarray, camera_id: str) -> List[FireDetection]:
        """使用模型检测"""
        import cv2
        
        h, w = image.shape[:2]
        img = cv2.resize(image, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        inputs = {self._model.get_inputs()[0].name: img}
        outputs = self._model.run(None, inputs)
        
        predictions = outputs[0][0]
        
        detections = []
        for pred in predictions:
            if len(pred) >= 6:
                x1, y1, x2, y2, conf, cls = pred[:6]
            else:
                continue
            
            if conf < self.confidence_threshold:
                continue
            
            cls_id = int(cls)
            fire_type = self.CLASS_MAPPING.get(cls_id, FireType.NONE)
            if fire_type == FireType.NONE:
                continue
            
            scale_x, scale_y = w / self.input_size[0], h / self.input_size[1]
            bbox = {
                'x': float(x1 * scale_x),
                'y': float(y1 * scale_y),
                'w': float((x2 - x1) * scale_x),
                'h': float((y2 - y1) * scale_y),
            }
            
            self._detection_count += 1
            detection = FireDetection(
                detection_id=f"FD{self._detection_count:06d}",
                fire_type=fire_type,
                confidence=float(conf),
                bbox=bbox,
                camera_id=camera_id,
                sources=[SensorType.VISUAL],
            )
            detections.append(detection)
        
        return self._apply_nms(detections)
    
    def _detect_simulated(self, image: np.ndarray, camera_id: str) -> List[FireDetection]:
        """模拟检测"""
        # 5%概率检测到火情
        if np.random.random() > 0.05:
            return []
        
        h, w = image.shape[:2]
        fire_type = np.random.choice([FireType.FLAME, FireType.SMOKE])
        
        x = np.random.uniform(0.1, 0.7) * w
        y = np.random.uniform(0.1, 0.7) * h
        box_w = np.random.uniform(50, 200)
        box_h = np.random.uniform(50, 200)
        
        self._detection_count += 1
        detection = FireDetection(
            detection_id=f"FD{self._detection_count:06d}",
            fire_type=fire_type,
            confidence=np.random.uniform(0.5, 0.95),
            bbox={'x': x, 'y': y, 'w': box_w, 'h': box_h},
            camera_id=camera_id,
            sources=[SensorType.VISUAL],
        )
        
        return [detection]
    
    def _apply_nms(self, detections: List[FireDetection]) -> List[FireDetection]:
        """非极大值抑制"""
        if len(detections) <= 1:
            return detections
        
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [
                d for d in detections
                if self._compute_iou(best.bbox, d.bbox) < self.nms_threshold
            ]
        
        return keep
    
    def _compute_iou(self, box1: Dict, box2: Dict) -> float:
        """计算IoU"""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
        y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1['w'] * box1['h']
        area2 = box2['w'] * box2['h']
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


# =============================================================================
# 多模态融合器
# =============================================================================
class MultiModalFusion:
    """
    多模态融合器
    
    融合视觉检测、温度传感器、烟雾传感器数据
    """
    
    def __init__(
        self,
        temp_threshold: float = 50.0,       # 温度报警阈值 (°C)
        temp_delta_threshold: float = 5.0,  # 温度变化率阈值 (°C/s)
        smoke_threshold: float = 0.3,       # 烟雾浓度阈值
        fusion_radius: float = 2.0,         # 融合半径 (米)
    ):
        self.temp_threshold = temp_threshold
        self.temp_delta_threshold = temp_delta_threshold
        self.smoke_threshold = smoke_threshold
        self.fusion_radius = fusion_radius
        
        # 传感器数据缓存
        self._sensor_data: Dict[str, deque] = {}
    
    def fuse(
        self,
        visual_detections: List[FireDetection],
        sensor_readings: List[SensorReading],
        thermal_image: Optional[np.ndarray] = None,
    ) -> List[FireDetection]:
        """
        融合多模态数据
        
        Args:
            visual_detections: 视觉检测结果
            sensor_readings: 传感器读数列表
            thermal_image: 热成像图像
            
        Returns:
            融合后的检测结果
        """
        # 更新传感器缓存
        for reading in sensor_readings:
            if reading.sensor_id not in self._sensor_data:
                self._sensor_data[reading.sensor_id] = deque(maxlen=100)
            self._sensor_data[reading.sensor_id].append(reading)
        
        # 提取异常传感器
        temp_alarms = [r for r in sensor_readings 
                       if r.sensor_type == SensorType.TEMP_SENSOR and r.value > self.temp_threshold]
        smoke_alarms = [r for r in sensor_readings 
                        if r.sensor_type == SensorType.SMOKE_SENSOR and r.value > self.smoke_threshold]
        
        # 融合视觉检测
        for det in visual_detections:
            # 融合热成像温度
            if thermal_image is not None:
                temp = self._get_thermal_temperature(thermal_image, det.bbox)
                det.temperature = temp
                det.temp_delta = self._compute_temp_delta(det.camera_id, temp)
                if SensorType.THERMAL not in det.sources:
                    det.sources.append(SensorType.THERMAL)
            
            # 融合温度传感器
            for alarm in temp_alarms:
                if self._is_nearby(det, alarm):
                    det.temperature = alarm.value
                    if SensorType.TEMP_SENSOR not in det.sources:
                        det.sources.append(SensorType.TEMP_SENSOR)
            
            # 融合烟雾传感器
            for alarm in smoke_alarms:
                if self._is_nearby(det, alarm):
                    det.smoke_density = alarm.value
                    if SensorType.SMOKE_SENSOR not in det.sources:
                        det.sources.append(SensorType.SMOKE_SENSOR)
        
        # 从传感器数据创建新检测 (如果没有视觉检测但传感器报警)
        if len(visual_detections) == 0:
            for alarm in smoke_alarms:
                if alarm.value > self.smoke_threshold * 1.5:  # 高浓度烟雾
                    det = self._create_sensor_detection(alarm, FireType.SMOKE)
                    visual_detections.append(det)
        
        return visual_detections
    
    def _get_thermal_temperature(self, thermal: np.ndarray, bbox: Dict) -> float:
        """从热成像获取温度"""
        try:
            x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
            region = thermal[y:y+h, x:x+w]
            if region.size > 0:
                return float(np.max(region))
        except:
            pass
        return 0.0
    
    def _compute_temp_delta(self, sensor_id: str, current_temp: float) -> float:
        """计算温度变化率"""
        if sensor_id not in self._sensor_data:
            return 0.0
        
        history = self._sensor_data[sensor_id]
        if len(history) < 2:
            return 0.0
        
        # 取最近5秒的数据
        recent = [r for r in history if time.time() - r.timestamp < 5]
        if len(recent) < 2:
            return 0.0
        
        # 线性拟合计算变化率
        times = [r.timestamp - recent[0].timestamp for r in recent]
        values = [r.value for r in recent]
        
        if max(times) > 0:
            return (values[-1] - values[0]) / max(times)
        return 0.0
    
    def _is_nearby(self, detection: FireDetection, reading: SensorReading) -> bool:
        """判断传感器是否在检测区域附近"""
        if detection.position is None or reading.location is None:
            return True  # 位置未知时假设相关
        
        dx = detection.position[0] - reading.location.get('x', 0)
        dy = detection.position[1] - reading.location.get('y', 0)
        dist = np.sqrt(dx*dx + dy*dy)
        
        return dist < self.fusion_radius
    
    def _create_sensor_detection(self, reading: SensorReading, fire_type: FireType) -> FireDetection:
        """从传感器读数创建检测"""
        return FireDetection(
            detection_id=f"SD{int(time.time()*1000) % 1000000:06d}",
            fire_type=fire_type,
            confidence=0.7,
            bbox={'x': 0, 'y': 0, 'w': 100, 'h': 100},
            position=(reading.location.get('x', 0), reading.location.get('y', 0)) if reading.location else None,
            smoke_density=reading.value if reading.sensor_type == SensorType.SMOKE_SENSOR else None,
            temperature=reading.value if reading.sensor_type == SensorType.TEMP_SENSOR else None,
            camera_id="",
            sources=[reading.sensor_type],
        )


# =============================================================================
# 火势跟踪器
# =============================================================================
class FireTracker:
    """
    火势跟踪器
    
    跟踪火焰/烟雾的扩散
    """
    
    def __init__(
        self,
        max_age: int = 50,
        min_hits: int = 2,
        distance_threshold: float = 150.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        
        self._tracks: Dict[str, Dict] = {}
        self._next_id = 1
    
    def update(self, detections: List[FireDetection]) -> List[FireDetection]:
        """更新跟踪"""
        if len(detections) == 0:
            stale = []
            for track_id, track in self._tracks.items():
                track['age'] += 1
                if track['age'] > self.max_age:
                    stale.append(track_id)
            for tid in stale:
                self._tracks.pop(tid, None)
            return []
        
        track_ids = list(self._tracks.keys())
        
        if len(track_ids) == 0:
            for det in detections:
                track_id = self._create_track(det)
                det.track_id = track_id
            return detections
        
        # 匹配
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            last_pos = track['last_position']
            
            for j, det in enumerate(detections):
                det_pos = (det.bbox['x'] + det.bbox['w']/2, det.bbox['y'] + det.bbox['h']/2)
                dist = np.sqrt((last_pos[0] - det_pos[0])**2 + (last_pos[1] - det_pos[1])**2)
                cost_matrix[i, j] = dist
        
        matched = set()
        matched_tracks = set()
        
        for _ in range(min(len(track_ids), len(detections))):
            if cost_matrix.size == 0:
                break
            
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            if cost_matrix[min_idx] < self.distance_threshold:
                track_id = track_ids[min_idx[0]]
                det_idx = min_idx[1]
                
                det = detections[det_idx]
                det.track_id = track_id
                self._update_track(track_id, det)
                
                matched.add(det_idx)
                matched_tracks.add(track_id)
                
                cost_matrix[min_idx[0], :] = np.inf
                cost_matrix[:, min_idx[1]] = np.inf
            else:
                break
        
        for j, det in enumerate(detections):
            if j not in matched:
                track_id = self._create_track(det)
                det.track_id = track_id
        
        stale = []
        for track_id in self._tracks:
            if track_id not in matched_tracks:
                self._tracks[track_id]['age'] += 1
                if self._tracks[track_id]['age'] > self.max_age:
                    stale.append(track_id)
        for tid in stale:
            self._tracks.pop(tid, None)
        
        return detections
    
    def _create_track(self, detection: FireDetection) -> str:
        """创建新跟踪"""
        track_id = f"FT{self._next_id:05d}"
        self._next_id += 1
        
        pos = (detection.bbox['x'] + detection.bbox['w']/2,
               detection.bbox['y'] + detection.bbox['h']/2)
        area = detection.bbox['w'] * detection.bbox['h']
        
        self._tracks[track_id] = {
            'track_id': track_id,
            'fire_type': detection.fire_type,
            'first_seen': detection.timestamp,
            'last_seen': detection.timestamp,
            'last_position': pos,
            'last_area': area,
            'positions': deque([pos], maxlen=50),
            'areas': deque([area], maxlen=50),
            'age': 0,
            'hits': 1,
        }
        
        detection.first_seen = detection.timestamp
        return track_id
    
    def _update_track(self, track_id: str, detection: FireDetection):
        """更新跟踪"""
        track = self._tracks[track_id]
        
        pos = (detection.bbox['x'] + detection.bbox['w']/2,
               detection.bbox['y'] + detection.bbox['h']/2)
        area = detection.bbox['w'] * detection.bbox['h']
        
        # 计算速度和扩散率
        dt = detection.timestamp - track['last_seen']
        if dt > 0:
            last_pos = track['last_position']
            vx = (pos[0] - last_pos[0]) / dt
            vy = (pos[1] - last_pos[1]) / dt
            detection.velocity = (vx, vy)
            
            # 扩散率 = 面积变化率
            area_delta = (area - track['last_area']) / dt
            detection.spread_rate = max(0, area_delta / 10000)  # m²/s估计
        
        track['last_seen'] = detection.timestamp
        track['last_position'] = pos
        track['last_area'] = area
        track['positions'].append(pos)
        track['areas'].append(area)
        track['age'] = 0
        track['hits'] += 1
        
        detection.first_seen = track['first_seen']
        detection.duration = detection.timestamp - track['first_seen']
    
    def get_track_count(self) -> int:
        """获取跟踪数量"""
        return len(self._tracks)


# =============================================================================
# 报警评估器
# =============================================================================
class AlarmEvaluator:
    """
    报警评估器
    
    评估火情严重程度并确定报警级别
    """
    
    def __init__(
        self,
        confidence_weight: float = 0.3,
        temperature_weight: float = 0.25,
        smoke_weight: float = 0.2,
        spread_weight: float = 0.15,
        duration_weight: float = 0.1,
    ):
        self.weights = {
            'confidence': confidence_weight,
            'temperature': temperature_weight,
            'smoke': smoke_weight,
            'spread': spread_weight,
            'duration': duration_weight,
        }
    
    def evaluate(self, detection: FireDetection) -> Tuple[AlarmLevel, float]:
        """
        评估报警级别
        
        Returns:
            (alarm_level, risk_score)
        """
        scores = {}
        
        # 置信度得分
        scores['confidence'] = detection.confidence
        
        # 温度得分 (基于绝对温度)
        if detection.temperature:
            temp_score = min(1.0, max(0, (detection.temperature - 30) / 70))
            scores['temperature'] = temp_score
        else:
            scores['temperature'] = 0.3 if detection.fire_type == FireType.FLAME else 0.1
        
        # 烟雾得分
        if detection.smoke_density:
            scores['smoke'] = min(1.0, detection.smoke_density)
        else:
            scores['smoke'] = 0.3 if detection.fire_type == FireType.SMOKE else 0.1
        
        # 扩散率得分
        scores['spread'] = min(1.0, detection.spread_rate * 10)
        
        # 持续时间得分
        scores['duration'] = min(1.0, detection.duration / 60)  # 60秒达到最大
        
        # 加权求和
        risk_score = sum(scores[k] * self.weights[k] for k in self.weights)
        
        # 多源确认加成
        if len(detection.sources) > 1:
            risk_score = min(1.0, risk_score * 1.2)
        
        # 火焰比烟雾更危险
        if detection.fire_type == FireType.FLAME:
            risk_score = min(1.0, risk_score * 1.3)
        
        # 确定报警级别
        if risk_score >= 0.8:
            alarm_level = AlarmLevel.CRITICAL
        elif risk_score >= 0.6:
            alarm_level = AlarmLevel.ALARM
        elif risk_score >= 0.4:
            alarm_level = AlarmLevel.WARNING
        elif risk_score >= 0.2:
            alarm_level = AlarmLevel.ATTENTION
        else:
            alarm_level = AlarmLevel.NORMAL
        
        return alarm_level, risk_score


# =============================================================================
# 消防监测插件
# =============================================================================
class FireDetectionPlugin:
    """
    消防监测插件 V1.0
    
    功能:
    - 深度学习火焰/烟雾检测
    - 多模态数据融合
    - 火势跟踪
    - 分级报警
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        temp_threshold: float = 50.0,
        smoke_threshold: float = 0.3,
        use_gpu: bool = False,
        event_callback: Optional[Callable[[FireEvent], None]] = None,
    ):
        self.event_callback = event_callback
        
        # 初始化组件
        self._detector = FireSmokeDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu,
        )
        self._fusion = MultiModalFusion(
            temp_threshold=temp_threshold,
            smoke_threshold=smoke_threshold,
        )
        self._tracker = FireTracker()
        self._evaluator = AlarmEvaluator()
        
        # 区域状态
        self._zone_status: Dict[str, FireZoneStatus] = {}
        
        # 统计
        self._total_detections = 0
        self._total_events = 0
        self._alarm_count: Dict[AlarmLevel, int] = {level: 0 for level in AlarmLevel}
        
        # 历史
        self._detection_history: deque = deque(maxlen=1000)
        self._event_history: deque = deque(maxlen=500)
        
        self._initialized = True
        logger.info(f"[FireDetectionPlugin] 初始化完成 V{self.VERSION}")
    
    def process(
        self,
        image: np.ndarray,
        thermal_image: Optional[np.ndarray] = None,
        sensor_readings: Optional[List[SensorReading]] = None,
        camera_id: str = "",
        zone_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理单帧
        
        Args:
            image: 可见光图像
            thermal_image: 热成像图像
            sensor_readings: 传感器读数
            camera_id: 相机ID
            zone_id: 区域ID
            
        Returns:
            处理结果
        """
        result = {
            'success': True,
            'timestamp': time.time(),
            'detections': [],
            'events': [],
            'zone_status': None,
            'alarm_level': AlarmLevel.NORMAL.value,
            'statistics': {},
        }
        
        try:
            sensor_readings = sensor_readings or []
            
            # 1. 视觉检测
            detections = self._detector.detect(image, camera_id)
            
            # 2. 多模态融合
            detections = self._fusion.fuse(detections, sensor_readings, thermal_image)
            
            # 3. 跟踪
            detections = self._tracker.update(detections)
            
            # 4. 设置区域
            for det in detections:
                det.zone_id = zone_id
            
            # 5. 评估报警级别
            max_alarm_level = AlarmLevel.NORMAL
            for det in detections:
                alarm_level, risk_score = self._evaluator.evaluate(det)
                det.alarm_level = alarm_level
                det.risk_score = risk_score
                
                if alarm_level.value > max_alarm_level.value:
                    max_alarm_level = alarm_level
            
            # 6. 生成事件
            for det in detections:
                if det.duration < 0.5:  # 新检测
                    event = self._create_event(det, "fire_detected")
                    self._emit_event(event)
                    result['events'].append(event.details)
                
                if det.alarm_level in [AlarmLevel.ALARM, AlarmLevel.CRITICAL]:
                    event = self._create_event(det, "alarm_triggered")
                    self._emit_event(event)
                    result['events'].append(event.details)
            
            # 7. 更新区域状态
            if zone_id:
                self._update_zone_status(zone_id, detections)
                result['zone_status'] = self._zone_status.get(zone_id)
            
            # 8. 更新统计
            self._total_detections += len(detections)
            for det in detections:
                self._alarm_count[det.alarm_level] += 1
                self._detection_history.append(det.to_dict())
            
            # 构造返回
            result['detections'] = [d.to_dict() for d in detections]
            result['alarm_level'] = max_alarm_level.value
            result['statistics'] = self.get_statistics()
            
        except Exception as e:
            logger.error(f"[FireDetectionPlugin] 处理失败: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def _create_event(self, detection: FireDetection, event_type: str) -> FireEvent:
        """创建事件"""
        self._total_events += 1
        event = FireEvent(
            event_id=f"FE{self._total_events:06d}",
            event_type=event_type,
            fire_type=detection.fire_type,
            alarm_level=detection.alarm_level,
            location={
                'x': detection.bbox['x'] + detection.bbox['w']/2,
                'y': detection.bbox['y'] + detection.bbox['h']/2,
                'camera_id': detection.camera_id,
                'zone_id': detection.zone_id,
            },
            timestamp=time.time(),
            details={
                'detection_id': detection.detection_id,
                'track_id': detection.track_id,
                'confidence': detection.confidence,
                'temperature': detection.temperature,
                'smoke_density': detection.smoke_density,
                'risk_score': detection.risk_score,
            }
        )
        self._event_history.append(event)
        return event
    
    def _emit_event(self, event: FireEvent):
        """发送事件"""
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"[FireDetectionPlugin] 事件回调失败: {e}")
    
    def _update_zone_status(self, zone_id: str, detections: List[FireDetection]):
        """更新区域状态"""
        fire_detected = any(d.fire_type == FireType.FLAME for d in detections)
        smoke_detected = any(d.fire_type == FireType.SMOKE for d in detections)
        
        # 最高温度
        temps = [d.temperature for d in detections if d.temperature]
        max_temp = max(temps) if temps else None
        
        # 最高烟雾浓度
        smokes = [d.smoke_density for d in detections if d.smoke_density]
        max_smoke = max(smokes) if smokes else None
        
        # 最高报警级别
        if detections:
            max_level = max((d.alarm_level for d in detections), key=lambda x: x.value)
        else:
            max_level = AlarmLevel.NORMAL
        
        self._zone_status[zone_id] = FireZoneStatus(
            zone_id=zone_id,
            zone_name=f"Zone {zone_id}",
            alarm_level=max_level,
            fire_detected=fire_detected,
            smoke_detected=smoke_detected,
            temperature=max_temp,
            smoke_density=max_smoke,
            active_detections=[d.detection_id for d in detections],
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_detections': self._total_detections,
            'total_events': self._total_events,
            'active_fires': self._tracker.get_track_count(),
            'alarm_counts': {k.value: v for k, v in self._alarm_count.items()},
        }
    
    def get_zone_status(self, zone_id: str) -> Optional[FireZoneStatus]:
        """获取区域状态"""
        return self._zone_status.get(zone_id)
    
    def get_all_zone_status(self) -> List[FireZoneStatus]:
        """获取所有区域状态"""
        return list(self._zone_status.values())
    
    def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        """获取最近检测"""
        return list(self._detection_history)[-limit:]
    
    def get_recent_events(self, limit: int = 50) -> List[FireEvent]:
        """获取最近事件"""
        return list(self._event_history)[-limit:]


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    # 类型
    'FireType',
    'AlarmLevel',
    'SensorType',
    # 数据类
    'FireDetection',
    'SensorReading',
    'FireZoneStatus',
    'FireEvent',
    # 组件
    'FireSmokeDetector',
    'MultiModalFusion',
    'FireTracker',
    'AlarmEvaluator',
    # 插件
    'FireDetectionPlugin',
]
