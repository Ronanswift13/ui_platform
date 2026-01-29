#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动物入侵检测插件 V1.0
=======================

基于迭代方案实现室内小动物入侵检测:
- YOLOv8小型模型检测
- 热成像二次确认
- 驱离措施联动
- 历史记录与统计

支持检测的动物类型:
- 老鼠 (mouse/rat)
- 猫 (cat)
- 蛇 (snake)
- 鸟类 (bird)
- 其他小动物 (other)

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
import hashlib
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
class AnimalType(Enum):
    """动物类型"""
    MOUSE = "mouse"         # 老鼠
    RAT = "rat"            # 大鼠
    CAT = "cat"            # 猫
    SNAKE = "snake"        # 蛇
    BIRD = "bird"          # 鸟
    INSECT = "insect"      # 昆虫
    DOG = "dog"            # 狗
    OTHER = "other"        # 其他
    UNKNOWN = "unknown"    # 未知


class ThreatLevel(Enum):
    """威胁等级"""
    LOW = "low"           # 低威胁 (如鸟类)
    MEDIUM = "medium"     # 中等威胁 (如猫)
    HIGH = "high"         # 高威胁 (如老鼠、蛇)
    CRITICAL = "critical" # 严重威胁


class DeterrentType(Enum):
    """驱离方式"""
    AUDIO = "audio"           # 超声波/高频声波
    LIGHT = "light"           # 强光闪烁
    VIBRATION = "vibration"   # 振动
    COMBINED = "combined"     # 组合驱离
    NONE = "none"            # 不驱离


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class AnimalDetection:
    """动物检测结果"""
    detection_id: str                       # 检测ID
    animal_type: AnimalType                 # 动物类型
    confidence: float                       # 置信度
    bbox: Dict[str, float]                  # 边界框 {x, y, w, h}
    position: Optional[Tuple[float, float]] = None  # 地面位置 (x, y)
    
    # 热成像确认
    thermal_confirmed: bool = False         # 热成像是否确认
    thermal_temperature: Optional[float] = None  # 热成像温度
    
    # 威胁评估
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    threat_score: float = 0.5
    
    # 轨迹
    track_id: Optional[str] = None          # 跟踪ID
    velocity: Tuple[float, float] = (0, 0)  # 速度
    direction: float = 0.0                  # 移动方向 (度)
    
    # 时间
    timestamp: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    duration: float = 0.0                   # 出现持续时间
    
    # 元数据
    camera_id: str = ""
    zone_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "detection_id": self.detection_id,
            "animal_type": self.animal_type.value,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "position": {"x": self.position[0], "y": self.position[1]} if self.position else None,
            "thermal_confirmed": self.thermal_confirmed,
            "thermal_temperature": self.thermal_temperature,
            "threat_level": self.threat_level.value,
            "threat_score": self.threat_score,
            "track_id": self.track_id,
            "velocity": {"vx": self.velocity[0], "vy": self.velocity[1]},
            "direction": self.direction,
            "timestamp": self.timestamp,
            "first_seen": self.first_seen,
            "duration": self.duration,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
        }


@dataclass
class DeterrentAction:
    """驱离动作"""
    action_id: str
    deterrent_type: DeterrentType
    target_detection_id: str
    start_time: float
    duration: float = 5.0           # 驱离持续时间(秒)
    intensity: float = 0.8          # 驱离强度 [0, 1]
    is_active: bool = True
    result: Optional[str] = None    # 驱离结果


@dataclass
class AnimalEvent:
    """动物入侵事件"""
    event_id: str
    event_type: str                 # intrusion, deterrent, clear
    animal_type: AnimalType
    threat_level: ThreatLevel
    location: Dict[str, float]
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 动物检测器
# =============================================================================
class AnimalDetector:
    """
    动物检测器
    
    基于YOLOv8进行小动物检测
    """
    
    # 类别映射
    CLASS_MAPPING = {
        0: AnimalType.MOUSE,
        1: AnimalType.RAT,
        2: AnimalType.CAT,
        3: AnimalType.SNAKE,
        4: AnimalType.BIRD,
        5: AnimalType.INSECT,
        6: AnimalType.DOG,
        7: AnimalType.OTHER,
    }
    
    # 通用COCO类别映射 (用于预训练模型)
    COCO_ANIMAL_CLASSES = {
        15: AnimalType.BIRD,      # bird
        16: AnimalType.CAT,       # cat
        17: AnimalType.DOG,       # dog
        # 老鼠等需要自定义训练
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
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
        """加载检测模型"""
        try:
            if self.model_path and Path(self.model_path).exists():
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if self.use_gpu else ['CPUExecutionProvider']
                self._model = ort.InferenceSession(self.model_path, providers=providers)
                self._initialized = True
                logger.info(f"[AnimalDetector] 模型加载成功: {self.model_path}")
            else:
                logger.info("[AnimalDetector] 未指定模型，使用模拟检测")
                self._initialized = True
        except Exception as e:
            logger.warning(f"[AnimalDetector] 模型加载失败: {e}")
            self._initialized = True
    
    def detect(self, image: np.ndarray, camera_id: str = "") -> List[AnimalDetection]:
        """
        检测图像中的动物
        
        Args:
            image: 输入图像 (H, W, C)
            camera_id: 相机ID
            
        Returns:
            检测结果列表
        """
        if not self._initialized:
            return []
        
        try:
            if self._model is not None:
                return self._detect_with_model(image, camera_id)
            else:
                return self._detect_simulated(image, camera_id)
        except Exception as e:
            logger.error(f"[AnimalDetector] 检测失败: {e}")
            return []
    
    def _detect_with_model(self, image: np.ndarray, camera_id: str) -> List[AnimalDetection]:
        """使用模型检测"""
        import cv2
        
        # 预处理
        h, w = image.shape[:2]
        img = cv2.resize(image, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        # 推理
        inputs = {self._model.get_inputs()[0].name: img}
        outputs = self._model.run(None, inputs)
        
        # 解析输出 (YOLOv8格式)
        predictions = outputs[0][0]  # (N, 6) or (N, 4+nc)
        
        detections = []
        for pred in predictions:
            if len(pred) >= 6:
                x1, y1, x2, y2, conf, cls = pred[:6]
            else:
                continue
            
            if conf < self.confidence_threshold:
                continue
            
            # 映射类别
            cls_id = int(cls)
            if cls_id in self.CLASS_MAPPING:
                animal_type = self.CLASS_MAPPING[cls_id]
            elif cls_id in self.COCO_ANIMAL_CLASSES:
                animal_type = self.COCO_ANIMAL_CLASSES[cls_id]
            else:
                continue
            
            # 转换坐标
            scale_x, scale_y = w / self.input_size[0], h / self.input_size[1]
            bbox = {
                'x': float(x1 * scale_x),
                'y': float(y1 * scale_y),
                'w': float((x2 - x1) * scale_x),
                'h': float((y2 - y1) * scale_y),
            }
            
            self._detection_count += 1
            detection = AnimalDetection(
                detection_id=f"AD{self._detection_count:06d}",
                animal_type=animal_type,
                confidence=float(conf),
                bbox=bbox,
                camera_id=camera_id,
                threat_level=self._assess_threat(animal_type, float(conf)),
            )
            detections.append(detection)
        
        # NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _detect_simulated(self, image: np.ndarray, camera_id: str) -> List[AnimalDetection]:
        """模拟检测 (用于测试)"""
        # 10%概率检测到动物
        if np.random.random() > 0.1:
            return []
        
        h, w = image.shape[:2]
        
        # 随机生成检测
        animal_types = [AnimalType.MOUSE, AnimalType.CAT, AnimalType.BIRD]
        animal_type = np.random.choice(animal_types)
        
        # 随机位置
        x = np.random.uniform(0.1, 0.8) * w
        y = np.random.uniform(0.1, 0.8) * h
        box_w = np.random.uniform(30, 100)
        box_h = np.random.uniform(30, 100)
        
        self._detection_count += 1
        detection = AnimalDetection(
            detection_id=f"AD{self._detection_count:06d}",
            animal_type=animal_type,
            confidence=np.random.uniform(0.6, 0.95),
            bbox={'x': x, 'y': y, 'w': box_w, 'h': box_h},
            camera_id=camera_id,
            threat_level=self._assess_threat(animal_type, 0.8),
        )
        
        return [detection]
    
    def _assess_threat(self, animal_type: AnimalType, confidence: float) -> ThreatLevel:
        """评估威胁等级"""
        threat_map = {
            AnimalType.MOUSE: ThreatLevel.HIGH,
            AnimalType.RAT: ThreatLevel.HIGH,
            AnimalType.SNAKE: ThreatLevel.CRITICAL,
            AnimalType.CAT: ThreatLevel.MEDIUM,
            AnimalType.BIRD: ThreatLevel.LOW,
            AnimalType.DOG: ThreatLevel.MEDIUM,
            AnimalType.INSECT: ThreatLevel.LOW,
            AnimalType.OTHER: ThreatLevel.MEDIUM,
        }
        
        base_threat = threat_map.get(animal_type, ThreatLevel.MEDIUM)
        
        # 高置信度提升威胁等级
        if confidence > 0.9 and base_threat != ThreatLevel.CRITICAL:
            threat_order = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            idx = threat_order.index(base_threat)
            if idx < len(threat_order) - 1:
                return threat_order[idx + 1]
        
        return base_threat
    
    def _apply_nms(self, detections: List[AnimalDetection]) -> List[AnimalDetection]:
        """应用非极大值抑制"""
        if len(detections) <= 1:
            return detections
        
        # 按置信度排序
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
# 热成像确认器
# =============================================================================
class ThermalConfirmer:
    """
    热成像确认器
    
    使用热成像数据确认动物检测结果
    """
    
    def __init__(
        self,
        min_animal_temp: float = 25.0,   # 最低动物体温
        max_animal_temp: float = 42.0,   # 最高动物体温
        ambient_temp: float = 20.0,       # 环境温度
        temp_threshold: float = 3.0,      # 温度差阈值
    ):
        self.min_animal_temp = min_animal_temp
        self.max_animal_temp = max_animal_temp
        self.ambient_temp = ambient_temp
        self.temp_threshold = temp_threshold
    
    def confirm(
        self,
        thermal_image: np.ndarray,
        detection: AnimalDetection
    ) -> Tuple[bool, Optional[float]]:
        """
        确认检测结果
        
        Args:
            thermal_image: 热成像图像 (温度矩阵)
            detection: 检测结果
            
        Returns:
            (是否确认, 平均温度)
        """
        try:
            bbox = detection.bbox
            x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
            
            # 裁剪区域
            region = thermal_image[y:y+h, x:x+w]
            
            if region.size == 0:
                return False, None
            
            # 计算统计
            max_temp = float(np.max(region))
            mean_temp = float(np.mean(region))
            
            # 判断是否为生物
            is_warm_blooded = (
                self.min_animal_temp <= max_temp <= self.max_animal_temp and
                (max_temp - self.ambient_temp) > self.temp_threshold
            )
            
            return is_warm_blooded, mean_temp
            
        except Exception as e:
            logger.warning(f"[ThermalConfirmer] 确认失败: {e}")
            return False, None
    
    def set_ambient_temp(self, temp: float):
        """更新环境温度"""
        self.ambient_temp = temp


# =============================================================================
# 动物跟踪器
# =============================================================================
class AnimalTracker:
    """
    动物跟踪器
    
    跟踪检测到的动物，维护轨迹
    """
    
    def __init__(
        self,
        max_age: int = 30,          # 最大未匹配帧数
        min_hits: int = 2,          # 确认最小帧数
        distance_threshold: float = 100.0,  # 匹配距离阈值 (像素)
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        
        self._tracks: Dict[str, Dict] = {}  # {track_id: track_info}
        self._next_id = 1
    
    def update(self, detections: List[AnimalDetection]) -> List[AnimalDetection]:
        """
        更新跟踪
        
        Args:
            detections: 当前帧检测结果
            
        Returns:
            带跟踪ID的检测结果
        """
        if len(detections) == 0:
            # 更新所有跟踪的年龄
            stale = []
            for track_id, track in self._tracks.items():
                track['age'] += 1
                if track['age'] > self.max_age:
                    stale.append(track_id)
            for tid in stale:
                self._tracks.pop(tid, None)
            return []
        
        # 匹配检测和跟踪
        track_ids = list(self._tracks.keys())
        
        if len(track_ids) == 0:
            # 全部创建新跟踪
            for det in detections:
                track_id = self._create_track(det)
                det.track_id = track_id
            return detections
        
        # 计算距离矩阵
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            last_pos = track['last_position']
            
            for j, det in enumerate(detections):
                det_pos = (det.bbox['x'] + det.bbox['w']/2, det.bbox['y'] + det.bbox['h']/2)
                dist = np.sqrt((last_pos[0] - det_pos[0])**2 + (last_pos[1] - det_pos[1])**2)
                cost_matrix[i, j] = dist
        
        # 贪婪匹配
        matched = set()
        matched_tracks = set()
        
        for _ in range(min(len(track_ids), len(detections))):
            if cost_matrix.size == 0:
                break
            
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            if cost_matrix[min_idx] < self.distance_threshold:
                track_id = track_ids[min_idx[0]]
                det_idx = min_idx[1]
                
                # 更新跟踪
                det = detections[det_idx]
                det.track_id = track_id
                self._update_track(track_id, det)
                
                matched.add(det_idx)
                matched_tracks.add(track_id)
                
                cost_matrix[min_idx[0], :] = np.inf
                cost_matrix[:, min_idx[1]] = np.inf
            else:
                break
        
        # 未匹配的检测创建新跟踪
        for j, det in enumerate(detections):
            if j not in matched:
                track_id = self._create_track(det)
                det.track_id = track_id
        
        # 未匹配的跟踪增加年龄
        stale = []
        for track_id in self._tracks:
            if track_id not in matched_tracks:
                self._tracks[track_id]['age'] += 1
                if self._tracks[track_id]['age'] > self.max_age:
                    stale.append(track_id)
        for tid in stale:
            self._tracks.pop(tid, None)
        
        return detections
    
    def _create_track(self, detection: AnimalDetection) -> str:
        """创建新跟踪"""
        track_id = f"AT{self._next_id:05d}"
        self._next_id += 1
        
        pos = (detection.bbox['x'] + detection.bbox['w']/2, 
               detection.bbox['y'] + detection.bbox['h']/2)
        
        self._tracks[track_id] = {
            'track_id': track_id,
            'animal_type': detection.animal_type,
            'first_seen': detection.timestamp,
            'last_seen': detection.timestamp,
            'last_position': pos,
            'positions': deque([pos], maxlen=30),
            'age': 0,
            'hits': 1,
        }
        
        detection.first_seen = detection.timestamp
        return track_id
    
    def _update_track(self, track_id: str, detection: AnimalDetection):
        """更新跟踪"""
        track = self._tracks[track_id]
        
        pos = (detection.bbox['x'] + detection.bbox['w']/2,
               detection.bbox['y'] + detection.bbox['h']/2)
        
        # 计算速度
        last_pos = track['last_position']
        dt = detection.timestamp - track['last_seen']
        if dt > 0:
            vx = (pos[0] - last_pos[0]) / dt
            vy = (pos[1] - last_pos[1]) / dt
            detection.velocity = (vx, vy)
            detection.direction = np.degrees(np.arctan2(vy, vx))
        
        # 更新
        track['last_seen'] = detection.timestamp
        track['last_position'] = pos
        track['positions'].append(pos)
        track['age'] = 0
        track['hits'] += 1
        
        detection.first_seen = track['first_seen']
        detection.duration = detection.timestamp - track['first_seen']
    
    def get_track_count(self) -> int:
        """获取当前跟踪数量"""
        return len(self._tracks)


# =============================================================================
# 驱离控制器
# =============================================================================
class DeterrentController:
    """
    驱离控制器
    
    控制驱离设备（声波、灯光等）
    """
    
    # 动物类型对应的最佳驱离方式
    DETERRENT_STRATEGY = {
        AnimalType.MOUSE: DeterrentType.AUDIO,      # 老鼠怕超声波
        AnimalType.RAT: DeterrentType.AUDIO,
        AnimalType.CAT: DeterrentType.LIGHT,        # 猫怕强光
        AnimalType.SNAKE: DeterrentType.VIBRATION,  # 蛇怕振动
        AnimalType.BIRD: DeterrentType.AUDIO,       # 鸟怕声音
        AnimalType.DOG: DeterrentType.AUDIO,
        AnimalType.INSECT: DeterrentType.LIGHT,
        AnimalType.OTHER: DeterrentType.COMBINED,
    }
    
    def __init__(
        self,
        audio_adapter: Optional[Any] = None,
        light_adapter: Optional[Any] = None,
        vibration_adapter: Optional[Any] = None,
        auto_deterrent: bool = True,
        min_deterrent_interval: float = 10.0,  # 最小驱离间隔(秒)
    ):
        self.audio_adapter = audio_adapter
        self.light_adapter = light_adapter
        self.vibration_adapter = vibration_adapter
        self.auto_deterrent = auto_deterrent
        self.min_deterrent_interval = min_deterrent_interval
        
        self._active_actions: Dict[str, DeterrentAction] = {}
        self._last_deterrent_time: Dict[str, float] = {}
        self._action_count = 0
    
    def should_deter(self, detection: AnimalDetection) -> bool:
        """判断是否应该驱离"""
        if not self.auto_deterrent:
            return False
        
        # 检查间隔
        key = f"{detection.camera_id}_{detection.animal_type.value}"
        last_time = self._last_deterrent_time.get(key, 0)
        if time.time() - last_time < self.min_deterrent_interval:
            return False
        
        # 根据威胁等级决定
        return detection.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
    
    def trigger_deterrent(
        self,
        detection: AnimalDetection,
        deterrent_type: Optional[DeterrentType] = None,
        duration: float = 5.0,
        intensity: float = 0.8,
    ) -> Optional[DeterrentAction]:
        """
        触发驱离
        
        Args:
            detection: 检测结果
            deterrent_type: 驱离类型 (None则自动选择)
            duration: 持续时间
            intensity: 强度
            
        Returns:
            驱离动作对象
        """
        if deterrent_type is None:
            deterrent_type = self.DETERRENT_STRATEGY.get(
                detection.animal_type, DeterrentType.COMBINED
            )
        
        self._action_count += 1
        action = DeterrentAction(
            action_id=f"DA{self._action_count:06d}",
            deterrent_type=deterrent_type,
            target_detection_id=detection.detection_id,
            start_time=time.time(),
            duration=duration,
            intensity=intensity,
            is_active=True,
        )
        
        # 执行驱离
        success = self._execute_deterrent(action)
        
        if success:
            self._active_actions[action.action_id] = action
            key = f"{detection.camera_id}_{detection.animal_type.value}"
            self._last_deterrent_time[key] = time.time()
            
            # 启动定时器停止驱离
            timer = threading.Timer(duration, self._stop_deterrent, args=[action.action_id])
            timer.daemon = True
            timer.start()
            
            logger.info(f"[Deterrent] 启动驱离: {action.action_id}, 类型: {deterrent_type.value}")
            return action
        
        return None
    
    def _execute_deterrent(self, action: DeterrentAction) -> bool:
        """执行驱离动作"""
        try:
            if action.deterrent_type == DeterrentType.AUDIO:
                if self.audio_adapter:
                    self.audio_adapter.play_deterrent(action.intensity)
                    return True
            
            elif action.deterrent_type == DeterrentType.LIGHT:
                if self.light_adapter:
                    self.light_adapter.flash(action.intensity)
                    return True
            
            elif action.deterrent_type == DeterrentType.VIBRATION:
                if self.vibration_adapter:
                    self.vibration_adapter.vibrate(action.intensity)
                    return True
            
            elif action.deterrent_type == DeterrentType.COMBINED:
                # 组合驱离
                success = False
                if self.audio_adapter:
                    self.audio_adapter.play_deterrent(action.intensity)
                    success = True
                if self.light_adapter:
                    self.light_adapter.flash(action.intensity)
                    success = True
                return success
            
            # 模拟成功
            logger.info(f"[Deterrent] 模拟驱离: {action.deterrent_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"[Deterrent] 执行失败: {e}")
            return False
    
    def _stop_deterrent(self, action_id: str):
        """停止驱离"""
        action = self._active_actions.pop(action_id, None)
        if action:
            action.is_active = False
            action.result = "completed"
            logger.info(f"[Deterrent] 驱离结束: {action_id}")
    
    def get_active_actions(self) -> List[DeterrentAction]:
        """获取活跃的驱离动作"""
        return list(self._active_actions.values())


# =============================================================================
# 动物入侵检测插件
# =============================================================================
class AnimalDetectionPlugin:
    """
    动物入侵检测插件 V1.0
    
    功能:
    - YOLOv8小目标检测
    - 热成像二次确认
    - 动物跟踪
    - 自动驱离
    - 事件记录
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        use_thermal: bool = True,
        auto_deterrent: bool = True,
        use_gpu: bool = False,
        event_callback: Optional[Callable[[AnimalEvent], None]] = None,
    ):
        """
        初始化插件
        
        Args:
            model_path: YOLOv8模型路径
            confidence_threshold: 检测置信度阈值
            use_thermal: 是否使用热成像确认
            auto_deterrent: 是否自动驱离
            use_gpu: 是否使用GPU
            event_callback: 事件回调函数
        """
        self.use_thermal = use_thermal
        self.event_callback = event_callback
        
        # 初始化组件
        self._detector = AnimalDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu,
        )
        self._thermal_confirmer = ThermalConfirmer() if use_thermal else None
        self._tracker = AnimalTracker()
        self._deterrent = DeterrentController(auto_deterrent=auto_deterrent)
        
        # 统计
        self._total_detections = 0
        self._total_events = 0
        self._detections_by_type: Dict[AnimalType, int] = {}
        
        # 历史
        self._detection_history: deque = deque(maxlen=1000)
        self._event_history: deque = deque(maxlen=500)
        
        self._initialized = True
        logger.info(f"[AnimalDetectionPlugin] 初始化完成 V{self.VERSION}")
    
    def process(
        self,
        image: np.ndarray,
        thermal_image: Optional[np.ndarray] = None,
        camera_id: str = "",
        zone_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理单帧图像
        
        Args:
            image: 可见光图像
            thermal_image: 热成像图像 (可选)
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
            'deterrent_actions': [],
            'statistics': {},
        }
        
        try:
            # 1. 检测
            detections = self._detector.detect(image, camera_id)
            
            # 2. 热成像确认
            if self.use_thermal and thermal_image is not None and self._thermal_confirmer:
                for det in detections:
                    confirmed, temp = self._thermal_confirmer.confirm(thermal_image, det)
                    det.thermal_confirmed = confirmed
                    det.thermal_temperature = temp
                
                # 过滤未确认的
                # 保留: 热确认的 或 高置信度的
                detections = [
                    d for d in detections 
                    if d.thermal_confirmed or d.confidence > 0.8
                ]
            
            # 3. 跟踪
            detections = self._tracker.update(detections)
            
            # 4. 设置区域
            for det in detections:
                det.zone_id = zone_id
            
            # 5. 驱离
            deterrent_actions = []
            for det in detections:
                if self._deterrent.should_deter(det):
                    action = self._deterrent.trigger_deterrent(det)
                    if action:
                        deterrent_actions.append(action)
                        # 生成驱离事件
                        event = self._create_event(det, "deterrent")
                        self._emit_event(event)
                        result['events'].append(event.details)
            
            # 6. 生成入侵事件
            for det in detections:
                if det.duration < 0.5:  # 新检测
                    event = self._create_event(det, "intrusion")
                    self._emit_event(event)
                    result['events'].append(event.details)
            
            # 7. 更新统计
            self._total_detections += len(detections)
            for det in detections:
                self._detections_by_type[det.animal_type] = \
                    self._detections_by_type.get(det.animal_type, 0) + 1
                self._detection_history.append(det.to_dict())
            
            # 构造返回
            result['detections'] = [d.to_dict() for d in detections]
            result['deterrent_actions'] = [
                {'action_id': a.action_id, 'type': a.deterrent_type.value}
                for a in deterrent_actions
            ]
            result['statistics'] = self.get_statistics()
            
        except Exception as e:
            logger.error(f"[AnimalDetectionPlugin] 处理失败: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def _create_event(self, detection: AnimalDetection, event_type: str) -> AnimalEvent:
        """创建事件"""
        self._total_events += 1
        event = AnimalEvent(
            event_id=f"AE{self._total_events:06d}",
            event_type=event_type,
            animal_type=detection.animal_type,
            threat_level=detection.threat_level,
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
                'thermal_confirmed': detection.thermal_confirmed,
            }
        )
        self._event_history.append(event)
        return event
    
    def _emit_event(self, event: AnimalEvent):
        """发送事件"""
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"[AnimalDetectionPlugin] 事件回调失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_detections': self._total_detections,
            'total_events': self._total_events,
            'active_tracks': self._tracker.get_track_count(),
            'active_deterrents': len(self._deterrent.get_active_actions()),
            'detections_by_type': {
                k.value: v for k, v in self._detections_by_type.items()
            },
        }
    
    def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        """获取最近的检测结果"""
        return list(self._detection_history)[-limit:]
    
    def get_recent_events(self, limit: int = 50) -> List[AnimalEvent]:
        """获取最近的事件"""
        return list(self._event_history)[-limit:]
    
    def trigger_manual_deterrent(
        self,
        deterrent_type: DeterrentType,
        duration: float = 5.0,
        intensity: float = 0.8
    ) -> bool:
        """手动触发驱离"""
        # 创建虚拟检测
        fake_detection = AnimalDetection(
            detection_id="manual",
            animal_type=AnimalType.UNKNOWN,
            confidence=1.0,
            bbox={'x': 0, 'y': 0, 'w': 0, 'h': 0},
        )
        action = self._deterrent.trigger_deterrent(
            fake_detection, deterrent_type, duration, intensity
        )
        return action is not None


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    # 类型
    'AnimalType',
    'ThreatLevel',
    'DeterrentType',
    # 数据类
    'AnimalDetection',
    'DeterrentAction',
    'AnimalEvent',
    # 组件
    'AnimalDetector',
    'ThermalConfirmer',
    'AnimalTracker',
    'DeterrentController',
    # 插件
    'AnimalDetectionPlugin',
]
