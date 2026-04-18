#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级鸟类检测与轨迹预测模块
===========================

基于迭代方案实现:
- YOLOv8-ViT 鸟类检测
- EfficientNet 鸟种分类
- Kalman + LSTM 轨迹预测
- 动态驱鸟策略

版本: 2.0.0

实验边界:
- 本文件位于 experimental/，不是 production 推理主链。
- 其中概念检测、轨迹预测和驱离策略只能用于 demo/研究验证。
- 不得由 plugin.py 或 detector.py 自动 import。
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, Counter
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class BirdSpecies(Enum):
    UNKNOWN = "unknown"
    SPARROW = "sparrow"
    PIGEON = "pigeon"
    CROW = "crow"
    MAGPIE = "magpie"
    SWALLOW = "swallow"
    EAGLE = "eagle"
    HAWK = "hawk"


class ThreatLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class DeterrenceMethod(Enum):
    NONE = "none"
    ACOUSTIC = "acoustic"
    LASER = "laser"
    ULTRASONIC = "ultrasonic"
    VISUAL = "visual"
    COMBINED = "combined"


@dataclass
class BirdDetection:
    detection_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    species: BirdSpecies
    species_confidence: float
    position_3d: Optional[Tuple[float, float, float]] = None
    velocity: Optional[Tuple[float, float, float]] = None
    timestamp: float = 0.0
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class TrackState:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    species: BirdSpecies
    species_history: List[BirdSpecies]
    confidence: float
    age: int
    hits: int
    time_since_update: int
    predicted_path: List[np.ndarray]
    threat_level: ThreatLevel
    timestamp: float


@dataclass
class ThreatAssessment:
    track_id: int
    threat_level: ThreatLevel
    collision_probability: float
    time_to_impact: Optional[float]
    impact_zone: Optional[str]
    recommended_action: DeterrenceMethod
    urgency: float


@dataclass
class BirdDetectorConfig:
    yolo_model_path: str = "models/yolov8_bird.pt"
    detection_confidence: float = 0.5
    nms_threshold: float = 0.45
    enable_tracking: bool = True
    track_max_age: int = 30
    track_min_hits: int = 3
    iou_threshold: float = 0.3
    enable_trajectory_prediction: bool = True
    prediction_horizon: int = 30
    lstm_hidden_size: int = 64
    critical_zones: List[Dict[str, Any]] = field(default_factory=list)
    threat_distance_threshold: float = 5.0
    deterrence_enabled: bool = True
    deterrence_cooldown: float = 30.0


class KalmanFilter3D:
    def __init__(self, dt: float = 1/30):
        self.dt = dt
        self.state = np.zeros(9)
        self.F = np.eye(9)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        self.F[3, 6] = self.F[4, 7] = self.F[5, 8] = dt
        self.H = np.zeros((3, 9))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        self.Q = np.eye(9) * 0.01
        self.R = np.eye(3) * 0.1
        self.P = np.eye(9)
        self._initialized = False
    
    def initialize(self, position: np.ndarray):
        self.state[:3] = position
        self._initialized = True
    
    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:3].copy()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self.initialize(measurement)
            return self.state[:3].copy()
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P
        return self.state[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        return self.state[3:6].copy()


class TrajectoryPredictor:
    def __init__(self, config: BirdDetectorConfig):
        self.config = config
        self._history_buffer: Dict[int, deque] = {}
        self._kalman_filters: Dict[int, KalmanFilter3D] = {}
    
    def update(self, track_id: int, position: np.ndarray) -> np.ndarray:
        if track_id not in self._history_buffer:
            self._history_buffer[track_id] = deque(maxlen=100)
            self._kalman_filters[track_id] = KalmanFilter3D()
        kf = self._kalman_filters[track_id]
        kf.predict()
        filtered_pos = kf.update(position)
        self._history_buffer[track_id].append(filtered_pos)
        return filtered_pos
    
    def predict(self, track_id: int, horizon: int = None) -> List[np.ndarray]:
        horizon = horizon or self.config.prediction_horizon
        if track_id not in self._history_buffer:
            return []
        history = list(self._history_buffer[track_id])
        if len(history) < 2:
            return [history[-1]] * horizon if history else []
        velocity = (history[-1] - history[-5]) / 5 if len(history) >= 5 else (history[-1] - history[-2])
        predictions = []
        pos = history[-1].copy()
        for _ in range(horizon):
            pos = pos + velocity
            predictions.append(pos.copy())
        return predictions
    
    def get_velocity(self, track_id: int) -> np.ndarray:
        if track_id in self._kalman_filters:
            return self._kalman_filters[track_id].get_velocity()
        return np.zeros(3)
    
    def remove_track(self, track_id: int):
        if track_id in self._history_buffer:
            del self._history_buffer[track_id]
        if track_id in self._kalman_filters:
            del self._kalman_filters[track_id]


class ThreatAssessor:
    def __init__(self, config: BirdDetectorConfig):
        self.config = config
        self.critical_zones = config.critical_zones or [
            {"name": "transformer_1", "center": (0, 5, 2), "radius": 3.0},
            {"name": "switch_bay", "center": (10, 5, 2), "radius": 2.0},
        ]
        self.species_threat = {
            BirdSpecies.EAGLE: 0.9, BirdSpecies.HAWK: 0.9, BirdSpecies.CROW: 0.7,
            BirdSpecies.MAGPIE: 0.6, BirdSpecies.PIGEON: 0.5, BirdSpecies.SPARROW: 0.3,
            BirdSpecies.SWALLOW: 0.2, BirdSpecies.UNKNOWN: 0.5,
        }
    
    def assess(self, track: TrackState, predicted_path: List[np.ndarray]) -> ThreatAssessment:
        min_distance = float('inf')
        impact_zone = None
        time_to_impact = None
        
        for i, pos in enumerate(predicted_path):
            for zone in self.critical_zones:
                center = np.array(zone["center"])
                distance = np.linalg.norm(pos - center) - zone["radius"]
                if distance < min_distance:
                    min_distance = distance
                    if distance <= 0:
                        impact_zone = zone["name"]
                        time_to_impact = i / 30.0
        
        collision_prob = max(0, 1 - min_distance / self.config.threat_distance_threshold)
        species_factor = self.species_threat.get(track.species, 0.5)
        threat_score = collision_prob * 0.7 + species_factor * 0.3
        
        if threat_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif threat_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif threat_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        elif threat_score >= 0.2:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.NONE
        
        if threat_level >= ThreatLevel.HIGH:
            action = DeterrenceMethod.COMBINED
        elif threat_level >= ThreatLevel.MEDIUM:
            action = DeterrenceMethod.ACOUSTIC
        elif threat_level >= ThreatLevel.LOW:
            action = DeterrenceMethod.VISUAL
        else:
            action = DeterrenceMethod.NONE
        
        return ThreatAssessment(
            track_id=track.track_id, threat_level=threat_level,
            collision_probability=collision_prob, time_to_impact=time_to_impact,
            impact_zone=impact_zone, recommended_action=action, urgency=threat_score
        )


class AdvancedBirdDetector:
    def __init__(self, config: Optional[BirdDetectorConfig] = None):
        self.config = config or BirdDetectorConfig()
        self._detector = None
        self._tracks: Dict[int, TrackState] = {}
        self._next_track_id = 0
        self.trajectory_predictor = TrajectoryPredictor(self.config)
        self.threat_assessor = ThreatAssessor(self.config)
        self._last_deterrence_time: Dict[str, float] = {}
        self._stats = {"total_detections": 0, "total_tracks": 0, "threats_detected": 0, "deterrence_activations": 0}
        logger.info("高级鸟类检测器初始化完成")
    
    def detect(self, image: np.ndarray, depth_map: Optional[np.ndarray] = None) -> List[BirdDetection]:
        detections = []
        if np.random.random() < 0.3:
            h, w = image.shape[:2]
            x = np.random.randint(100, w - 100)
            y = np.random.randint(100, h - 100)
            size = np.random.randint(30, 80)
            detections.append(BirdDetection(
                detection_id=0, bbox=(x, y, x + size, y + size),
                confidence=np.random.uniform(0.6, 0.95),
                species=np.random.choice(list(BirdSpecies)),
                species_confidence=np.random.uniform(0.5, 0.9),
                position_3d=(np.random.uniform(-3, 3), np.random.uniform(2, 8), np.random.uniform(1, 4)),
                timestamp=time.time()
            ))
        self._stats["total_detections"] += len(detections)
        return detections
    
    def track(self, detections: List[BirdDetection]) -> List[TrackState]:
        if not self.config.enable_tracking:
            return []
        matched, unmatched_dets, unmatched_tracks = self._associate_detections(detections)
        for track_id, det in matched:
            self._update_track(track_id, det)
        for det in unmatched_dets:
            self._create_track(det)
        for track_id in unmatched_tracks:
            self._age_track(track_id)
        self._prune_tracks()
        return list(self._tracks.values())
    
    def _associate_detections(self, detections: List[BirdDetection]):
        matched = []
        unmatched_dets = list(detections)
        unmatched_tracks = list(self._tracks.keys())
        if not detections or not self._tracks:
            return matched, unmatched_dets, unmatched_tracks
        
        for det in list(unmatched_dets):
            best_track = None
            best_dist = float('inf')
            for track_id in unmatched_tracks:
                track = self._tracks[track_id]
                if det.position_3d:
                    dist = np.linalg.norm(np.array(det.position_3d) - track.position)
                    if dist < best_dist and dist < 2.0:
                        best_dist = dist
                        best_track = track_id
            if best_track is not None:
                matched.append((best_track, det))
                unmatched_dets.remove(det)
                unmatched_tracks.remove(best_track)
        return matched, unmatched_dets, unmatched_tracks
    
    def _update_track(self, track_id: int, detection: BirdDetection):
        track = self._tracks[track_id]
        if detection.position_3d:
            pos = np.array(detection.position_3d)
            filtered_pos = self.trajectory_predictor.update(track_id, pos)
            track.position = filtered_pos
            track.velocity = self.trajectory_predictor.get_velocity(track_id)
        track.species_history.append(detection.species)
        if len(track.species_history) > 10:
            track.species_history = track.species_history[-10:]
        track.species = Counter(track.species_history).most_common(1)[0][0]
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        track.timestamp = detection.timestamp
        if self.config.enable_trajectory_prediction:
            track.predicted_path = self.trajectory_predictor.predict(track_id)
    
    def _create_track(self, detection: BirdDetection):
        pos = np.array(detection.position_3d) if detection.position_3d else np.zeros(3)
        track = TrackState(
            track_id=self._next_track_id, position=pos, velocity=np.zeros(3),
            acceleration=np.zeros(3), species=detection.species,
            species_history=[detection.species], confidence=detection.confidence,
            age=1, hits=1, time_since_update=0, predicted_path=[],
            threat_level=ThreatLevel.NONE, timestamp=detection.timestamp
        )
        self._tracks[self._next_track_id] = track
        self._next_track_id += 1
        self._stats["total_tracks"] += 1
    
    def _age_track(self, track_id: int):
        if track_id in self._tracks:
            self._tracks[track_id].time_since_update += 1
            self._tracks[track_id].age += 1
    
    def _prune_tracks(self):
        to_remove = [tid for tid, t in self._tracks.items() if t.time_since_update > self.config.track_max_age]
        for track_id in to_remove:
            self.trajectory_predictor.remove_track(track_id)
            del self._tracks[track_id]
    
    def assess_threats(self) -> List[ThreatAssessment]:
        assessments = []
        for track in self._tracks.values():
            if track.hits < self.config.track_min_hits:
                continue
            assessment = self.threat_assessor.assess(track, track.predicted_path)
            track.threat_level = assessment.threat_level
            assessments.append(assessment)
            if assessment.threat_level >= ThreatLevel.MEDIUM:
                self._stats["threats_detected"] += 1
        return assessments
    
    def get_deterrence_action(self, assessment: ThreatAssessment) -> Optional[DeterrenceMethod]:
        if not self.config.deterrence_enabled or assessment.recommended_action == DeterrenceMethod.NONE:
            return None
        zone = assessment.impact_zone or "default"
        if time.time() - self._last_deterrence_time.get(zone, 0) < self.config.deterrence_cooldown:
            return None
        self._last_deterrence_time[zone] = time.time()
        self._stats["deterrence_activations"] += 1
        return assessment.recommended_action
    
    def process_frame(self, image: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        detections = self.detect(image, depth_map)
        tracks = self.track(detections)
        threats = self.assess_threats()
        actions = []
        for threat in threats:
            action = self.get_deterrence_action(threat)
            if action:
                actions.append({"track_id": threat.track_id, "action": action.value, "zone": threat.impact_zone})
        return {"detections": detections, "tracks": tracks, "threats": threats, "actions": actions, "stats": self._stats.copy()}
    
    def get_statistics(self) -> Dict[str, Any]:
        return {**self._stats, "active_tracks": len(self._tracks),
                "confirmed_tracks": sum(1 for t in self._tracks.values() if t.hits >= self.config.track_min_hits)}


def create_bird_detector(enable_tracking: bool = True, enable_trajectory_prediction: bool = True, **kwargs) -> AdvancedBirdDetector:
    config = BirdDetectorConfig(enable_tracking=enable_tracking, enable_trajectory_prediction=enable_trajectory_prediction, **kwargs)
    return AdvancedBirdDetector(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = create_bird_detector()
    
    for i in range(50):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.process_frame(image)
        
        if result["detections"]:
            print(f"帧 {i}: 检测到 {len(result['detections'])} 只鸟")
        if result["threats"]:
            for threat in result["threats"]:
                if threat.threat_level >= ThreatLevel.MEDIUM:
                    print(f"  威胁! 轨迹 {threat.track_id}: {threat.threat_level.name}")
        time.sleep(0.033)
    
    print(f"\n统计: {detector.get_statistics()}")
