"""EnhancedBirdDetector — experimental only.

该检测器从 `detector.py::BirdDetectorEnhanced` 迁移过来，保留完整的 3D 距离估计、
轨迹跟踪、威胁评估、驱鸟建议等能力，但**不得被 `plugin.py` 自动加载**。

启用方式（仅限 experimental / bench 用途）:
  1. 在外部脚本显式 `from plugins.bird_monitoring.experimental.enhanced_detector
     import EnhancedBirdDetector` 并自行实例化；
  2. 或通过 `detector.py::BirdDetectorEnhanced` 的 opt-in shim（需要
     `BIRD_ENABLE_ENHANCED_DETECTOR=1` 环境变量）。

该模块内部的 `RepelController` 已在 detector.py 中明确为硬件禁用；此处沿用。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from plugins.bird_monitoring.detector import (
    AlertManager,
    BirdBehavior,
    BirdDetection,
    BirdDetector,
    RepelController,
    ThreatLevel,
)

logger = logging.getLogger(__name__)


class EnhancedBirdDetector(BirdDetector):
    """增强版鸟类检测器（experimental）。"""

    DEFAULT_DANGER_ZONES = [
        {"name": "transmission_line_1", "x": 0.0, "y": 0.3, "w": 1.0, "h": 0.1},
        {"name": "transmission_line_2", "x": 0.0, "y": 0.5, "w": 1.0, "h": 0.1},
        {"name": "insulator_area", "x": 0.4, "y": 0.2, "w": 0.2, "h": 0.6},
    ]

    SPECIES_RISK_FACTOR = {
        "crow": 1.5,
        "magpie": 1.3,
        "eagle": 1.4,
        "heron": 1.2,
        "sparrow": 0.8,
        "swallow": 0.7,
        "egret": 1.1,
        "dove": 0.9,
        "nest": 2.0,
        "bird_generic": 1.0,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        camera_config = config.get("camera", {})
        self.camera_height_m = camera_config.get("height_m", 15.0)
        self.camera_fov_deg = camera_config.get("fov_deg", 60.0)
        self.reference_bird_size_m = 0.3

        self.danger_zones = config.get("danger_zones", self.DEFAULT_DANGER_ZONES)

        threat_config = config.get("threat", {})
        self.distance_thresholds = {
            ThreatLevel.CRITICAL: threat_config.get("critical_distance_m", 3.0),
            ThreatLevel.HIGH: threat_config.get("high_distance_m", 8.0),
            ThreatLevel.MEDIUM: threat_config.get("medium_distance_m", 15.0),
            ThreatLevel.LOW: threat_config.get("low_distance_m", 30.0),
        }
        self.perch_time_threshold_s = threat_config.get("perch_time_threshold_s", 10.0)

        self._trajectory_history: Dict[int, List[Dict]] = {}
        self._max_trajectory_length = 30
        self._zone_entry_time: Dict[int, float] = {}

        self.alert_manager = AlertManager(config.get("alert", {}))
        self.repel_controller = RepelController(config.get("repel", {}))

        self._stats = {
            "total_detections": 0,
            "high_threat_count": 0,
            "repel_triggered": 0,
            "alerts_sent": 0,
        }

        logger.info("EnhancedBirdDetector (experimental) 初始化完成")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """增强检测 — 兼容原始接口，返回附加字段后的 dict 列表。"""
        raw_detections = BirdDetector.detect(self, image)
        current_time = time.time()
        image_size = image.shape[:2]

        for det in raw_detections:
            det["distance"] = self._estimate_distance(det, image_size)
            self._update_trajectory(det)
            det["speed"], det["heading"] = self._estimate_velocity(det["track_id"])
            behavior = self._determine_behavior(det, det["distance"])
            det["status"] = behavior.value
            time_in_zone = self._update_zone_time(
                det["track_id"], det["bbox"], current_time
            )
            det["time_in_zone"] = time_in_zone
            threat = self._assess_threat(det, det["distance"], behavior, time_in_zone)
            det["threat_level"] = threat.name
            self._stats["total_detections"] += 1

        return raw_detections

    def _assess_threat(
        self,
        detection: Dict,
        distance: float,
        behavior: BirdBehavior,
        time_in_zone: float,
    ) -> ThreatLevel:
        base_level = ThreatLevel.NONE
        for level in [
            ThreatLevel.CRITICAL,
            ThreatLevel.HIGH,
            ThreatLevel.MEDIUM,
            ThreatLevel.LOW,
        ]:
            if distance <= self.distance_thresholds[level]:
                base_level = level
                break

        species = detection.get("class_name", "bird_generic")
        risk_factor = self.SPECIES_RISK_FACTOR.get(species, 1.0)

        behavior_factor = 1.0
        if behavior == BirdBehavior.NESTING:
            behavior_factor = 2.0
        elif behavior == BirdBehavior.PERCHED:
            behavior_factor = 1.5
        elif behavior == BirdBehavior.APPROACHING:
            behavior_factor = 1.3

        time_factor = 1.0 + (time_in_zone / 60.0)
        total_score = base_level.value * risk_factor * behavior_factor * time_factor

        if total_score >= ThreatLevel.CRITICAL.value * 1.5:
            return ThreatLevel.CRITICAL
        if total_score >= ThreatLevel.HIGH.value * 1.3:
            return ThreatLevel.HIGH
        if total_score >= ThreatLevel.MEDIUM.value * 1.2:
            return ThreatLevel.MEDIUM
        if total_score >= ThreatLevel.LOW.value:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _determine_behavior(self, detection: Dict, distance: float) -> BirdBehavior:
        track_id = detection["track_id"]
        class_name = detection.get("class_name", "")

        if class_name == "nest":
            return BirdBehavior.NESTING

        if track_id in self._trajectory_history:
            history = self._trajectory_history[track_id]
            if len(history) >= 5:
                recent = history[-5:]
                x_var = np.var([h["x"] for h in recent])
                y_var = np.var([h["y"] for h in recent])
                if x_var < 0.0005 and y_var < 0.0005:
                    return BirdBehavior.PERCHED
                if len(history) >= 10:
                    early = history[-10:-5]
                    late = history[-5:]
                    early_dist = np.mean([self._point_to_center_dist(p) for p in early])
                    late_dist = np.mean([self._point_to_center_dist(p) for p in late])
                    if late_dist > early_dist + 0.1:
                        return BirdBehavior.LEAVING

        if distance < 15:
            return BirdBehavior.APPROACHING
        return BirdBehavior.FLYING

    @staticmethod
    def _point_to_center_dist(point: Dict) -> float:
        return float(np.sqrt((point["x"] - 0.5) ** 2 + (point["y"] - 0.5) ** 2))

    def _update_zone_time(
        self, track_id: int, bbox: Dict, current_time: float
    ) -> float:
        in_danger_zone = self._is_in_danger_zone(bbox)
        if in_danger_zone:
            if track_id not in self._zone_entry_time:
                self._zone_entry_time[track_id] = current_time
            return current_time - self._zone_entry_time[track_id]
        if track_id in self._zone_entry_time:
            del self._zone_entry_time[track_id]
        return 0.0

    def _is_in_danger_zone(self, bbox: Dict) -> bool:
        cx = bbox["x"] + bbox["width"] / 2
        cy = bbox["y"] + bbox["height"] / 2
        for zone in self.danger_zones:
            if (
                zone["x"] <= cx <= zone["x"] + zone["w"]
                and zone["y"] <= cy <= zone["y"] + zone["h"]
            ):
                return True
        return False

    def _estimate_distance(
        self, detection: Dict, image_size: Tuple[int, int]
    ) -> float:
        bbox = detection["bbox"]
        box_height = bbox["height"] * image_size[0]
        if box_height < 1:
            return 50.0
        focal_length = image_size[1] / (
            2 * np.tan(np.radians(self.camera_fov_deg / 2))
        )
        distance = (self.reference_bird_size_m * focal_length) / box_height
        return float(np.clip(distance, 1.0, 100.0))

    def _update_trajectory(self, detection: Dict) -> None:
        track_id = detection["track_id"]
        bbox = detection["bbox"]
        point = {
            "x": bbox["x"] + bbox["width"] / 2,
            "y": bbox["y"] + bbox["height"] / 2,
            "frame": self._frame_count,
        }
        self._trajectory_history.setdefault(track_id, []).append(point)
        if len(self._trajectory_history[track_id]) > self._max_trajectory_length:
            self._trajectory_history[track_id].pop(0)

    def _estimate_velocity(self, track_id: int) -> Tuple[float, float]:
        history = self._trajectory_history.get(track_id, [])
        if len(history) < 2:
            return 0.0, 0.0
        p1, p2 = history[-2], history[-1]
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        dt = p2["frame"] - p1["frame"]
        if dt == 0:
            return 0.0, 0.0
        speed_px = np.sqrt(dx ** 2 + dy ** 2) / dt
        speed_ms = speed_px * 30 * 0.1
        heading = np.degrees(np.arctan2(dy, dx))
        return float(speed_ms), float(heading)

    def cleanup(self) -> None:
        super().cleanup()
        self._trajectory_history.clear()
