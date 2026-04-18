"""
Indoor Fence V3.0 - Behavior Recognizer
=========================================

Temporal behavior recognition using sliding window of pose observations.
Classifies: normal walk, prolonged stay, climbing, crossing, fallen.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .pose_estimator import PoseResult, PostureType


class BehaviorType(str, Enum):
    NORMAL_WALK = "normal_walk"
    PROLONGED_STAY = "prolonged_stay"
    CLIMBING = "climbing"
    CROSSING = "crossing"
    FALLEN = "fallen"
    UNKNOWN = "unknown"


@dataclass
class BehaviorResult:
    """Result of behavior recognition."""
    behavior: BehaviorType
    confidence: float
    duration_s: float = 0.0


@dataclass
class _TrackEntry:
    pose: PoseResult
    position: Tuple[float, float]
    timestamp: float


class BehaviorRecognizerV3:
    """Temporal behavior recognizer using sliding window."""

    def __init__(
        self,
        window_size: int = 30,
        prolonged_stay_threshold_s: float = 30.0,
        movement_threshold_m: float = 0.3,
    ):
        self.window_size = window_size
        self._prolonged_stay_threshold = prolonged_stay_threshold_s
        self._movement_threshold = movement_threshold_m
        self._tracks: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def update(
        self,
        track_id: str,
        pose: PoseResult,
        position: Tuple[float, float],
    ) -> Optional[BehaviorResult]:
        """Update with new pose observation and classify behavior."""
        ts = time.time()
        entry = _TrackEntry(pose=pose, position=position, timestamp=ts)
        self._tracks[track_id].append(entry)

        window = self._tracks[track_id]
        if len(window) < 2:
            return BehaviorResult(behavior=BehaviorType.UNKNOWN, confidence=0.3)

        return self._classify(window)

    def _classify(self, window: deque) -> BehaviorResult:
        """Classify behavior from observation window."""
        # Check for fallen posture (majority vote)
        fallen_count = sum(
            1 for e in window if e.pose.posture == PostureType.FALLEN
        )
        if fallen_count > len(window) * 0.5:
            return BehaviorResult(
                behavior=BehaviorType.FALLEN,
                confidence=fallen_count / len(window),
            )

        # Check for climbing posture
        climbing_count = sum(
            1 for e in window if e.pose.posture == PostureType.CLIMBING
        )
        if climbing_count > len(window) * 0.4:
            return BehaviorResult(
                behavior=BehaviorType.CLIMBING,
                confidence=climbing_count / len(window),
            )

        # Check movement distance
        first = window[0]
        last = window[-1]
        dx = last.position[0] - first.position[0]
        dy = last.position[1] - first.position[1]
        total_dist = math.sqrt(dx * dx + dy * dy)
        duration = last.timestamp - first.timestamp

        # Prolonged stay detection
        if duration > self._prolonged_stay_threshold and total_dist < self._movement_threshold:
            return BehaviorResult(
                behavior=BehaviorType.PROLONGED_STAY,
                confidence=min(0.95, 0.5 + duration / (self._prolonged_stay_threshold * 2)),
                duration_s=duration,
            )

        # Normal walk
        if total_dist >= self._movement_threshold:
            return BehaviorResult(
                behavior=BehaviorType.NORMAL_WALK,
                confidence=0.8,
            )

        return BehaviorResult(behavior=BehaviorType.UNKNOWN, confidence=0.4)

    def clear_track(self, track_id: str) -> None:
        """Clear observation history for a track."""
        if track_id in self._tracks:
            del self._tracks[track_id]
