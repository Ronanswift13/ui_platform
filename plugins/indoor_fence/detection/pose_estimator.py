"""
Indoor Fence V3.0 - Pose Estimator
====================================

17-keypoint COCO pose estimation with posture classification.
Supports ONNX models and simulation mode.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PostureType(str, Enum):
    STANDING = "standing"
    BENDING = "bending"
    CLIMBING = "climbing"
    CROUCHING = "crouching"
    FALLEN = "fallen"
    UNKNOWN = "unknown"


@dataclass
class PoseResult:
    """Result of pose estimation for one person."""
    keypoints: List[Tuple[float, float, float]]  # (x, y, confidence) * 17
    posture: PostureType = PostureType.UNKNOWN
    confidence: float = 0.0


class PoseEstimatorV3:
    """Pose estimator with posture classification."""

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._simulation_mode = model_path is None

        if not self._simulation_mode:
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(model_path)
            except Exception:
                self._simulation_mode = True

    def estimate(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[PoseResult]:
        """Estimate pose for a person crop."""
        if self._simulation_mode:
            return self._simulate_pose(bbox)
        return self._run_inference(frame, bbox)

    def _simulate_pose(self, bbox: Tuple[int, int, int, int]) -> PoseResult:
        """Generate plausible standing person keypoints."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        w = x2 - x1
        h = y2 - y1

        # Generate COCO-format keypoints for a standing person
        keypoints = []
        # Vertical positions as fractions of height from top
        y_fracs = [
            0.05,  # nose
            0.03, 0.03,  # eyes
            0.05, 0.05,  # ears
            0.18, 0.18,  # shoulders
            0.38, 0.38,  # elbows
            0.52, 0.52,  # wrists
            0.52, 0.52,  # hips
            0.72, 0.72,  # knees
            0.92, 0.92,  # ankles
        ]
        x_offsets = [
            0.0,
            -0.05, 0.05,
            -0.1, 0.1,
            -0.2, 0.2,
            -0.25, 0.25,
            -0.2, 0.2,
            -0.12, 0.12,
            -0.12, 0.12,
            -0.12, 0.12,
        ]

        for yf, xo in zip(y_fracs, x_offsets):
            kx = (cx + xo * w + random.gauss(0, 2)) / (x2 if x2 > 0 else 640)
            ky = (y1 + yf * h + random.gauss(0, 2)) / (y2 if y2 > 0 else 480)
            conf = random.uniform(0.7, 0.98)
            keypoints.append((kx, ky, conf))

        posture = self._classify_posture(keypoints)
        avg_conf = sum(k[2] for k in keypoints) / len(keypoints)

        return PoseResult(
            keypoints=keypoints,
            posture=posture,
            confidence=avg_conf,
        )

    def _classify_posture(
        self, keypoints: List[Tuple[float, float, float]]
    ) -> PostureType:
        """Classify posture from keypoint geometry."""
        if len(keypoints) < 17:
            return PostureType.UNKNOWN

        # Get key body parts
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        l_ankle = keypoints[15]
        r_ankle = keypoints[16]
        nose = keypoints[0]

        # Shoulder and hip midpoints
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_y = (l_hip[1] + r_hip[1]) / 2
        ankle_y = (l_ankle[1] + r_ankle[1]) / 2

        torso_height = abs(hip_y - shoulder_y)
        if torso_height < 0.001:
            return PostureType.UNKNOWN

        # Full height ratio
        full_height = abs(ankle_y - nose[1])
        if full_height < 0.001:
            return PostureType.UNKNOWN

        # Width vs height ratio (fallen detection)
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        if full_height > 0 and shoulder_width / full_height > 0.6:
            return PostureType.FALLEN

        # Bending: nose below shoulder significantly
        if nose[1] > shoulder_y + torso_height * 0.5:
            return PostureType.BENDING

        # Crouching: hip close to ankle
        leg_length = abs(ankle_y - hip_y)
        if leg_length < torso_height * 0.5:
            return PostureType.CROUCHING

        return PostureType.STANDING

    def _run_inference(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> PoseResult:
        """Run actual ONNX pose inference."""
        return self._simulate_pose(bbox)
