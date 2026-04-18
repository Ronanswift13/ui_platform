"""Tests for pose estimation module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from plugins.indoor_fence.detection.pose_estimator import (
    PoseEstimatorV3, PoseResult, PostureType, KEYPOINT_NAMES,
)


def test_pose_result():
    result = PoseResult(
        keypoints=[(0.5, 0.3, 0.9)] * 17,
        posture=PostureType.STANDING,
        confidence=0.85,
    )
    assert len(result.keypoints) == 17
    assert result.posture == PostureType.STANDING


def test_pose_estimator_simulation():
    estimator = PoseEstimatorV3(model_path=None)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (100, 100, 200, 400)
    result = estimator.estimate(frame, bbox)
    assert result is not None
    assert len(result.keypoints) == 17
    assert result.posture in PostureType


def test_posture_types():
    assert PostureType.STANDING.value == "standing"
    assert PostureType.BENDING.value == "bending"
    assert PostureType.CLIMBING.value == "climbing"
    assert PostureType.FALLEN.value == "fallen"
    assert PostureType.CROUCHING.value == "crouching"


def test_keypoint_names():
    assert len(KEYPOINT_NAMES) == 17
    assert "nose" in KEYPOINT_NAMES
    assert "left_ankle" in KEYPOINT_NAMES
