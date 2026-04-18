"""Tests for behavior recognition module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import time
import pytest
from plugins.indoor_fence.detection.behavior_recognizer import (
    BehaviorRecognizerV3, BehaviorType, BehaviorResult,
)
from plugins.indoor_fence.detection.pose_estimator import PoseResult, PostureType


def test_behavior_types():
    assert BehaviorType.NORMAL_WALK.value == "normal_walk"
    assert BehaviorType.PROLONGED_STAY.value == "prolonged_stay"
    assert BehaviorType.CLIMBING.value == "climbing"
    assert BehaviorType.CROSSING.value == "crossing"
    assert BehaviorType.FALLEN.value == "fallen"


def test_behavior_recognizer_creation():
    rec = BehaviorRecognizerV3(
        window_size=30,
        prolonged_stay_threshold_s=30.0,
    )
    assert rec.window_size == 30


def test_behavior_normal_walk():
    rec = BehaviorRecognizerV3(window_size=5)
    # Simulate walking person - different positions each frame
    for i in range(5):
        pose = PoseResult(
            keypoints=[(0.3 + i * 0.01, 0.5, 0.9)] * 17,
            posture=PostureType.STANDING,
            confidence=0.9,
        )
        result = rec.update("person_1", pose, position=(float(i), 1.0))

    assert result is not None
    assert result.behavior in (BehaviorType.NORMAL_WALK, BehaviorType.UNKNOWN)


def test_behavior_prolonged_stay():
    rec = BehaviorRecognizerV3(
        window_size=20,
        prolonged_stay_threshold_s=0.1,  # Very short for testing
        movement_threshold_m=0.3,
    )
    for i in range(15):
        pose = PoseResult(
            keypoints=[(0.5, 0.5, 0.9)] * 17,
            posture=PostureType.STANDING,
            confidence=0.9,
        )
        result = rec.update("person_1", pose, position=(5.0, 3.0))
        time.sleep(0.02)

    assert result.behavior == BehaviorType.PROLONGED_STAY


def test_behavior_fallen():
    rec = BehaviorRecognizerV3(window_size=3)
    for _ in range(5):
        pose = PoseResult(
            keypoints=[(0.5, 0.5, 0.9)] * 17,
            posture=PostureType.FALLEN,
            confidence=0.9,
        )
        result = rec.update("person_1", pose, position=(5.0, 3.0))

    assert result.behavior == BehaviorType.FALLEN


def test_behavior_clear_track():
    rec = BehaviorRecognizerV3(window_size=5)
    pose = PoseResult(
        keypoints=[(0.5, 0.5, 0.9)] * 17,
        posture=PostureType.STANDING,
        confidence=0.9,
    )
    rec.update("person_1", pose, position=(5.0, 3.0))
    rec.clear_track("person_1")
    # After clearing, should start fresh
    result = rec.update("person_1", pose, position=(5.0, 3.0))
    assert result is not None
