"""Tests for V3 multi-target tracker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.tracking.multi_target_tracker_v3 import (
    MultiTargetTrackerV3, TrackState,
)
from plugins.indoor_fence.protocols import FusionOutput, SensorType


def test_tracker_creation():
    tracker = MultiTargetTrackerV3(max_age=30, min_hits=3)
    assert tracker.get_active_tracks() == []


def test_tracker_single_target():
    tracker = MultiTargetTrackerV3(max_age=30, min_hits=1)
    for i in range(5):
        outputs = [FusionOutput(
            track_id=0, position_3d=(float(i), 2.0, 0.0),
            velocity_3d=(1.0, 0.0, 0.0), confidence=0.9,
            sources=[SensorType.CAMERA],
        )]
        tracks = tracker.update(outputs)

    active = tracker.get_active_tracks()
    assert len(active) >= 1
    assert active[0].state == TrackState.ACTIVE


def test_tracker_multiple_targets():
    tracker = MultiTargetTrackerV3(max_age=30, min_hits=1)
    for i in range(5):
        outputs = [
            FusionOutput(track_id=0, position_3d=(float(i), 2.0, 0.0),
                        velocity_3d=(1.0, 0.0, 0.0), confidence=0.9,
                        sources=[SensorType.CAMERA]),
            FusionOutput(track_id=1, position_3d=(8.0 - float(i), 4.0, 0.0),
                        velocity_3d=(-1.0, 0.0, 0.0), confidence=0.9,
                        sources=[SensorType.CAMERA]),
        ]
        tracks = tracker.update(outputs)

    active = tracker.get_active_tracks()
    assert len(active) == 2


def test_tracker_lost_target():
    tracker = MultiTargetTrackerV3(max_age=3, min_hits=1)
    # Target appears
    for i in range(3):
        tracker.update([FusionOutput(
            track_id=0, position_3d=(float(i), 2.0, 0.0),
            confidence=0.9, sources=[SensorType.CAMERA],
        )])

    # Target disappears
    for i in range(5):
        tracker.update([])

    active = tracker.get_active_tracks()
    assert len(active) == 0  # Should be deleted after max_age
