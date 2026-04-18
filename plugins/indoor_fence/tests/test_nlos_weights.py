"""Tests for NLOS detection and dynamic weight management."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.fusion.nlos_detector import NLOSDetector
from plugins.indoor_fence.core.fusion.weight_manager import (
    WeightManager, SensorWeight,
)
from plugins.indoor_fence.protocols import SensorType


def test_nlos_detector_los():
    detector = NLOSDetector()
    # Line-of-sight: low delay spread, high first-path power
    prob = detector.classify(
        first_path_power=-70.0,
        total_power=-65.0,
        delay_spread_ns=5.0,
    )
    assert prob < 0.3  # Should be classified as LOS


def test_nlos_detector_nlos():
    detector = NLOSDetector()
    # Non-line-of-sight: high delay spread, low first-path power
    prob = detector.classify(
        first_path_power=-90.0,
        total_power=-65.0,
        delay_spread_ns=50.0,
    )
    assert prob > 0.7  # Should be classified as NLOS


def test_weight_manager_default():
    wm = WeightManager()
    weights = wm.get_weights()
    assert SensorType.CAMERA in weights
    assert SensorType.LIDAR in weights
    assert all(0.0 <= w.weight <= 1.0 for w in weights.values())


def test_weight_manager_sensor_failure():
    wm = WeightManager()
    wm.report_sensor_status(SensorType.CAMERA, healthy=False)
    weights = wm.get_weights()
    assert weights[SensorType.CAMERA].weight < weights[SensorType.LIDAR].weight


def test_weight_manager_nlos_degrades_uwb():
    wm = WeightManager()
    wm.report_nlos(SensorType.UWB, nlos_probability=0.9)
    weights = wm.get_weights()
    assert weights[SensorType.UWB].weight < 1.0


def test_weight_manager_night_mode():
    wm = WeightManager()
    wm.set_environment(low_light=True)
    weights = wm.get_weights()
    # Camera should be degraded in low light
    assert weights[SensorType.CAMERA].weight < weights[SensorType.LIDAR].weight
