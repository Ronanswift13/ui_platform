"""Tests for V3.0 data protocol models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.protocols import (
    SensorData, SensorType, AdapterMode,
    DetectionResult, PoseKeypoint,
    RiskAssessment, PersonStateV3,
    FusionInput, FusionOutput, GlobalAlarmLevelV3,
)


def test_sensor_data_creation():
    sd = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=1000.0,
        data={"bbox": [0.1, 0.2, 0.3, 0.4]},
        confidence=0.95,
    )
    assert sd.sensor_type == SensorType.CAMERA
    assert sd.is_simulated is False


def test_sensor_data_simulated():
    sd = SensorData(
        sensor_type=SensorType.UWB,
        timestamp=1000.0,
        data={"x": 1.0, "y": 2.0, "z": 0.5},
        confidence=0.8,
        is_simulated=True,
    )
    assert sd.is_simulated is True


def test_detection_result():
    dr = DetectionResult(
        track_id=1,
        position_3d=(1.0, 2.0, 0.0),
        velocity_3d=(0.1, 0.0, 0.0),
        confidence=0.9,
        fusion_sources=["camera", "lidar"],
    )
    assert dr.position_3d == (1.0, 2.0, 0.0)
    assert dr.bbox is None
    assert dr.behavior is None


def test_detection_result_with_pose():
    kp = PoseKeypoint(x=0.5, y=0.3, confidence=0.9, name="nose")
    dr = DetectionResult(
        track_id=2,
        position_3d=(3.0, 4.0, 0.0),
        velocity_3d=(0.0, 0.0, 0.0),
        confidence=0.85,
        fusion_sources=["camera"],
        pose_keypoints=[kp],
        behavior="normal",
    )
    assert len(dr.pose_keypoints) == 1
    assert dr.behavior == "normal"


def test_risk_assessment():
    ra = RiskAssessment(
        person_state=PersonStateV3.CLIMBING,
        risk_score=0.8,
        zone_id="cabinet_1",
        violations=["unauthorized_zone", "climbing"],
        recommended_action="alarm_red",
    )
    assert ra.risk_score == 0.8
    assert PersonStateV3.CLIMBING.value == "climbing"


def test_person_state_v3_has_all_states():
    states = [s.value for s in PersonStateV3]
    assert "normal" in states
    assert "climbing" in states
    assert "prolonged_stay" in states
    assert "fallen" in states
    assert "multi_person" in states


def test_adapter_mode_enum():
    assert AdapterMode.LIVE.value == "live"
    assert AdapterMode.SIMULATED.value == "simulated"
    assert AdapterMode.REPLAY.value == "replay"


def test_fusion_input_output():
    fi = FusionInput(
        sensor_type=SensorType.LIDAR,
        timestamp=1000.0,
        position_2d=(2.0, 3.0),
        confidence=0.9,
    )
    assert fi.position_3d is None

    fo = FusionOutput(
        track_id=1,
        position_3d=(2.0, 3.0, 0.0),
        velocity_3d=(0.0, 0.0, 0.0),
        confidence=0.85,
        sources=[SensorType.LIDAR, SensorType.CAMERA],
        covariance_diag=(0.01, 0.01, 0.05),
    )
    assert len(fo.sources) == 2
