"""Tests for V3 multi-sensor fusion manager."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import time
from plugins.indoor_fence.core.fusion.sensor_fusion_v3 import (
    SensorFusionV3,
)
from plugins.indoor_fence.protocols import SensorType, SensorData, FusionOutput


def test_fusion_v3_creation():
    fusion = SensorFusionV3()
    assert fusion is not None


def test_fusion_camera_only():
    fusion = SensorFusionV3()
    ts = time.time()
    camera_data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        data={"detections": [{"id": 0, "position_2d": (3.0, 2.0), "bbox": [100, 200, 200, 400]}]},
        confidence=0.9,
    )
    outputs = fusion.update({SensorType.CAMERA: camera_data})
    assert len(outputs) >= 1
    assert isinstance(outputs[0], FusionOutput)


def test_fusion_camera_lidar():
    fusion = SensorFusionV3()
    ts = time.time()
    camera_data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        data={"detections": [{"id": 0, "position_2d": (3.0, 2.0)}]},
        confidence=0.9,
    )
    lidar_data = SensorData(
        sensor_type=SensorType.LIDAR,
        timestamp=ts,
        data={"clusters": [{"id": 0, "position_2d": (3.1, 2.1)}]},
        confidence=0.95,
    )
    outputs = fusion.update({
        SensorType.CAMERA: camera_data,
        SensorType.LIDAR: lidar_data,
    })
    assert len(outputs) >= 1
    # Fused position should be between camera and lidar
    pos = outputs[0].position_3d
    assert 2.9 <= pos[0] <= 3.2


def test_fusion_with_uwb():
    fusion = SensorFusionV3()
    ts = time.time()
    # First frame: establish camera track
    camera_data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        data={"detections": [{"id": 0, "position_2d": (3.0, 2.0)}]},
        confidence=0.9,
    )
    fusion.update({SensorType.CAMERA: camera_data})

    # Second frame: camera + UWB together, UWB updates existing track with z
    uwb_data = SensorData(
        sensor_type=SensorType.UWB,
        timestamp=ts + 0.1,
        data={"positions": [{"id": 0, "x": 3.0, "y": 2.0, "z": 0.5}]},
        confidence=0.85,
    )
    camera_data2 = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts + 0.1,
        data={"detections": [{"id": 0, "position_2d": (3.0, 2.0)}]},
        confidence=0.9,
    )
    outputs = fusion.update({
        SensorType.CAMERA: camera_data2,
        SensorType.UWB: uwb_data,
    })
    assert len(outputs) >= 1
    # At least one output should have z component from UWB
    assert any(o.position_3d[2] > 0.0 for o in outputs)


def test_fusion_tracking_over_time():
    fusion = SensorFusionV3()
    for i in range(10):
        ts = time.time() + i * 0.1
        camera_data = SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=ts,
            data={"detections": [{"id": 0, "position_2d": (1.0 + i * 0.1, 2.0)}]},
            confidence=0.9,
        )
        outputs = fusion.update({SensorType.CAMERA: camera_data})

    assert len(outputs) >= 1
    # Track should have velocity
    assert outputs[0].velocity_3d[0] != 0.0


def test_fusion_multi_target_hungarian():
    """Test multi-target association uses Hungarian for optimal matching."""
    fusion = SensorFusionV3()
    ts = time.time()

    # First frame: establish two targets far apart
    cam1 = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        data={"detections": [
            {"position_2d": (1.0, 1.0)},
            {"position_2d": (8.0, 4.0)},
        ]},
        confidence=0.9,
    )
    outputs1 = fusion.update({SensorType.CAMERA: cam1})
    assert len(outputs1) == 2

    # Second frame: slightly shifted
    cam2 = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts + 0.1,
        data={"detections": [
            {"position_2d": (1.1, 1.0)},
            {"position_2d": (7.9, 4.1)},
        ]},
        confidence=0.9,
    )
    outputs2 = fusion.update({SensorType.CAMERA: cam2})
    # Should still be 2 tracks (not 4)
    assert len(outputs2) == 2


def test_fusion_frame_loss_recovery():
    """Test track prediction during frame loss and recovery."""
    fusion = SensorFusionV3()
    ts = time.time()

    # Establish track
    for i in range(5):
        cam = SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=ts + i * 0.1,
            data={"detections": [{"position_2d": (2.0 + i * 0.1, 3.0)}]},
            confidence=0.9,
        )
        outputs = fusion.update({SensorType.CAMERA: cam})

    # Frame loss: send empty data for several frames
    for i in range(3):
        empty = SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=ts + 0.5 + i * 0.1,
            data={"detections": []},
            confidence=0.0,
        )
        outputs = fusion.update({SensorType.CAMERA: empty})

    # Track should still exist (predicted via EKF)
    assert len(outputs) >= 1

    # Recovery: detection returns
    recovery = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts + 0.8,
        data={"detections": [{"position_2d": (2.8, 3.0)}]},
        confidence=0.9,
    )
    outputs = fusion.update({SensorType.CAMERA: recovery})
    assert len(outputs) >= 1


def test_fusion_imu_integration():
    """Test IMU velocity data integration."""
    fusion = SensorFusionV3()
    ts = time.time()

    # First frame: camera only
    cam = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        data={"detections": [{"position_2d": (3.0, 2.0)}]},
        confidence=0.9,
    )
    fusion.update({SensorType.CAMERA: cam})

    # Second frame: camera + IMU with significant acceleration
    cam2 = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts + 0.1,
        data={"detections": [{"position_2d": (3.1, 2.0)}]},
        confidence=0.9,
    )
    imu = SensorData(
        sensor_type=SensorType.IMU,
        timestamp=ts + 0.1,
        data={
            "acceleration": {"x": 2.0, "y": 1.0, "z": 9.81},
            "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        confidence=0.95,
    )
    outputs = fusion.update({SensorType.CAMERA: cam2, SensorType.IMU: imu})
    assert len(outputs) >= 1
    # Should have IMU in sources
    assert any(SensorType.IMU in o.sources for o in outputs)
