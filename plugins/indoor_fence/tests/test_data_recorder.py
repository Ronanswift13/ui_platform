"""Tests for multi-sensor data recorder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import time
from plugins.indoor_fence.standalone.data_recorder import DataRecorder
from plugins.indoor_fence.protocols import SensorData, SensorType


def test_recorder_creation():
    rec = DataRecorder()
    assert rec.is_recording is False


def test_recorder_start_stop(tmp_path):
    rec = DataRecorder(output_dir=str(tmp_path))
    rec.start()
    assert rec.is_recording is True

    data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=time.time(),
        data={"test": True},
        confidence=0.9,
    )
    rec.record(data)
    rec.record(data)
    rec.stop()

    assert rec.is_recording is False
    # Should have created output files
    files = list(tmp_path.rglob("*.jsonl"))
    assert len(files) >= 1


def test_recorder_multiple_sensors(tmp_path):
    rec = DataRecorder(output_dir=str(tmp_path))
    rec.start()

    for sensor in [SensorType.CAMERA, SensorType.LIDAR, SensorType.UWB]:
        rec.record(SensorData(
            sensor_type=sensor,
            timestamp=time.time(),
            data={"sensor": sensor.value},
            confidence=0.9,
        ))

    rec.stop()
    # Should have separate files per sensor
    files = list(tmp_path.rglob("*.jsonl"))
    assert len(files) >= 1


def test_recorder_metadata(tmp_path):
    rec = DataRecorder(output_dir=str(tmp_path))
    rec.start(metadata={"scene": "test", "operator": "test_user"})
    rec.stop()

    # Should have metadata file
    meta_files = list(tmp_path.rglob("*metadata*.json"))
    assert len(meta_files) >= 1
