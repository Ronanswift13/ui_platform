"""Tests for IMU adapter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.adapters.imu_adapter import IMUAdapter, IMUConfig
from plugins.indoor_fence.protocols import SensorType


def test_imu_adapter_creation():
    config = IMUConfig(protocol="simulate")
    adapter = IMUAdapter(config)
    assert adapter.sensor_type == SensorType.IMU


def test_imu_adapter_simulation():
    config = IMUConfig(protocol="simulate")
    adapter = IMUAdapter(config)
    adapter.connect()
    data = adapter.get_data()
    assert data is not None
    assert "acceleration" in data.data
    assert "gyroscope" in data.data
    acc = data.data["acceleration"]
    assert "x" in acc and "y" in acc and "z" in acc
