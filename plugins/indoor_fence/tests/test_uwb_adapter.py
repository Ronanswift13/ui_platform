"""Tests for UWB adapter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.adapters.uwb_adapter import UWBAdapter, UWBConfig
from plugins.indoor_fence.adapters.base_adapter import AdapterStatus
from plugins.indoor_fence.protocols import SensorType, AdapterMode


def test_uwb_adapter_creation():
    config = UWBConfig(protocol="udp", host="127.0.0.1", port=5000)
    adapter = UWBAdapter(config)
    assert adapter.sensor_type == SensorType.UWB
    assert adapter.mode == AdapterMode.LIVE


def test_uwb_adapter_simulation():
    config = UWBConfig(protocol="simulate")
    adapter = UWBAdapter(config)
    adapter.connect()
    assert adapter.is_connected
    data = adapter.get_data()
    assert data is not None
    assert data.sensor_type == SensorType.UWB
    assert "positions" in data.data
    positions = data.data["positions"]
    assert len(positions) > 0
    pos = positions[0]
    assert "x" in pos and "y" in pos and "z" in pos


def test_uwb_adapter_nlos_indicator():
    config = UWBConfig(protocol="simulate", simulate_nlos=True)
    adapter = UWBAdapter(config)
    adapter.connect()
    data = adapter.get_data()
    positions = data.data["positions"]
    for pos in positions:
        assert "nlos_probability" in pos
        assert 0.0 <= pos["nlos_probability"] <= 1.0


def test_uwb_adapter_disconnect():
    config = UWBConfig(protocol="simulate")
    adapter = UWBAdapter(config)
    adapter.connect()
    adapter.disconnect()
    assert not adapter.is_connected
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_uwb_adapter_weight_with_nlos():
    config = UWBConfig(protocol="simulate", simulate_nlos=True, nlos_ratio=0.8)
    adapter = UWBAdapter(config)
    adapter.connect()
    for _ in range(10):
        adapter.get_data()
    weight = adapter.get_weight()
    assert weight < 1.0
