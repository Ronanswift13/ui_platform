"""Tests for enhanced BaseAdapter interface."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import time
import json
from plugins.indoor_fence.adapters.base_adapter import (
    BaseAdapter, AdapterConfig, AdapterStatus, AdapterHealth,
)
from plugins.indoor_fence.protocols import SensorData, SensorType, AdapterMode


class MockAdapter(BaseAdapter):
    """Concrete adapter for testing."""

    def __init__(self):
        super().__init__(AdapterConfig(adapter_type="mock"))
        self._connected = False

    @property
    def sensor_type(self) -> SensorType:
        return SensorType.CAMERA

    def connect(self) -> bool:
        self._connected = True
        self._status = AdapterStatus.CONNECTED
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def _read_live(self) -> SensorData:
        return SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=time.time(),
            data={"test": True},
            confidence=1.0,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected


def test_adapter_lifecycle():
    adapter = MockAdapter()
    assert adapter.status == AdapterStatus.DISCONNECTED
    assert adapter.mode == AdapterMode.LIVE

    adapter.connect()
    assert adapter.status == AdapterStatus.CONNECTED
    assert adapter.is_connected is True

    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_adapter_get_data_live():
    adapter = MockAdapter()
    adapter.connect()
    data = adapter.get_data()
    assert data is not None
    assert data.sensor_type == SensorType.CAMERA
    assert data.is_simulated is False


def test_adapter_simulation_mode():
    adapter = MockAdapter()
    adapter.set_mode(AdapterMode.SIMULATED)
    assert adapter.mode == AdapterMode.SIMULATED
    data = adapter.get_data()
    assert data is not None
    assert data.is_simulated is True


def test_adapter_health():
    adapter = MockAdapter()
    adapter.connect()
    for _ in range(3):
        adapter.get_data()
    health = adapter.get_health()
    assert isinstance(health, AdapterHealth)
    assert health.frames_read >= 3
    assert health.error_count == 0


def test_adapter_recording(tmp_path):
    adapter = MockAdapter()
    adapter.connect()
    rec_path = str(tmp_path / "recording.jsonl")
    adapter.start_recording(rec_path)
    assert adapter.is_recording is True

    for _ in range(3):
        adapter.get_data()

    adapter.stop_recording()
    assert adapter.is_recording is False

    with open(rec_path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    entry = json.loads(lines[0])
    assert "sensor_type" in entry


def test_adapter_replay(tmp_path):
    rec_path = str(tmp_path / "replay.jsonl")
    with open(rec_path, "w") as f:
        for i in range(3):
            entry = SensorData(
                sensor_type=SensorType.CAMERA,
                timestamp=1000.0 + i,
                data={"frame": i},
                confidence=0.9,
                is_simulated=False,
            )
            f.write(entry.model_dump_json() + "\n")

    adapter = MockAdapter()
    adapter.load_replay(rec_path)
    assert adapter.mode == AdapterMode.REPLAY

    data = adapter.get_data()
    assert data is not None
    assert data.data["frame"] == 0

    data = adapter.get_data()
    assert data.data["frame"] == 1


def test_adapter_weight():
    adapter = MockAdapter()
    assert 0.0 <= adapter.get_weight() <= 1.0
