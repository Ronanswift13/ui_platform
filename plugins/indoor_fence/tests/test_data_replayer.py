"""Tests for data replay engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import json
import time
from plugins.indoor_fence.standalone.data_replayer import DataReplayer
from plugins.indoor_fence.protocols import SensorData, SensorType


@pytest.fixture
def replay_dir(tmp_path):
    """Create sample recording data."""
    data_file = tmp_path / "camera.jsonl"
    with open(data_file, "w") as f:
        for i in range(10):
            entry = SensorData(
                sensor_type=SensorType.CAMERA,
                timestamp=1000.0 + i * 0.1,
                data={"frame": i, "detections": []},
                confidence=0.9,
            )
            f.write(entry.model_dump_json() + "\n")
    return tmp_path


def test_replayer_load(replay_dir):
    replayer = DataReplayer(str(replay_dir))
    assert replayer.total_frames >= 10


def test_replayer_next(replay_dir):
    replayer = DataReplayer(str(replay_dir))
    data = replayer.next()
    assert data is not None
    assert data[SensorType.CAMERA].data["frame"] == 0

    data = replayer.next()
    assert data[SensorType.CAMERA].data["frame"] == 1


def test_replayer_reset(replay_dir):
    replayer = DataReplayer(str(replay_dir))
    replayer.next()
    replayer.next()
    replayer.reset()
    data = replayer.next()
    assert data[SensorType.CAMERA].data["frame"] == 0


def test_replayer_exhausted(replay_dir):
    replayer = DataReplayer(str(replay_dir))
    for _ in range(10):
        replayer.next()
    data = replayer.next()
    assert data is None  # No more frames
