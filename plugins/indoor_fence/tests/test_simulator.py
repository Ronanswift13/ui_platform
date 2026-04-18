"""Tests for unified simulator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.adapters.simulator import (
    Simulator, SimulatorConfig, PersonPath, PathType,
)
from plugins.indoor_fence.protocols import SensorType


def test_simulator_creation():
    config = SimulatorConfig(
        num_persons=2,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
    )
    sim = Simulator(config)
    assert sim.num_persons == 2


def test_simulator_step():
    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
    )
    sim = Simulator(config)
    data = sim.step()
    assert SensorType.CAMERA in data
    assert SensorType.LIDAR in data
    for sd in data.values():
        assert sd.is_simulated is True


def test_simulator_with_uwb():
    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR, SensorType.UWB],
    )
    sim = Simulator(config)
    data = sim.step()
    assert SensorType.UWB in data
    uwb_data = data[SensorType.UWB]
    assert "positions" in uwb_data.data


def test_simulator_random_path():
    config = SimulatorConfig(
        num_persons=3,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA],
        path_type=PathType.RANDOM,
    )
    sim = Simulator(config)
    data1 = sim.step()
    for _ in range(10):
        sim.step()
    data2 = sim.step()
    pos1 = data1[SensorType.CAMERA].data["detections"][0]["position"]
    pos2 = data2[SensorType.CAMERA].data["detections"][0]["position"]
    assert pos1 != pos2


def test_simulator_cross_line_scenario():
    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA],
        path_type=PathType.CROSS_LINE,
        yellow_line_y=2.5,
    )
    sim = Simulator(config)
    crossed = False
    for _ in range(100):
        data = sim.step()
        pos = data[SensorType.CAMERA].data["detections"][0]["position"]
        if pos[1] > 2.5:
            crossed = True
            break
    assert crossed, "Person should cross yellow line in CROSS_LINE scenario"


def test_simulator_fault_injection():
    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
    )
    sim = Simulator(config)
    sim.inject_fault(SensorType.CAMERA, "offline")
    data = sim.step()
    assert data[SensorType.CAMERA].confidence == 0.0
    assert SensorType.LIDAR in data

    sim.clear_fault(SensorType.CAMERA)
    data = sim.step()
    assert data[SensorType.CAMERA].confidence > 0.0


def test_load_scenario_file():
    scenario_dir = Path(__file__).parent.parent / "configs" / "scenarios"
    scenario_file = scenario_dir / "basic_patrol.json"
    if not scenario_file.exists():
        pytest.skip("Scenario file not created yet")

    sim = Simulator.from_scenario(str(scenario_file))
    assert sim.num_persons >= 1
    data = sim.step()
    assert len(data) > 0
