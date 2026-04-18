"""Simulator 与 CameraAdapter 集成测试"""
import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from plugins.indoor_fence.adapters.simulator import (
    Simulator,
    SimulatorConfig,
    PathType,
    PersonPath,
)
from plugins.indoor_fence.adapters.camera_adapter import CameraAdapter, CameraConfig
from plugins.indoor_fence.protocols import SensorType


class TestSimulatorCameraIntegration:
    """CameraAdapter + Simulator 集成"""

    def _make_adapter(self):
        cfg = CameraConfig(simulate_if_unavailable=True)
        adapter = CameraAdapter(cfg)
        adapter.connect()
        return adapter

    def test_simulator_provides_detections(self):
        adapter = self._make_adapter()
        sim = Simulator(SimulatorConfig(
            num_persons=2,
            sensor_types=[SensorType.CAMERA],
        ))
        adapter.set_simulator(sim)

        dets = adapter._simulate_raw_detections((480, 640, 3))
        assert len(dets) == 2
        for d in dets:
            assert 'bbox' in d
            assert 'confidence' in d

    def test_fallback_without_simulator(self):
        adapter = self._make_adapter()
        assert adapter._simulator is None
        dets = adapter._simulate_raw_detections((480, 640, 3))
        assert 1 <= len(dets) <= 3

    def test_renderer_frame_generation(self):
        from plugins.indoor_fence.standalone.simulation_renderer import SimulationRenderer
        adapter = self._make_adapter()
        sim = Simulator(SimulatorConfig(
            num_persons=1,
            sensor_types=[SensorType.CAMERA],
        ))
        renderer = SimulationRenderer(
            scene_bounds=(0.0, 0.0, 10.0, 6.0),
            resolution=(640, 480),
        )
        adapter.set_simulator(sim, renderer)

        # Advance one step so persons have positions
        sim.step()
        frame = adapter.get_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_legacy_frame_without_renderer(self):
        adapter = self._make_adapter()
        frame = adapter.get_frame()
        assert frame is not None
        assert frame.shape[2] == 3


class TestScenarioLoading:
    """场景 JSON 文件加载"""

    def _scenario_dir(self):
        return Path(__file__).parent.parent / "configs" / "scenarios"

    def test_basic_patrol_loads(self):
        path = self._scenario_dir() / "basic_patrol.json"
        sim = Simulator.from_scenario(str(path))
        assert sim.num_persons == 1

    def test_cross_line_loads(self):
        path = self._scenario_dir() / "cross_line.json"
        sim = Simulator.from_scenario(str(path))
        assert sim.num_persons >= 1

    def test_multi_person_loads(self):
        path = self._scenario_dir() / "multi_person.json"
        sim = Simulator.from_scenario(str(path))
        assert sim.num_persons >= 2

    def test_all_scenarios_produce_camera_data(self):
        for name in ["basic_patrol", "cross_line", "multi_person"]:
            path = self._scenario_dir() / f"{name}.json"
            sim = Simulator.from_scenario(str(path))
            data = sim.step()
            assert SensorType.CAMERA in data
            cam = data[SensorType.CAMERA]
            assert "detections" in cam.data
            assert len(cam.data["detections"]) == sim.num_persons


class TestTrajectoryPhysics:
    """轨迹连续性验证"""

    def test_fixed_path_continuous(self):
        sim = Simulator(SimulatorConfig(
            num_persons=1,
            path_type=PathType.FIXED,
            waypoints=[[1.0, 1.0], [5.0, 1.0], [9.0, 1.0]],
            speed_m_per_step=0.1,
            sensor_types=[SensorType.CAMERA],
        ))
        prev_pos = None
        for _ in range(100):
            data = sim.step()
            det = data[SensorType.CAMERA].data["detections"][0]
            pos = det["position"]
            if prev_pos is not None:
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                displacement = math.sqrt(dx * dx + dy * dy)
                assert displacement < 0.5, f"Teleport detected: {displacement:.3f}m"
            prev_pos = pos

    def test_random_path_bounded(self):
        bounds = (0.0, 0.0, 10.0, 6.0)
        sim = Simulator(SimulatorConfig(
            num_persons=1,
            path_type=PathType.RANDOM,
            scene_bounds=bounds,
            sensor_types=[SensorType.CAMERA],
        ))
        for _ in range(200):
            data = sim.step()
            det = data[SensorType.CAMERA].data["detections"][0]
            pos = det["position"]
            assert -1.0 <= pos[0] <= 11.0, f"X out of bounds: {pos[0]}"
            assert -1.0 <= pos[1] <= 7.0, f"Y out of bounds: {pos[1]}"

    def test_cross_line_oscillates(self):
        sim = Simulator(SimulatorConfig(
            num_persons=1,
            path_type=PathType.CROSS_LINE,
            scene_bounds=(0.0, 0.0, 10.0, 6.0),
            yellow_line_y=3.0,
            sensor_types=[SensorType.CAMERA],
        ))
        min_y = float('inf')
        max_y = float('-inf')
        for _ in range(200):
            data = sim.step()
            det = data[SensorType.CAMERA].data["detections"][0]
            y = det["position"][1]
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        assert max_y - min_y > 2.0, "Cross-line path should oscillate across a range"
