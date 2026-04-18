#!/usr/bin/env python3
"""SLAM Mapping - Standalone Tests"""
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


class TestSLAMMappingStandalone(unittest.TestCase):
    """Test SLAM mapping plugin in standalone mode."""

    @classmethod
    def setUpClass(cls):
        from plugins.slam_mapping.plugin import Plugin
        cls.plugin = Plugin.create_standalone()

    def test_create_standalone(self):
        self.assertIsNotNone(self.plugin)
        self.assertEqual(self.plugin.PLUGIN_ID, "slam_mapping")

    def test_healthcheck(self):
        from darkbreaker_sdk.interfaces import HealthStatus
        health = self.plugin.healthcheck()
        self.assertIsInstance(health, HealthStatus)

    def test_infer_returns_list(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.plugin.infer(frame, [], None)
        self.assertIsInstance(results, list)

    def test_process_point_cloud(self):
        points = np.random.randn(100, 3)
        result = self.plugin.process_point_cloud(points)
        self.assertIsInstance(result, dict)


def test_runner_creates_app():
    from darkbreaker_sdk.standalone import StandalonePluginRunner
    from plugins.slam_mapping.plugin import Plugin

    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent.parent / "standalone" / "templates",
        plugin_static_dir=Path(__file__).parent.parent / "standalone" / "static",
    )
    assert runner is not None


def test_create_runner_registers_simulation_routes():
    from fastapi.testclient import TestClient
    from plugins.slam_mapping.standalone.app import create_runner

    runner = create_runner()
    client = TestClient(runner.app)

    scenarios_resp = client.get("/api/simulator/scenarios")
    assert scenarios_resp.status_code == 200
    scenarios = scenarios_resp.json()["scenarios"]
    assert len(scenarios) >= 2

    step_resp = client.post("/api/simulator/step")
    assert step_resp.status_code == 200
    step_payload = step_resp.json()
    assert step_payload["success"] is True
    assert step_payload["runtime"]["isolation"] == "standalone_isolated_plugin"
    assert "sensor_points" in step_payload["point_cloud"]


def test_template_declares_simulation_isolation_and_modes():
    template = (
        Path(__file__).parent.parent
        / "standalone"
        / "templates"
        / "slam_mapping.html"
    ).read_text(encoding="utf-8")

    for marker in (
        "mode-simulated",
        "mode-real",
        "scenario-select",
        "btn-play-sim",
        "btn-step-sim",
        "独立仿真链路仅使用 standalone 内部实例推演",
        "真实监测模式只读取插件当前状态",
    ):
        assert marker in template


if __name__ == "__main__":
    unittest.main()
