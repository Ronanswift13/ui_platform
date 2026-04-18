#!/usr/bin/env python3
"""Temperature Monitoring - Standalone Tests"""
import sys
import unittest
from unittest import mock
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from fastapi.testclient import TestClient


class TestTemperatureMonitoringStandalone(unittest.TestCase):
    """Test temperature monitoring plugin in standalone mode."""

    @classmethod
    def setUpClass(cls):
        from plugins.temperature_monitoring.plugin import Plugin
        from darkbreaker_sdk.standalone import StandalonePluginRunner

        cls.plugin = Plugin.create_standalone()
        cls.runner = StandalonePluginRunner(
            cls.plugin,
            plugin_templates_dir=Path(__file__).parent.parent / "standalone" / "templates",
            plugin_static_dir=Path(__file__).parent.parent / "standalone" / "static",
        )
        cls.client = TestClient(cls.runner.app)

    def test_create_standalone(self):
        """Test plugin is created via create_standalone."""
        self.assertIsNotNone(self.plugin)
        self.assertEqual(self.plugin.id, "temperature_monitoring")

    def test_healthcheck(self):
        """Test healthcheck returns HealthStatus."""
        from darkbreaker_sdk.interfaces import HealthStatus
        health = self.plugin.healthcheck()
        self.assertIsInstance(health, HealthStatus)

    def test_infer_returns_list(self):
        """Test infer returns a list."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.plugin.infer(frame, [], None)
        self.assertIsInstance(results, list)

    def test_runner_creates_app(self):
        """Test StandalonePluginRunner can create Flask app."""
        self.assertIsNotNone(self.runner)

    def test_runner_registers_simulation_routes(self):
        """Standalone runner should expose plugin simulation APIs."""
        paths = {getattr(route, "path", "") for route in self.runner.app.router.routes}
        self.assertIn("/api/simulation/bootstrap", paths)
        self.assertIn("/api/simulation/scenarios", paths)
        self.assertIn("/api/simulation/load", paths)
        self.assertIn("/api/simulation/step", paths)

    def test_dashboard_contains_simulation_controls(self):
        """Dashboard should render dedicated simulation controls."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("仿真场景", response.text)
        self.assertIn("simulation-scenario", response.text)
        self.assertIn("/api/simulation/bootstrap", response.text)
        self.assertIn("/api/simulation/step", response.text)

    def test_simulation_bootstrap_returns_ready_payload(self):
        """Bootstrap should let the dashboard render without chained 404s."""
        response = self.client.get("/api/simulation/bootstrap")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertIn("scenario", payload)
        self.assertIn("scenarios", payload)
        self.assertIn("initial_data", payload)
        self.assertTrue(payload["initial_data"]["success"])

    def test_simulation_step_uses_standalone_inputs_not_detector_fallback(self):
        """Simulation mode should not depend on detector's internal fallback heatmap."""
        with mock.patch.object(
            self.plugin._detector,
            "_simulate_heatmap",
            side_effect=AssertionError("detector fallback should stay unused in standalone simulation"),
        ):
            response = self.client.post("/api/simulation/step")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["simulation"]["mode"], "simulation")
        self.assertFalse(payload["simulation"]["detector_fallback"])

    def test_simulation_supports_sensor_array_mode(self):
        """Standalone simulation should cover the detector's sensor_readings path."""
        load_response = self.client.post("/api/simulation/load?scenario_id=cable_trench_rise")
        self.assertEqual(load_response.status_code, 200)

        response = self.client.post("/api/simulation/step")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["simulation"]["input_mode"], "sensor_readings")
        self.assertIn("max_temp", payload)


if __name__ == "__main__":
    unittest.main()
