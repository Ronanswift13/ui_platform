#!/usr/bin/env python3
"""Device Monitoring - Standalone Tests"""
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from fastapi.testclient import TestClient


class TestDeviceMonitoringStandalone(unittest.TestCase):
    """Test device monitoring plugin in standalone mode."""

    @classmethod
    def setUpClass(cls):
        from plugins.device_monitoring.plugin import Plugin
        cls.plugin = Plugin.create_standalone()

    def test_create_standalone(self):
        """Test plugin is created via create_standalone."""
        self.assertIsNotNone(self.plugin)
        self.assertEqual(self.plugin.id, "device_monitoring")

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
        from darkbreaker_sdk.standalone import StandalonePluginRunner
        runner = StandalonePluginRunner(
            self.plugin,
            plugin_templates_dir=Path(__file__).parent.parent / "standalone" / "templates",
            plugin_static_dir=Path(__file__).parent.parent / "standalone" / "static",
        )
        self.assertIsNotNone(runner)

    def test_standalone_smoke_route_accepts_simulated_sample(self):
        """Standalone smoke route should submit a simulated sample."""
        from darkbreaker_sdk.standalone import StandalonePluginRunner

        runner = StandalonePluginRunner(
            self.plugin,
            plugin_templates_dir=Path(__file__).parent.parent / "standalone" / "templates",
            plugin_static_dir=Path(__file__).parent.parent / "standalone" / "static",
        )
        client = TestClient(runner.app)

        response = client.post("/api/device/smoke", json={"device_id": "edge_smoke"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["metadata"]["modality"], "device")


if __name__ == "__main__":
    unittest.main()
