#!/usr/bin/env python3
"""
Acoustic Monitoring - Standalone Tests
"""
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from fastapi.testclient import TestClient


class TestAcousticMonitoringStandalone(unittest.TestCase):
    """Test acoustic monitoring plugin in standalone mode."""

    @classmethod
    def setUpClass(cls):
        from plugins.acoustic_monitoring.plugin import Plugin
        cls.plugin = Plugin.create_standalone()

    @classmethod
    def tearDownClass(cls):
        cls.plugin.shutdown()

    def test_plugin_creation(self):
        """Test plugin is created and initialized."""
        self.assertIsNotNone(self.plugin)
        self.assertEqual(self.plugin.id, "acoustic_monitoring")
        self.assertTrue(self.plugin._is_initialized)

    def test_healthcheck(self):
        """Test healthcheck returns proper HealthStatus."""
        from darkbreaker_sdk.interfaces import HealthStatus
        health = self.plugin.healthcheck()
        self.assertIsInstance(health, HealthStatus)
        self.assertTrue(health.healthy)

    def test_process_normal_audio(self):
        """Test processing normal audio data."""
        sample_rate = 16000
        duration = 2.0
        rng = np.random.default_rng(20260417)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = (np.sin(2 * np.pi * 50 * t) + 0.1 * rng.standard_normal(len(t))).astype(np.float32)

        result = self.plugin.process({
            "audio": audio,
            "sample_rate": sample_rate,
            "device_id": "test_device",
        })

        self.assertTrue(result["success"])
        self.assertIn("anomaly_detected", result)
        self.assertIn("anomaly_score", result)
        self.assertIn("frequency_analysis", result)

    def test_process_without_audio(self):
        """Test processing with no audio generates mock data."""
        result = self.plugin.process({
            "device_id": "test_device",
        })
        self.assertTrue(result["success"])
        self.assertIn("anomaly_detected", result)

    def test_plugin_status(self):
        """Test get_plugin_status returns expected fields."""
        status = self.plugin.get_plugin_status()
        self.assertEqual(status["plugin_id"], "acoustic_monitoring")
        self.assertIn("capabilities", status)

    def test_plugin_info(self):
        """Test plugin_info property."""
        info = self.plugin.plugin_info
        self.assertEqual(info["id"], "acoustic_monitoring")
        self.assertIn("capabilities", info)

    def test_shutdown_and_reinit(self):
        """Test shutdown and reinitialization."""
        from plugins.acoustic_monitoring.plugin import Plugin
        plugin = Plugin.create_standalone()
        self.assertTrue(plugin._is_initialized)
        plugin.shutdown()
        self.assertFalse(plugin._is_initialized)

    def test_standalone_smoke_route_accepts_simulated_sample(self):
        """Standalone smoke route should submit a simulated sample."""
        from darkbreaker_sdk.standalone import StandalonePluginRunner

        runner = StandalonePluginRunner(
            self.plugin,
            plugin_templates_dir=Path(__file__).parent.parent / "standalone" / "templates",
            plugin_static_dir=Path(__file__).parent.parent / "standalone" / "static",
        )
        client = TestClient(runner.app)

        response = client.post("/api/acoustic/smoke", json={"device_id": "acoustic_smoke"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["metadata"]["modality"], "acoustic")


if __name__ == "__main__":
    unittest.main()
