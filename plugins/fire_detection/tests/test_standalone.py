#!/usr/bin/env python3
"""Fire Detection - Standalone Tests"""
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


class TestFireDetectionStandalone(unittest.TestCase):
    """Test fire detection plugin in standalone mode."""

    @classmethod
    def setUpClass(cls):
        from plugins.fire_detection.plugin import Plugin
        cls.plugin = Plugin.create_standalone()

    def test_create_standalone(self):
        """Test plugin is created via create_standalone."""
        self.assertIsNotNone(self.plugin)
        self.assertEqual(self.plugin.id, "fire_detection")

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

    def test_detect_returns_contract_keys(self):
        """Standalone smoke test should expose minimal contract semantics."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.plugin.detect(frame)
        for key in [
            "semantic_type",
            "severity",
            "confidence",
            "reason",
            "recommended_action",
            "runtime_mode",
            "capability_states",
        ]:
            self.assertIn(key, result)
        self.assertEqual(result["semantic_type"], "visual_detection")
        self.assertEqual(result["runtime_mode"]["analysis_mode"], "simulation_only")

    def test_runner_creates_app(self):
        """Test StandalonePluginRunner can create Flask app."""
        from darkbreaker_sdk.standalone import StandalonePluginRunner
        runner = StandalonePluginRunner(
            self.plugin,
            plugin_templates_dir=Path(__file__).parent.parent / "standalone" / "templates",
            plugin_static_dir=Path(__file__).parent.parent / "standalone" / "static",
        )
        self.assertIsNotNone(runner)


if __name__ == "__main__":
    unittest.main()
