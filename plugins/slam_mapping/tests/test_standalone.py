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

    def test_runner_creates_app(self):
        from darkbreaker_sdk.standalone import StandalonePluginRunner
        runner = StandalonePluginRunner(
            self.plugin,
            plugin_templates_dir=Path(__file__).parent.parent / "standalone" / "templates",
        )
        self.assertIsNotNone(runner)


if __name__ == "__main__":
    unittest.main()
