#!/usr/bin/env python3
"""Animal Detection - Standalone Tests"""
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


class TestAnimalDetectionStandalone(unittest.TestCase):
    """Test animal detection plugin in standalone mode."""

    @classmethod
    def setUpClass(cls):
        from plugins.animal_detection.plugin import Plugin
        cls.plugin = Plugin.create_standalone()

    def test_create_standalone(self):
        """Test plugin is created via create_standalone."""
        self.assertIsNotNone(self.plugin)
        self.assertEqual(self.plugin.id, "animal_detection")

    def test_healthcheck(self):
        """Test healthcheck returns valid response."""
        health = self.plugin.healthcheck()
        self.assertIsNotNone(health)
        if isinstance(health, dict):
            self.assertIn("status", health)
        else:
            self.assertTrue(hasattr(health, "healthy"))

    def test_infer_returns_result(self):
        """Test infer returns a result object."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.plugin.infer(frame, [], None)
        self.assertIsNotNone(result)

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
