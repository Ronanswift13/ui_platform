#!/usr/bin/env python3
"""Meter Reading - Standalone smoke tests."""
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


class TestMeterReadingStandalone(unittest.TestCase):
    """Test meter_reading plugin in standalone mode."""

    @classmethod
    def setUpClass(cls):
        from plugins.meter_reading.plugin import MeterReadingPlugin

        cls.plugin = MeterReadingPlugin.create_standalone()

    def test_create_standalone(self):
        """Plugin is created via create_standalone."""
        self.assertIsNotNone(self.plugin)
        self.assertEqual(self.plugin.id, "meter_reading")

    def test_healthcheck(self):
        """healthcheck() returns HealthStatus."""
        from darkbreaker_sdk.interfaces import HealthStatus

        health = self.plugin.healthcheck()
        self.assertIsInstance(health, HealthStatus)

    def test_infer_returns_list(self):
        """infer() returns a list on a blank frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.plugin.infer(frame, [], None)
        self.assertIsInstance(results, list)

    def test_infer_result_elements(self):
        """Each infer result should have label and confidence attributes."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.plugin.infer(frame, [], None)
        for r in results:
            self.assertTrue(hasattr(r, "label"))
            self.assertTrue(hasattr(r, "confidence"))


if __name__ == "__main__":
    unittest.main()
