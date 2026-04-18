#!/usr/bin/env python3
"""Temperature Monitoring - Detect Path Tests"""
import unittest
from unittest import mock

import numpy as np


def build_sensor_grid(base_temp: float, hotspot: tuple[int, int, float] | None = None):
    readings = []
    for row in range(6):
        for col in range(8):
            temp = base_temp
            if hotspot and (row, col) == hotspot[:2]:
                temp = hotspot[2]
            readings.append(
                {
                    "sensor_id": f"sensor_{row}_{col}",
                    "position": {"row": row, "col": col},
                    "temperature": temp,
                    "timestamp": 1710000000 + row * 8 + col,
                }
            )
    return readings


class TestTemperatureMonitoringDetect(unittest.TestCase):
    """Validate detect() behavior for simulation, thermal, sensor and training flows."""

    @classmethod
    def setUpClass(cls):
        from plugins.temperature_monitoring.plugin import Plugin

        cls.plugin_cls = Plugin

    def setUp(self):
        self.plugin = self.plugin_cls.create_standalone()

    def test_detect_sensor_readings_exposes_sensor_metadata(self):
        """Real sensor input should report the measured temperature path."""
        sensors = build_sensor_grid(34.0, hotspot=(4, 6, 58.4))
        result = self.plugin.detect(sensor_readings=sensors)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["max_temp"], 58.4)
        self.assertEqual(result["metadata"]["input_mode"], "sensor_readings")
        self.assertEqual(result["metadata"]["sensor_summary"]["count"], len(sensors))

    def test_detect_source_isolation_prevents_false_critical(self):
        """Simulation history should not inflate the next real thermal detection."""
        cool_heatmap = np.full((48, 64), 30.0, dtype=np.float32)
        warm_frame = np.full((48, 64), 56.0, dtype=np.float32)

        with mock.patch.object(self.plugin._detector, "_simulate_heatmap", return_value=cool_heatmap):
            sim_result = self.plugin.detect()

        real_result = self.plugin.detect(thermal_frame=warm_frame)

        self.assertTrue(sim_result["success"])
        self.assertTrue(real_result["success"])
        self.assertEqual(real_result["metadata"]["source_key"], "thermal_frame")
        self.assertEqual(real_result["status"], "warning")
        self.assertLess(real_result["trend"]["rise_rate"], 2.0)

    def test_detect_short_interval_does_not_create_fake_rise_rate(self):
        """Back-to-back same-source reads should not alarm only because the calls are milliseconds apart."""
        frame_a = np.full((48, 64), 40.0, dtype=np.float32)
        frame_b = np.full((48, 64), 41.0, dtype=np.float32)

        first = self.plugin.detect(thermal_frame=frame_a)
        second = self.plugin.detect(thermal_frame=frame_b)

        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertEqual(second["status"], "normal")
        self.assertLess(second["trend"]["rise_rate"], 2.0)

    def test_training_profile_flags_future_sensor_anomaly(self):
        """Baseline training should surface AI anomaly insight for a later abnormal sample."""
        baseline_a = build_sensor_grid(34.0)
        baseline_b = build_sensor_grid(34.5)
        baseline_c = build_sensor_grid(35.0)

        self.plugin.upload_training_data({"sensor_readings": baseline_a}, {"status": "normal"})
        self.plugin.upload_training_data({"sensor_readings": baseline_b}, {"status": "normal"})
        self.plugin.upload_training_data({"sensor_readings": baseline_c}, {"status": "normal"})

        training_result = self.plugin.start_training({"profile_name": "sensor_baseline"})
        self.assertTrue(training_result["success"])
        self.assertEqual(training_result["profile"]["sample_count"], 3)

        anomaly = build_sensor_grid(35.0, hotspot=(4, 6, 72.0))
        result = self.plugin.detect(sensor_readings=anomaly, context={"source_key": "field_sensor"})

        self.assertTrue(result["success"])
        self.assertIn("ai_insight", result["metadata"])
        self.assertTrue(result["metadata"]["ai_insight"]["enabled"])
        self.assertTrue(result["metadata"]["ai_insight"]["triggered"])
        ai_alarm_types = {alarm["type"] for alarm in result["alarms"]}
        self.assertIn("ai_temperature_anomaly", ai_alarm_types)


if __name__ == "__main__":
    unittest.main()
