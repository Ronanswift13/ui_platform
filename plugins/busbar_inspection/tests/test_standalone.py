#!/usr/bin/env python3
"""Busbar Inspection - Standalone Tests"""
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


class TestBusbarInspectionStandalone(unittest.TestCase):
    """Test busbar inspection plugin in standalone mode."""

    @classmethod
    def setUpClass(cls):
        from plugins.busbar_inspection.plugin import Plugin
        cls.plugin = Plugin.create_standalone()

    def test_create_standalone(self):
        """Test plugin is created via create_standalone."""
        self.assertIsNotNone(self.plugin)
        self.assertEqual(self.plugin.id, "busbar_inspection")

    def test_create_standalone_initializes_detector(self):
        """Regression: standalone create must initialize detector main chain."""
        self.assertTrue(getattr(self.plugin._detector, "_initialized", False))

    def test_healthcheck(self):
        """Test healthcheck returns HealthStatus."""
        from darkbreaker_sdk.interfaces import HealthStatus
        health = self.plugin.healthcheck()
        self.assertIsInstance(health, HealthStatus)

    def test_infer_returns_list(self):
        """Test infer returns a list."""
        from darkbreaker_sdk.interfaces import PluginContext
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = PluginContext(task_id="t", site_id="s", device_id="d")
        results = self.plugin.infer(frame, [], ctx)
        self.assertIsInstance(results, list)

    def test_infer_empty_rois_runs_fullframe_pipeline(self):
        """Empty ROI list must synthesize a full-frame ROI and run the pipeline.

        Regression guard for the standalone upload path: runner calls
        plugin.infer(frame, [], context) with an empty ROI list. Before the
        fix, this returned an empty list so no quality / no results were ever
        surfaced in the UI. After the fix, pipeline must actually run and
        return at least one result (defect or quality_failed).
        """
        from darkbreaker_sdk.interfaces import PluginContext
        # A zero-brightness frame will trip the underexposure quality gate
        # and return a `quality_failed` result — proving the pipeline ran.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = PluginContext(task_id="t", site_id="s", device_id="d")
        results = self.plugin.infer(frame, [], ctx)
        self.assertGreaterEqual(
            len(results), 1,
            "Empty ROI list must synthesize a full-frame ROI and run the pipeline",
        )
        # Output must satisfy contract: metadata.quality present
        first = results[0]
        self.assertIsNotNone(first.metadata)
        # Either top-level quality (defect path) or nested details.quality (quality_failed)
        quality = first.metadata.get("quality") or (
            first.metadata.get("details", {}).get("quality") if isinstance(first.metadata.get("details"), dict) else None
        )
        self.assertIsNotNone(quality, "Result metadata must expose quality info")

    def test_infer_empty_rois_natural_image_produces_quality_result(self):
        """Natural uploaded image with empty ROIs must still produce output.

        Simulates the runner/_detect path for a realistic photo: pipeline must
        always return at least a quality assessment, never an empty list.
        """
        from darkbreaker_sdk.interfaces import PluginContext
        # Non-trivial synthetic image that should pass the quality gate
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        # Add some edges so clarity score is not 0
        frame[100:600, 200:1000] = 180
        frame[300:320, :] = 60
        ctx = PluginContext(task_id="t", site_id="s", device_id="d")
        results = self.plugin.infer(frame, [], ctx)
        self.assertIsInstance(results, list)
        # Must produce at least one result (quality info or detection)
        self.assertGreaterEqual(len(results), 1)

    def test_infer_empty_rois_crack_upload_returns_detection(self):
        """回归: 真实上传小图应使用非轴向裂纹样例，避免结构线被误当成 crack。"""
        from darkbreaker_sdk.interfaces import PluginContext
        try:
            import cv2
        except ImportError:
            self.skipTest("cv2 not installed")

        frame = np.random.default_rng(1).integers(
            95, 160, size=(480, 640, 3), dtype=np.uint8
        )
        pts = np.array(
            [[80, 350], [160, 300], [240, 340], [330, 250], [430, 310], [520, 230]],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts], False, (20, 20, 20), 2)

        ctx = PluginContext(task_id="t", site_id="s", device_id="d")
        results = self.plugin.infer(frame, [], ctx)

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(
            any(r.label == "crack" for r in results),
            "Uploaded crack-like image should surface crack detections in standalone path",
        )

    def test_simulated_defect_upload_reaches_detection_stage(self):
        """Regression: defect upload frames should not all die at quality gate."""
        from darkbreaker_sdk.interfaces import PluginContext
        from plugins.busbar_inspection.standalone.busbar_simulator import BusbarSimulator

        sim = BusbarSimulator(seed=42)
        loaded = sim.load_scenario("crack_detected")
        self.assertTrue(loaded, "Scenario name must match simulator builtin case")
        step = sim.step()
        frame = sim.render_frame(defects=step.defects)
        ctx = PluginContext(task_id="t", site_id="s", device_id="d")

        results = self.plugin.infer(frame, [], ctx)

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(
            any(r.label != "quality_failed" for r in results),
            "Defect upload frame should reach detection stage instead of stopping at quality gate",
        )

    def test_simulator_api_marks_standalone_simulation_runtime_mode(self):
        """Standalone 仿真接口必须显式标记 standalone_simulation。"""
        from plugins.busbar_inspection.standalone.busbar_simulator import BusbarSimulator

        sim = BusbarSimulator(seed=7)
        sim.load_scenario("pin_missing_single")
        payload = sim.result_to_api_format(sim.step())

        self.assertEqual(payload["runtime_mode"], "standalone_simulation")
        self.assertTrue(payload["detections"])
        self.assertTrue(
            all(det["runtime_mode"] == "standalone_simulation" for det in payload["detections"])
        )

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
