"""Plugin contract tests for busbar_inspection.

Validates that the plugin's infer interface conforms to the expected
contract: infer returns a list of RecognitionResult, postprocess
returns a list of Alarm.
"""
from __future__ import annotations

import numpy as np


def _make_plugin():
    from plugins.busbar_inspection.plugin import BusbarInspectionPlugin

    return BusbarInspectionPlugin.create_standalone()


def _make_context():
    """Minimal PluginContext-like object for standalone testing."""
    from types import SimpleNamespace

    return SimpleNamespace(
        task_id="test-task",
        site_id="test-site",
        device_id="test-device",
        component_id="busbar-001",
    )


def test_infer_returns_list():
    """infer() should always return a list."""
    plugin = _make_plugin()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx = _make_context()
    results = plugin.infer(frame, [], ctx)
    assert isinstance(results, list)


def test_infer_results_are_recognition_results():
    """Each element returned by infer() should be a RecognitionResult."""
    from darkbreaker_sdk.schemas import RecognitionResult

    plugin = _make_plugin()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx = _make_context()
    results = plugin.infer(frame, [], ctx)
    for r in results:
        assert isinstance(r, RecognitionResult), (
            f"expected RecognitionResult, got {type(r).__name__}"
        )


def test_infer_result_has_label_and_confidence():
    """RecognitionResult items should carry label and confidence."""
    plugin = _make_plugin()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx = _make_context()
    results = plugin.infer(frame, [], ctx)
    for r in results:
        assert hasattr(r, "label")
        assert hasattr(r, "confidence")
        assert 0.0 <= r.confidence <= 1.0


def test_healthcheck_returns_health_status():
    """healthcheck() should return HealthStatus."""
    from darkbreaker_sdk.interfaces import HealthStatus

    plugin = _make_plugin()
    health = plugin.healthcheck()
    assert isinstance(health, HealthStatus)
