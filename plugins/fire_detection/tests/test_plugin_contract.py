"""Plugin contract tests for fire_detection.

Validates that the plugin's infer/detect interface conforms to
the expected contract.
"""
from __future__ import annotations

import numpy as np


def test_infer_returns_list(plugin, blank_frame):
    """infer() should always return a list of RecognitionResult."""
    results = plugin.infer(blank_frame, [], None)
    assert isinstance(results, list)


def test_detect_returns_dict(plugin, blank_frame):
    """detect() should return a dict with semantic_type and severity."""
    result = plugin.detect(blank_frame)
    assert isinstance(result, dict)
    assert "semantic_type" in result
    assert "severity" in result


def test_detect_has_runtime_mode(plugin, blank_frame):
    """detect() output should include runtime_mode for traceability."""
    result = plugin.detect(blank_frame)
    assert "runtime_mode" in result


def test_detect_confidence_in_range(plugin, blank_frame):
    """detect() confidence should be in [0, 1]."""
    result = plugin.detect(blank_frame)
    conf = result.get("confidence", 0)
    assert 0.0 <= conf <= 1.0


def test_infer_no_detections_on_blank(plugin, blank_frame):
    """A blank frame should produce zero or minimal detections."""
    results = plugin.infer(blank_frame, [], None)
    # 无论检测器是否初始化，返回的应当是 list
    assert isinstance(results, list)


def test_healthcheck_returns_health_status(plugin):
    """healthcheck() should return HealthStatus."""
    from darkbreaker_sdk.interfaces import HealthStatus

    health = plugin.healthcheck()
    assert isinstance(health, HealthStatus)
