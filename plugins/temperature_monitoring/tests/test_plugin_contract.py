"""Plugin contract tests for temperature_monitoring.

Validates that the plugin's infer/detect/postprocess interface conforms
to the expected contract.
"""
from __future__ import annotations

import numpy as np


def _make_plugin():
    from plugins.temperature_monitoring.plugin import TemperatureMonitoringPlugin

    return TemperatureMonitoringPlugin.create_standalone()


def test_infer_returns_list():
    """infer() should return a list of RecognitionResult."""
    plugin = _make_plugin()
    # 温度监测使用热成像帧 (float32 or uint8 均可)
    frame = np.full((48, 64), 30.0, dtype=np.float32)
    results = plugin.infer(frame, [], None)
    assert isinstance(results, list)


def test_detect_returns_dict():
    """detect() should return a dict with expected keys."""
    plugin = _make_plugin()
    frame = np.full((48, 64), 30.0, dtype=np.float32)
    result = plugin.detect(thermal_frame=frame)
    assert isinstance(result, dict)


def test_detect_has_hotspots_key():
    """detect() output should contain a 'hotspots' key."""
    plugin = _make_plugin()
    frame = np.full((48, 64), 30.0, dtype=np.float32)
    result = plugin.detect(thermal_frame=frame)
    assert "hotspots" in result
    assert isinstance(result["hotspots"], list)


def test_postprocess_returns_list():
    """postprocess() should return a list of Alarm objects."""
    plugin = _make_plugin()
    frame = np.full((48, 64), 30.0, dtype=np.float32)
    results = plugin.infer(frame, [], None)
    alarms = plugin.postprocess(results, [])
    assert isinstance(alarms, list)


def test_healthcheck_returns_health_status():
    """healthcheck() should return HealthStatus."""
    from darkbreaker_sdk.interfaces import HealthStatus

    plugin = _make_plugin()
    health = plugin.healthcheck()
    assert isinstance(health, HealthStatus)
