"""Plugin contract tests for hyperspectral_detection."""
from __future__ import annotations

def test_infer_returns_list(plugin, visual_frame, sample_context, make_roi):
    rois = [make_roi()]
    results = plugin.infer(frame=visual_frame, rois=rois, context=sample_context)
    assert isinstance(results, list)

def test_process_returns_dict(plugin):
    result = plugin.process({"device_id": "test-device"})
    assert isinstance(result, dict)
    assert "success" in result

def test_postprocess_returns_list(plugin):
    alarms = plugin.postprocess([], rules=None)
    assert isinstance(alarms, list)

def test_healthcheck(plugin):
    health = plugin.healthcheck()
    assert health.healthy is True

def test_plugin_info(plugin):
    info = plugin.plugin_info
    assert info["id"] == "hyperspectral_detection"
    assert "capabilities" in info
