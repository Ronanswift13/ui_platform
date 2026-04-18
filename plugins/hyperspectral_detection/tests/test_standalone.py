"""Standalone smoke tests for hyperspectral_detection."""
from __future__ import annotations

def test_create_standalone(plugin):
    assert plugin is not None
    assert plugin.id == "hyperspectral_detection"
    assert plugin.healthcheck().healthy is True

def test_runner_creation(plugin):
    from darkbreaker_sdk.standalone import StandalonePluginRunner
    runner = StandalonePluginRunner(
        plugin=plugin,
        title="Hyperspectral Detection - Test Runner",
        port=18095,
    )
    assert runner is not None
    assert runner.app is not None

def test_infer_returns_list(plugin, visual_frame, sample_context, make_roi):
    rois = [make_roi()]
    results = plugin.infer(frame=visual_frame, rois=rois, context=sample_context)
    assert isinstance(results, list)
