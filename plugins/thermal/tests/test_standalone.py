"""Standalone smoke tests for thermal."""
from __future__ import annotations


def test_create_standalone(plugin):
    assert plugin is not None
    assert plugin.id == "thermal"
    assert plugin.healthcheck().healthy is True


def test_runner_creation(plugin):
    from darkbreaker_sdk.standalone import StandalonePluginRunner

    runner = StandalonePluginRunner(
        plugin=plugin,
        title="Thermal Analysis - Test Runner",
        port=18096,
    )
    assert runner is not None
    assert runner.app is not None


def test_infer_returns_list(plugin, thermal_frame, sample_context, make_roi):
    rois = [make_roi(roi_id="roi-standalone", name="thermal_region")]
    results = plugin.infer(frame=thermal_frame, rois=rois, context=sample_context)
    assert isinstance(results, list)
