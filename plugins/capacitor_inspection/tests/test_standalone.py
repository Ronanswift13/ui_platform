"""Standalone smoke tests for capacitor_inspection."""

from __future__ import annotations

from darkbreaker_sdk.schemas import ROIType


def test_create_standalone(plugin):
    """Standalone factory should return a ready plugin."""
    assert plugin is not None
    assert plugin.id == "capacitor_inspection"
    assert plugin.healthcheck().healthy is True


def test_runner_creation(plugin):
    """Standalone runner should wrap the plugin via the current SDK API."""
    from darkbreaker_sdk.standalone import StandalonePluginRunner

    runner = StandalonePluginRunner(
        plugin=plugin,
        title="Capacitor Inspection - Test Runner",
        port=18090,
    )
    assert runner is not None
    assert runner.app is not None


def test_infer_returns_list(plugin, structural_frame, sample_context, make_roi):
    """Minimal inference call should always return a list."""
    rois = [make_roi(roi_id="roi-standalone", name="capacitor_bank", roi_type=ROIType.DEFECT)]
    results = plugin.infer(frame=structural_frame, rois=rois, context=sample_context)
    assert isinstance(results, list)
