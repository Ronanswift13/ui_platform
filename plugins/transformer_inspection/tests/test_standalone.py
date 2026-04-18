"""Standalone smoke tests for transformer_inspection."""

from __future__ import annotations

from darkbreaker_sdk.schemas import ROIType


def test_create_standalone(plugin):
    """Standalone factory should return a ready plugin."""
    assert plugin is not None
    assert plugin.id == "transformer_inspection"
    assert plugin.healthcheck().healthy is True


def test_runner_creation(plugin):
    """Standalone runner should wrap the plugin without old plugin_dir API."""
    from darkbreaker_sdk.standalone import StandalonePluginRunner

    runner = StandalonePluginRunner(
        plugin=plugin,
        title="Transformer Inspection - Test Runner",
        port=18087,
    )
    assert runner is not None
    assert runner.app is not None


def test_infer_returns_list(plugin, defect_frame, sample_context, make_roi):
    """Standalone plugin should support a minimal inference call."""
    rois = [make_roi(roi_id="roi-standalone", name="radiator", roi_type=ROIType.DEFECT)]
    results = plugin.infer(frame=defect_frame, rois=rois, context=sample_context)
    assert isinstance(results, list)
