"""Plugin contract tests for transformer_inspection."""

from __future__ import annotations

import copy

from darkbreaker_sdk.interfaces import PluginContext
from darkbreaker_sdk.schemas import ROIType


def test_infer_maps_defect_results(plugin, defect_frame, sample_context, make_roi):
    """Defect ROI should produce SDK RecognitionResult objects from detector dataclasses."""
    rois = [make_roi(roi_id="roi-defect", name="radiator", roi_type=ROIType.DEFECT)]
    results = plugin.infer(frame=defect_frame, rois=rois, context=sample_context)

    assert results
    assert any(result.label in {"oil_leak", "rust", "damage", "foreign_object"} for result in results)
    assert all(hasattr(result, "bbox") for result in results)
    assert all(0 <= result.confidence <= 1 for result in results)


def test_infer_maps_silica_state(plugin, blue_silica_frame, sample_context, make_roi):
    """Breather ROI should use the detector's silica-gel path."""
    rois = [make_roi(roi_id="roi-breather", name="breather", roi_type=ROIType.STATE)]
    results = plugin.infer(frame=blue_silica_frame, rois=rois, context=sample_context)

    assert any(result.label == "silica_gel_normal" for result in results)


def test_infer_accepts_none_context(plugin, defect_frame, make_roi):
    """Standalone/demo mode may omit PluginContext and should still return results."""
    rois = [make_roi(roi_id="roi-standalone", name="radiator", roi_type=ROIType.DEFECT)]
    results = plugin.infer(frame=defect_frame, rois=rois, context=None)

    assert isinstance(results, list)
    assert all(result.task_id for result in results)


def test_infer_emits_overtemp_when_thermal_enabled(
    loaded_config,
    defect_frame,
    thermal_frame,
    make_roi,
):
    """Thermal results should be adapted into overtemp RecognitionResult objects."""
    from plugins.transformer_inspection.plugin import TransformerInspectionPlugin

    config = copy.deepcopy(loaded_config)
    config.setdefault("thermal", {})["enabled"] = True

    plugin = TransformerInspectionPlugin.create_standalone(config=config)
    context = PluginContext(
        task_id="thermal-task",
        site_id="test-site",
        device_id="test-camera",
        component_id="transformer-01",
        metadata={"thermal_frame": thermal_frame},
    )
    rois = [make_roi(roi_id="roi-thermal", name="radiator", roi_type=ROIType.DEFECT)]

    results = plugin.infer(frame=defect_frame, rois=rois, context=context)
    assert any(result.label == "overtemp" for result in results)
