"""Plugin contract tests for thermal."""
from __future__ import annotations

import numpy as np


def test_infer_returns_list(plugin, thermal_frame, sample_context, make_roi):
    rois = [make_roi(roi_id="roi-thermal-01", name="transformer_region")]
    results = plugin.infer(frame=thermal_frame, rois=rois, context=sample_context)
    assert isinstance(results, list)


def test_infer_normal_frame_returns_normal_label(plugin, thermal_frame, sample_context, make_roi):
    """A frame with normal temperatures should return normal label."""
    normal_frame = np.full((480, 640), 35.0, dtype=np.float32)
    rois = [make_roi(roi_id="roi-normal", name="busbar_region")]
    results = plugin.infer(frame=normal_frame, rois=rois, context=sample_context)
    assert len(results) >= 1
    assert results[0].label == "normal"
    assert 0 <= results[0].confidence <= 1


def test_infer_hotspot_frame_returns_anomaly(
    plugin, thermal_frame_with_hotspot, sample_context, make_roi
):
    """A frame with a hotspot should detect anomaly."""
    rois = [make_roi(roi_id="roi-hot", name="transformer_region")]
    results = plugin.infer(frame=thermal_frame_with_hotspot, rois=rois, context=sample_context)
    assert len(results) >= 1
    # At least one result should be anomaly-related
    labels = [r.label for r in results]
    assert any(l != "normal" for l in labels), f"Expected anomaly detection, got: {labels}"


def test_result_metadata_contract(plugin, thermal_frame, sample_context, make_roi):
    """Results must contain required metadata fields."""
    rois = [make_roi(roi_id="roi-meta", name="region")]
    results = plugin.infer(frame=thermal_frame, rois=rois, context=sample_context)
    assert len(results) >= 1
    meta = results[0].metadata
    assert meta.get("modality") == "thermal"
    assert meta.get("algorithm_stage") in (
        "baseline",
        "fallback",
        "placeholder",
        "real_model",
    )
    assert meta.get("quality_gate_status") in (
        "pass",
        "soft_fail",
        "hard_fail",
        "not_applicable",
    )


def test_result_bbox_valid(plugin, thermal_frame, sample_context, make_roi):
    """Bbox coordinates must be normalized 0-1."""
    rois = [make_roi(roi_id="roi-bbox", name="region")]
    results = plugin.infer(frame=thermal_frame, rois=rois, context=sample_context)
    for r in results:
        assert 0 <= r.bbox.x <= 1
        assert 0 <= r.bbox.y <= 1
        assert 0 <= r.bbox.width <= 1
        assert 0 <= r.bbox.height <= 1


def test_postprocess_generates_alarms(
    plugin, thermal_frame_with_hotspot, sample_context, make_roi
):
    """Postprocess should generate alarms for anomaly results."""
    rois = [make_roi(roi_id="roi-alarm", name="region")]
    results = plugin.infer(frame=thermal_frame_with_hotspot, rois=rois, context=sample_context)
    alarms = plugin.postprocess(results, rules=None)
    assert isinstance(alarms, list)


def test_infer_none_frame_returns_empty(plugin, sample_context, make_roi):
    """None frame should return empty results, not crash."""
    rois = [make_roi(roi_id="roi-none", name="region")]
    results = plugin.infer(frame=None, rois=rois, context=sample_context)
    assert isinstance(results, list)
    assert len(results) == 0
