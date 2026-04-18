"""Plugin contract tests for capacitor_inspection."""

from __future__ import annotations

from darkbreaker_sdk.schemas import BoundingBox, ROIType


def test_structural_roi_routes_and_maps_dataclass(
    monkeypatch,
    plugin,
    structural_frame,
    sample_context,
    make_roi,
):
    """DEFECT ROI should map structural detector dataclasses into RecognitionResult."""
    from plugins.capacitor_inspection.detector_enhanced import CapacitorDefectType, CapacitorDetection

    monkeypatch.setattr(
        plugin._detector,
        "detect_structural_defects",
        lambda image: [
            CapacitorDetection(
                defect_type=CapacitorDefectType.TILT_WARNING,
                bbox={"x": 0.2, "y": 0.1, "width": 0.3, "height": 0.5},
                confidence=0.88,
                class_name="tilt_warning",
                tilt_angle=4.2,
                metadata={"source": "stub_structural"},
            )
        ],
    )
    monkeypatch.setattr(plugin._detector, "detect_intrusion", lambda *args, **kwargs: [])

    roi = make_roi(
        roi_id="roi-structural",
        name="capacitor_bank",
        roi_type=ROIType.DEFECT,
        bbox=BoundingBox(x=0.1, y=0.2, width=0.5, height=0.5),
    )
    results = plugin.infer(frame=structural_frame, rois=[roi], context=sample_context)

    assert len(results) == 1
    assert results[0].label == "tilt_warning"
    assert results[0].metadata["source"] == "stub_structural"
    assert results[0].bbox.x == 0.2
    assert results[0].bbox.y == 0.25


def test_intrusion_roi_routes_and_maps_dataclass(
    monkeypatch,
    plugin,
    intrusion_frame,
    sample_context,
    make_roi,
):
    """INTRUSION ROI should map intrusion dataclasses into RecognitionResult."""
    from plugins.capacitor_inspection.detector_enhanced import IntrusionDetection, IntrusionType, ZoneType

    monkeypatch.setattr(plugin._detector, "detect_structural_defects", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        plugin._detector,
        "detect_intrusion",
        lambda image, timestamp: [
            IntrusionDetection(
                intrusion_type=IntrusionType.PERSON,
                bbox={"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.4},
                confidence=0.91,
                zone=ZoneType.RESTRICTED,
                track_id=7,
                duration_sec=3.2,
                confirmed=True,
                metadata={"source": "stub_intrusion"},
            )
        ],
    )

    roi = make_roi(
        roi_id="roi-intrusion",
        name="restricted_zone",
        roi_type=ROIType.INTRUSION,
    )
    results = plugin.infer(frame=intrusion_frame, rois=[roi], context=sample_context)

    assert len(results) == 1
    assert results[0].label == "intrusion_person"
    assert results[0].metadata["confirmed"] is True
    assert results[0].metadata["zone"] == "restricted"


def test_infer_accepts_none_context(monkeypatch, plugin, structural_frame, make_roi):
    """Standalone/demo mode may omit PluginContext and should still return results."""
    from plugins.capacitor_inspection.detector_enhanced import CapacitorDefectType, CapacitorDetection

    monkeypatch.setattr(
        plugin._detector,
        "detect_structural_defects",
        lambda image: [
            CapacitorDetection(
                defect_type=CapacitorDefectType.COLLAPSE,
                bbox={"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
                confidence=0.9,
                class_name="collapse",
                metadata={"source": "standalone"},
            )
        ],
    )
    monkeypatch.setattr(plugin._detector, "detect_intrusion", lambda *args, **kwargs: [])

    roi = make_roi(roi_id="roi-none-context", name="capacitor_bank", roi_type=ROIType.DEFECT)
    results = plugin.infer(frame=structural_frame, rois=[roi], context=None)

    assert len(results) == 1
    assert results[0].task_id == "standalone-task"
