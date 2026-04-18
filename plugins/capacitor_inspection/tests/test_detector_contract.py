"""Detector contract tests for capacitor_inspection."""

from __future__ import annotations


def test_detect_structural_defects_classifies_tilt(monkeypatch, detector, structural_frame):
    """Detector should classify tilt severity based on configured thresholds."""
    monkeypatch.setattr(
        detector,
        "_detect_capacitor_units",
        lambda image: [
            {
                "bbox": {"x": 0.2, "y": 0.1, "width": 0.15, "height": 0.7},
                "confidence": 0.86,
                "class_name": "capacitor",
            }
        ],
    )
    monkeypatch.setattr(detector, "_calculate_tilt_angle", lambda image, bbox: 6.0)
    monkeypatch.setattr(detector, "_detect_collapse", lambda image, units: [])
    monkeypatch.setattr(detector, "_detect_missing_units", lambda image, units: [])

    defects = detector.detect_structural_defects(structural_frame)

    assert len(defects) == 1
    assert defects[0].defect_type.value == "tilt_error"
    assert defects[0].tilt_angle == 6.0


def test_detect_intrusion_wraps_tracker_results(monkeypatch, detector, intrusion_frame):
    """Detector should convert tracked intrusion dicts into IntrusionDetection dataclasses."""
    monkeypatch.setattr(
        detector,
        "_detect_intrusion_targets",
        lambda image: [{"bbox": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.3}, "confidence": 0.95, "type": "person"}],
    )
    monkeypatch.setattr(
        detector._intrusion_tracker,
        "update",
        lambda detections, timestamp: [
            {
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.3},
                "confidence": 0.95,
                "type": "person",
                "track_id": 42,
                "hits": 3,
                "confirmed": True,
                "duration": 3.5,
            }
        ],
    )

    intrusions = detector.detect_intrusion(intrusion_frame, timestamp=100.0)

    assert len(intrusions) == 1
    assert intrusions[0].intrusion_type.value == "person"
    assert intrusions[0].confirmed is True
    assert intrusions[0].track_id == 42


def test_analyze_bank_status_uses_expected_grid(monkeypatch, detector, structural_frame):
    """Bank-status helper should report configured expected totals."""
    monkeypatch.setattr(detector, "_detect_capacitor_units", lambda image: [])
    monkeypatch.setattr(detector, "_find_missing_positions", lambda units: [])

    status = detector.analyze_bank_status(structural_frame)

    assert status.total_units == 12
    assert status.detected_units == 0
