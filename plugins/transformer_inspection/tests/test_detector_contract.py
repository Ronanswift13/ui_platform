"""Detector contract tests for transformer_inspection."""

from __future__ import annotations


def test_detect_defects_returns_detection_objects(detector, defect_frame):
    """Traditional fallback should return Detection dataclasses with normalized boxes."""
    detections = detector.detect_defects(defect_frame)

    assert detections
    assert all(hasattr(item, "bbox") for item in detections)
    assert all(0 <= item.confidence <= 1 for item in detections)
    assert all(0 <= item.bbox["x"] <= 1 for item in detections)
    assert any(item.defect_type.value in {"oil_leak", "rust", "damage", "foreign"} for item in detections)


def test_detect_oil_level_returns_structured_result(detector, oil_level_frame):
    """Oil-level helper should return ratio, status, and confidence."""
    result = detector.detect_oil_level(oil_level_frame)

    assert 0 <= result.level_ratio <= 1
    assert isinstance(result.level_status, str)
    assert 0 <= result.confidence <= 1


def test_recognize_silica_gel_returns_structured_result(detector, blue_silica_frame):
    """Silica-gel helper should expose enum-backed state and confidence."""
    result = detector.recognize_silica_gel(blue_silica_frame)

    assert result.state.value == "normal"
    assert result.confidence > 0


def test_analyze_thermal_returns_hotspots(detector, thermal_frame, defect_frame):
    """Thermal analysis should identify the synthetic hot block."""
    result = detector.analyze_thermal(thermal_frame, visible_image=defect_frame)

    assert result.max_temperature > 100
    assert result.hotspot_count >= 1
    assert result.level.value in {"alarm", "critical"}
