"""Fallback-chain tests for transformer_inspection detector."""

from __future__ import annotations


def test_detect_defects_falls_back_to_traditional(monkeypatch, detector, defect_frame):
    """If deep learning yields nothing, detect_defects should still use traditional logic."""
    from plugins.transformer_inspection.detector_enhanced import DefectType, Detection

    monkeypatch.setattr(detector, "_detect_by_deep_learning", lambda image: [])
    monkeypatch.setattr(
        detector,
        "_detect_by_traditional",
        lambda image: [
            Detection(
                defect_type=DefectType.RUST,
                bbox={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.2},
                confidence=0.8,
                class_name="rust",
                metadata={"source": "traditional_stub"},
            )
        ],
    )
    detector._use_deep_learning = True

    detections = detector.detect_defects(defect_frame)

    assert len(detections) == 1
    assert detections[0].metadata["source"] == "traditional_stub"
    assert detections[0].defect_type.value == "rust"
