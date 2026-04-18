"""Fallback-chain tests for capacitor_inspection detector."""

from __future__ import annotations


def test_detect_capacitor_units_falls_back_to_traditional(monkeypatch, detector, structural_frame):
    """If deep-learning detection yields nothing, detector should still use traditional logic."""
    detector._use_deep_learning = True
    detector._dl_initialized = True
    monkeypatch.setattr(
        detector._yolov8_vit_detector,
        "detect",
        lambda image: type("Result", (), {"detections": [], "inference_time_ms": 1.0})(),
        raising=False,
    )
    monkeypatch.setattr(
        detector,
        "_detect_units_traditional",
        lambda image: [
            {
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.4},
                "confidence": 0.7,
                "class_name": "capacitor",
                "source": "traditional_stub",
            }
        ],
    )

    detections = detector._detect_capacitor_units(structural_frame)

    assert len(detections) == 1
    assert detections[0]["source"] == "traditional_stub"


def test_intrusion_tracker_confirms_after_min_hits():
    """IntrusionTracker should confirm a stable target after enough hits."""
    from plugins.capacitor_inspection.detector_enhanced import IntrusionTracker

    tracker = IntrusionTracker(max_age=5, min_hits=3, iou_threshold=0.1)
    detection = {"bbox": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.2}, "confidence": 0.9, "type": "person"}

    tracker.update([detection], timestamp=1.0)
    tracker.update([detection], timestamp=2.0)
    tracked = tracker.update([detection], timestamp=4.0)

    assert tracked
    assert tracked[0]["confirmed"] is True
    assert tracked[0]["duration"] == 3.0
