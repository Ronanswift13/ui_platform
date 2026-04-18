"""Tests for object detection module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
import tempfile
import os
from plugins.indoor_fence.detection.object_detector import ObjectDetector, DetectionBox
from plugins.indoor_fence.detection.yolo_detector import YOLODetector


def test_detection_box():
    box = DetectionBox(
        x1=100, y1=200, x2=200, y2=400,
        confidence=0.85,
        class_id=0,
        class_name="person",
    )
    assert box.width == 100
    assert box.height == 200
    assert box.center == (150, 300)
    assert box.foot_point == (150, 400)


def test_yolo_detector_simulation():
    """YOLO detector in simulation mode (no model file)."""
    detector = YOLODetector(model_path=None, device="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    # Simulation mode should return at least one detection
    assert len(detections) >= 1
    for d in detections:
        assert isinstance(d, DetectionBox)
        assert d.class_name == "person"
        assert 0.0 <= d.confidence <= 1.0


def test_yolo_detector_model_missing_fallback():
    """Test fallback when model file does not exist."""
    nonexistent_path = "/tmp/nonexistent_model_12345.onnx"
    assert not os.path.exists(nonexistent_path)

    detector = YOLODetector(model_path=nonexistent_path, device="cpu")

    # Should fallback to simulation mode
    assert detector.is_simulation_mode()
    assert detector.get_fallback_reason() == "model_file_not_found"

    # Should still work in simulation mode
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    assert len(detections) >= 1


def test_yolo_detector_model_exists_recovery():
    """Test that detector works when model file exists."""
    # Create a temporary file to simulate model existence
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"fake model content")

    try:
        # This will fail to load (not a real ONNX model) but should handle gracefully
        detector = YOLODetector(model_path=tmp_path, device="cpu")

        # Should fallback due to load error, not file missing
        assert detector.is_simulation_mode()
        assert detector.get_fallback_reason() in ["model_load_failed", "onnxruntime_not_installed"]

        # Should still work in simulation mode
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)
        assert len(detections) >= 1
    finally:
        os.unlink(tmp_path)


def test_yolo_detector_no_path_provided():
    """Test explicit None path triggers simulation mode."""
    detector = YOLODetector(model_path=None, device="cpu")
    assert detector.is_simulation_mode()
    assert detector.get_fallback_reason() == "no_model_path_provided"


def test_object_detector_interface():
    """Verify ObjectDetector is an ABC with detect method."""
    with pytest.raises(TypeError):
        ObjectDetector()  # Cannot instantiate abstract class
