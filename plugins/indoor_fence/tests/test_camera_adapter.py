"""Tests for camera adapter fallback behavior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import logging
from plugins.indoor_fence.adapters.camera_adapter import CameraAdapter, CameraConfig


def test_camera_model_missing_fallback(caplog):
    """Test camera adapter fallback when model file is missing."""
    caplog.set_level(logging.ERROR)

    config = CameraConfig(
        source="/dev/nonexistent_camera",  # Force hardware failure
        model_path="/nonexistent/model.onnx",
        simulate_if_unavailable=True
    )

    adapter = CameraAdapter(config)
    result = adapter.connect()

    # Should succeed with simulation mode
    assert result is True
    assert adapter.is_simulated

    # Should have logged fallback (camera open failed comes first)
    assert any("FALLBACK" in record.message for record in caplog.records)


def test_camera_hardware_unavailable_fallback(caplog):
    """Test camera adapter fallback when hardware is unavailable."""
    caplog.set_level(logging.ERROR)

    config = CameraConfig(
        source="/dev/nonexistent_camera",
        model_path="models/indoor/person_yolov8n.onnx",
        simulate_if_unavailable=True
    )

    adapter = CameraAdapter(config)
    result = adapter.connect()

    # Should succeed with simulation mode
    assert result is True
    assert adapter.is_simulated

    # Should have logged fallback
    assert any("FALLBACK" in record.message for record in caplog.records)


def test_camera_no_fallback_when_disabled(caplog):
    """Test camera adapter fails when fallback is disabled."""
    caplog.set_level(logging.ERROR)

    config = CameraConfig(
        source="/dev/nonexistent_camera",
        model_path="/nonexistent/model.onnx",
        simulate_if_unavailable=False
    )

    adapter = CameraAdapter(config)
    result = adapter.connect()

    # Should fail
    assert result is False
    assert not adapter.is_connected


def test_camera_simulation_detections():
    """Test camera adapter can generate detections in simulation mode."""
    config = CameraConfig(
        source="/dev/nonexistent_camera",
        model_path="/nonexistent/model.onnx",
        simulate_if_unavailable=True,
        tracking_enabled=True,
        min_hits=1  # Allow immediate tracking
    )

    adapter = CameraAdapter(config)
    adapter.connect()

    assert adapter.is_simulated

    # Should be able to get detections
    detections = adapter.get_person_detections()
    assert isinstance(detections, list)
    # Simulation mode generates 1-3 detections
    assert len(detections) >= 1

