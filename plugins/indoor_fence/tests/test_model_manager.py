"""Tests for model lifecycle manager."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.detection.model_manager import (
    ModelManager, ModelInfo, ModelStatus,
)


def test_model_manager_creation():
    mm = ModelManager()
    assert mm.list_models() == []


def test_register_model():
    mm = ModelManager()
    mm.register("yolov8n", "/path/to/model.onnx", model_type="detector")
    models = mm.list_models()
    assert len(models) == 1
    assert models[0].model_id == "yolov8n"
    assert models[0].status == ModelStatus.REGISTERED


def test_model_not_found():
    mm = ModelManager()
    assert mm.get("nonexistent") is None


def test_model_status_transitions():
    mm = ModelManager()
    mm.register("test_model", "/path/to/model.onnx", model_type="detector")
    mm.set_status("test_model", ModelStatus.LOADED)
    info = mm.get("test_model")
    assert info.status == ModelStatus.LOADED


def test_model_status_enum():
    assert ModelStatus.REGISTERED.value == "registered"
    assert ModelStatus.LOADED.value == "loaded"
    assert ModelStatus.FAILED.value == "failed"
