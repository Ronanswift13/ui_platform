"""Tests for training pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from plugins.indoor_fence.standalone.training_pipeline import (
    TrainingPipeline, TrainingConfig, TrainingStatus, DatasetInfo,
)


def test_training_config():
    config = TrainingConfig(
        model_type="yolov8n",
        epochs=10,
        batch_size=16,
        learning_rate=0.001,
    )
    assert config.model_type == "yolov8n"


def test_pipeline_creation():
    pipeline = TrainingPipeline()
    assert pipeline.status == TrainingStatus.IDLE


def test_register_dataset(tmp_path):
    pipeline = TrainingPipeline(data_dir=str(tmp_path))
    # Create a minimal dataset directory
    ds_dir = tmp_path / "ds1"
    ds_dir.mkdir()
    (ds_dir / "images").mkdir()
    (ds_dir / "labels").mkdir()
    # Create a dummy image and label
    np.save(str(ds_dir / "images" / "0001.npy"), np.zeros((480, 640, 3)))
    (ds_dir / "labels" / "0001.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    info = pipeline.register_dataset("ds1", str(ds_dir))
    assert isinstance(info, DatasetInfo)
    assert info.num_images >= 1


def test_training_status_enum():
    assert TrainingStatus.IDLE.value == "idle"
    assert TrainingStatus.TRAINING.value == "training"
    assert TrainingStatus.COMPLETED.value == "completed"
    assert TrainingStatus.FAILED.value == "failed"
