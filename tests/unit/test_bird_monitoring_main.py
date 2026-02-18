from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from plugins.bird_monitoring.main import (
    infer_once,
    normalize_input_files,
    train_once,
)


def _write_test_image(image_path: Path) -> None:
    pillow = pytest.importorskip("PIL")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    pillow.Image.fromarray(image).save(image_path)


def _create_minimal_yolo_dataset(base_dir: Path) -> Path:
    (base_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (base_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (base_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    train_img = base_dir / "images" / "train" / "sample_train.jpg"
    val_img = base_dir / "images" / "val" / "sample_val.jpg"
    _write_test_image(train_img)
    _write_test_image(val_img)

    (base_dir / "labels" / "train" / "sample_train.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )
    (base_dir / "labels" / "val" / "sample_val.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )

    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["bird"],
    }
    data_yaml_path = base_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    return data_yaml_path


def test_normalize_input_files_supports_single_value() -> None:
    assert normalize_input_files("a.jpg") == ["a.jpg"]
    assert normalize_input_files(Path("b.jpg")) == ["b.jpg"]


def test_infer_once_supports_single_image(tmp_path: Path) -> None:
    image_path = tmp_path / "single.jpg"
    output_path = tmp_path / "inference" / "summary.json"
    _write_test_image(image_path)

    summary = infer_once(
        image_paths=[image_path],
        output_json=output_path,
        plugin_config={},
    )

    assert summary["success"] is True
    assert summary["total_images"] == 1
    assert len(summary["items"]) == 1
    assert summary["items"][0]["input"].endswith("single.jpg")
    assert output_path.exists()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["total_images"] == 1


def test_train_once_dry_run(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    data_yaml_path = _create_minimal_yolo_dataset(dataset_dir)
    output_dir = tmp_path / "runs"

    summary = train_once(
        data_yaml=data_yaml_path,
        output_dir=output_dir,
        model="yolov8n.pt",
        epochs=1,
        imgsz=64,
        batch=1,
        device="cpu",
        run_name="dry_run_case",
        dry_run=True,
    )

    assert summary["success"] is True
    assert summary["backend"] == "dry_run"
    assert summary["executed"] is False
    assert summary["validation"]["valid"] is True

    summary_path = Path(summary["summary_path"])
    assert summary_path.exists()
