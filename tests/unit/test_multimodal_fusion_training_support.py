from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.multimodal_fusion import validate_multimodal_sample_record
from training.pipelines.ingestion.data_router import DataRouter
from training.pipelines.preprocessing.multimodal_preprocessor import MultimodalPreprocessor
from training.schemas.dataset_manifest import DatasetManifest, validate_manifest
from training.training_api_v2 import TrainRequest, TrainingManagerV2


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def test_multimodal_manifest_supports_sparse_modalities() -> None:
    manifest = DatasetManifest(
        name="multimodal-sparse",
        plugin_id="multimodal_fusion",
        task_type="multimodal_feature_fusion",
        version="1.0.0",
        format="multimodal_aligned",
        input_modality="multimodal",
        num_samples=2,
        fusion_strategy="feature_level",
        supported_modalities=["visual", "thermal", "gas"],
        required_modalities=["visual", "thermal"],
        missing_modality_policy="mask_and_gate",
        classes=["normal", "warning", "alarm"],
        multimodal_schema={
            "supported_modalities": ["visual", "thermal", "gas"],
            "required_modalities": ["visual", "thermal"],
            "fusion_strategy": "feature_level",
            "missing_modality_policy": "mask_and_gate",
            "runtime_mode": "rule_model_hybrid",
            "modality_specs": [
                {"modality": "visual", "path": "visual", "format": "json"},
                {"modality": "thermal", "path": "thermal", "format": "json"},
                {"modality": "gas", "path": "gas", "format": "json"},
            ],
        },
        diagnosis_target_schema={
            "outputs": [
                {"name": "overall_status", "type": "classification"},
                {"name": "confidence", "type": "confidence"},
                {"name": "detections", "type": "detection_list"},
                {"name": "modality_contributions", "type": "attribution"},
                {"name": "diagnostic_report", "type": "structured_report"},
            ]
        },
    )

    assert validate_manifest(manifest) == []


def test_multimodal_sample_record_allows_partial_modalities() -> None:
    record = {
        "sample_id": "sample_001",
        "device_id": "transformer_01",
        "timestamp": "2026-04-14T10:00:00Z",
        "available_modalities": ["visual", "thermal", "gas"],
        "modality_paths": {
            "visual": "visual/sample_001.json",
            "thermal": "thermal/sample_001.json",
            "gas": "gas/sample_001.json",
        },
        "label": "alarm",
        "diagnosis_target": {"overall_status": "alarm"},
    }

    assert validate_multimodal_sample_record(record) == []


def test_data_router_routes_multimodal_tasks_with_version(tmp_path: Path) -> None:
    router = DataRouter(training_root=tmp_path / "training")
    manifest = DatasetManifest(
        name="mmf",
        plugin_id="multimodal_fusion",
        task_type="multimodal_decision_fusion",
        version="2.0.0",
        format="multimodal_sparse",
        input_modality="multimodal",
        num_samples=3,
        fusion_strategy="decision_level",
        supported_modalities=["visual", "thermal"],
        required_modalities=["visual"],
        missing_modality_policy="allow_sparse",
        classes=["normal", "warning"],
        multimodal_schema={
            "supported_modalities": ["visual", "thermal"],
            "required_modalities": ["visual"],
            "fusion_strategy": "decision_level",
            "missing_modality_policy": "allow_sparse",
            "runtime_mode": "rule_model_hybrid",
            "modality_specs": [
                {"modality": "visual", "path": "visual", "format": "json"},
                {"modality": "thermal", "path": "thermal", "format": "json"},
            ],
        },
        diagnosis_target_schema={
            "outputs": [
                {"name": "overall_status", "type": "classification"},
                {"name": "confidence", "type": "confidence"},
                {"name": "detections", "type": "detection_list"},
                {"name": "modality_contributions", "type": "attribution"},
                {"name": "diagnostic_report", "type": "structured_report"},
            ]
        },
        diagnostic_rule_pack={"path": "metadata/diagnostic_rules.json"},
    )

    result = router.route(manifest)
    assert result.success is True
    assert result.storage_family == "multimodal_fusion"
    assert result.destination == (
        tmp_path
        / "training"
        / "datasets"
        / "multimodal_fusion"
        / "multimodal_fusion"
        / "decision_fusion"
        / "2.0.0"
    )


def test_multimodal_preprocessor_keeps_sparse_records(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    (input_dir / "visual").mkdir(parents=True)
    (input_dir / "thermal").mkdir(parents=True)
    (input_dir / "gas").mkdir(parents=True)
    (input_dir / "labels").mkdir(parents=True)
    (input_dir / "metadata").mkdir(parents=True)

    _write_json(input_dir / "visual" / "sample_001.json", {"confidence": 0.8})
    _write_json(input_dir / "thermal" / "sample_001.json", {"confidence": 0.7})
    _write_json(input_dir / "visual" / "sample_002.json", {"confidence": 0.6})
    _write_json(input_dir / "gas" / "sample_002.json", {"confidence": 0.9})
    _write_json(input_dir / "metadata" / "diagnostic_rules.json", {"rules": []})

    _write_json(
        input_dir / "manifest.json",
        {
            "name": "mmf-preprocess",
            "plugin_id": "multimodal_fusion",
            "task_type": "multimodal_feature_fusion",
            "version": "1.0.0",
            "format": "multimodal_aligned",
            "input_modality": "multimodal",
            "num_samples": 2,
            "fusion_strategy": "feature_level",
            "supported_modalities": ["visual", "thermal", "gas"],
            "required_modalities": ["visual"],
            "missing_modality_policy": "mask_and_gate",
            "classes": ["normal", "alarm"],
            "multimodal_schema": {
                "supported_modalities": ["visual", "thermal", "gas"],
                "required_modalities": ["visual"],
                "fusion_strategy": "feature_level",
                "missing_modality_policy": "mask_and_gate",
                "runtime_mode": "rule_model_hybrid",
                "modality_specs": [
                    {"modality": "visual", "path": "visual", "format": "json"},
                    {"modality": "thermal", "path": "thermal", "format": "json"},
                    {"modality": "gas", "path": "gas", "format": "json"}
                ]
            },
            "diagnosis_target_schema": {
                "outputs": [
                    {"name": "overall_status", "type": "classification"},
                    {"name": "confidence", "type": "confidence"},
                    {"name": "detections", "type": "detection_list"},
                    {"name": "modality_contributions", "type": "attribution"},
                    {"name": "diagnostic_report", "type": "structured_report"}
                ]
            }
        },
    )

    samples = [
        {
            "sample_id": "sample_001",
            "device_id": "transformer_01",
            "timestamp": "2026-04-14T10:00:00Z",
            "available_modalities": ["visual", "thermal"],
            "modality_paths": {
                "visual": "visual/sample_001.json",
                "thermal": "thermal/sample_001.json",
            },
            "label": "alarm",
            "diagnosis_target": {"overall_status": "alarm"},
        },
        {
            "sample_id": "sample_002",
            "device_id": "transformer_02",
            "timestamp": "2026-04-14T10:05:00Z",
            "available_modalities": ["visual", "gas"],
            "modality_paths": {
                "visual": "visual/sample_002.json",
                "gas": "gas/sample_002.json",
            },
            "label": "normal",
            "diagnosis_target": {"overall_status": "normal"},
        },
    ]
    with open(input_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = MultimodalPreprocessor(config={"random_seed": 7}).preprocess(
        input_dir, tmp_path / "prepared"
    )

    assert result.success is True
    assert (tmp_path / "prepared" / "dataset_config.json").exists()
    assert (tmp_path / "prepared" / "train" / "samples.jsonl").exists()


def test_training_manager_v2_accepts_multimodal_task() -> None:
    manager = TrainingManagerV2()
    task = manager.create_task(
        TrainRequest(
            plugin_id="multimodal_fusion",
            task_type="multimodal_feature_fusion",
            fusion_strategy="feature_level",
            supported_modalities=["visual", "thermal", "gas"],
            required_modalities=["visual", "thermal"],
            missing_modality_policy="mask_and_gate",
        )
    )

    assert task.plugin_id == "multimodal_fusion"
    assert task.task_type == "multimodal_feature_fusion"
