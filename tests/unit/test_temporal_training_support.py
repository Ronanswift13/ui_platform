from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.pipelines.export.model_exporter import ModelExporter
from training.pipelines.ingestion.data_router import DataRouter
from training.pipelines.preprocessing.temporal_preprocessor import TemporalPreprocessor
from training.schemas.dataset_manifest import DatasetManifest, validate_manifest
from training.training_api_v2 import TrainRequest, TrainingManagerV2


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def test_acoustic_manifest_requires_dual_path_contract() -> None:
    manifest = DatasetManifest(
        name="acoustic-dual-path",
        plugin_id="acoustic_monitoring",
        task_type="acoustic_time_frequency_anomaly",
        version="1.0.0",
        format="dual_audio_views",
        input_modality="audio_waveform+spectrogram",
        num_samples=10,
        sample_rate_hz=16000,
        window_size=2.0,
        feature_views=["raw_waveform"],
        classes=["normal", "partial_discharge"],
        temporal_schema={
            "sources": [
                {
                    "name": "waveform_main",
                    "kind": "waveform",
                    "path": "waveforms",
                    "format": "wav",
                    "sample_rate_hz": 16000,
                }
            ]
        },
    )

    errors = validate_manifest(manifest)
    assert any("双路径方案" in error for error in errors)


def test_device_monitoring_requires_three_output_heads() -> None:
    manifest = DatasetManifest(
        name="device-health",
        plugin_id="device_monitoring",
        task_type="equipment_health_prediction",
        version="1.0.0",
        format="csv_series",
        input_modality="multivariate_timeseries",
        num_samples=12,
        sequence_length=64,
        prediction_horizon=24,
        sensor_columns=["cpu_temp", "cpu_usage"],
        temporal_schema={
            "sources": [
                {
                    "name": "timeseries_main",
                    "kind": "timeseries",
                    "path": "timeseries",
                    "format": "csv",
                    "sensor_columns": ["cpu_temp", "cpu_usage"],
                },
                {
                    "name": "labels_main",
                    "kind": "labels",
                    "path": "labels",
                    "format": "jsonl",
                },
            ]
        },
        target_schema={
            "outputs": [
                {"name": "health_index", "type": "regression"},
                {"name": "anomaly_score", "type": "anomaly_score"},
            ]
        },
    )

    errors = validate_manifest(manifest)
    assert any("predicted_failure" in error for error in errors)


def test_data_router_routes_temporal_tasks_with_version(tmp_path: Path) -> None:
    router = DataRouter(training_root=tmp_path / "training")
    manifest = DatasetManifest(
        name="gas-series",
        plugin_id="gas_detection",
        task_type="multivariate_sensor_anomaly",
        version="1.2.0",
        format="csv_series",
        input_modality="multivariate_timeseries",
        num_samples=20,
        sequence_length=48,
        sensor_columns=["SF6", "H2"],
        temporal_schema={
            "sources": [
                {
                    "name": "gas_series",
                    "kind": "timeseries",
                    "path": "timeseries",
                    "format": "csv",
                    "sensor_columns": ["SF6", "H2"],
                }
            ]
        },
    )

    result = router.route(manifest)
    assert result.success is True
    assert result.storage_family == "temporal_anomaly"
    assert result.destination == (
        tmp_path
        / "training"
        / "datasets"
        / "temporal_anomaly"
        / "gas_detection"
        / "sensor_anomaly"
        / "1.2.0"
    )


def test_temporal_preprocessor_creates_split_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    timeseries_dir = input_dir / "timeseries"
    labels_dir = input_dir / "labels"
    timeseries_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for idx in range(6):
        (timeseries_dir / f"sample_{idx}.csv").write_text("ts,value\n1,0.1\n", encoding="utf-8")
        (labels_dir / f"sample_{idx}.json").write_text(
            json.dumps({"sample_id": f"sample_{idx}", "label": "normal"}),
            encoding="utf-8",
        )

    _write_json(
        input_dir / "manifest.json",
        {
            "name": "device-samples",
            "plugin_id": "device_monitoring",
            "task_type": "multivariate_sensor_anomaly",
            "version": "1.0.0",
            "format": "csv_series",
            "input_modality": "multivariate_timeseries",
            "num_samples": 6,
            "sequence_length": 32,
            "sensor_columns": ["cpu_temp"],
            "temporal_schema": {
                "sources": [
                    {
                        "name": "series",
                        "kind": "timeseries",
                        "path": "timeseries",
                        "format": "csv",
                        "sensor_columns": ["cpu_temp"]
                    }
                ]
            }
        },
    )

    result = TemporalPreprocessor(config={"random_seed": 7}).preprocess(
        input_dir, tmp_path / "prepared"
    )

    assert result.success is True
    assert (tmp_path / "prepared" / "dataset_config.json").exists()
    assert (tmp_path / "prepared" / "train" / "samples.jsonl").exists()


def test_temporal_export_bundle_writes_versioned_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_text("onnx", encoding="utf-8")

    exporter = ModelExporter(exports_root=tmp_path / "exports")
    result = exporter.export_bundle(
        plugin_id="device_monitoring",
        task_type="equipment_health_prediction",
        version="health-v1",
        model_path=model_path,
        modality="multivariate_timeseries",
        compatible_runtime=[{"runtime": "onnxruntime"}],
        target_schema={
            "outputs": [
                {"name": "health_index", "type": "regression"},
                {"name": "anomaly_score", "type": "anomaly_score"},
                {"name": "predicted_failure", "type": "classification"},
            ]
        },
    )

    bundle_dir = tmp_path / "exports" / "device_monitoring" / "equipment_health_prediction" / "health-v1"
    assert result.success is True
    assert (bundle_dir / "model.onnx").exists()
    assert (bundle_dir / "bundle.json").exists()


def test_training_manager_v2_accepts_temporal_plugin_task() -> None:
    manager = TrainingManagerV2()
    task = manager.create_task(
        TrainRequest(
            plugin_id="device_monitoring",
            task_type="equipment_health_prediction",
            sequence_length=64,
            prediction_horizon=24,
            sensor_columns=["cpu_temp", "cpu_usage"],
        )
    )

    assert task.plugin_id == "device_monitoring"
    assert task.task_type == "equipment_health_prediction"


def test_health_index_calibration_manifest_is_valid() -> None:
    manifest = DatasetManifest(
        name="device-health-calibration",
        plugin_id="device_monitoring",
        task_type="health_index_calibration",
        version="1.0.0",
        format="csv_series",
        input_modality="multivariate_timeseries",
        num_samples=64,
        sequence_length=64,
        sensor_columns=["cpu_temp", "cpu_usage"],
        temporal_schema={
            "sources": [
                {
                    "name": "series_main",
                    "kind": "timeseries",
                    "path": "timeseries",
                    "format": "csv",
                    "sensor_columns": ["cpu_temp", "cpu_usage"],
                },
                {
                    "name": "labels_main",
                    "kind": "labels",
                    "path": "labels",
                    "format": "jsonl",
                },
            ]
        },
        target_schema={
            "outputs": [
                {"name": "health_index", "type": "regression", "range": "[0,100]"}
            ]
        },
    )

    assert validate_manifest(manifest) == []
