from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from darkbreaker_sdk.interfaces import PluginManifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _create_bundle(
    training_root: Path,
    *,
    plugin_id: str,
    task_type: str = "detection",
    version: str = "det-v1",
    modality: str = "rgb",
) -> Path:
    bundle_dir = training_root / "exports" / plugin_id / task_type / version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.onnx").write_text("onnx-model", encoding="utf-8")
    _write_json(bundle_dir / "label_map.json", {"0": "ok"})
    (bundle_dir / "preprocess.yaml").write_text("resize: [640, 640]\n", encoding="utf-8")
    (bundle_dir / "postprocess.yaml").write_text("nms: 0.5\n", encoding="utf-8")
    _write_json(
        bundle_dir / "bundle.json",
        {
            "plugin_id": plugin_id,
            "task_type": task_type,
            "modality": modality,
            "version": version,
            "compatible_runtime": [
                {
                    "runtime": "onnxruntime",
                    "providers": ["CPUExecutionProvider"],
                    "file_extensions": [".onnx"],
                }
            ],
            "labels": ["ok"],
        },
    )
    return bundle_dir


class _CapturingDetector:
    last_config = None

    def __init__(self, config, *args, **kwargs):
        type(self).last_config = config

    def initialize(self) -> bool:
        return True

    def get_runtime_status(self, quality_blocked: bool = False) -> dict:
        config = type(self).last_config or {}
        model_cfg = config.get("model", {})
        model_path = model_cfg.get("model_path") or model_cfg.get("path")
        return {
            "runtime_mode": "real_dl" if not quality_blocked else "quality_blocked",
            "model_path_configured": model_path,
            "model_path_resolved": model_path,
            "model_file_exists": True,
            "real_model_loaded": True,
            "onnx_session_ready": True,
            "fallback_enabled": True,
            "dl_preflight_checked": True,
            "dl_preflight_passed": True,
            "dl_failure_reason": None,
            "dl_failure_details": [],
            "manifest_path": None,
            "manifest_exists": False,
            "class_map_path": model_cfg.get("class_map_path"),
            "class_map_exists": bool(model_cfg.get("class_map_path")),
            "class_map_compatible": True,
            "model_version_validated": True,
            "input_size_validated": True,
            "output_format_validated": True,
            "validated_class_names": [],
            "requested_providers": ["CPUExecutionProvider"],
            "session_providers": ["CPUExecutionProvider"],
            "input_tensor_shape": [1, 3, 640, 640],
            "output_tensor_shapes": [[1, 84, 8400]],
            "output_probe_shape": [1, 84, 8400],
            "output_structure_compatible": True,
            "session_error": None,
            "runtime_supported_labels": [],
            "runtime_supported_defect_labels": [],
        }


@pytest.mark.parametrize(
    ("module_name", "class_name", "plugin_id"),
    [
        ("plugins.busbar_inspection.plugin", "BusbarInspectionPlugin", "busbar_inspection"),
        ("plugins.capacitor_inspection.plugin", "CapacitorInspectionPlugin", "capacitor_inspection"),
        ("plugins.switch_inspection.plugin", "SwitchInspectionPlugin", "switch_inspection"),
        ("plugins.transformer_inspection.plugin", "TransformerInspectionPlugin", "transformer_inspection"),
    ],
)
def test_outdoor_plugins_inject_resolved_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    plugin_id: str,
) -> None:
    training_root = tmp_path / "training"
    bundle_dir = _create_bundle(training_root, plugin_id=plugin_id)

    plugin_dir = REPO_ROOT / "plugins" / plugin_id
    config = yaml.safe_load((plugin_dir / "configs" / "default.yaml").read_text(encoding="utf-8"))
    config.setdefault("model", {})
    config["model"]["registry"] = {
        "training_root": str(training_root),
        "version": "det-v1",
        "strict": True,
    }

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "get_detector_class", lambda: _CapturingDetector)
    _CapturingDetector.last_config = None

    manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
    plugin_cls = getattr(module, class_name)
    plugin = plugin_cls(manifest, plugin_dir)

    assert plugin.init(copy.deepcopy(config)) is True

    captured = _CapturingDetector.last_config
    assert captured is not None
    assert captured["model"]["path"] == str(bundle_dir / "model.onnx")
    assert captured["model"]["model_path"] == str(bundle_dir / "model.onnx")
    assert captured["model"]["label_map_path"] == str(bundle_dir / "label_map.json")
    assert captured["model"]["preprocess_config_path"] == str(bundle_dir / "preprocess.yaml")
    assert captured["model"]["postprocess_config_path"] == str(bundle_dir / "postprocess.yaml")
    assert captured["model_resolution"]["resolved"] is True
    assert captured["model_resolution"]["source"] == "normalized"

    health = plugin.healthcheck()
    assert health.details["model_resolution"]["resolved"] is True

    if plugin_id in {"busbar_inspection", "switch_inspection"}:
        assert captured["yolov8_model_path"] == str(bundle_dir / "model.onnx")


def test_outdoor_plugin_keeps_original_path_when_resolution_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_id = "capacitor_inspection"
    plugin_dir = REPO_ROOT / "plugins" / plugin_id
    config = yaml.safe_load((plugin_dir / "configs" / "default.yaml").read_text(encoding="utf-8"))
    original_path = config["model"]["path"]
    config["model"]["registry"] = {
        "training_root": str(tmp_path / "training"),
        "version": "missing-v1",
        "strict": False,
    }

    module = importlib.import_module("plugins.capacitor_inspection.plugin")
    monkeypatch.setattr(module, "get_detector_class", lambda: _CapturingDetector)
    _CapturingDetector.last_config = None

    manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
    plugin = module.CapacitorInspectionPlugin(manifest, plugin_dir)

    assert plugin.init(copy.deepcopy(config)) is True

    captured = _CapturingDetector.last_config
    assert captured is not None
    assert captured["model"]["path"] == original_path
    assert captured["model_resolution"]["resolved"] is False
    assert captured["model_resolution"]["error_code"] == "MODEL_NOT_FOUND"
