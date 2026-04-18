from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "training" / "registry" / "visual_model_registry.py"
SPEC = importlib.util.spec_from_file_location("visual_model_registry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ModelRegistryEntry = MODULE.ModelRegistryEntry
ModelArtifacts = MODULE.ModelArtifacts
RuntimeCompatibility = MODULE.RuntimeCompatibility
PluginModelResolver = MODULE.PluginModelResolver
VisualModelRegistry = MODULE.VisualModelRegistry
ModelResolutionRequest = MODULE.ModelResolutionRequest
ModelNotFoundError = MODULE.ModelNotFoundError
ModelVersionIncompatibleError = MODULE.ModelVersionIncompatibleError
LabelContractMismatchError = MODULE.LabelContractMismatchError
InputModalityMismatchError = MODULE.InputModalityMismatchError
PreprocessConfigMissingError = MODULE.PreprocessConfigMissingError


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_training_root(tmp_path: Path) -> Path:
    training_root = tmp_path / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        training_root / "registry" / "plugin_training_mapping.json",
        {
            "plugins": {
                "busbar_inspection": {
                    "legacy_alias": "busbar",
                    "input_modality": "rgb",
                }
            },
            "legacy_alias_map": {
                "busbar": "busbar_inspection",
            },
        },
    )
    return training_root


def _create_normalized_bundle(
    training_root: Path,
    *,
    plugin_id: str = "busbar_inspection",
    task_type: str = "detection",
    version: str = "det-v1",
    modality: str = "rgb",
    with_preprocess: bool = True,
) -> Path:
    bundle_dir = training_root / "exports" / plugin_id / task_type / version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.onnx").write_text("model", encoding="utf-8")
    _write_json(bundle_dir / "label_map.json", {"0": "pin_missing", "1": "crack"})
    if with_preprocess:
        (bundle_dir / "preprocess.yaml").write_text("resize: [640, 640]\n", encoding="utf-8")
    (bundle_dir / "postprocess.yaml").write_text("nms: 0.5\n", encoding="utf-8")
    _write_json(
        bundle_dir / "bundle.json",
        {
            "plugin_id": plugin_id,
            "task_type": task_type,
            "modality": modality,
            "version": version,
            "metrics": {"mAP@0.5": 0.82},
            "labels": ["pin_missing", "crack"],
            "compatible_runtime": [
                {
                    "runtime": "onnxruntime",
                    "min_version": "1.16.0",
                    "providers": ["CPUExecutionProvider"],
                    "file_extensions": [".onnx"],
                }
            ],
        },
    )
    return bundle_dir


def _register_bundle(training_root: Path, bundle_dir: Path, *, version: str = "det-v1") -> None:
    entry = ModelRegistryEntry(
        plugin_id="busbar_inspection",
        task_type="detection",
        modality="rgb",
        version=version,
        metrics={"mAP@0.5": 0.82},
        compatible_runtime=(
            RuntimeCompatibility(
                runtime="onnxruntime",
                min_version="1.16.0",
                providers=("CPUExecutionProvider",),
                file_extensions=(".onnx",),
            ),
        ),
        artifacts=ModelArtifacts(
            export_dir=bundle_dir,
            model_path=bundle_dir / "model.onnx",
            label_map_path=bundle_dir / "label_map.json",
            preprocess_config_path=bundle_dir / "preprocess.yaml",
            postprocess_config_path=bundle_dir / "postprocess.yaml",
            metadata_path=bundle_dir / "bundle.json",
        ),
        labels=("pin_missing", "crack"),
        export_path=bundle_dir,
        source="registry",
    )
    registry = VisualModelRegistry(training_root)
    registry.upsert(entry)


def test_resolve_normalized_bundle_from_registry(tmp_path: Path) -> None:
    training_root = _build_training_root(tmp_path)
    bundle_dir = _create_normalized_bundle(training_root)
    _register_bundle(training_root, bundle_dir)

    resolver = PluginModelResolver("busbar_inspection", training_root=training_root)
    bundle = resolver.resolve_bundle(
        "detection",
        version="det-v1",
        runtime="onnxruntime",
        runtime_version="1.18.0",
        provider="CPUExecutionProvider",
        expected_modality="rgb",
        expected_labels=("pin_missing", "crack"),
        require_preprocess=True,
    )

    plugin_config = bundle.to_plugin_config()
    assert plugin_config["model"]["path"].endswith("model.onnx")
    assert plugin_config["model"]["label_map_path"].endswith("label_map.json")
    assert bundle.entry.source == "registry"


def test_raise_when_runtime_version_is_incompatible(tmp_path: Path) -> None:
    training_root = _build_training_root(tmp_path)
    bundle_dir = _create_normalized_bundle(training_root)
    _register_bundle(training_root, bundle_dir)

    resolver = PluginModelResolver("busbar_inspection", training_root=training_root)

    try:
        resolver.resolve_bundle(
            "detection",
            version="det-v1",
            runtime="onnxruntime",
            runtime_version="1.14.0",
        )
    except ModelVersionIncompatibleError as exc:
        assert exc.code.value == "VERSION_INCOMPATIBLE"
    else:
        raise AssertionError("expected ModelVersionIncompatibleError")


def test_raise_when_labels_do_not_match(tmp_path: Path) -> None:
    training_root = _build_training_root(tmp_path)
    bundle_dir = _create_normalized_bundle(training_root)
    _register_bundle(training_root, bundle_dir)

    resolver = PluginModelResolver("busbar_inspection", training_root=training_root)

    try:
        resolver.resolve_bundle(
            "detection",
            version="det-v1",
            expected_labels=("pin_missing", "foreign_object"),
        )
    except LabelContractMismatchError as exc:
        assert exc.code.value == "LABEL_MISMATCH"
    else:
        raise AssertionError("expected LabelContractMismatchError")


def test_raise_when_modality_is_wrong(tmp_path: Path) -> None:
    training_root = _build_training_root(tmp_path)
    bundle_dir = _create_normalized_bundle(training_root, modality="rgb")
    _register_bundle(training_root, bundle_dir)

    resolver = PluginModelResolver("busbar_inspection", training_root=training_root)

    try:
        resolver.resolve_bundle(
            "detection",
            version="det-v1",
            expected_modality="thermal",
        )
    except InputModalityMismatchError as exc:
        assert exc.code.value == "MODALITY_MISMATCH"
    else:
        raise AssertionError("expected InputModalityMismatchError")


def test_raise_when_preprocess_is_missing(tmp_path: Path) -> None:
    training_root = _build_training_root(tmp_path)
    bundle_dir = _create_normalized_bundle(training_root, with_preprocess=False)

    registry = VisualModelRegistry(training_root)
    registry.resolve(
        ModelResolutionRequest.create(
            plugin_id="busbar_inspection",
            task_type="detection",
            version="latest",
            allow_legacy=False,
        )
    )

    resolver = PluginModelResolver("busbar_inspection", training_root=training_root)

    try:
        resolver.resolve_bundle(
            "detection",
            version="latest",
            require_preprocess=True,
            allow_legacy=False,
        )
    except PreprocessConfigMissingError as exc:
        assert exc.code.value == "PREPROCESS_MISSING"
    else:
        raise AssertionError("expected PreprocessConfigMissingError")


def test_fallback_to_legacy_best_pt(tmp_path: Path) -> None:
    training_root = _build_training_root(tmp_path)
    legacy_model = training_root / "checkpoints" / "busbar" / "HV_220kV" / "best.pt"
    legacy_model.parent.mkdir(parents=True, exist_ok=True)
    legacy_model.write_text("weights", encoding="utf-8")

    resolver = PluginModelResolver("busbar_inspection", training_root=training_root)
    bundle = resolver.resolve_bundle(
        "detection",
        version="latest",
        runtime="pytorch",
    )

    assert bundle.entry.source == "legacy_checkpoint"
    assert bundle.entry.artifacts.model_path == legacy_model
    assert "legacy path fallback" in " ".join(bundle.warnings)


def test_raise_when_exact_version_is_missing(tmp_path: Path) -> None:
    training_root = _build_training_root(tmp_path)
    _create_normalized_bundle(training_root, version="det-v1")

    resolver = PluginModelResolver("busbar_inspection", training_root=training_root)

    try:
        resolver.resolve_bundle("detection", version="det-v2", allow_legacy=False)
    except ModelNotFoundError as exc:
        assert exc.code.value == "MODEL_NOT_FOUND"
        assert "det-v2" in str(exc)
    else:
        raise AssertionError("expected ModelNotFoundError")
