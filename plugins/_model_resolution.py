"""
Shared model-resolution helper for visual plugins.

Phase 1 integration goals:
- try the unified registry first
- normalize resolved artifact paths back into legacy plugin config shapes
- preserve current plugin behavior when resolution fails unless `strict=true`
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

try:
    from training.registry.visual_model_registry import (
        ModelRegistryError,
        PluginModelResolver,
    )
except ImportError:  # pragma: no cover - defensive fallback for partial environments
    ModelRegistryError = None  # type: ignore[assignment]
    PluginModelResolver = None  # type: ignore[assignment]


def _coerce_project_relative(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    project_root = base_dir.parents[1]
    return str((project_root / path).resolve(strict=False))


def resolve_plugin_model_config(
    *,
    plugin_id: str,
    plugin_dir: Path,
    config: dict[str, Any] | None,
    default_task_type: str,
    expected_modality: str | None,
    model_path_keys: Iterable[str] = (),
    top_level_path_keys: Iterable[str] = (),
    default_runtime: str | None = "onnxruntime",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Try to resolve a model bundle for a plugin and map it into the plugin config.

    The returned resolution dict is intended for logging and health surfaces.
    """
    normalized_config = copy.deepcopy(config or {})
    model_cfg = normalized_config.setdefault("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        normalized_config["model"] = model_cfg

    registry_cfg = model_cfg.get("registry", {})
    if not isinstance(registry_cfg, dict):
        registry_cfg = {}
    model_cfg["registry"] = registry_cfg

    resolution: dict[str, Any] = {
        "enabled": bool(registry_cfg.get("enabled", True)),
        "attempted": False,
        "resolved": False,
        "plugin_id": plugin_id,
        "task_type": str(registry_cfg.get("task_type") or default_task_type),
        "version": str(registry_cfg.get("version") or "latest"),
        "runtime": registry_cfg.get("runtime", default_runtime),
        "source": None,
        "model_path": None,
        "label_map_path": None,
        "preprocess_config_path": None,
        "postprocess_config_path": None,
        "warnings": [],
        "error_code": None,
        "error_message": None,
        "details": {},
    }

    if not resolution["enabled"]:
        normalized_config["model_resolution"] = resolution
        return normalized_config, resolution

    if PluginModelResolver is None or ModelRegistryError is None:
        resolution.update(
            {
                "attempted": True,
                "error_code": "RESOLVER_UNAVAILABLE",
                "error_message": "training.registry.visual_model_registry is unavailable",
            }
        )
        normalized_config["model_resolution"] = resolution
        return normalized_config, resolution

    training_root = _coerce_project_relative(plugin_dir, registry_cfg.get("training_root"))
    strict = bool(registry_cfg.get("strict", False))
    allow_legacy = bool(registry_cfg.get("allow_legacy", True))
    require_preprocess = bool(registry_cfg.get("require_preprocess", False))
    require_postprocess = bool(registry_cfg.get("require_postprocess", False))
    runtime = registry_cfg.get("runtime", default_runtime)
    runtime_version = registry_cfg.get("runtime_version")
    provider = registry_cfg.get("provider")
    model_role = registry_cfg.get("model_role")
    expected_labels = registry_cfg.get("expected_labels")

    resolver = PluginModelResolver(
        plugin_id,
        training_root=training_root,
        plugin_dir=plugin_dir,
    )

    resolution["attempted"] = True

    try:
        bundle = resolver.resolve_bundle(
            resolution["task_type"],
            version=resolution["version"],
            runtime=runtime,
            runtime_version=runtime_version,
            provider=provider,
            expected_modality=expected_modality,
            expected_labels=expected_labels,
            require_preprocess=require_preprocess,
            require_postprocess=require_postprocess,
            allow_legacy=allow_legacy,
            model_role=model_role,
        )
    except ModelRegistryError as exc:
        resolution.update(
            {
                "error_code": exc.code.value,
                "error_message": str(exc),
                "details": dict(exc.details),
            }
        )
        normalized_config["model_resolution"] = resolution
        if strict:
            raise
        return normalized_config, resolution

    artifacts = bundle.entry.artifacts
    assert artifacts is not None
    model_path = str(artifacts.model_path)

    model_cfg["path"] = model_path
    model_cfg["model_path"] = model_path
    for key in model_path_keys:
        model_cfg[key] = model_path
    for key in top_level_path_keys:
        normalized_config[key] = model_path

    label_map_path = str(artifacts.label_map_path) if artifacts.label_map_path else None
    preprocess_path = (
        str(artifacts.preprocess_config_path) if artifacts.preprocess_config_path else None
    )
    postprocess_path = (
        str(artifacts.postprocess_config_path) if artifacts.postprocess_config_path else None
    )

    if label_map_path:
        model_cfg["label_map_path"] = label_map_path
        model_cfg["class_map_path"] = label_map_path
    if preprocess_path:
        model_cfg["preprocess_config_path"] = preprocess_path
    if postprocess_path:
        model_cfg["postprocess_config_path"] = postprocess_path

    resolution.update(
        {
            "resolved": True,
            "version": bundle.entry.version,
            "source": bundle.entry.source,
            "model_path": model_path,
            "label_map_path": label_map_path,
            "preprocess_config_path": preprocess_path,
            "postprocess_config_path": postprocess_path,
            "warnings": list(bundle.warnings),
            "details": {
                "model_role": bundle.entry.model_role,
                "compatible_runtime": [
                    item.to_dict() for item in bundle.entry.compatible_runtime
                ],
                "labels": list(bundle.entry.labels),
            },
        }
    )

    normalized_config["model_resolution"] = resolution
    return normalized_config, resolution
