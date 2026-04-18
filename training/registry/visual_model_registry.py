"""
Unified visual model registry and plugin-side resolver.

This module defines a filesystem-first registry contract for visual plugins.
It is intentionally narrow for phase 1:

- find model bundles by plugin_id + task_type + version
- load resolved artifact paths for plugins
- classify resolution failures with stable error codes
- fall back to legacy `best.pt` / `.onnx` paths during migration

Recommended normalized export layout:

    training/exports/{plugin_id}/{task_type}/{version}/
      model.onnx | model.engine | model.pt | model.pth
      label_map.json
      preprocess.yaml | preprocess.json
      postprocess.yaml | postprocess.json
      bundle.json

Recommended registry index:

    training/registry/model_registry.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_REGISTRY_FILENAME = "model_registry.json"
DEFAULT_PLUGIN_MAPPING_FILENAME = "plugin_training_mapping.json"
DEFAULT_SCHEMA_VERSION = "2.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _normalize_modality(value: str | None) -> str:
    if not value:
        return ""
    parts = [part.strip().lower() for part in value.split("+") if part.strip()]
    return "+".join(sorted(parts))


def _normalize_labels(labels: Iterable[str] | None) -> tuple[str, ...]:
    if not labels:
        return ()
    return tuple(str(item).strip() for item in labels if str(item).strip())


def _version_tokens(value: str) -> tuple[Any, ...]:
    text = value.strip().lstrip("vV")
    if not text:
        return ()
    tokens: list[Any] = []
    for part in re.split(r"[^0-9A-Za-z]+", text):
        if not part:
            continue
        if part.isdigit():
            tokens.append(int(part))
        else:
            tokens.append(part.lower())
    return tuple(tokens)


def _compare_versions(left: str, right: str) -> int:
    left_tokens = list(_version_tokens(left))
    right_tokens = list(_version_tokens(right))
    max_len = max(len(left_tokens), len(right_tokens))
    left_tokens.extend([0] * (max_len - len(left_tokens)))
    right_tokens.extend([0] * (max_len - len(right_tokens)))
    for lhs, rhs in zip(left_tokens, right_tokens):
        if lhs == rhs:
            continue
        if isinstance(lhs, int) and isinstance(rhs, int):
            return -1 if lhs < rhs else 1
        return -1 if str(lhs) < str(rhs) else 1
    return 0


class ResolutionErrorCode(str, Enum):
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    VERSION_INCOMPATIBLE = "VERSION_INCOMPATIBLE"
    LABEL_MISMATCH = "LABEL_MISMATCH"
    MODALITY_MISMATCH = "MODALITY_MISMATCH"
    PREPROCESS_MISSING = "PREPROCESS_MISSING"
    POSTPROCESS_MISSING = "POSTPROCESS_MISSING"
    RUNTIME_INCOMPATIBLE = "RUNTIME_INCOMPATIBLE"


class ModelRegistryError(RuntimeError):
    def __init__(
        self,
        code: ResolutionErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ModelNotFoundError(ModelRegistryError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ResolutionErrorCode.MODEL_NOT_FOUND, message, details=details)


class ModelVersionIncompatibleError(ModelRegistryError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ResolutionErrorCode.VERSION_INCOMPATIBLE, message, details=details)


class LabelContractMismatchError(ModelRegistryError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ResolutionErrorCode.LABEL_MISMATCH, message, details=details)


class InputModalityMismatchError(ModelRegistryError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ResolutionErrorCode.MODALITY_MISMATCH, message, details=details)


class PreprocessConfigMissingError(ModelRegistryError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ResolutionErrorCode.PREPROCESS_MISSING, message, details=details)


class PostprocessConfigMissingError(ModelRegistryError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ResolutionErrorCode.POSTPROCESS_MISSING, message, details=details)


class RuntimeCompatibilityError(ModelRegistryError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ResolutionErrorCode.RUNTIME_INCOMPATIBLE, message, details=details)


@dataclass(frozen=True)
class RuntimeCompatibility:
    runtime: str
    min_version: str | None = None
    max_version: str | None = None
    providers: tuple[str, ...] = ()
    file_extensions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeCompatibility":
        providers = tuple(str(item) for item in data.get("providers", []) if item)
        file_extensions = tuple(
            str(item) for item in data.get("file_extensions", []) if item
        )
        return cls(
            runtime=str(data.get("runtime", "")).strip(),
            min_version=data.get("min_version"),
            max_version=data.get("max_version"),
            providers=providers,
            file_extensions=file_extensions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "min_version": self.min_version,
            "max_version": self.max_version,
            "providers": list(self.providers),
            "file_extensions": list(self.file_extensions),
        }


@dataclass(frozen=True)
class ModelArtifacts:
    export_dir: Path
    model_path: Path
    label_map_path: Path | None = None
    preprocess_config_path: Path | None = None
    postprocess_config_path: Path | None = None
    metadata_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_dir": str(self.export_dir),
            "model_path": str(self.model_path),
            "label_map_path": str(self.label_map_path) if self.label_map_path else None,
            "preprocess_config_path": (
                str(self.preprocess_config_path) if self.preprocess_config_path else None
            ),
            "postprocess_config_path": (
                str(self.postprocess_config_path) if self.postprocess_config_path else None
            ),
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "ModelArtifacts":
        def _coerce(path_value: str | None) -> Path | None:
            if not path_value:
                return None
            path = Path(path_value)
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            return path

        export_dir = _coerce(data.get("export_dir"))
        model_path = _coerce(data.get("model_path"))
        if export_dir is None or model_path is None:
            raise ValueError("artifacts.export_dir and artifacts.model_path are required")

        return cls(
            export_dir=export_dir,
            model_path=model_path,
            label_map_path=_coerce(data.get("label_map_path")),
            preprocess_config_path=_coerce(data.get("preprocess_config_path")),
            postprocess_config_path=_coerce(data.get("postprocess_config_path")),
            metadata_path=_coerce(data.get("metadata_path")),
        )


@dataclass(frozen=True)
class ModelRegistryEntry:
    plugin_id: str
    task_type: str
    modality: str
    version: str
    metrics: dict[str, float] = field(default_factory=dict)
    compatible_runtime: tuple[RuntimeCompatibility, ...] = ()
    artifacts: ModelArtifacts | None = None
    model_role: str | None = None
    labels: tuple[str, ...] = ()
    export_path: Path | None = None
    created_at: str | None = None
    source: str = "registry"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return build_entry_key(
            self.plugin_id,
            self.task_type,
            self.version,
            model_role=self.model_role,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "task_type": self.task_type,
            "modality": self.modality,
            "version": self.version,
            "metrics": self.metrics,
            "compatible_runtime": [item.to_dict() for item in self.compatible_runtime],
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "model_role": self.model_role,
            "labels": list(self.labels),
            "export_path": str(self.export_path) if self.export_path else None,
            "created_at": self.created_at,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "ModelRegistryEntry":
        runtimes = tuple(
            RuntimeCompatibility.from_dict(item)
            for item in data.get("compatible_runtime", [])
        )
        artifacts_data = data.get("artifacts")
        artifacts = None
        if isinstance(artifacts_data, dict):
            artifacts = ModelArtifacts.from_dict(artifacts_data, base_dir=base_dir)

        export_path = data.get("export_path")
        export_path_obj: Path | None = None
        if export_path:
            export_path_obj = Path(export_path)
            if base_dir is not None and not export_path_obj.is_absolute():
                export_path_obj = base_dir / export_path_obj

        return cls(
            plugin_id=str(data["plugin_id"]),
            task_type=str(data["task_type"]),
            modality=_normalize_modality(data.get("modality")),
            version=str(data["version"]),
            metrics={str(k): float(v) for k, v in data.get("metrics", {}).items()},
            compatible_runtime=runtimes,
            artifacts=artifacts,
            model_role=data.get("model_role"),
            labels=_normalize_labels(data.get("labels")),
            export_path=export_path_obj,
            created_at=data.get("created_at"),
            source=str(data.get("source", "registry")),
            metadata=dict(data.get("metadata", {})),
        )


def build_entry_key(
    plugin_id: str,
    task_type: str,
    version: str,
    *,
    model_role: str | None = None,
) -> str:
    if model_role:
        return f"{plugin_id}:{task_type}:{model_role}:{version}"
    return f"{plugin_id}:{task_type}:{version}"


@dataclass(frozen=True)
class ModelResolutionRequest:
    plugin_id: str
    task_type: str
    version: str | None = "latest"
    runtime: str | None = None
    runtime_version: str | None = None
    provider: str | None = None
    expected_modality: str | None = None
    expected_labels: tuple[str, ...] = ()
    require_preprocess: bool = False
    require_postprocess: bool = False
    allow_legacy: bool = True
    plugin_dir: Path | None = None
    model_role: str | None = None

    @classmethod
    def create(
        cls,
        plugin_id: str,
        task_type: str,
        *,
        version: str | None = "latest",
        runtime: str | None = None,
        runtime_version: str | None = None,
        provider: str | None = None,
        expected_modality: str | None = None,
        expected_labels: Iterable[str] | None = None,
        require_preprocess: bool = False,
        require_postprocess: bool = False,
        allow_legacy: bool = True,
        plugin_dir: Path | None = None,
        model_role: str | None = None,
    ) -> "ModelResolutionRequest":
        return cls(
            plugin_id=plugin_id,
            task_type=task_type,
            version=version,
            runtime=runtime,
            runtime_version=runtime_version,
            provider=provider,
            expected_modality=expected_modality,
            expected_labels=_normalize_labels(expected_labels),
            require_preprocess=require_preprocess,
            require_postprocess=require_postprocess,
            allow_legacy=allow_legacy,
            plugin_dir=plugin_dir,
            model_role=model_role,
        )


@dataclass(frozen=True)
class ResolvedModelBundle:
    entry: ModelRegistryEntry
    request: ModelResolutionRequest
    warnings: tuple[str, ...] = ()

    def to_plugin_config(self) -> dict[str, Any]:
        artifacts = self.entry.artifacts
        if artifacts is None:
            raise ModelNotFoundError(
                "resolved entry does not contain artifacts",
                details={"entry_key": self.entry.key},
            )

        return {
            "registry": {
                "plugin_id": self.entry.plugin_id,
                "task_type": self.entry.task_type,
                "version": self.entry.version,
                "model_role": self.entry.model_role,
                "source": self.entry.source,
            },
            "model": {
                "path": str(artifacts.model_path),
                "export_dir": str(artifacts.export_dir),
                "label_map_path": (
                    str(artifacts.label_map_path) if artifacts.label_map_path else None
                ),
                "preprocess_config_path": (
                    str(artifacts.preprocess_config_path)
                    if artifacts.preprocess_config_path
                    else None
                ),
                "postprocess_config_path": (
                    str(artifacts.postprocess_config_path)
                    if artifacts.postprocess_config_path
                    else None
                ),
                "modality": self.entry.modality,
                "compatible_runtime": [
                    item.to_dict() for item in self.entry.compatible_runtime
                ],
                "labels": list(self.entry.labels),
            },
        }


class VisualModelRegistry:
    def __init__(
        self,
        training_root: Path | str | None = None,
        *,
        registry_path: Path | str | None = None,
        plugin_mapping_path: Path | str | None = None,
    ) -> None:
        self.training_root = Path(training_root) if training_root else Path(__file__).resolve().parents[1]
        self.registry_path = (
            Path(registry_path)
            if registry_path
            else self.training_root / "registry" / DEFAULT_REGISTRY_FILENAME
        )
        self.plugin_mapping_path = (
            Path(plugin_mapping_path)
            if plugin_mapping_path
            else self.training_root / "registry" / DEFAULT_PLUGIN_MAPPING_FILENAME
        )
        self._registry_cache: dict[str, ModelRegistryEntry] | None = None
        self._plugin_mapping_cache: dict[str, Any] | None = None

    def load_registry(self, *, force_reload: bool = False) -> dict[str, ModelRegistryEntry]:
        if self._registry_cache is not None and not force_reload:
            return self._registry_cache

        records: dict[str, ModelRegistryEntry] = {}
        if self.registry_path.exists():
            raw = _read_json(self.registry_path)
            base_dir = self.registry_path.parent
            if isinstance(raw.get("records"), dict):
                for key, item in raw["records"].items():
                    if not isinstance(item, dict):
                        continue
                    entry = ModelRegistryEntry.from_dict(item, base_dir=base_dir)
                    records[key] = entry
        self._registry_cache = records
        return records

    def save_registry(self) -> None:
        records = self.load_registry()
        payload = {
            "schema_version": DEFAULT_SCHEMA_VERSION,
            "updated_at": _utc_now(),
            "records": {
                key: entry.to_dict()
                for key, entry in sorted(records.items(), key=lambda item: item[0])
            },
        }
        _write_json(self.registry_path, payload)

    def upsert(self, entry: ModelRegistryEntry) -> str:
        records = self.load_registry()
        records[entry.key] = entry
        self.save_registry()
        return entry.key

    def resolve(self, request: ModelResolutionRequest) -> ResolvedModelBundle:
        canonical_plugin_id = self.resolve_alias(request.plugin_id)
        request = ModelResolutionRequest(
            plugin_id=canonical_plugin_id,
            task_type=request.task_type,
            version=request.version,
            runtime=request.runtime,
            runtime_version=request.runtime_version,
            provider=request.provider,
            expected_modality=request.expected_modality,
            expected_labels=request.expected_labels,
            require_preprocess=request.require_preprocess,
            require_postprocess=request.require_postprocess,
            allow_legacy=request.allow_legacy,
            plugin_dir=request.plugin_dir,
            model_role=request.model_role,
        )

        candidates = self._collect_candidates(request)
        if not candidates:
            raise ModelNotFoundError(
                f"no model bundle found for {request.plugin_id}/{request.task_type}",
                details={
                    "plugin_id": request.plugin_id,
                    "task_type": request.task_type,
                    "version": request.version,
                },
            )

        entry = self._select_candidate(candidates, request)
        warnings = self._validate_entry(entry, request)
        return ResolvedModelBundle(entry=entry, request=request, warnings=warnings)

    def resolve_alias(self, plugin_id_or_alias: str) -> str:
        mapping = self._load_plugin_mapping()
        alias_map = mapping.get("legacy_alias_map", {})
        return str(alias_map.get(plugin_id_or_alias, plugin_id_or_alias))

    def _load_plugin_mapping(self) -> dict[str, Any]:
        if self._plugin_mapping_cache is None:
            if self.plugin_mapping_path.exists():
                self._plugin_mapping_cache = _read_json(self.plugin_mapping_path)
            else:
                self._plugin_mapping_cache = {}
        return self._plugin_mapping_cache

    def _collect_candidates(self, request: ModelResolutionRequest) -> list[ModelRegistryEntry]:
        records = self.load_registry()
        candidates: dict[str, ModelRegistryEntry] = {}

        for entry in records.values():
            if entry.plugin_id != request.plugin_id or entry.task_type != request.task_type:
                continue
            if request.model_role and entry.model_role != request.model_role:
                continue
            candidates[entry.key] = entry

        normalized = self._discover_normalized_entries(
            request.plugin_id,
            request.task_type,
            model_role=request.model_role,
        )
        for entry in normalized:
            candidates.setdefault(entry.key, entry)

        if request.allow_legacy:
            legacy = self._discover_legacy_entries(
                request.plugin_id,
                request.task_type,
                plugin_dir=request.plugin_dir,
                model_role=request.model_role,
            )
            for entry in legacy:
                candidates.setdefault(entry.key, entry)

        return list(candidates.values())

    def _select_candidate(
        self,
        candidates: list[ModelRegistryEntry],
        request: ModelResolutionRequest,
    ) -> ModelRegistryEntry:
        if request.version and request.version not in ("latest", "*"):
            exact = [item for item in candidates if item.version == request.version]
            if exact:
                return sorted(exact, key=self._candidate_sort_key, reverse=True)[0]
            available = sorted({item.version for item in candidates})
            raise ModelNotFoundError(
                (
                    f"version '{request.version}' not found for "
                    f"{request.plugin_id}/{request.task_type}"
                ),
                details={
                    "plugin_id": request.plugin_id,
                    "task_type": request.task_type,
                    "requested_version": request.version,
                    "available_versions": available,
                },
            )

        return sorted(candidates, key=self._candidate_sort_key, reverse=True)[0]

    def _candidate_sort_key(self, entry: ModelRegistryEntry) -> tuple[Any, ...]:
        source_priority = {
            "registry": 50,
            "normalized": 40,
            "plugin_embedded": 30,
            "legacy_export": 20,
            "legacy_checkpoint": 10,
        }.get(entry.source, 0)
        created = entry.created_at or ""
        try:
            mtime = (
                entry.artifacts.model_path.stat().st_mtime
                if entry.artifacts and entry.artifacts.model_path.exists()
                else 0.0
            )
        except OSError:
            mtime = 0.0
        return (
            source_priority,
            _version_tokens(entry.version),
            created,
            mtime,
        )

    def _discover_normalized_entries(
        self,
        plugin_id: str,
        task_type: str,
        *,
        model_role: str | None = None,
    ) -> list[ModelRegistryEntry]:
        task_root = self.training_root / "exports" / plugin_id / task_type
        if not task_root.exists():
            return []

        entries: list[ModelRegistryEntry] = []
        for version_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
            entry = self._entry_from_bundle_dir(
                version_dir,
                plugin_id=plugin_id,
                task_type=task_type,
                version=version_dir.name,
                source="normalized",
                model_role=model_role,
            )
            if entry is not None:
                entries.append(entry)

        if entries:
            return entries

        # Transitional support: exports/{plugin}/{task}/{model.onnx}
        entry = self._entry_from_bundle_dir(
            task_root,
            plugin_id=plugin_id,
            task_type=task_type,
            version="legacy-unversioned",
            source="normalized",
            model_role=model_role,
        )
        return [entry] if entry else []

    def _discover_legacy_entries(
        self,
        plugin_id: str,
        task_type: str,
        *,
        plugin_dir: Path | None,
        model_role: str | None = None,
    ) -> list[ModelRegistryEntry]:
        candidates: list[ModelRegistryEntry] = []
        legacy_names = {plugin_id}

        mapping = self._load_plugin_mapping()
        plugin_info = mapping.get("plugins", {}).get(plugin_id, {})
        legacy_alias = plugin_info.get("legacy_alias")
        if legacy_alias:
            legacy_names.add(str(legacy_alias))

        for legacy_name in sorted(legacy_names):
            checkpoint_root = self.training_root / "checkpoints" / legacy_name
            if checkpoint_root.exists():
                for model_path in sorted(checkpoint_root.rglob("best.pt")):
                    export_dir = model_path.parent
                    version = f"legacy-{export_dir.name}-best"
                    candidates.append(
                        self._build_discovered_entry(
                            plugin_id=plugin_id,
                            task_type=task_type,
                            version=version,
                            model_path=model_path,
                            export_dir=export_dir,
                            source="legacy_checkpoint",
                            model_role=model_role,
                        )
                    )

            exports_root = self.training_root / "exports" / legacy_name
            if exports_root.exists():
                for model_path in sorted(exports_root.rglob("*.onnx")):
                    export_dir = model_path.parent
                    version = f"legacy-{model_path.stem}"
                    candidates.append(
                        self._build_discovered_entry(
                            plugin_id=plugin_id,
                            task_type=task_type,
                            version=version,
                            model_path=model_path,
                            export_dir=export_dir,
                            source="legacy_export",
                            model_role=model_role,
                        )
                    )

        if plugin_dir is not None:
            models_root = plugin_dir / "models"
            if models_root.exists():
                for model_path in sorted(models_root.rglob("*.onnx")):
                    export_dir = model_path.parent
                    version = f"plugin-local-{model_path.stem}"
                    candidates.append(
                        self._build_discovered_entry(
                            plugin_id=plugin_id,
                            task_type=task_type,
                            version=version,
                            model_path=model_path,
                            export_dir=export_dir,
                            source="plugin_embedded",
                            model_role=model_role,
                        )
                    )

        return candidates

    def _entry_from_bundle_dir(
        self,
        bundle_dir: Path,
        *,
        plugin_id: str,
        task_type: str,
        version: str,
        source: str,
        model_role: str | None = None,
    ) -> ModelRegistryEntry | None:
        model_path = self._detect_model_path(bundle_dir)
        if model_path is None:
            return None

        metadata_path = self._detect_metadata_path(bundle_dir)
        metadata: dict[str, Any] = {}
        if metadata_path and metadata_path.exists():
            metadata = _read_json(metadata_path)

        artifacts = ModelArtifacts(
            export_dir=bundle_dir,
            model_path=model_path,
            label_map_path=self._detect_label_map_path(bundle_dir),
            preprocess_config_path=self._detect_config_path(bundle_dir, "preprocess"),
            postprocess_config_path=self._detect_config_path(bundle_dir, "postprocess"),
            metadata_path=metadata_path,
        )

        compatible_runtime = tuple(
            RuntimeCompatibility.from_dict(item)
            for item in metadata.get("compatible_runtime", [])
            if isinstance(item, dict)
        )

        if not compatible_runtime:
            compatible_runtime = self._infer_runtime_from_model(model_path)

        labels = _normalize_labels(metadata.get("labels")) or self._load_labels_from_map(
            artifacts.label_map_path
        )

        entry_modality = _normalize_modality(metadata.get("modality")) or _normalize_modality(
            metadata.get("input_modality")
        )
        if not entry_modality:
            entry_modality = _normalize_modality(
                self._lookup_plugin_modality(plugin_id, task_type)
            )

        created_at = metadata.get("created_at") or self._safe_created_at(model_path)

        return ModelRegistryEntry(
            plugin_id=plugin_id,
            task_type=task_type,
            modality=entry_modality,
            version=metadata.get("version", version),
            metrics={
                str(k): float(v)
                for k, v in metadata.get("metrics", {}).items()
                if isinstance(v, (int, float))
            },
            compatible_runtime=compatible_runtime,
            artifacts=artifacts,
            model_role=metadata.get("model_role", model_role),
            labels=labels,
            export_path=bundle_dir,
            created_at=created_at,
            source=source,
            metadata=metadata,
        )

    def _build_discovered_entry(
        self,
        *,
        plugin_id: str,
        task_type: str,
        version: str,
        model_path: Path,
        export_dir: Path,
        source: str,
        model_role: str | None = None,
    ) -> ModelRegistryEntry:
        artifacts = ModelArtifacts(
            export_dir=export_dir,
            model_path=model_path,
            label_map_path=self._detect_label_map_path(export_dir),
            preprocess_config_path=self._detect_config_path(export_dir, "preprocess"),
            postprocess_config_path=self._detect_config_path(export_dir, "postprocess"),
            metadata_path=self._detect_metadata_path(export_dir),
        )

        return ModelRegistryEntry(
            plugin_id=plugin_id,
            task_type=task_type,
            modality=_normalize_modality(self._lookup_plugin_modality(plugin_id, task_type)),
            version=version,
            compatible_runtime=self._infer_runtime_from_model(model_path),
            artifacts=artifacts,
            model_role=model_role,
            labels=self._load_labels_from_map(artifacts.label_map_path),
            export_path=export_dir,
            created_at=self._safe_created_at(model_path),
            source=source,
            metadata={},
        )

    def _validate_entry(
        self,
        entry: ModelRegistryEntry,
        request: ModelResolutionRequest,
    ) -> tuple[str, ...]:
        artifacts = entry.artifacts
        if artifacts is None or not artifacts.model_path.exists():
            raise ModelNotFoundError(
                f"resolved model file is missing: {artifacts.model_path if artifacts else 'unknown'}",
                details={"entry_key": entry.key},
            )

        warnings: list[str] = []

        if request.runtime:
            self._validate_runtime(entry, request)

        if request.expected_modality:
            expected = _normalize_modality(request.expected_modality)
            actual = _normalize_modality(entry.modality)
            if expected != actual:
                raise InputModalityMismatchError(
                    (
                        f"modality mismatch for {entry.key}: "
                        f"expected '{expected}', got '{actual}'"
                    ),
                    details={
                        "entry_key": entry.key,
                        "expected_modality": expected,
                        "actual_modality": actual,
                    },
                )

        if request.expected_labels:
            actual_labels = entry.labels or self._load_labels_from_map(artifacts.label_map_path)
            if tuple(request.expected_labels) != tuple(actual_labels):
                raise LabelContractMismatchError(
                    f"label mismatch for {entry.key}",
                    details={
                        "entry_key": entry.key,
                        "expected_labels": list(request.expected_labels),
                        "actual_labels": list(actual_labels),
                    },
                )

        if request.require_preprocess and not (
            artifacts.preprocess_config_path and artifacts.preprocess_config_path.exists()
        ):
            raise PreprocessConfigMissingError(
                f"preprocess config missing for {entry.key}",
                details={"entry_key": entry.key},
            )

        if request.require_postprocess and not (
            artifacts.postprocess_config_path and artifacts.postprocess_config_path.exists()
        ):
            raise PostprocessConfigMissingError(
                f"postprocess config missing for {entry.key}",
                details={"entry_key": entry.key},
            )

        if entry.source.startswith("legacy"):
            warnings.append("resolved through legacy path fallback")

        if not artifacts.label_map_path:
            warnings.append("label map not found")
        if not artifacts.postprocess_config_path:
            warnings.append("postprocess config not found")

        return tuple(warnings)

    def _validate_runtime(
        self,
        entry: ModelRegistryEntry,
        request: ModelResolutionRequest,
    ) -> None:
        runtime_name = str(request.runtime or "").strip()
        if not runtime_name:
            return

        compatible_items = entry.compatible_runtime or self._infer_runtime_from_model(
            entry.artifacts.model_path if entry.artifacts else Path("model.onnx")
        )
        matching = [item for item in compatible_items if item.runtime == runtime_name]

        if not matching:
            raise RuntimeCompatibilityError(
                f"runtime '{runtime_name}' is not compatible with {entry.key}",
                details={
                    "entry_key": entry.key,
                    "requested_runtime": runtime_name,
                    "compatible_runtime": [item.to_dict() for item in compatible_items],
                },
            )

        if request.provider:
            provider = str(request.provider)
            if not any((not item.providers) or (provider in item.providers) for item in matching):
                raise RuntimeCompatibilityError(
                    f"provider '{provider}' is not compatible with {entry.key}",
                    details={
                        "entry_key": entry.key,
                        "requested_provider": provider,
                        "compatible_runtime": [item.to_dict() for item in matching],
                    },
                )

        if request.runtime_version:
            runtime_version = str(request.runtime_version)
            for item in matching:
                if item.min_version and _compare_versions(runtime_version, item.min_version) < 0:
                    raise ModelVersionIncompatibleError(
                        (
                            f"runtime version '{runtime_version}' is below minimum "
                            f"'{item.min_version}' for {entry.key}"
                        ),
                        details={
                            "entry_key": entry.key,
                            "requested_runtime": runtime_name,
                            "requested_runtime_version": runtime_version,
                            "min_version": item.min_version,
                        },
                    )
                if item.max_version and _compare_versions(runtime_version, item.max_version) > 0:
                    raise ModelVersionIncompatibleError(
                        (
                            f"runtime version '{runtime_version}' is above maximum "
                            f"'{item.max_version}' for {entry.key}"
                        ),
                        details={
                            "entry_key": entry.key,
                            "requested_runtime": runtime_name,
                            "requested_runtime_version": runtime_version,
                            "max_version": item.max_version,
                        },
                    )

    def _lookup_plugin_modality(self, plugin_id: str, task_type: str) -> str | None:
        mapping = self._load_plugin_mapping()
        plugin_info = mapping.get("plugins", {}).get(plugin_id, {})
        if not plugin_info:
            return None

        # Prefer explicit per-task modality if present.
        task_modalities = plugin_info.get("task_modalities", {})
        if isinstance(task_modalities, dict) and task_modalities.get(task_type):
            return str(task_modalities[task_type])
        return plugin_info.get("input_modality")

    def _infer_runtime_from_model(
        self,
        model_path: Path,
    ) -> tuple[RuntimeCompatibility, ...]:
        extension = model_path.suffix.lower()
        if extension == ".onnx":
            return (
                RuntimeCompatibility(
                    runtime="onnxruntime",
                    file_extensions=(".onnx",),
                ),
            )
        if extension == ".engine":
            return (
                RuntimeCompatibility(
                    runtime="tensorrt",
                    file_extensions=(".engine",),
                ),
            )
        if extension in (".pt", ".pth"):
            return (
                RuntimeCompatibility(
                    runtime="pytorch",
                    file_extensions=(extension,),
                ),
            )
        return ()

    def _detect_model_path(self, bundle_dir: Path) -> Path | None:
        preferred_names = (
            "model.onnx",
            "model.engine",
            "model.pt",
            "model.pth",
            "best.pt",
            "best.pth",
        )
        for name in preferred_names:
            path = bundle_dir / name
            if path.exists():
                return path

        for pattern in ("*.onnx", "*.engine", "*.pt", "*.pth"):
            matches = sorted(bundle_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _detect_label_map_path(self, bundle_dir: Path) -> Path | None:
        for name in ("label_map.json", "class_mapping.json", "classes.json"):
            path = bundle_dir / name
            if path.exists():
                return path
        return None

    def _detect_config_path(self, bundle_dir: Path, stem: str) -> Path | None:
        for suffix in (".json", ".yaml", ".yml"):
            path = bundle_dir / f"{stem}{suffix}"
            if path.exists():
                return path
        return None

    def _detect_metadata_path(self, bundle_dir: Path) -> Path | None:
        for name in ("bundle.json", "export_meta.json", "model_meta.json"):
            path = bundle_dir / name
            if path.exists():
                return path
        return None

    def _load_labels_from_map(self, label_map_path: Path | None) -> tuple[str, ...]:
        if not label_map_path or not label_map_path.exists():
            return ()
        raw = _read_json(label_map_path)
        if isinstance(raw, list):
            return _normalize_labels(raw)
        if isinstance(raw, dict):
            if all(str(key).isdigit() for key in raw.keys()):
                ordered = [label for _, label in sorted(raw.items(), key=lambda item: int(item[0]))]
                return _normalize_labels(ordered)
            if "labels" in raw and isinstance(raw["labels"], list):
                return _normalize_labels(raw["labels"])
            ordered = [label for _, label in sorted(raw.items(), key=lambda item: str(item[1]))]
            return _normalize_labels(ordered)
        return ()

    def _safe_created_at(self, path: Path) -> str | None:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
                microsecond=0
            ).isoformat()
        except OSError:
            return None


class PluginModelResolver:
    """
    Thin plugin-side adapter around VisualModelRegistry.

    Plugins should keep detector/session construction logic local.
    This resolver only discovers, validates, and returns normalized paths.
    """

    def __init__(
        self,
        plugin_id: str,
        *,
        training_root: Path | str | None = None,
        plugin_dir: Path | str | None = None,
        registry: VisualModelRegistry | None = None,
    ) -> None:
        self.plugin_id = plugin_id
        self.plugin_dir = Path(plugin_dir) if plugin_dir else None
        self.registry = registry or VisualModelRegistry(training_root)

    def resolve_bundle(
        self,
        task_type: str,
        *,
        version: str | None = "latest",
        runtime: str | None = None,
        runtime_version: str | None = None,
        provider: str | None = None,
        expected_modality: str | None = None,
        expected_labels: Iterable[str] | None = None,
        require_preprocess: bool = False,
        require_postprocess: bool = False,
        allow_legacy: bool = True,
        model_role: str | None = None,
    ) -> ResolvedModelBundle:
        request = ModelResolutionRequest.create(
            plugin_id=self.plugin_id,
            task_type=task_type,
            version=version,
            runtime=runtime,
            runtime_version=runtime_version,
            provider=provider,
            expected_modality=expected_modality,
            expected_labels=expected_labels,
            require_preprocess=require_preprocess,
            require_postprocess=require_postprocess,
            allow_legacy=allow_legacy,
            plugin_dir=self.plugin_dir,
            model_role=model_role,
        )
        return self.registry.resolve(request)
