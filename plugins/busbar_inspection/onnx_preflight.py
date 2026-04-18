"""ONNX preflight gates for busbar real_dl validation.

The goal of this module is to make real DL enablement:
- verifiable
- fail-fast
- observable
- non-silent

It does not claim anything about model quality or delivery readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@dataclass
class OnnxPreflightResult:
    """Serializable real_dl preflight snapshot."""

    checked: bool = False
    passed: bool = False
    failure_reason: str = ""
    failure_details: list[str] = field(default_factory=list)

    model_path_configured: Optional[str] = None
    model_path_resolved: Optional[str] = None
    model_file_exists: bool = False

    manifest_path: Optional[str] = None
    manifest_exists: bool = False
    class_map_path: Optional[str] = None
    class_map_exists: bool = False

    model_version: Optional[str] = None
    input_size: Optional[list[int]] = None
    output_format: Optional[str] = None
    class_names: list[str] = field(default_factory=list)
    class_map_compatible: bool = False

    onnxruntime_available: bool = False
    requested_providers: list[str] = field(default_factory=list)
    session_providers: list[str] = field(default_factory=list)
    session_ready: bool = False
    session_error: str = ""
    input_tensor_shape: Optional[list[Any]] = None
    output_tensor_shapes: list[list[Any]] = field(default_factory=list)
    output_probe_shape: Optional[list[int]] = None
    output_structure_compatible: bool = False

    def to_runtime_status(self) -> dict[str, Any]:
        """Expose fields through detector/plugin health surfaces."""
        return {
            "dl_preflight_checked": self.checked,
            "dl_preflight_passed": self.passed,
            "dl_failure_reason": self.failure_reason or None,
            "dl_failure_details": list(self.failure_details),
            "manifest_path": self.manifest_path,
            "manifest_exists": self.manifest_exists,
            "class_map_path": self.class_map_path,
            "class_map_exists": self.class_map_exists,
            "class_map_compatible": self.class_map_compatible,
            "model_version_validated": self.model_version,
            "input_size_validated": list(self.input_size) if self.input_size else None,
            "output_format_validated": self.output_format,
            "validated_class_names": list(self.class_names),
            "requested_providers": list(self.requested_providers),
            "session_providers": list(self.session_providers),
            "input_tensor_shape": list(self.input_tensor_shape) if self.input_tensor_shape else None,
            "output_tensor_shapes": [list(shape) for shape in self.output_tensor_shapes],
            "output_probe_shape": list(self.output_probe_shape) if self.output_probe_shape else None,
            "output_structure_compatible": self.output_structure_compatible,
            "session_error": self.session_error or None,
        }


def run_onnx_preflight(
    *,
    plugin_dir: Path,
    model_path_configured: Optional[str],
    runtime_providers: list[str],
    runtime_supported_defect_labels: tuple[str, ...],
    canonicalize_label,
    is_runtime_supported_defect_label,
    model_config: Optional[dict[str, Any]] = None,
) -> OnnxPreflightResult:
    """Validate whether a configured ONNX asset is safe to enter real_dl."""
    result = OnnxPreflightResult(
        checked=True,
        model_path_configured=model_path_configured,
        onnxruntime_available=ort is not None,
        requested_providers=list(runtime_providers or []),
    )
    model_cfg = model_config or {}

    model_path = _resolve_model_path(plugin_dir, model_path_configured)
    if model_path is not None:
        result.model_path_resolved = str(model_path)
        result.model_file_exists = model_path.exists()

    if not model_path_configured:
        result.failure_details.append("model_path is not configured")
    elif model_path is None or not result.model_file_exists:
        result.failure_details.append("configured model file does not exist")

    manifest_path = _resolve_manifest_path(model_path, model_cfg)
    if manifest_path is not None:
        result.manifest_path = str(manifest_path)
        result.manifest_exists = manifest_path.exists()
    if not result.manifest_exists:
        result.failure_details.append("manifest is missing")

    manifest: dict[str, Any] = {}
    if result.manifest_exists and manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            result.failure_details.append(f"manifest is invalid: {exc}")

    class_names, class_map_path = _resolve_class_names(model_path, manifest, model_cfg)
    if class_map_path is not None:
        result.class_map_path = str(class_map_path)
        result.class_map_exists = class_map_path.exists()
    elif class_names:
        result.class_map_path = "manifest:inline"
        result.class_map_exists = True
    else:
        result.class_map_exists = False

    if not result.class_map_exists:
        result.failure_details.append("class map / class names are missing")

    normalized_class_names = [canonicalize_label(name) for name in class_names if str(name or "").strip()]
    result.class_names = normalized_class_names
    if normalized_class_names:
        incompatible = [
            label for label in normalized_class_names
            if not is_runtime_supported_defect_label(label)
        ]
        result.class_map_compatible = len(incompatible) == 0
        if incompatible:
            result.failure_details.append(
                "label contract mismatch: " + ", ".join(sorted(dict.fromkeys(incompatible)))
            )
    else:
        result.class_map_compatible = False

    manifest_version = manifest.get("model_version")
    if isinstance(manifest_version, str) and manifest_version.strip():
        result.model_version = manifest_version.strip()
    else:
        result.failure_details.append("model_version is missing from manifest")

    parsed_input_size = _parse_input_size(manifest.get("input_size"))
    if parsed_input_size is not None:
        result.input_size = list(parsed_input_size)
    else:
        result.failure_details.append("input_size is missing or invalid")

    output_format = str(manifest.get("output_format") or "").strip()
    if output_format:
        result.output_format = output_format

    session = None
    if ort is None:
        result.failure_details.append("onnxruntime is not available")
    elif result.model_file_exists and model_path is not None:
        selected_providers = _select_runtime_providers(runtime_providers)
        try:
            session = ort.InferenceSession(str(model_path), providers=selected_providers)
            result.session_ready = session is not None
            result.session_providers = list(session.get_providers())
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if inputs:
                result.input_tensor_shape = list(inputs[0].shape)
            result.output_tensor_shapes = [list(output.shape) for output in outputs]
        except Exception as exc:
            result.session_error = str(exc)
            result.failure_details.append(f"onnx session failed: {exc}")

    if session is not None and result.input_tensor_shape is not None:
        if result.input_size is not None and not _input_size_matches(result.input_tensor_shape, result.input_size):
            result.failure_details.append("input tensor shape is incompatible with manifest input_size")
        try:
            output = _probe_session_output(session, result.input_tensor_shape)
            result.output_probe_shape = list(output.shape)
            result.output_structure_compatible = _is_output_structure_compatible(
                output_shape=output.shape,
                num_classes=len(normalized_class_names),
            )
            if not result.output_structure_compatible:
                result.failure_details.append(
                    f"output shape is incompatible with detector postprocess: {tuple(output.shape)}"
                )
        except Exception as exc:
            result.failure_details.append(f"output probe failed: {exc}")

    result.failure_reason = _select_primary_failure_reason(
        model_path_configured=model_path_configured,
        result=result,
    )
    result.passed = result.failure_reason == ""
    if result.passed:
        result.failure_details = []
    return result


def _resolve_model_path(plugin_dir: Path, model_path_configured: Optional[str]) -> Optional[Path]:
    if not model_path_configured:
        return None
    candidate = Path(model_path_configured)
    if not candidate.is_absolute():
        candidate = plugin_dir / candidate
    return candidate.resolve(strict=False)


def _resolve_manifest_path(model_path: Optional[Path], model_cfg: dict[str, Any]) -> Optional[Path]:
    explicit = model_cfg.get("manifest_path")
    if explicit:
        return Path(explicit).resolve(strict=False)
    if model_path is None:
        return None
    candidates = [
        model_path.with_name(f"{model_path.stem}.manifest.json"),
        model_path.with_name(f"{model_path.stem}_manifest.json"),
        model_path.parent / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_class_names(
    model_path: Optional[Path],
    manifest: dict[str, Any],
    model_cfg: dict[str, Any],
) -> tuple[list[str], Optional[Path]]:
    inline = manifest.get("class_names") or manifest.get("classes")
    if isinstance(inline, list):
        return [str(name).strip() for name in inline if str(name).strip()], None

    explicit = model_cfg.get("class_map_path")
    if explicit:
        path = Path(explicit).resolve(strict=False)
        return _load_class_names_file(path), path

    manifest_mapping = manifest.get("class_mapping") or manifest.get("class_map")
    if isinstance(manifest_mapping, str) and manifest_mapping.strip():
        path = Path(manifest_mapping)
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve(strict=False)
        return _load_class_names_file(path), path

    if model_path is None:
        return [], None

    candidates = [
        model_path.with_name(f"{model_path.stem}.class_map.json"),
        model_path.with_name(f"{model_path.stem}_class_map.json"),
        model_path.with_name(f"{model_path.stem}.classes.json"),
        model_path.parent / "class_map.json",
        model_path.parent / "class_names.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_class_names_file(candidate), candidate
    return [], candidates[0]


def _load_class_names_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [str(name).strip() for name in data if str(name).strip()]
    if isinstance(data, dict):
        if "class_names" in data and isinstance(data["class_names"], list):
            return [str(name).strip() for name in data["class_names"] if str(name).strip()]
        items: list[tuple[int, str]] = []
        for key, value in data.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            label = str(value).strip()
            if label:
                items.append((index, label))
        return [label for _, label in sorted(items, key=lambda item: item[0])]
    return []


def _parse_input_size(value: Any) -> Optional[tuple[int, int]]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)):
        ints = [int(v) for v in value]
        if len(ints) == 2:
            return (ints[0], ints[1])
        if len(ints) == 4:
            return (ints[-2], ints[-1])
    return None


def _select_runtime_providers(requested_providers: list[str]) -> list[str]:
    if ort is None:
        return []
    available = list(ort.get_available_providers())
    selected = [provider for provider in requested_providers if provider in available]
    if selected:
        return selected
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available


def _input_size_matches(input_tensor_shape: list[Any], manifest_input_size: list[int]) -> bool:
    if len(input_tensor_shape) != 4:
        return False
    tensor_h = _safe_int(input_tensor_shape[-2])
    tensor_w = _safe_int(input_tensor_shape[-1])
    return tensor_h == manifest_input_size[0] and tensor_w == manifest_input_size[1]


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _probe_session_output(session, input_tensor_shape: list[Any]) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    resolved_shape = [_resolve_dim(dim) for dim in input_tensor_shape]
    dummy = np.zeros(resolved_shape, dtype=np.float32)
    outputs = session.run(None, {input_meta.name: dummy})
    if not outputs:
        raise RuntimeError("session returned no outputs")
    output = outputs[0]
    if not isinstance(output, np.ndarray):
        output = np.asarray(output)
    return output


def _resolve_dim(dim: Any) -> int:
    resolved = _safe_int(dim)
    if resolved is None or resolved <= 0:
        return 1
    return resolved


def _is_output_structure_compatible(output_shape: tuple[int, ...], num_classes: int) -> bool:
    if len(output_shape) not in {2, 3}:
        return False
    dims = output_shape[1:] if len(output_shape) == 3 else output_shape
    if len(dims) != 2:
        return False
    first, second = int(dims[0]), int(dims[1])
    if first <= 0 or second <= 0:
        return False
    row_width = first if first < second else second
    return row_width >= (4 + max(num_classes, 1))


def _select_primary_failure_reason(
    *,
    model_path_configured: Optional[str],
    result: OnnxPreflightResult,
) -> str:
    if not model_path_configured:
        return "model_missing"
    if not result.model_file_exists:
        return "model_missing"
    if not result.session_ready:
        return "onnx_session_failed"
    if not result.manifest_exists:
        return "manifest_missing"
    if not result.class_map_exists:
        return "class_map_missing"
    if not result.class_map_compatible:
        return "label_contract_mismatch"
    if result.input_size is None:
        return "input_size_missing"
    if not result.output_structure_compatible:
        return "output_shape_incompatible"
    return ""
