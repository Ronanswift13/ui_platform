"""Small helpers for non-visual sensor plugin contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


VIRTUAL_BBOX = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
MODEL_UPGRADE_PLACEHOLDERS = {
    "model_features_placeholder": [],
    "sequence_embedding_placeholder": None,
    "temporal_pattern_placeholder": None,
    "anomaly_score_trace_placeholder": [],
    "root_cause_feature_placeholder": None,
}


def clamp_confidence(value: Any, default: float = 1.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def timestamp_to_iso(value: Any = None) -> str:
    parsed = timestamp_diagnostics(value)
    return parsed["iso"]


def timestamp_diagnostics(value: Any = None) -> dict[str, Any]:
    if isinstance(value, (int, float)):
        try:
            return {
                "iso": datetime.fromtimestamp(float(value)).isoformat(),
                "valid": True,
                "source": "numeric_timestamp",
                "reason": "",
            }
        except (OverflowError, OSError, ValueError):
            pass
    if isinstance(value, str) and value:
        try:
            normalized = value.replace("Z", "+00:00")
            return {
                "iso": datetime.fromisoformat(normalized).isoformat(),
                "valid": True,
                "source": "iso_timestamp",
                "reason": "",
            }
        except ValueError:
            return {
                "iso": datetime.now().isoformat(),
                "valid": False,
                "source": "bad_timestamp_defaulted",
                "reason": f"Invalid timestamp: {value}",
            }
    return {
        "iso": datetime.now().isoformat(),
        "valid": False,
        "source": "missing_timestamp_defaulted",
        "reason": "Missing timestamp",
    }


def normalize_context(payload: Mapping[str, Any] | None, device_id: str) -> dict[str, str]:
    payload = payload or {}
    context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
    return {
        "task_id": str(context.get("task_id") or payload.get("task_id") or "virtual-sensor-task"),
        "site_id": str(context.get("site_id") or payload.get("site_id") or "virtual-site"),
        "device_id": str(context.get("device_id") or payload.get("device_id") or device_id or "virtual-device"),
    }


def build_common_metadata(
    *,
    modality: str,
    sensor_type: str,
    threshold_snapshot: Mapping[str, Any] | None,
    runtime_mode: str,
    algorithm_stage: str,
    model_status: str,
    fallback_level: str,
    trend_prediction_available: bool,
    upgrade_placeholders: Any,
    sampling_rate: Any = None,
    sample_interval: Any = None,
    window_size: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "modality": modality,
        "sensor_type": sensor_type,
        "window_size": window_size,
        "threshold_snapshot": dict(threshold_snapshot or {}),
        "runtime_mode": runtime_mode,
        "algorithm_stage": algorithm_stage,
        "model_status": model_status,
        "fallback_level": fallback_level,
        "trend_prediction_available": bool(trend_prediction_available),
        "upgrade_placeholders": upgrade_placeholders,
    }
    if sampling_rate is not None:
        metadata["sampling_rate"] = sampling_rate
    if sample_interval is not None:
        metadata["sample_interval"] = sample_interval
    if extra:
        metadata.update(extra)
    return metadata


def detect_input_type(payload: Mapping[str, Any] | None) -> str:
    payload = payload or {}
    for key in (
        "sampled_sequence",
        "sensor_window",
        "structured_timeseries",
        "state_change_events",
        "protocol_ingested_data",
        "audio_buffer",
        "readings",
        "gas_readings",
        "device_readings",
        "events",
        "signal_changes",
    ):
        if key in payload:
            return key
    return "direct_sample"


def normalize_placeholders(placeholders: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(MODEL_UPGRADE_PLACEHOLDERS)
    if placeholders:
        merged.update(dict(placeholders))
    return merged


def build_time_window(
    payload: Mapping[str, Any] | None,
    *,
    window_size: Any = None,
    sample_interval: Any = None,
    timestamp_value: Any = None,
) -> dict[str, Any]:
    payload = payload or {}
    ts = timestamp_diagnostics(timestamp_value if timestamp_value is not None else payload.get("timestamp"))
    duration = None
    try:
        if window_size is not None and sample_interval is not None:
            duration = float(window_size) * float(sample_interval)
    except (TypeError, ValueError):
        duration = None
    return {
        "start": ts["iso"],
        "end": ts["iso"],
        "duration_seconds": duration,
        "window_size": window_size,
        "sample_interval": sample_interval,
        "timestamp_valid": ts["valid"],
        "timestamp_source": ts["source"],
        "timestamp_reason": ts["reason"],
    }


def build_unified_temporal_output(
    *,
    plugin_name: str,
    task_type: str,
    payload: Mapping[str, Any] | None,
    status: str,
    label: str,
    severity: str | None = None,
    confidence: Any = 1.0,
    summary: Mapping[str, Any] | None = None,
    anomaly_events: list[dict[str, Any]] | None = None,
    abnormal_intervals: list[dict[str, Any]] | None = None,
    reason_codes: list[str] | None = None,
    recommended_actions: list[str] | None = None,
    trend_diagnosis: Mapping[str, Any] | None = None,
    evidence: list[dict[str, Any]] | Mapping[str, Any] | None = None,
    review_required: bool | None = None,
    model_info: Mapping[str, Any] | None = None,
    placeholders: Mapping[str, Any] | None = None,
    time_window: Mapping[str, Any] | None = None,
    data_quality: Mapping[str, Any] | None = None,
    input_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    events = list(anomaly_events or [])
    intervals = list(abnormal_intervals or [])
    reasons = list(reason_codes or [])
    actions = list(recommended_actions or [])
    evidence_items = evidence if isinstance(evidence, list) else ([dict(evidence)] if evidence else [])
    severity_value = severity or ("error" if status == "error" else "normal" if label == "normal" else status)
    quality = dict(data_quality or {})
    if time_window and not time_window.get("timestamp_valid", True):
        quality.setdefault("status", "degraded")
        quality.setdefault("issues", [])
        quality["issues"] = list(quality["issues"]) + [time_window.get("timestamp_reason", "timestamp defaulted")]
    else:
        quality.setdefault("status", "ok")
        quality.setdefault("issues", [])

    protocol = {
        "observed_input_type": detect_input_type(payload),
        "supported_inputs": [
            "sampled_sequence",
            "sensor_window",
            "structured_timeseries",
            "state_change_events",
            "protocol_ingested_data",
        ],
        "time_window": dict(time_window or {}),
        "missing_value_policy": "default_or_explainable_failure",
        "abnormal_value_policy": "flag_event_with_reason_code",
        "data_quality": quality,
    }
    if input_protocol:
        protocol.update(dict(input_protocol))

    return {
        "plugin_name": plugin_name,
        "task_type": task_type,
        "summary": dict(summary or {}),
        "anomaly_events": events,
        "abnormal_intervals": intervals,
        "severity": severity_value,
        "confidence": clamp_confidence(confidence),
        "reason_codes": reasons,
        "recommended_actions": actions,
        "trend_diagnosis": dict(trend_diagnosis or {
            "available": False,
            "direction": "unknown",
            "confidence": 0.0,
            "reason": "trend model unavailable or insufficient history",
        }),
        "evidence": evidence_items,
        "review_required": bool(review_required) if review_required is not None else severity_value in ("alarm", "critical", "error"),
        "model_info": dict(model_info or {}),
        "placeholders": normalize_placeholders(placeholders),
        "input_protocol": protocol,
    }


def build_virtual_result(
    *,
    payload: Mapping[str, Any] | None,
    plugin_id: str,
    plugin_version: str,
    code_hash: str,
    device_id: str,
    roi_id: str,
    label: str,
    value: Any,
    confidence: Any,
    metadata: Mapping[str, Any],
    component_id: str = "sensor",
    failure_reason: str | None = None,
) -> dict[str, Any]:
    context = normalize_context(payload, device_id)
    result_metadata = dict(metadata)
    result_metadata["virtual_roi"] = True
    return {
        "task_id": context["task_id"],
        "site_id": context["site_id"],
        "device_id": context["device_id"],
        "component_id": component_id,
        "roi_id": roi_id or context["device_id"],
        "bbox": dict(VIRTUAL_BBOX),
        "label": label,
        "value": value,
        "confidence": clamp_confidence(confidence),
        "evidence_path": "",
        "model_version": str(result_metadata.get("model_status", "")),
        "code_version": code_hash,
        "timestamp": timestamp_to_iso((payload or {}).get("timestamp")),
        "metadata": result_metadata,
        "failure_reason": failure_reason,
    }
