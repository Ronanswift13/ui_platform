"""
时序训练数据契约

统一描述:
- 原始波形
- 频谱图 / 梅尔谱
- 多变量数值时序
- 带时间戳的事件序列
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .task_profiles import (
    DEVICE_MONITORING_OUTPUT_HEADS,
    TEMPORAL_MODALITIES,
    TEMPORAL_SOURCE_KINDS,
    get_temporal_task_profile,
)


@dataclass
class TemporalSourceSpec:
    """单个输入源定义"""

    name: str
    kind: str
    path: str
    format: str
    channels: list[str] = field(default_factory=list)
    sensor_columns: list[str] = field(default_factory=list)
    sample_rate_hz: int | None = None
    timestamp_field: str = ""
    value_field: str = ""
    description: str = ""


@dataclass
class TemporalFeatureViewSpec:
    """预计算特征视图定义"""

    name: str
    source: str
    transform: str
    path: str
    shape: list[int] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalTargetSpec:
    """监督目标定义"""

    name: str
    target_type: str
    path: str = ""
    horizon: str = ""
    classes: list[str] = field(default_factory=list)
    range_hint: str = ""
    description: str = ""


@dataclass
class TemporalDatasetContract:
    """统一时序数据契约"""

    schema_version: str = "1.0.0"
    timestamp_field: str = ""
    entity_id_field: str = ""
    sequence_id_field: str = ""
    sources: list[TemporalSourceSpec] = field(default_factory=list)
    feature_views: list[TemporalFeatureViewSpec] = field(default_factory=list)
    targets: list[TemporalTargetSpec] = field(default_factory=list)
    metadata_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_temporal_contract(payload: dict[str, Any] | None) -> TemporalDatasetContract:
    """将 dict 归一化为 TemporalDatasetContract"""

    payload = payload or {}
    sources = [
        TemporalSourceSpec(**item)
        for item in payload.get("sources", [])
        if isinstance(item, dict)
    ]
    feature_views = [
        TemporalFeatureViewSpec(**item)
        for item in payload.get("feature_views", [])
        if isinstance(item, dict)
    ]
    targets = [
        TemporalTargetSpec(**item)
        for item in payload.get("targets", [])
        if isinstance(item, dict)
    ]
    return TemporalDatasetContract(
        schema_version=str(payload.get("schema_version", "1.0.0")),
        timestamp_field=str(payload.get("timestamp_field", "")),
        entity_id_field=str(payload.get("entity_id_field", "")),
        sequence_id_field=str(payload.get("sequence_id_field", "")),
        sources=sources,
        feature_views=feature_views,
        targets=targets,
        metadata_fields=[
            str(item) for item in payload.get("metadata_fields", []) if str(item).strip()
        ],
    )


def validate_temporal_contract(
    *,
    plugin_id: str,
    task_type: str,
    input_modality: str,
    temporal_schema: dict[str, Any] | None,
    target_schema: dict[str, Any] | None,
    feature_views: list[str] | None,
    sensor_columns: list[str] | None,
    event_types: list[str] | None,
    timestamp_field: str,
) -> list[str]:
    """校验 manifest 中的时序契约定义"""

    errors: list[str] = []
    profile = get_temporal_task_profile(task_type)
    if profile is None:
        return [f"未知时序 task_type: '{task_type}'"]

    if input_modality not in TEMPORAL_MODALITIES:
        errors.append(
            f"时序任务 input_modality '{input_modality}' 不合法，"
            f"合法值: {sorted(TEMPORAL_MODALITIES)}"
        )

    contract = normalize_temporal_contract(temporal_schema)
    if not contract.sources:
        errors.append("temporal_schema.sources 不能为空")
        return errors

    seen_source_kinds = {source.kind for source in contract.sources}
    for source in contract.sources:
        if source.kind not in TEMPORAL_SOURCE_KINDS:
            errors.append(
                f"temporal_schema.sources[{source.name}].kind='{source.kind}' 不合法"
            )
        if not source.path:
            errors.append(f"temporal_schema.sources[{source.name}] 缺少 path")
        if source.kind == "waveform" and not source.sample_rate_hz:
            errors.append(f"waveform source '{source.name}' 必须提供 sample_rate_hz")
        if source.kind == "timeseries" and not (source.sensor_columns or sensor_columns):
            errors.append(f"timeseries source '{source.name}' 必须提供 sensor_columns")
        if source.kind == "event_sequence":
            effective_timestamp = source.timestamp_field or timestamp_field
            if not effective_timestamp:
                errors.append(f"event_sequence source '{source.name}' 必须提供 timestamp_field")

    for required_kind in profile.required_source_kinds:
        if required_kind not in seen_source_kinds:
            errors.append(
                f"{task_type} 必须包含 source kind '{required_kind}'，"
                f"当前仅有: {sorted(seen_source_kinds)}"
            )

    feature_view_names = {name for name in (feature_views or []) if name}
    feature_view_names.update(view.name for view in contract.feature_views if view.name)

    if task_type == "acoustic_time_frequency_anomaly":
        if input_modality == "audio_waveform+spectrogram":
            has_waveform = "waveform" in seen_source_kinds
            has_spectrogram = (
                "spectrogram" in seen_source_kinds
                or any(
                    name in feature_view_names
                    for name in ("spectrogram", "mel_spectrogram", "log_mel")
                )
            )
            if not has_waveform or not has_spectrogram:
                errors.append(
                    "acoustic_monitoring 双路径方案要求同时提供原始波形与频谱图/梅尔谱视图"
                )

    if task_type in {"multivariate_sensor_anomaly", "equipment_health_prediction"}:
        if not sensor_columns:
            errors.append(f"{task_type} 必须提供 sensor_columns")
    if task_type == "health_index_calibration" and not sensor_columns:
        errors.append("health_index_calibration 必须提供 sensor_columns")

    if task_type == "action_event_sequence_recognition" and not event_types:
        errors.append("action_event_sequence_recognition 必须提供 event_types")

    target_schema = target_schema or {}
    outputs = target_schema.get("outputs", [])
    output_names = {
        str(item.get("name"))
        for item in outputs
        if isinstance(item, dict) and item.get("name")
    }

    if task_type in {"equipment_health_prediction", "health_index_calibration"} and not outputs:
        errors.append(f"{task_type} 必须提供 target_schema.outputs")

    if plugin_id == "device_monitoring":
        if task_type == "equipment_health_prediction":
            required_outputs = {head.name for head in DEVICE_MONITORING_OUTPUT_HEADS}
            missing = sorted(required_outputs - output_names)
            if missing:
                errors.append(
                    "device_monitoring 必须定义三个输出层: "
                    "health_index / anomaly_score / predicted_failure; "
                    f"当前缺少: {missing}"
                )
        if task_type == "health_index_calibration" and "health_index" not in output_names:
            errors.append("device_monitoring 的 health_index_calibration 必须输出 health_index")

    return errors
