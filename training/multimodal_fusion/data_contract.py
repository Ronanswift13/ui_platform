"""
多模态融合数据契约

支持单样本中多模态对齐数据上传，并兼容模态缺失场景。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .task_profiles import (
    MULTIMODAL_FUSION_STRATEGIES,
    MULTIMODAL_MISSING_MODALITY_POLICIES,
    MULTIMODAL_RUNTIME_MODES,
    MULTIMODAL_SUPPORTED_MODALITIES,
    get_multimodal_task_profile,
)


@dataclass
class ModalityPathSpec:
    modality: str
    path: str
    format: str
    is_required: bool = False
    feature_dim: int | None = None
    plugin_dependency: str | None = None
    notes: str = ""


@dataclass
class DiagnosticRulePackRef:
    path: str
    version: str = ""
    source: str = "plugin_manifest"


@dataclass
class MultimodalAlignedSample:
    sample_id: str
    device_id: str
    timestamp: str
    available_modalities: list[str]
    modality_paths: dict[str, str]
    label: str = ""
    diagnosis_target: dict[str, Any] = field(default_factory=dict)
    decision_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalDatasetContract:
    schema_version: str = "1.0.0"
    supported_modalities: list[str] = field(default_factory=list)
    required_modalities: list[str] = field(default_factory=list)
    fusion_strategy: str = "hybrid"
    missing_modality_policy: str = "mask_and_gate"
    alignment_tolerance_ms: int = 5000
    runtime_mode: str = "rule_model_hybrid"
    modality_specs: list[ModalityPathSpec] = field(default_factory=list)
    diagnosis_target_schema: dict[str, Any] = field(default_factory=dict)
    diagnostic_rule_pack: DiagnosticRulePackRef | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_multimodal_contract(
    payload: dict[str, Any] | None,
) -> MultimodalDatasetContract:
    payload = payload or {}
    modality_specs = [
        ModalityPathSpec(**item)
        for item in payload.get("modality_specs", [])
        if isinstance(item, dict)
    ]
    diagnostic_rule_pack = None
    rule_pack_payload = payload.get("diagnostic_rule_pack")
    if isinstance(rule_pack_payload, dict):
        diagnostic_rule_pack = DiagnosticRulePackRef(**rule_pack_payload)

    return MultimodalDatasetContract(
        schema_version=str(payload.get("schema_version", "1.0.0")),
        supported_modalities=[
            str(item)
            for item in payload.get("supported_modalities", [])
            if str(item).strip()
        ],
        required_modalities=[
            str(item)
            for item in payload.get("required_modalities", [])
            if str(item).strip()
        ],
        fusion_strategy=str(payload.get("fusion_strategy", "hybrid")),
        missing_modality_policy=str(payload.get("missing_modality_policy", "mask_and_gate")),
        alignment_tolerance_ms=int(payload.get("alignment_tolerance_ms", 5000)),
        runtime_mode=str(payload.get("runtime_mode", "rule_model_hybrid")),
        modality_specs=modality_specs,
        diagnosis_target_schema=payload.get("diagnosis_target_schema", {}) or {},
        diagnostic_rule_pack=diagnostic_rule_pack,
    )


def validate_multimodal_contract(
    *,
    task_type: str,
    input_modality: str,
    fusion_strategy: str,
    supported_modalities: list[str] | None,
    required_modalities: list[str] | None,
    missing_modality_policy: str,
    multimodal_schema: dict[str, Any] | None,
    diagnosis_target_schema: dict[str, Any] | None,
    diagnostic_rule_pack: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    profile = get_multimodal_task_profile(task_type)
    if profile is None:
        return [f"未知多模态 task_type: '{task_type}'"]

    if input_modality != "multimodal":
        errors.append("多模态任务 input_modality 必须为 'multimodal'")

    if supported_modalities:
        unknown = sorted(set(supported_modalities) - MULTIMODAL_SUPPORTED_MODALITIES)
        if unknown:
            errors.append(f"supported_modalities 包含未知模态: {unknown}")
    if required_modalities:
        unknown_required = sorted(set(required_modalities) - MULTIMODAL_SUPPORTED_MODALITIES)
        if unknown_required:
            errors.append(f"required_modalities 包含未知模态: {unknown_required}")
    if supported_modalities and required_modalities:
        missing_supported = sorted(set(required_modalities) - set(supported_modalities))
        if missing_supported:
            errors.append(f"required_modalities 必须是 supported_modalities 的子集: {missing_supported}")

    if fusion_strategy and fusion_strategy not in MULTIMODAL_FUSION_STRATEGIES:
        errors.append(
            f"fusion_strategy '{fusion_strategy}' 不合法，"
            f"合法值: {sorted(MULTIMODAL_FUSION_STRATEGIES)}"
        )

    if missing_modality_policy not in MULTIMODAL_MISSING_MODALITY_POLICIES:
        errors.append(
            f"missing_modality_policy '{missing_modality_policy}' 不合法，"
            f"合法值: {sorted(MULTIMODAL_MISSING_MODALITY_POLICIES)}"
        )

    contract = normalize_multimodal_contract(multimodal_schema)
    effective_supported = set(supported_modalities or contract.supported_modalities)
    effective_required = set(required_modalities or contract.required_modalities)

    if not effective_supported:
        errors.append("多模态任务必须提供 supported_modalities")
    if not contract.modality_specs:
        errors.append("multimodal_schema.modality_specs 不能为空")
    else:
        declared_modalities = set()
        for spec in contract.modality_specs:
            declared_modalities.add(spec.modality)
            if spec.modality not in MULTIMODAL_SUPPORTED_MODALITIES:
                errors.append(f"modality_specs 中包含未知模态: {spec.modality}")
            if not spec.path:
                errors.append(f"modality_specs[{spec.modality}] 缺少 path")
            if not spec.format:
                errors.append(f"modality_specs[{spec.modality}] 缺少 format")
        missing_specs = sorted(effective_supported - declared_modalities)
        if missing_specs:
            errors.append(f"supported_modalities 未在 modality_specs 中定义: {missing_specs}")

    if effective_required and missing_modality_policy == "required_modalities_only":
        if not effective_required:
            errors.append("required_modalities_only 策略必须提供 required_modalities")

    if contract.runtime_mode not in MULTIMODAL_RUNTIME_MODES:
        errors.append(
            f"runtime_mode '{contract.runtime_mode}' 不合法，"
            f"合法值: {sorted(MULTIMODAL_RUNTIME_MODES)}"
        )

    diagnosis_target_schema = diagnosis_target_schema or {}
    outputs = diagnosis_target_schema.get("outputs", [])
    if not outputs:
        errors.append("diagnosis_target_schema.outputs 不能为空")
    else:
        output_names = {str(item.get("name")) for item in outputs if isinstance(item, dict)}
        required_output_names = {
            "overall_status",
            "confidence",
            "detections",
            "modality_contributions",
            "diagnostic_report",
        }
        missing_outputs = sorted(required_output_names - output_names)
        if missing_outputs:
            errors.append(
                "诊断目标必须至少覆盖 overall_status / confidence / detections / "
                f"modality_contributions / diagnostic_report；当前缺少: {missing_outputs}"
            )

    if task_type == "multimodal_decision_fusion":
        pack_payload = diagnostic_rule_pack or (
            {"path": contract.diagnostic_rule_pack.path}
            if contract.diagnostic_rule_pack is not None
            else {}
        )
        if not pack_payload or not pack_payload.get("path"):
            errors.append("multimodal_decision_fusion 必须提供 diagnostic_rule_pack.path")

    return errors


def validate_multimodal_sample_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = {
        "sample_id",
        "device_id",
        "timestamp",
        "available_modalities",
        "modality_paths",
        "label",
        "diagnosis_target",
    }
    missing = sorted(field for field in required_fields if field not in record)
    if missing:
        errors.append(f"缺少字段: {missing}")
        return errors

    available_modalities = record.get("available_modalities", [])
    if not isinstance(available_modalities, list) or not available_modalities:
        errors.append("available_modalities 必须是非空数组")
    else:
        unknown = sorted(set(available_modalities) - MULTIMODAL_SUPPORTED_MODALITIES)
        if unknown:
            errors.append(f"available_modalities 包含未知模态: {unknown}")

    modality_paths = record.get("modality_paths", {})
    if not isinstance(modality_paths, dict):
        errors.append("modality_paths 必须是对象")
    else:
        for modality in available_modalities:
            path = modality_paths.get(modality)
            if not path:
                errors.append(f"模态 {modality} 缺少 modality_paths[{modality}]")

    if not isinstance(record.get("diagnosis_target"), (dict, str)):
        errors.append("diagnosis_target 必须是对象或字符串")

    return errors
