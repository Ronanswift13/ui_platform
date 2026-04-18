"""
多模态融合训练任务画像

聚焦:
- feature-level fusion
- decision-level fusion
- 规则 + 模型混合诊断
"""

from __future__ import annotations

from dataclasses import dataclass, field


MULTIMODAL_DATASET_FAMILY = "multimodal_fusion"

MULTIMODAL_TASK_TYPES = {
    "multimodal_feature_fusion",
    "multimodal_decision_fusion",
}

MULTIMODAL_SUPPORTED_MODALITIES = {
    "visual",
    "thermal",
    "acoustic",
    "ultrasonic",
    "gas",
    "hyperspectral",
    "vibration",
}

MULTIMODAL_FUSION_STRATEGIES = {
    "feature_level",
    "decision_level",
    "hybrid",
}

MULTIMODAL_TRAINING_PARADIGMS = {
    "attention_pooling",
    "transformer_encoder",
    "cross_modal_projection",
    "stacking_ensemble",
    "bayesian_calibration",
    "gradient_boosted_meta_learner",
}

MULTIMODAL_MISSING_MODALITY_POLICIES = {
    "allow_sparse",
    "required_modalities_only",
    "mask_and_gate",
    "teacher_distillation_fallback",
}

MULTIMODAL_RUNTIME_MODES = {
    "rules_only",
    "model_only",
    "rule_model_hybrid",
}


@dataclass(frozen=True)
class FusionOutputHead:
    name: str
    head_type: str
    description: str


@dataclass(frozen=True)
class MultimodalTaskProfile:
    task_type: str
    fusion_strategy: str
    description: str
    storage_family: str = MULTIMODAL_DATASET_FAMILY
    supported_modalities: tuple[str, ...] = ()
    required_manifest_fields: tuple[str, ...] = ()
    supported_training_paradigms: tuple[str, ...] = ()
    default_model_family: str = ""
    default_runtime_mode: str = "rule_model_hybrid"
    evaluation_metrics: tuple[str, ...] = ()
    output_heads: tuple[FusionOutputHead, ...] = ()


COMMON_OUTPUT_HEADS = (
    FusionOutputHead("overall_status", "classification", "融合后的总体状态"),
    FusionOutputHead("confidence", "confidence", "总体置信度"),
    FusionOutputHead("detections", "detection_list", "关联后的融合检测结果"),
    FusionOutputHead("modality_contributions", "attribution", "各模态贡献度"),
    FusionOutputHead("diagnostic_report", "structured_report", "规则 + 模型联合诊断报告"),
)


MULTIMODAL_TASK_PROFILES: dict[str, MultimodalTaskProfile] = {
    "multimodal_feature_fusion": MultimodalTaskProfile(
        task_type="multimodal_feature_fusion",
        fusion_strategy="feature_level",
        description="特征级融合，将多模态编码后的向量映射到统一嵌入空间后再分类/诊断。",
        supported_modalities=tuple(sorted(MULTIMODAL_SUPPORTED_MODALITIES)),
        required_manifest_fields=(
            "supported_modalities",
            "multimodal_schema",
            "missing_modality_policy",
            "fusion_strategy",
        ),
        supported_training_paradigms=(
            "attention_pooling",
            "transformer_encoder",
            "cross_modal_projection",
        ),
        default_model_family="multimodal_attention_fusion",
        default_runtime_mode="rule_model_hybrid",
        evaluation_metrics=(
            "overall_accuracy",
            "macro_f1",
            "ece",
            "modality_dropout_robustness",
            "diagnosis_hit_rate",
        ),
        output_heads=COMMON_OUTPUT_HEADS,
    ),
    "multimodal_decision_fusion": MultimodalTaskProfile(
        task_type="multimodal_decision_fusion",
        fusion_strategy="decision_level",
        description="决策级融合，对各模态独立检测结果进行加权投票、贝叶斯推断和规则校准。",
        supported_modalities=tuple(sorted(MULTIMODAL_SUPPORTED_MODALITIES)),
        required_manifest_fields=(
            "supported_modalities",
            "multimodal_schema",
            "diagnostic_rule_pack",
            "fusion_strategy",
        ),
        supported_training_paradigms=(
            "stacking_ensemble",
            "bayesian_calibration",
            "gradient_boosted_meta_learner",
        ),
        default_model_family="multimodal_decision_ensemble",
        default_runtime_mode="rule_model_hybrid",
        evaluation_metrics=(
            "overall_accuracy",
            "macro_f1",
            "rule_consistency",
            "modality_dropout_robustness",
            "alarm_precision",
        ),
        output_heads=COMMON_OUTPUT_HEADS,
    ),
}


PLUGIN_MODALITY_DEPENDENCY_MAP: dict[str, str | None] = {
    "visual": "transformer_monitoring",
    "thermal": "transformer_monitoring",
    "acoustic": "acoustic_monitoring",
    "ultrasonic": "acoustic_monitoring",
    "gas": "gas_detection",
    "hyperspectral": "hyperspectral_detection",
    "vibration": None,
}


PLUGIN_MODEL_ROLES: dict[str, tuple[str, ...]] = {
    "multimodal_fusion": (
        "multimodal_fusion",
        "multimodal_feature_fusion",
        "multimodal_decision_fusion",
    )
}


@dataclass(frozen=True)
class MultimodalUploadLayout:
    task_type: str
    required_dirs: tuple[str, ...]
    optional_dirs: tuple[str, ...] = ()
    sample_index_file: str = "samples.jsonl"
    notes: tuple[str, ...] = field(default_factory=tuple)


MULTIMODAL_UPLOAD_LAYOUTS: dict[str, MultimodalUploadLayout] = {
    "multimodal_feature_fusion": MultimodalUploadLayout(
        task_type="multimodal_feature_fusion",
        required_dirs=("labels", "metadata"),
        optional_dirs=tuple(sorted(MULTIMODAL_SUPPORTED_MODALITIES)),
        notes=(
            "每个样本通过 samples.jsonl 声明 available_modalities 和 modality_paths。",
            "允许部分模态缺失，不要求所有样本都具备完整模态。",
        ),
    ),
    "multimodal_decision_fusion": MultimodalUploadLayout(
        task_type="multimodal_decision_fusion",
        required_dirs=("labels", "metadata"),
        optional_dirs=tuple(sorted(MULTIMODAL_SUPPORTED_MODALITIES)),
        notes=(
            "决策级融合可上传各模态原始文件，也可上传预提取检测结果。",
            "diagnostic_rule_pack 推荐与数据包一同上传，用于训练/评估规则一致性。",
        ),
    ),
}


def is_multimodal_task(task_type: str) -> bool:
    return task_type in MULTIMODAL_TASK_TYPES


def get_multimodal_task_profile(task_type: str) -> MultimodalTaskProfile | None:
    return MULTIMODAL_TASK_PROFILES.get(task_type)


def get_multimodal_upload_layout(task_type: str) -> MultimodalUploadLayout | None:
    return MULTIMODAL_UPLOAD_LAYOUTS.get(task_type)
