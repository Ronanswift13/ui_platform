"""
训练配置 Schema 定义与校验

校验用户提交的训练任务配置是否完整且合法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfigSchema:
    """训练任务配置结构"""

    # 目标
    plugin_id: str = ""
    task_type: str = ""
    voltage_level: str = ""

    # 模型
    model_family: str = ""
    backbone: str = ""
    pretrained_weights: str = ""
    paradigm: str = ""

    # 数据
    dataset_path: str = ""
    dataset_manifest: str = ""

    # 超参
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    image_size: int = 640
    sequence_length: int = 0
    window_size: int | float = 0
    prediction_horizon: int | float = 0
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    weight_decay: float = 0.0005

    # 增强
    augmentation: dict[str, Any] = field(default_factory=lambda: {
        "mosaic": 1.0,
        "mixup": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "flipud": 0.0,
        "fliplr": 0.5,
    })

    # 输出
    checkpoint_dir: str = ""
    export_formats: list[str] = field(default_factory=lambda: ["onnx"])

    # 高光谱专用
    pca_components: int | None = None
    spatial_patch_size: int | None = None
    spectral_preprocessing: str = ""  # none / pca / mnf / ica

    # 热像专用
    temperature_normalization: bool = True
    thermal_colormap: str = "jet"

    # 时序专用
    sample_rate_hz: int | None = None
    feature_views: list[str] = field(default_factory=list)
    sensor_columns: list[str] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)

    # 多模态专用
    supported_modalities: list[str] = field(default_factory=list)
    required_modalities: list[str] = field(default_factory=list)
    fusion_strategy: str = "hybrid"
    missing_modality_policy: str = "mask_and_gate"


def validate_training_config(config: TrainingConfigSchema) -> list[str]:
    """校验训练配置，返回错误列表"""
    from .dataset_manifest import VALID_TASK_TYPES, VALID_PLUGIN_IDS
    from training.temporal_anomaly import TEMPORAL_PARADIGMS, is_temporal_task
    from training.multimodal_fusion import (
        MULTIMODAL_FUSION_STRATEGIES,
        MULTIMODAL_MISSING_MODALITY_POLICIES,
        MULTIMODAL_SUPPORTED_MODALITIES,
        MULTIMODAL_TRAINING_PARADIGMS,
        is_multimodal_task,
    )

    errors: list[str] = []

    if not config.plugin_id:
        errors.append("plugin_id 不能为空")
    elif config.plugin_id not in VALID_PLUGIN_IDS:
        errors.append(f"未知 plugin_id: '{config.plugin_id}'")

    if not config.task_type:
        errors.append("task_type 不能为空")
    elif config.task_type not in VALID_TASK_TYPES:
        errors.append(f"未知 task_type: '{config.task_type}'")

    if config.epochs < 1:
        errors.append(f"epochs={config.epochs} 必须 >= 1")

    if config.batch_size < 1:
        errors.append(f"batch_size={config.batch_size} 必须 >= 1")

    if config.learning_rate <= 0:
        errors.append(f"learning_rate={config.learning_rate} 必须 > 0")

    if config.image_size < 32:
        errors.append(f"image_size={config.image_size} 必须 >= 32")

    if config.paradigm:
        valid_paradigms = set(TEMPORAL_PARADIGMS)
        if is_multimodal_task(config.task_type):
            valid_paradigms |= set(MULTIMODAL_TRAINING_PARADIGMS)
        if config.paradigm not in valid_paradigms:
            errors.append(f"paradigm '{config.paradigm}' 不合法")

    # 高光谱校验
    if config.task_type in ("hyperspectral_classification", "hyperspectral_anomaly"):
        if not config.pca_components:
            errors.append("高光谱任务需要设置 pca_components")
        if config.spectral_preprocessing and config.spectral_preprocessing not in (
            "none", "pca", "mnf", "ica"
        ):
            errors.append(f"spectral_preprocessing '{config.spectral_preprocessing}' 不合法")

    if is_temporal_task(config.task_type):
        if config.sequence_length < 0:
            errors.append(f"sequence_length={config.sequence_length} 不能为负数")
        if config.window_size and config.window_size < 0:
            errors.append(f"window_size={config.window_size} 不能为负数")
        if (
            config.task_type in {
                "multivariate_sensor_anomaly",
                "equipment_health_prediction",
                "health_index_calibration",
            }
            and not config.sensor_columns
        ):
            errors.append(f"{config.task_type} 需要提供 sensor_columns")
        if config.task_type == "equipment_health_prediction" and config.prediction_horizon <= 0:
            errors.append("equipment_health_prediction 需要提供 prediction_horizon")
        if config.task_type == "acoustic_time_frequency_anomaly" and not config.sample_rate_hz:
            errors.append("acoustic_time_frequency_anomaly 需要提供 sample_rate_hz")
        if (
            config.task_type == "action_event_sequence_recognition"
            and not config.event_types
        ):
            errors.append("action_event_sequence_recognition 需要提供 event_types")

    if is_multimodal_task(config.task_type):
        if not config.supported_modalities:
            errors.append(f"{config.task_type} 需要提供 supported_modalities")
        else:
            unknown = sorted(set(config.supported_modalities) - MULTIMODAL_SUPPORTED_MODALITIES)
            if unknown:
                errors.append(f"supported_modalities 包含未知模态: {unknown}")
        if config.required_modalities:
            missing = sorted(set(config.required_modalities) - set(config.supported_modalities))
            if missing:
                errors.append(f"required_modalities 必须是 supported_modalities 子集: {missing}")
        if config.fusion_strategy not in MULTIMODAL_FUSION_STRATEGIES:
            errors.append(f"fusion_strategy '{config.fusion_strategy}' 不合法")
        if config.missing_modality_policy not in MULTIMODAL_MISSING_MODALITY_POLICIES:
            errors.append(
                f"missing_modality_policy '{config.missing_modality_policy}' 不合法"
            )

    return errors
