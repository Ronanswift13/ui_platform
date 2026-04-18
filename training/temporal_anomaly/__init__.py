"""
时序异常训练模块

为非视觉任务提供统一的数据契约、任务画像和上传布局。
"""

from .data_contract import (
    TemporalDatasetContract,
    TemporalFeatureViewSpec,
    TemporalSourceSpec,
    TemporalTargetSpec,
    normalize_temporal_contract,
    validate_temporal_contract,
)
from .task_profiles import (
    DEVICE_MONITORING_OUTPUT_HEADS,
    PLUGIN_TEMPORAL_DEFAULT_TASKS,
    PLUGIN_TEMPORAL_MODEL_ROLES,
    TEMPORAL_DATASET_FAMILY,
    TEMPORAL_MODALITIES,
    TEMPORAL_PARADIGMS,
    TEMPORAL_SOURCE_KINDS,
    TEMPORAL_TASK_PROFILES,
    TEMPORAL_TASK_TYPES,
    TEMPORAL_UPLOAD_LAYOUTS,
    OutputHeadSpec,
    TemporalTaskProfile,
    TemporalUploadLayout,
    get_temporal_task_profile,
    get_temporal_upload_layout,
    is_temporal_task,
)

__all__ = [
    "DEVICE_MONITORING_OUTPUT_HEADS",
    "PLUGIN_TEMPORAL_DEFAULT_TASKS",
    "PLUGIN_TEMPORAL_MODEL_ROLES",
    "TEMPORAL_DATASET_FAMILY",
    "TEMPORAL_MODALITIES",
    "TEMPORAL_PARADIGMS",
    "TEMPORAL_SOURCE_KINDS",
    "TEMPORAL_TASK_PROFILES",
    "TEMPORAL_TASK_TYPES",
    "TEMPORAL_UPLOAD_LAYOUTS",
    "OutputHeadSpec",
    "TemporalDatasetContract",
    "TemporalFeatureViewSpec",
    "TemporalSourceSpec",
    "TemporalTargetSpec",
    "TemporalTaskProfile",
    "TemporalUploadLayout",
    "get_temporal_task_profile",
    "get_temporal_upload_layout",
    "is_temporal_task",
    "normalize_temporal_contract",
    "validate_temporal_contract",
]
