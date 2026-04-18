"""
训练模块 - Schema 定义与校验
"""

from .dataset_manifest import DatasetManifest, validate_manifest
from .label_schemas import (
    LabelSchema,
    DetectionBBoxSchema,
    SegmentationMaskSchema,
    ClassificationSchema,
    OCRReadingSchema,
    ThermalAnomalySchema,
    HyperspectralSchema,
    TemporalSeriesSchema,
    ActionEventSequenceSchema,
    MultimodalSampleSchema,
    get_schema_for_task_type,
)
from .training_config_schema import TrainingConfigSchema, validate_training_config

__all__ = [
    "DatasetManifest",
    "validate_manifest",
    "LabelSchema",
    "DetectionBBoxSchema",
    "SegmentationMaskSchema",
    "ClassificationSchema",
    "OCRReadingSchema",
    "ThermalAnomalySchema",
    "HyperspectralSchema",
    "TemporalSeriesSchema",
    "ActionEventSequenceSchema",
    "MultimodalSampleSchema",
    "get_schema_for_task_type",
    "TrainingConfigSchema",
    "validate_training_config",
]
