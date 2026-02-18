"""
DarkBreaker SDK Schemas

Pydantic v2 data models for detection, alarms, and plugin I/O.
"""

from darkbreaker_sdk.schemas.common import (
    BaseEntity,
    ROI,
    ROIType,
    Evidence,
    EvidenceType,
    generate_id,
)
from darkbreaker_sdk.schemas.detection import (
    BoundingBox,
    RecognitionResult,
)
from darkbreaker_sdk.schemas.alarm import (
    Alarm,
    AlarmLevel,
    AlarmRule,
    AlarmStatus,
)
from darkbreaker_sdk.schemas.plugin_io import PluginOutput

__all__ = [
    "BaseEntity",
    "ROI",
    "ROIType",
    "Evidence",
    "EvidenceType",
    "generate_id",
    "BoundingBox",
    "RecognitionResult",
    "Alarm",
    "AlarmLevel",
    "AlarmRule",
    "AlarmStatus",
    "PluginOutput",
]
