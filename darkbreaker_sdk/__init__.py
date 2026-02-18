"""
DarkBreaker SDK

Lightweight SDK for building standalone DarkBreaker plugins.
Provides interfaces, schemas, standalone runner, and utilities.
"""

__version__ = "1.0.0"

from darkbreaker_sdk.interfaces.lifecycle import (
    PluginCapability,
    PluginStatus,
    HealthStatus,
)
from darkbreaker_sdk.interfaces.base_plugin import (
    BasePlugin,
    PluginManifest,
    PluginContext,
)
from darkbreaker_sdk.interfaces.base_adapter import (
    BaseAdapter,
    AdapterStatus,
)
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
    # Interfaces
    "PluginCapability",
    "PluginStatus",
    "HealthStatus",
    "BasePlugin",
    "PluginManifest",
    "PluginContext",
    "BaseAdapter",
    "AdapterStatus",
    # Schemas
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
