"""
DarkBreaker SDK

Lightweight SDK for building standalone DarkBreaker plugins.
Provides interfaces, schemas, standalone runner, local services, and utilities.

Each plugin can operate in two modes:
  - Integrated Mode: Loaded by PluginManager within the full platform
  - Standalone Mode: Runs independently with SDK-provided local services

Local services (darkbreaker_sdk.services):
  - LocalInferenceEngine: ONNX/PyTorch inference
  - LocalModelRegistry: Model discovery and management
  - LocalDataManager: Data persistence with voltage level support
  - LocalAlarmManager: Alarm lifecycle management
  - LocalConfigManager: Configuration with device type support
"""

__version__ = "2.0.0"

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
