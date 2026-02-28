"""
DarkBreaker SDK - Local Services

Lightweight, self-contained versions of platform_core services.
These allow plugins to run independently without the full platform.

When running inside the platform, these are replaced by the full versions.
When running standalone, these provide essential functionality locally.
"""

from darkbreaker_sdk.services.inference import LocalInferenceEngine
from darkbreaker_sdk.services.model_registry import LocalModelRegistry
from darkbreaker_sdk.services.data_manager import LocalDataManager
from darkbreaker_sdk.services.alarm_manager import LocalAlarmManager
from darkbreaker_sdk.services.config_manager import LocalConfigManager

__all__ = [
    "LocalInferenceEngine",
    "LocalModelRegistry",
    "LocalDataManager",
    "LocalAlarmManager",
    "LocalConfigManager",
]
