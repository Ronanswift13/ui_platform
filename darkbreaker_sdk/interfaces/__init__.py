"""
DarkBreaker SDK Interfaces

Plugin lifecycle, base plugin, and adapter abstractions.
"""

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

__all__ = [
    "PluginCapability",
    "PluginStatus",
    "HealthStatus",
    "BasePlugin",
    "PluginManifest",
    "PluginContext",
    "BaseAdapter",
    "AdapterStatus",
]
