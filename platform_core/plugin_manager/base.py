"""
插件基类定义 - 兼容层

所有接口定义已迁移到 darkbreaker_sdk.interfaces
本模块保留为向后兼容的 re-export 层。

插件开发者应使用:
    from darkbreaker_sdk.interfaces import BasePlugin, PluginManifest, ...
    from darkbreaker_sdk.schemas import Alarm, RecognitionResult, ...
"""

# ===== Re-exports from darkbreaker_sdk (backward compatibility) =====

from darkbreaker_sdk.interfaces.lifecycle import (  # noqa: F401
    PluginCapability,
    PluginStatus,
    HealthStatus,
)

from darkbreaker_sdk.interfaces.base_plugin import (  # noqa: F401
    BasePlugin,
    PluginManifest,
    PluginContext,
)

from darkbreaker_sdk.schemas import (  # noqa: F401
    Alarm,
    AlarmRule,
    PluginOutput,
    RecognitionResult,
    ROI,
    BoundingBox,
)

__all__ = [
    "BasePlugin",
    "PluginManifest",
    "PluginContext",
    "PluginCapability",
    "PluginStatus",
    "HealthStatus",
    "Alarm",
    "AlarmRule",
    "PluginOutput",
    "RecognitionResult",
    "ROI",
    "BoundingBox",
]
