"""
输变电激光星芒破夜绘明监测平台 - 核心模块

平台核心提供以下功能:
- 配置管理 (config)
- 插件管理 (plugin_manager)
- 任务调度 (scheduler)
- Schema校验 (schema)
- 证据链管理 (evidence)
- 统一日志 (logging)
- 设备适配 (device_adapter)
- 回放功能 (replay)
"""


from __future__ import annotations
__version__ = "1.0.0"
__author__ = "Power Station Team"

from platform_core.config import PlatformConfig, get_config
from platform_core.exceptions import (
    PlatformError,
    PluginError,
    SchemaValidationError,
    TaskError,
)

__all__ = [
    "PlatformConfig",
    "get_config",
    "PlatformError",
    "PluginError",
    "SchemaValidationError",
    "TaskError",
]
