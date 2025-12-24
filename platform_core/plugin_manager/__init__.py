"""
插件管理模块

负责:
- 插件发现和加载
- 插件生命周期管理
- 插件版本管理
- 插件健康检查
"""


from __future__ import annotations
from platform_core.plugin_manager.base import BasePlugin, PluginContext, PluginCapability
from platform_core.plugin_manager.manager import PluginManager
from platform_core.plugin_manager.registry import PluginRegistry

__all__ = [
    "BasePlugin",
    "PluginContext",
    "PluginCapability",
    "PluginManager",
    "PluginRegistry",
]
