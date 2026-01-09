#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件状态混入类 (Plugin Status Mixin)

为独立插件提供 set_status 等必要方法，以便与 platform_core 兼容。
将此混入类添加到各插件中，解决 'set_status' 属性缺失问题。
"""

from enum import Enum
from typing import Optional


class PluginStatus(str, Enum):
    """插件状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class PluginStatusMixin:
    """
    插件状态混入类
    
    为插件提供以下功能:
    - status 属性
    - set_status() 方法
    - get_plugin_status() 方法
    """
    
    def __init_status__(self):
        """初始化状态相关属性 - 在插件 __init__ 中调用"""
        self._status: PluginStatus = PluginStatus.UNLOADED
        self._last_error: str = ""
    
    @property
    def status(self) -> PluginStatus:
        """获取插件状态"""
        if not hasattr(self, '_status'):
            self._status = PluginStatus.UNLOADED
        return self._status
    
    @status.setter
    def status(self, value: PluginStatus):
        """设置插件状态"""
        self._status = value
    
    def set_status(self, status: PluginStatus, error: str = "") -> None:
        """
        设置插件状态
        
        Args:
            status: 插件状态
            error: 错误信息 (可选)
        """
        self._status = status
        if error:
            self._last_error = error
    
    def get_plugin_status(self) -> dict:
        """获取插件状态详情"""
        return {
            'plugin_id': getattr(self, 'PLUGIN_ID', 'unknown'),
            'name': getattr(self, 'PLUGIN_NAME', 'Unknown Plugin'),
            'version': getattr(self, 'PLUGIN_VERSION', '0.0.0'),
            'status': self._status.value if hasattr(self, '_status') else 'unknown',
            'initialized': getattr(self, '_is_initialized', False),
            'last_error': getattr(self, '_last_error', '')
        }
