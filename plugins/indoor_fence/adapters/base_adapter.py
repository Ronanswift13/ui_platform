#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配器基类
==========================================

定义统一的硬件适配器接口
支持: 相机、激光雷达、灯光控制

作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class AdapterStatus(str, Enum):
    """适配器状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RUNNING = "running"
    ERROR = "error"
    SIMULATED = "simulated"
    SIMULATING = "simulating"  # 模拟模式别名


@dataclass
class AdapterHealth:
    """适配器健康状态"""
    status: AdapterStatus
    message: str = ""
    last_data_time: Optional[datetime] = None
    error_count: int = 0
    data_rate_hz: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterConfig:
    """适配器配置基类"""
    enabled: bool = True
    device_id: str = ""
    device_name: str = ""
    connection_timeout: float = 5.0
    retry_count: int = 3
    simulate_if_unavailable: bool = True
    simulation: bool = False  # 模拟模式开关
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """
    适配器基类
    
    所有硬件适配器需要继承此类并实现抽象方法
    """
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self._status = AdapterStatus.DISCONNECTED
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._connected = False
        self._simulated = False
        self._simulation_mode = getattr(config, 'simulation', False)
        self._last_data_time: Optional[datetime] = None
        self._data_count = 0
        self._start_time: Optional[datetime] = None
        self._stats = {
            'connect_attempts': 0,
            'successful_reads': 0,
            'failed_reads': 0,
            'last_read_time': 0,
        }

        logger.info(f"[{self.__class__.__name__}] 适配器初始化: {getattr(config, 'device_id', 'unknown')}")
    
    @property
    def status(self) -> AdapterStatus:
        return self._status
    
    @property
    def is_connected(self) -> bool:
        return self._connected or self._simulated
    
    @property
    def is_simulated(self) -> bool:
        return self._simulated

    @property
    def is_simulating(self) -> bool:
        """是否在模拟模式"""
        return self._simulation_mode or self._status in (AdapterStatus.SIMULATED, AdapterStatus.SIMULATING)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
    
    @abstractmethod
    def connect(self) -> bool:
        """连接设备"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def healthcheck(self) -> bool:
        """健康检查"""
        pass
    
    def _set_error(self, message: str):
        """设置错误状态"""
        self._last_error = message
        self._status = AdapterStatus.ERROR
        logger.error(f"[{self.__class__.__name__}] {message}")
    
    def _set_connected(self):
        """设置连接状态"""
        self._connected = True
        self._simulated = False
        self._status = AdapterStatus.CONNECTED
        self._last_error = None
    
    def _set_simulated(self):
        """设置模拟状态"""
        self._connected = False
        self._simulated = True
        self._status = AdapterStatus.SIMULATED
        logger.warning(f"[{self.__class__.__name__}] 使用模拟模式")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()

    def get_health(self) -> AdapterHealth:
        """获取健康状态"""
        data_rate = 0.0
        if self._start_time and self._data_count > 0:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            if elapsed > 0:
                data_rate = self._data_count / elapsed

        return AdapterHealth(
            status=self._status,
            message=self._last_error or "",
            last_data_time=self._last_data_time,
            error_count=self._error_count,
            data_rate_hz=data_rate,
        )

    def reset_error(self):
        """重置错误状态"""
        self._error_count = 0
        self._last_error = None

    def _update_data_time(self):
        """更新数据时间"""
        self._last_data_time = datetime.now()
        self._data_count += 1
        if self._start_time is None:
            self._start_time = self._last_data_time

    def enable_simulation(self, enable: bool = True):
        """启用/禁用模拟模式"""
        self._simulation_mode = enable
        if enable:
            self._status = AdapterStatus.SIMULATING
            self._simulated = True
            logger.info(f"[{self.__class__.__name__}] 模拟模式已启用")
        else:
            self._status = AdapterStatus.DISCONNECTED
            self._simulated = False
            logger.info(f"[{self.__class__.__name__}] 模拟模式已禁用")


__all__ = ['AdapterStatus', 'AdapterHealth', 'AdapterConfig', 'BaseAdapter']
