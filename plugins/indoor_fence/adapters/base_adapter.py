#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配器基类 V3.0
==========================================

定义统一的硬件适配器接口
支持: 相机、激光雷达、灯光控制
模式: LIVE / SIMULATED / REPLAY
功能: 数据录制(JSONL)、回放、健康监控

作者: G组 | 版本: 3.0.0
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import time

# Import protocol types for the new interface
from ..protocols import SensorData, SensorType, AdapterMode

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
    frames_read: int = 0
    data_rate_hz: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterConfig:
    """适配器配置基类"""
    adapter_type: str = ""
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
    适配器基类 V3.0

    所有硬件适配器需要继承此类并实现抽象方法。
    支持三种运行模式:
      - LIVE: 从真实硬件读取数据
      - SIMULATED: 生成模拟数据
      - REPLAY: 从录制文件回放数据
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

        # V3.0: Mode, recording, replay, health counters
        self._mode: AdapterMode = AdapterMode.LIVE
        self._recording_file = None
        self._is_recording: bool = False
        self._replay_data: List[SensorData] = []
        self._replay_index: int = 0
        self._frames_read: int = 0

        logger.info(f"[{self.__class__.__name__}] 适配器初始化: {getattr(config, 'device_id', 'unknown')}")

    # ------------------------------------------------------------------
    # Existing properties (backward compatible)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # V3.0 properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> AdapterMode:
        """Current adapter operating mode."""
        return self._mode

    @property
    def is_recording(self) -> bool:
        """Whether data recording is active."""
        return self._is_recording

    @property
    def sensor_type(self) -> SensorType:
        """Sensor type for this adapter. Override in subclasses."""
        return SensorType.CAMERA

    # ------------------------------------------------------------------
    # Abstract methods (backward compatible)
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """连接设备"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    def healthcheck(self) -> bool:
        """健康检查 - default implementation returns connection status."""
        return self.is_connected

    # ------------------------------------------------------------------
    # V3.0: Mode management
    # ------------------------------------------------------------------

    def set_mode(self, mode: AdapterMode) -> None:
        """Switch adapter operating mode."""
        old_mode = self._mode
        self._mode = mode
        logger.info(f"[{self.__class__.__name__}] 模式切换: {old_mode.value} -> {mode.value}")

    # ------------------------------------------------------------------
    # V3.0: Data read dispatch
    # ------------------------------------------------------------------

    def _read_live(self) -> Optional[SensorData]:
        """Read data from live hardware. Override in subclasses."""
        return None

    def _read_simulated(self) -> SensorData:
        """Generate simulated data. Override for sensor-specific simulation."""
        return SensorData(
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            data={"simulated": True},
            confidence=0.5,
            is_simulated=True,
        )

    def _read_replay(self) -> Optional[SensorData]:
        """Read next frame from replay buffer."""
        if not self._replay_data or self._replay_index >= len(self._replay_data):
            return None
        data = self._replay_data[self._replay_index]
        self._replay_index += 1
        return data

    def get_data(self) -> Optional[SensorData]:
        """
        Main data retrieval method.

        Routes to the appropriate _read method based on current mode.
        Also handles recording and health counter updates.
        """
        data: Optional[SensorData] = None
        try:
            if self._mode == AdapterMode.LIVE:
                data = self._read_live()
            elif self._mode == AdapterMode.SIMULATED:
                data = self._read_simulated()
            elif self._mode == AdapterMode.REPLAY:
                data = self._read_replay()

            if data is not None:
                self._frames_read += 1
                self._update_data_time()
                self._stats['successful_reads'] += 1

                # Record if active
                if self._is_recording and self._recording_file is not None:
                    try:
                        self._recording_file.write(data.model_dump_json() + "\n")
                        self._recording_file.flush()
                    except Exception as e:
                        logger.warning(f"[{self.__class__.__name__}] 录制写入失败: {e}")

            return data

        except Exception as e:
            self._error_count += 1
            self._stats['failed_reads'] += 1
            self._set_error(str(e))
            return None

    # ------------------------------------------------------------------
    # V3.0: Recording
    # ------------------------------------------------------------------

    def start_recording(self, path: str) -> None:
        """Start recording sensor data to a JSONL file."""
        if self._is_recording:
            logger.warning(f"[{self.__class__.__name__}] 已在录制中，先停止当前录制")
            self.stop_recording()
        self._recording_file = open(path, "w")
        self._is_recording = True
        logger.info(f"[{self.__class__.__name__}] 开始录制: {path}")

    def stop_recording(self) -> None:
        """Stop recording and close the file."""
        if self._recording_file is not None:
            try:
                self._recording_file.close()
            except Exception:
                pass
            self._recording_file = None
        self._is_recording = False
        logger.info(f"[{self.__class__.__name__}] 停止录制")

    # ------------------------------------------------------------------
    # V3.0: Replay
    # ------------------------------------------------------------------

    def load_replay(self, path: str) -> None:
        """Load a JSONL recording file and switch to REPLAY mode."""
        self._replay_data = []
        self._replay_index = 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._replay_data.append(SensorData.model_validate_json(line))
        self._mode = AdapterMode.REPLAY
        logger.info(
            f"[{self.__class__.__name__}] 加载回放数据: {len(self._replay_data)} 帧, 模式切换到 REPLAY"
        )

    # ------------------------------------------------------------------
    # V3.0: Weight
    # ------------------------------------------------------------------

    def get_weight(self) -> float:
        """
        Return adapter weight for fusion layer (0.0 - 1.0).
        Override in subclasses for sensor-specific weighting.
        """
        return 1.0

    # ------------------------------------------------------------------
    # Existing helper methods (backward compatible)
    # ------------------------------------------------------------------

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
            frames_read=self._frames_read,
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


__all__ = [
    'AdapterStatus', 'AdapterHealth', 'AdapterConfig', 'BaseAdapter',
    'AdapterMode',
]
