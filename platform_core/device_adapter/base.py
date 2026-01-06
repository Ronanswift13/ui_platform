"""
设备基类

所有设备适配器的基类
"""


from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class DeviceStatus(str, Enum):
    """设备状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_type: str
    name: str
    manufacturer: str = ""
    model: str = ""
    firmware_version: str = ""
    serial_number: str = ""
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDevice(ABC):
    """
    设备基类

    所有设备适配器必须继承此类
    """

    def __init__(self, device_id: str, config: dict[str, Any]):
        self.device_id = device_id
        self.config = config
        self._status = DeviceStatus.DISCONNECTED
        self._info: Optional[DeviceInfo] = None
        self._last_error: str = ""
        self._connected_at: Optional[datetime] = None

    @property
    def status(self) -> DeviceStatus:
        return self._status

    @property
    def info(self) -> Optional[DeviceInfo]:
        return self._info

    @property
    def is_connected(self) -> bool:
        return self._status == DeviceStatus.CONNECTED

    @abstractmethod
    def connect(self) -> bool:
        """连接设备"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    def healthcheck(self) -> bool:
        """健康检查"""
        pass

    def get_status_info(self) -> dict[str, Any]:
        """获取状态信息"""
        return {
            "device_id": self.device_id,
            "status": self._status.value,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "last_error": self._last_error,
            "info": self._info.__dict__ if self._info else None,
        }

    def _set_status(self, status: DeviceStatus, error: str = "") -> None:
        """设置状态"""
        self._status = status
        if error:
            self._last_error = error
        if status == DeviceStatus.CONNECTED:
            self._connected_at = datetime.now()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.device_id} [{self._status.value}]>"
