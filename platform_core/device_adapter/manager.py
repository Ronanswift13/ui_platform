"""
设备管理器

统一管理所有设备的连接和调用
"""


from __future__ import annotations
from typing import Any, Optional, Type

from platform_core.device_adapter.base import BaseDevice, DeviceStatus
from platform_core.device_adapter.camera import CameraDevice
from platform_core.device_adapter.ptz import PTZDevice
from platform_core.logging import get_logger

logger = get_logger(__name__)

# 设备类型注册表
DEVICE_TYPES: dict[str, Type[BaseDevice]] = {
    "camera": CameraDevice,
    "ptz": PTZDevice,
}


class DeviceManager:
    """
    设备管理器

    管理所有设备的生命周期
    """

    _instance: Optional["DeviceManager"] = None

    def __new__(cls) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._devices: dict[str, BaseDevice] = {}
        self._initialized = True
        logger.info("设备管理器初始化完成")

    def register_device_type(self, type_name: str, device_class: Type[BaseDevice]) -> None:
        """注册设备类型"""
        DEVICE_TYPES[type_name] = device_class
        logger.info(f"设备类型已注册: {type_name}")

    def create_device(
        self,
        device_id: str,
        device_type: str,
        config: dict[str, Any],
    ) -> BaseDevice:
        """
        创建设备实例

        Args:
            device_id: 设备ID
            device_type: 设备类型
            config: 设备配置

        Returns:
            设备实例
        """
        if device_type not in DEVICE_TYPES:
            raise ValueError(f"未知设备类型: {device_type}")

        device_class = DEVICE_TYPES[device_type]
        device = device_class(device_id, config)
        self._devices[device_id] = device
        logger.info(f"设备已创建: {device}")
        return device

    def get_device(self, device_id: str) -> Optional[BaseDevice]:
        """获取设备"""
        return self._devices.get(device_id)

    def remove_device(self, device_id: str) -> bool:
        """移除设备"""
        if device_id not in self._devices:
            return False

        device = self._devices[device_id]
        if device.is_connected:
            device.disconnect()

        del self._devices[device_id]
        logger.info(f"设备已移除: {device_id}")
        return True

    def list_devices(self) -> list[BaseDevice]:
        """列出所有设备"""
        return list(self._devices.values())

    def connect_device(self, device_id: str) -> bool:
        """连接设备"""
        device = self.get_device(device_id)
        if device is None:
            return False
        return device.connect()

    def disconnect_device(self, device_id: str) -> bool:
        """断开设备"""
        device = self.get_device(device_id)
        if device is None:
            return False
        return device.disconnect()

    def connect_all(self) -> dict[str, bool]:
        """连接所有设备"""
        results = {}
        for device_id, device in self._devices.items():
            results[device_id] = device.connect()
        return results

    def disconnect_all(self) -> dict[str, bool]:
        """断开所有设备"""
        results = {}
        for device_id, device in self._devices.items():
            results[device_id] = device.disconnect()
        return results

    def healthcheck_all(self) -> dict[str, bool]:
        """检查所有设备健康状态"""
        results = {}
        for device_id, device in self._devices.items():
            results[device_id] = device.healthcheck()
        return results

    def get_status_summary(self) -> dict[str, Any]:
        """获取状态摘要"""
        summary = {
            "total": len(self._devices),
            "connected": 0,
            "disconnected": 0,
            "error": 0,
            "devices": [],
        }

        for device in self._devices.values():
            status_info = device.get_status_info()
            summary["devices"].append(status_info)

            if device.status == DeviceStatus.CONNECTED:
                summary["connected"] += 1
            elif device.status == DeviceStatus.DISCONNECTED:
                summary["disconnected"] += 1
            elif device.status == DeviceStatus.ERROR:
                summary["error"] += 1

        return summary


def get_device_manager() -> DeviceManager:
    """获取设备管理器单例"""
    return DeviceManager()
