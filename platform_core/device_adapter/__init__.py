"""
设备适配层

统一适配各种设备:
- 摄像头 (Camera)
- 云台 (PTZ)
- 热成像 (Thermal)
- 可扩展其他设备 (LiDAR等)
"""


from __future__ import annotations
from platform_core.device_adapter.base import BaseDevice, DeviceStatus
from platform_core.device_adapter.camera import CameraDevice, CameraConfig
from platform_core.device_adapter.ptz import PTZDevice, PTZPreset
from platform_core.device_adapter.manager import DeviceManager

__all__ = [
    "BaseDevice",
    "DeviceStatus",
    "CameraDevice",
    "CameraConfig",
    "PTZDevice",
    "PTZPreset",
    "DeviceManager",
]
