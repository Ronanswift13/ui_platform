"""
云台设备适配器

支持:
- PTZ控制 (平移/俯仰/缩放)
- 预置位管理
- 自动巡航
"""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from platform_core.device_adapter.base import BaseDevice, DeviceInfo, DeviceStatus
from platform_core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PTZPosition:
    """云台位置"""
    pan: float = 0.0  # 水平角度 (-180 ~ 180)
    tilt: float = 0.0  # 垂直角度 (-90 ~ 90)
    zoom: float = 1.0  # 缩放倍数


@dataclass
class PTZPreset:
    """云台预置位"""
    id: str
    name: str
    position: PTZPosition
    metadata: dict[str, Any] = field(default_factory=dict)


class PTZDevice(BaseDevice):
    """
    云台设备

    控制摄像头的PTZ功能
    """

    def __init__(self, device_id: str, config: dict[str, Any]):
        super().__init__(device_id, config)
        self._current_position = PTZPosition()
        self._presets: dict[str, PTZPreset] = {}
        self._is_mock = config.get("mock", True)

    def connect(self) -> bool:
        """连接云台"""
        self._set_status(DeviceStatus.CONNECTING)

        try:
            if self._is_mock:
                # Mock模式
                self._set_status(DeviceStatus.CONNECTED)
                self._info = DeviceInfo(
                    device_id=self.device_id,
                    device_type="ptz",
                    name="Mock PTZ",
                    capabilities=["pan", "tilt", "zoom", "preset"],
                )
                logger.info(f"Mock云台已连接: {self.device_id}")
                return True

            # TODO: 实现实际的云台连接协议 (ONVIF/私有协议)
            self._set_status(DeviceStatus.ERROR, "未实现实际PTZ协议")
            return False

        except Exception as e:
            self._set_status(DeviceStatus.ERROR, str(e))
            logger.error(f"云台连接失败 [{self.device_id}]: {e}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        self._set_status(DeviceStatus.DISCONNECTED)
        logger.info(f"云台已断开: {self.device_id}")
        return True

    def healthcheck(self) -> bool:
        """健康检查"""
        return self.is_connected

    def get_position(self) -> PTZPosition:
        """获取当前位置"""
        return self._current_position

    def move_to(self, position: PTZPosition, speed: float = 1.0) -> bool:
        """
        移动到指定位置

        Args:
            position: 目标位置
            speed: 移动速度 (0-1)

        Returns:
            是否成功
        """
        if not self.is_connected:
            return False

        try:
            if self._is_mock:
                # Mock模式直接设置位置
                self._current_position = position
                logger.debug(f"云台移动到: pan={position.pan}, tilt={position.tilt}, zoom={position.zoom}")
                return True

            # TODO: 实际PTZ控制
            return False

        except Exception as e:
            logger.error(f"云台移动失败: {e}")
            return False

    def move_relative(
        self,
        pan_delta: float = 0,
        tilt_delta: float = 0,
        zoom_delta: float = 0,
        speed: float = 1.0,
    ) -> bool:
        """相对移动"""
        new_position = PTZPosition(
            pan=self._current_position.pan + pan_delta,
            tilt=self._current_position.tilt + tilt_delta,
            zoom=max(1.0, self._current_position.zoom + zoom_delta),
        )
        return self.move_to(new_position, speed)

    def zoom_in(self, factor: float = 0.5) -> bool:
        """放大"""
        return self.move_relative(zoom_delta=factor)

    def zoom_out(self, factor: float = 0.5) -> bool:
        """缩小"""
        return self.move_relative(zoom_delta=-factor)

    def go_to_preset(self, preset_id: str) -> bool:
        """
        移动到预置位

        Args:
            preset_id: 预置位ID

        Returns:
            是否成功
        """
        if preset_id not in self._presets:
            logger.warning(f"预置位不存在: {preset_id}")
            return False

        preset = self._presets[preset_id]
        return self.move_to(preset.position)

    def save_preset(self, preset_id: str, name: str, metadata: Optional[dict] = None) -> PTZPreset:
        """
        保存当前位置为预置位

        Args:
            preset_id: 预置位ID
            name: 预置位名称
            metadata: 元数据

        Returns:
            预置位对象
        """
        preset = PTZPreset(
            id=preset_id,
            name=name,
            position=PTZPosition(
                pan=self._current_position.pan,
                tilt=self._current_position.tilt,
                zoom=self._current_position.zoom,
            ),
            metadata=metadata or {},
        )
        self._presets[preset_id] = preset
        logger.info(f"预置位已保存: {preset_id} - {name}")
        return preset

    def delete_preset(self, preset_id: str) -> bool:
        """删除预置位"""
        if preset_id in self._presets:
            del self._presets[preset_id]
            return True
        return False

    def list_presets(self) -> list[PTZPreset]:
        """列出所有预置位"""
        return list(self._presets.values())

    def home(self) -> bool:
        """回到原点"""
        return self.move_to(PTZPosition())
