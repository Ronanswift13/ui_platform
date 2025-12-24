"""
摄像头设备适配器

支持:
- RTSP流
- USB摄像头
- 图片文件 (测试用)
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum

import cv2
import numpy as np

from platform_core.device_adapter.base import BaseDevice, DeviceInfo, DeviceStatus
from platform_core.logging import get_logger

logger = get_logger(__name__)


class CameraType(str, Enum):
    """摄像头类型"""
    RTSP = "rtsp"
    USB = "usb"
    FILE = "file"
    MOCK = "mock"


@dataclass
class CameraConfig:
    """摄像头配置"""
    camera_type: CameraType = CameraType.MOCK
    url: str = ""  # RTSP URL 或文件路径
    device_index: int = 0  # USB设备索引
    width: int = 1920
    height: int = 1080
    fps: int = 25
    buffer_size: int = 1
    timeout: int = 10000  # 毫秒
    reconnect_delay: int = 5  # 秒

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        camera_type = data.get("camera_type", "mock")
        if isinstance(camera_type, str):
            camera_type = CameraType(camera_type)
        return cls(
            camera_type=camera_type,
            url=data.get("url", ""),
            device_index=data.get("device_index", 0),
            width=data.get("width", 1920),
            height=data.get("height", 1080),
            fps=data.get("fps", 25),
            buffer_size=data.get("buffer_size", 1),
            timeout=data.get("timeout", 10000),
            reconnect_delay=data.get("reconnect_delay", 5),
        )


class CameraDevice(BaseDevice):
    """
    摄像头设备

    统一封装各种视频源
    """

    def __init__(self, device_id: str, config: dict[str, Any]):
        super().__init__(device_id, config)
        self.camera_config = CameraConfig.from_dict(config)
        self._capture: Optional[cv2.VideoCapture] = None
        self._mock_image: Optional[np.ndarray] = None

    def connect(self) -> bool:
        """连接摄像头"""
        self._set_status(DeviceStatus.CONNECTING)

        try:
            if self.camera_config.camera_type == CameraType.MOCK:
                # Mock模式,生成测试图像
                self._mock_image = self._create_mock_image()
                self._set_status(DeviceStatus.CONNECTED)
                self._info = DeviceInfo(
                    device_id=self.device_id,
                    device_type="camera",
                    name="Mock Camera",
                    capabilities=["capture", "stream"],
                )
                logger.info(f"Mock摄像头已连接: {self.device_id}")
                return True

            elif self.camera_config.camera_type == CameraType.RTSP:
                self._capture = cv2.VideoCapture(self.camera_config.url)

            elif self.camera_config.camera_type == CameraType.USB:
                self._capture = cv2.VideoCapture(self.camera_config.device_index)

            elif self.camera_config.camera_type == CameraType.FILE:
                self._capture = cv2.VideoCapture(self.camera_config.url)

            if self._capture is None or not self._capture.isOpened():
                self._set_status(DeviceStatus.ERROR, "无法打开视频源")
                return False

            # 设置参数
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.camera_config.buffer_size)

            self._set_status(DeviceStatus.CONNECTED)
            self._info = DeviceInfo(
                device_id=self.device_id,
                device_type="camera",
                name=f"{self.camera_config.camera_type.value} Camera",
                capabilities=["capture", "stream"],
            )
            logger.info(f"摄像头已连接: {self.device_id}")
            return True

        except Exception as e:
            self._set_status(DeviceStatus.ERROR, str(e))
            logger.error(f"摄像头连接失败 [{self.device_id}]: {e}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._mock_image = None
        self._set_status(DeviceStatus.DISCONNECTED)
        logger.info(f"摄像头已断开: {self.device_id}")
        return True

    def healthcheck(self) -> bool:
        """健康检查"""
        if self.camera_config.camera_type == CameraType.MOCK:
            return self._mock_image is not None

        if self._capture is None:
            return False

        return self._capture.isOpened()

    def capture(self) -> Optional[np.ndarray]:
        """
        捕获一帧

        Returns:
            BGR格式的图像数组
        """
        if not self.is_connected:
            return None

        if self.camera_config.camera_type == CameraType.MOCK:
            if self._mock_image is None:
                return None
            return self._mock_image.copy()

        if self._capture is None:
            return None

        ret, frame = self._capture.read()
        if not ret:
            logger.warning(f"摄像头读取失败: {self.device_id}")
            return None

        return frame

    def capture_multiple(self, count: int = 5, interval_ms: int = 100) -> list[np.ndarray]:
        """
        连续捕获多帧

        Args:
            count: 捕获数量
            interval_ms: 帧间隔(毫秒)

        Returns:
            图像列表
        """
        import time

        frames = []
        for _ in range(count):
            frame = self.capture()
            if frame is not None:
                frames.append(frame)
            time.sleep(interval_ms / 1000)

        return frames

    def _create_mock_image(self) -> np.ndarray:
        """创建Mock测试图像"""
        image = np.zeros(
            (self.camera_config.height, self.camera_config.width, 3),
            dtype=np.uint8,
        )
        # 添加网格
        for i in range(0, self.camera_config.width, 100):
            cv2.line(image, (i, 0), (i, self.camera_config.height), (50, 50, 50), 1)
        for i in range(0, self.camera_config.height, 100):
            cv2.line(image, (0, i), (self.camera_config.width, i), (50, 50, 50), 1)

        # 添加文字
        cv2.putText(
            image,
            f"Mock Camera: {self.device_id}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Resolution: {self.camera_config.width}x{self.camera_config.height}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
        )

        return image
