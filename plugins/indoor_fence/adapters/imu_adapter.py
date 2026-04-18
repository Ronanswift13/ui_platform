"""
Indoor Fence V3.0 - IMU Adapter
================================

Inertial Measurement Unit adapter for acceleration and gyroscope data.
Supports serial/I2C connections and simulation mode.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

from .base_adapter import BaseAdapter, AdapterConfig, AdapterStatus
from ..protocols import SensorData, SensorType, AdapterMode

logger = logging.getLogger(__name__)


@dataclass
class IMUConfig(AdapterConfig):
    """Configuration for IMU adapter."""
    adapter_type: str = "imu"
    protocol: str = "simulate"  # serial, i2c, simulate
    device_path: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    sample_rate_hz: float = 100.0
    noise_sigma_accel: float = 0.05
    noise_sigma_gyro: float = 0.01


class IMUAdapter(BaseAdapter):
    """IMU adapter with 6-axis acceleration + gyroscope data."""

    def __init__(self, config: IMUConfig):
        super().__init__(config)
        self._imu_config = config
        self._connected = False

    @property
    def sensor_type(self) -> SensorType:
        return SensorType.IMU

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Connect to IMU device."""
        if self._imu_config.protocol == "simulate":
            self.set_mode(AdapterMode.SIMULATED)
            self._connected = True
            self._status = AdapterStatus.CONNECTED
            logger.info("IMU adapter connected (simulation mode)")
            return True

        try:
            self._connected = True
            self._status = AdapterStatus.CONNECTED
            logger.info(f"IMU adapter connected via {self._imu_config.protocol}")
            return True
        except Exception as e:
            logger.error(f"IMU connection failed: {e}")
            self._status = AdapterStatus.ERROR
            return False

    def disconnect(self) -> None:
        """Disconnect from IMU device."""
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        logger.info("IMU adapter disconnected")

    def _read_live(self) -> SensorData:
        """Read from real IMU hardware."""
        raise NotImplementedError(
            "Live IMU reading requires hardware-specific protocol implementation. "
            "Set protocol='simulate' for simulation mode."
        )

    def _read_simulated(self) -> SensorData:
        """Generate simulated IMU data."""
        ts = time.time()
        s_a = self._imu_config.noise_sigma_accel
        s_g = self._imu_config.noise_sigma_gyro

        return SensorData(
            sensor_type=SensorType.IMU,
            timestamp=ts,
            data={
                "acceleration": {
                    "x": random.gauss(0, s_a),
                    "y": random.gauss(0, s_a),
                    "z": 9.81 + random.gauss(0, s_a),
                },
                "gyroscope": {
                    "x": random.gauss(0, s_g),
                    "y": random.gauss(0, s_g),
                    "z": random.gauss(0, s_g),
                },
                "timestamp": ts,
            },
            confidence=0.95,
            is_simulated=True,
        )
