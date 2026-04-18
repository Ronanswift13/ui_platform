"""
Indoor Fence V3.0 - Dynamic Sensor Weight Manager
===================================================

Manages per-sensor weights based on health, NLOS status,
and environmental conditions. Weights feed into the EKF
as observation noise scaling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ...protocols import SensorType


@dataclass
class SensorWeight:
    """Weight and metadata for a single sensor."""
    sensor_type: SensorType
    weight: float = 1.0
    healthy: bool = True
    nlos_probability: float = 0.0


class WeightManager:
    """Manages dynamic sensor weights for fusion."""

    # Default base weights per sensor type
    _DEFAULT_WEIGHTS: Dict[SensorType, float] = {
        SensorType.CAMERA: 0.8,
        SensorType.LIDAR: 0.9,
        SensorType.UWB: 0.85,
        SensorType.IMU: 0.7,
    }

    def __init__(self):
        self._weights: Dict[SensorType, SensorWeight] = {}
        self._low_light = False

        # Initialize with defaults
        for st, w in self._DEFAULT_WEIGHTS.items():
            self._weights[st] = SensorWeight(sensor_type=st, weight=w)

    def get_weights(self) -> Dict[SensorType, SensorWeight]:
        """Get current weights for all sensors."""
        self._recalculate()
        return dict(self._weights)

    def report_sensor_status(self, sensor_type: SensorType, healthy: bool) -> None:
        """Report whether a sensor is healthy."""
        if sensor_type in self._weights:
            self._weights[sensor_type].healthy = healthy

    def report_nlos(self, sensor_type: SensorType, nlos_probability: float) -> None:
        """Report NLOS probability for a sensor (typically UWB)."""
        if sensor_type in self._weights:
            self._weights[sensor_type].nlos_probability = nlos_probability

    def set_environment(self, low_light: bool = False) -> None:
        """Set environmental conditions that affect sensor weights."""
        self._low_light = low_light

    def _recalculate(self) -> None:
        """Recalculate all weights based on current conditions."""
        for st, sw in self._weights.items():
            base = self._DEFAULT_WEIGHTS.get(st, 0.5)

            # Health penalty
            if not sw.healthy:
                base *= 0.1

            # NLOS penalty
            base *= (1.0 - sw.nlos_probability * 0.8)

            # Environmental adjustments
            if self._low_light and st == SensorType.CAMERA:
                base *= 0.3

            sw.weight = max(0.0, min(1.0, base))
