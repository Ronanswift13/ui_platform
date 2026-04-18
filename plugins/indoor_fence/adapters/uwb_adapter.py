"""
Indoor Fence V3.0 - UWB Adapter
================================

Ultra-Wideband positioning adapter supporting DW1000/DW3000 protocols.
Provides 3D position with NLOS detection and dynamic weight degradation.
"""
from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_adapter import BaseAdapter, AdapterConfig, AdapterStatus
from ..protocols import SensorData, SensorType, AdapterMode

logger = logging.getLogger(__name__)


@dataclass
class UWBConfig(AdapterConfig):
    """Configuration for UWB adapter."""
    adapter_type: str = "uwb"
    protocol: str = "simulate"  # udp, serial, dw3000, simulate
    host: str = "127.0.0.1"
    port: int = 5000
    tag_ids: List[str] = field(default_factory=lambda: ["tag_0"])
    simulate_nlos: bool = False
    nlos_ratio: float = 0.2  # fraction of readings with NLOS
    num_simulated_tags: int = 1
    noise_sigma_los: float = 0.1
    noise_sigma_nlos: float = 0.5


class UWBAdapter(BaseAdapter):
    """UWB positioning adapter with NLOS detection."""

    def __init__(self, config: UWBConfig):
        super().__init__(config)
        self._uwb_config = config
        self._connected = False
        self._nlos_history: deque = deque(maxlen=50)
        self._sim_positions: Dict[str, Dict[str, float]] = {}

        # Initialize simulated tag positions
        for i in range(config.num_simulated_tags):
            tag_id = f"tag_{i}"
            self._sim_positions[tag_id] = {
                "x": random.uniform(1.0, 9.0),
                "y": random.uniform(1.0, 5.0),
                "z": random.uniform(0.0, 0.3),
            }

    @property
    def sensor_type(self) -> SensorType:
        return SensorType.UWB

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Connect to UWB system."""
        if self._uwb_config.protocol == "simulate":
            self.set_mode(AdapterMode.SIMULATED)
            self._connected = True
            self._status = AdapterStatus.CONNECTED
            logger.info("UWB adapter connected (simulation mode)")
            return True

        # For real protocols (UDP, serial, DW3000), stub for now
        try:
            self._connected = True
            self._status = AdapterStatus.CONNECTED
            logger.info(f"UWB adapter connected via {self._uwb_config.protocol}")
            return True
        except Exception as e:
            logger.error(f"UWB connection failed: {e}")
            self._status = AdapterStatus.ERROR
            return False

    def disconnect(self) -> None:
        """Disconnect from UWB system."""
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        logger.info("UWB adapter disconnected")

    def _read_live(self) -> SensorData:
        """Read from real UWB hardware."""
        raise NotImplementedError(
            "Live UWB reading requires hardware-specific protocol implementation. "
            "Set protocol='simulate' for simulation mode."
        )

    def _read_simulated(self) -> SensorData:
        """Generate simulated UWB data."""
        ts = time.time()
        positions = []

        for tag_id, pos in self._sim_positions.items():
            is_nlos = (
                self._uwb_config.simulate_nlos
                and random.random() < self._uwb_config.nlos_ratio
            )

            sigma = (
                self._uwb_config.noise_sigma_nlos if is_nlos
                else self._uwb_config.noise_sigma_los
            )
            nlos_prob = random.uniform(0.6, 0.95) if is_nlos else random.uniform(0.0, 0.15)

            self._nlos_history.append(nlos_prob)

            # Drift simulated position slightly
            pos["x"] += random.gauss(0, 0.01)
            pos["y"] += random.gauss(0, 0.01)

            positions.append({
                "x": pos["x"] + random.gauss(0, sigma),
                "y": pos["y"] + random.gauss(0, sigma),
                "z": pos["z"] + random.gauss(0, sigma * 0.5),
                "tag_id": tag_id,
                "nlos_probability": nlos_prob,
                "timestamp": ts,
            })

        confidence = max(0.3, 1.0 - self._avg_nlos())
        return SensorData(
            sensor_type=SensorType.UWB,
            timestamp=ts,
            data={"positions": positions},
            confidence=confidence,
            is_simulated=True,
        )

    def get_weight(self) -> float:
        """Weight decreases with NLOS probability."""
        avg_nlos = self._avg_nlos()
        return max(0.1, 1.0 - avg_nlos)

    def _avg_nlos(self) -> float:
        if not self._nlos_history:
            return 0.0
        return sum(self._nlos_history) / len(self._nlos_history)
