"""
Indoor Fence V3.0 - Unified Multi-Sensor Simulator
====================================================

Generates synchronized simulated sensor data for all configured
sensor types. Supports configurable person paths, fault injection,
and scenario loading from JSON files.
"""
from __future__ import annotations

import json
import math
import random
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..protocols import SensorData, SensorType


class PathType(str, Enum):
    RANDOM = "random"
    FIXED = "fixed"
    CROSS_LINE = "cross_line"
    CLIMBING = "climbing"


class SimulatorConfig(BaseModel):
    """Configuration for unified simulator."""
    num_persons: int = 1
    scene_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 6.0)
    sensor_types: List[SensorType] = Field(default_factory=lambda: [SensorType.CAMERA])
    path_type: PathType = PathType.RANDOM
    waypoints: Optional[List[List[float]]] = None
    speed_m_per_step: float = 0.1
    yellow_line_y: float = 3.0
    loop: bool = True
    noise_sigma: float = 0.05


class PersonPath:
    """Tracks a simulated person's position over time."""

    def __init__(
        self,
        person_id: int,
        bounds: Tuple[float, float, float, float],
        path_type: PathType,
        speed: float = 0.1,
        waypoints: Optional[List[List[float]]] = None,
        yellow_line_y: float = 3.0,
    ):
        self.person_id = person_id
        self.bounds = bounds
        self.path_type = path_type
        self.speed = speed
        self.yellow_line_y = yellow_line_y

        x_min, y_min, x_max, y_max = bounds
        if path_type == PathType.FIXED and waypoints:
            self.waypoints = waypoints
            self._wp_idx = 0
            self.x, self.y = waypoints[0]
        elif path_type == PathType.CROSS_LINE:
            self.x = random.uniform(x_min + 1, x_max - 1)
            self.y = y_min + 0.5
            self._direction = 1.0
        elif path_type == PathType.CLIMBING:
            self.x = random.uniform(x_min + 1, x_max - 1)
            self.y = random.uniform(y_min + 1, y_max - 1)
            self.z = 0.0
        else:
            self.x = random.uniform(x_min + 1, x_max - 1)
            self.y = random.uniform(y_min + 1, y_max - 1)
            self._vx = random.uniform(-0.2, 0.2)
            self._vy = random.uniform(-0.2, 0.2)

        if not hasattr(self, 'z'):
            self.z = 0.0
        if not hasattr(self, 'waypoints'):
            self.waypoints = None
        self._prev_x = self.x
        self._prev_y = self.y

    def step(self) -> Tuple[float, float, float]:
        """Advance one step, return (x, y, z)."""
        self._prev_x = self.x
        self._prev_y = self.y
        x_min, y_min, x_max, y_max = self.bounds

        if self.path_type == PathType.FIXED and self.waypoints:
            tx, ty = self.waypoints[self._wp_idx]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.speed:
                self.x, self.y = tx, ty
                self._wp_idx = (self._wp_idx + 1) % len(self.waypoints)
            else:
                self.x += self.speed * dx / dist
                self.y += self.speed * dy / dist

        elif self.path_type == PathType.CROSS_LINE:
            self.y += self.speed * self._direction
            if self.y > y_max - 0.5:
                self._direction = -1.0
            elif self.y < y_min + 0.5:
                self._direction = 1.0

        elif self.path_type == PathType.CLIMBING:
            self.z += self.speed * 0.3
            self.x += random.gauss(0, 0.02)
            self.y += random.gauss(0, 0.02)

        else:  # RANDOM
            self.x += self._vx + random.gauss(0, 0.02)
            self.y += self._vy + random.gauss(0, 0.02)
            self._vx += random.gauss(0, 0.01)
            self._vy += random.gauss(0, 0.01)
            self._vx = max(-0.3, min(0.3, self._vx))
            self._vy = max(-0.3, min(0.3, self._vy))
            self.x = max(x_min + 0.2, min(x_max - 0.2, self.x))
            self.y = max(y_min + 0.2, min(y_max - 0.2, self.y))

        return (self.x, self.y, self.z)


class Simulator:
    """Unified multi-sensor simulator."""

    def __init__(self, config: SimulatorConfig):
        self._config = config
        self._persons: List[PersonPath] = []
        self._faults: Dict[SensorType, str] = {}
        self._step_count = 0

        for i in range(config.num_persons):
            self._persons.append(PersonPath(
                person_id=i,
                bounds=config.scene_bounds,
                path_type=config.path_type,
                speed=config.speed_m_per_step,
                waypoints=config.waypoints,
                yellow_line_y=config.yellow_line_y,
            ))

    @property
    def num_persons(self) -> int:
        return len(self._persons)

    def step(self) -> Dict[SensorType, SensorData]:
        """Advance simulation one step, return sensor data for all types."""
        self._step_count += 1
        ts = time.time()

        positions = []
        for p in self._persons:
            x, y, z = p.step()
            positions.append({"id": p.person_id, "x": x, "y": y, "z": z})

        result: Dict[SensorType, SensorData] = {}
        for st in self._config.sensor_types:
            if st in self._faults and self._faults[st] == "offline":
                result[st] = SensorData(
                    sensor_type=st,
                    timestamp=ts,
                    data={"detections": [], "positions": [], "clusters": []},
                    confidence=0.0,
                    is_simulated=True,
                )
            else:
                result[st] = self._generate_sensor_data(st, positions, ts)

        return result

    def _generate_sensor_data(
        self, sensor_type: SensorType, positions: List[Dict], ts: float
    ) -> SensorData:
        sigma = self._config.noise_sigma

        if sensor_type == SensorType.CAMERA:
            detections = []
            for p in positions:
                nx = p["x"] + random.gauss(0, sigma)
                ny = p["y"] + random.gauss(0, sigma)
                cx = int(nx * 64)
                cy = int(ny * 80)
                w, h = random.randint(40, 80), random.randint(100, 200)
                detections.append({
                    "id": p["id"],
                    "position": [nx, ny],
                    "bbox": [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
                    "foot_point": [cx, cy + h // 2],
                    "confidence": random.uniform(0.75, 0.98),
                })
            return SensorData(
                sensor_type=SensorType.CAMERA,
                timestamp=ts,
                data={"detections": detections},
                confidence=random.uniform(0.8, 0.95),
                is_simulated=True,
            )

        elif sensor_type == SensorType.LIDAR:
            clusters = []
            for p in positions:
                nx = p["x"] + random.gauss(0, sigma * 0.5)
                ny = p["y"] + random.gauss(0, sigma * 0.5)
                dist = math.sqrt(nx * nx + ny * ny)
                angle = math.atan2(ny, nx)
                clusters.append({
                    "id": p["id"],
                    "position_2d": [nx, ny],
                    "distance": dist,
                    "angle_rad": angle,
                    "num_points": random.randint(5, 30),
                })
            return SensorData(
                sensor_type=SensorType.LIDAR,
                timestamp=ts,
                data={"clusters": clusters},
                confidence=random.uniform(0.85, 0.98),
                is_simulated=True,
            )

        elif sensor_type == SensorType.UWB:
            uwb_positions = []
            for p in positions:
                nx = p["x"] + random.gauss(0, 0.1)
                ny = p["y"] + random.gauss(0, 0.1)
                nz = p["z"] + random.gauss(0, 0.05)
                uwb_positions.append({
                    "id": p["id"],
                    "x": nx, "y": ny, "z": nz,
                    "tag_id": f"tag_{p['id']}",
                    "nlos_probability": random.uniform(0.0, 0.2),
                    "timestamp": ts,
                })
            return SensorData(
                sensor_type=SensorType.UWB,
                timestamp=ts,
                data={"positions": uwb_positions},
                confidence=random.uniform(0.8, 0.95),
                is_simulated=True,
            )

        elif sensor_type == SensorType.IMU:
            return SensorData(
                sensor_type=SensorType.IMU,
                timestamp=ts,
                data={
                    "acceleration": {
                        "x": random.gauss(0, 0.05),
                        "y": random.gauss(0, 0.05),
                        "z": 9.81 + random.gauss(0, 0.05),
                    },
                    "gyroscope": {
                        "x": random.gauss(0, 0.01),
                        "y": random.gauss(0, 0.01),
                        "z": random.gauss(0, 0.01),
                    },
                },
                confidence=0.95,
                is_simulated=True,
            )

        else:
            return SensorData(
                sensor_type=sensor_type,
                timestamp=ts,
                data={},
                confidence=0.5,
                is_simulated=True,
            )

    def inject_fault(self, sensor_type: SensorType, fault_type: str) -> None:
        """Inject a fault for a sensor type (e.g., 'offline')."""
        self._faults[sensor_type] = fault_type

    def clear_fault(self, sensor_type: SensorType) -> None:
        """Clear fault for a sensor type."""
        self._faults.pop(sensor_type, None)

    @classmethod
    def default(cls) -> "Simulator":
        """Create a default simulator."""
        return cls(SimulatorConfig(
            num_persons=1,
            sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
        ))

    @classmethod
    def from_scenario(cls, scenario_path: str) -> "Simulator":
        """Load simulator from a JSON scenario file."""
        with open(scenario_path) as f:
            data = json.load(f)

        sensor_types = [SensorType(s) for s in data.get("sensor_types", ["camera"])]
        path_type = PathType(data.get("path_type", "random"))
        bounds = tuple(data.get("scene_bounds", [0.0, 0.0, 10.0, 6.0]))

        config = SimulatorConfig(
            num_persons=data.get("num_persons", 1),
            scene_bounds=bounds,
            sensor_types=sensor_types,
            path_type=path_type,
            waypoints=data.get("waypoints"),
            speed_m_per_step=data.get("speed_m_per_step", 0.1),
            loop=data.get("loop", True),
        )
        return cls(config)
