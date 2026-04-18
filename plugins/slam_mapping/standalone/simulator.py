#!/usr/bin/env python3
"""Standalone SLAM simulation routes and isolated scene generator."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import Query
from fastapi.responses import JSONResponse

from plugins.slam_mapping.plugin import Plugin


@dataclass(frozen=True)
class DeviceSpec:
    device_id: str
    device_type: str
    label: str
    location: Tuple[float, float, float]
    size: Tuple[float, float, float]


@dataclass(frozen=True)
class MonitorSpec:
    point_id: str
    location: Tuple[float, float]
    initial_height: float
    settlement_mm: float


@dataclass(frozen=True)
class LineSpec:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    density: int


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    name: str
    summary: str
    station_area: str
    voltage_level: str
    focus: str
    bounds: Tuple[float, float, float, float]
    sensor_track: Tuple[Dict[str, float], ...]
    structures: Tuple[DeviceSpec, ...]
    monitors: Tuple[MonitorSpec, ...]
    overhead_lines: Tuple[LineSpec, ...]
    obstacle_size: Tuple[float, float, float]
    seed_offset: int
    base_noise: float = 0.035


class StandaloneSLAMSimulator:
    """Standalone-only simulator that uses an isolated SLAM plugin instance."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.scenarios = self._build_scenarios()
        self.current_scenario = self.scenarios["transformer_bay_220kv"]
        self.sim_plugin = None
        self.frame_index = 0
        self.latest_snapshot: Dict[str, Any] | None = None
        self.history: deque[Dict[str, float]] = deque(maxlen=18)
        self.load_scenario(self.current_scenario.scenario_id)

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": spec.scenario_id,
                "name": spec.name,
                "summary": spec.summary,
                "station_area": spec.station_area,
                "voltage_level": spec.voltage_level,
                "focus": spec.focus,
            }
            for spec in self.scenarios.values()
        ]

    def load_scenario(self, scenario_id: str) -> Dict[str, Any]:
        spec = self.scenarios.get(scenario_id)
        if spec is None:
            raise KeyError(f"Unknown scenario: {scenario_id}")

        self.current_scenario = spec
        self.frame_index = 0
        self.history.clear()
        self.sim_plugin = Plugin.create_standalone()

        for device in spec.structures:
            self.sim_plugin.register_device(
                device.device_id,
                {"x": device.location[0], "y": device.location[1], "z": device.location[2]},
                device.device_type,
                metadata={"label": device.label, "source": "standalone_simulator"},
            )

        for monitor in spec.monitors:
            self.sim_plugin.add_subsidence_monitor(
                monitor.point_id,
                monitor.location,
                monitor.initial_height,
            )

        self.latest_snapshot = self.step()
        return self.latest_snapshot

    def reset(self) -> Dict[str, Any]:
        return self.load_scenario(self.current_scenario.scenario_id)

    def status(self) -> Dict[str, Any]:
        if self.latest_snapshot is None:
            return self.load_scenario(self.current_scenario.scenario_id)
        return self.latest_snapshot

    def step(self) -> Dict[str, Any]:
        spec = self.current_scenario
        pose = spec.sensor_track[self.frame_index % len(spec.sensor_track)]
        world_points = self._compose_world_scene(spec, self.frame_index)
        sensor_points = self._world_to_sensor_frame(world_points, pose)
        result = self.sim_plugin.process_point_cloud(sensor_points)
        status = self.sim_plugin.get_status()

        self.history.append(
            {
                "frame_id": float(result.get("frame_id", self.frame_index + 1)),
                "processed_points": float(result.get("processed_points", 0)),
                "detected_objects": float(result.get("detected_objects", 0)),
                "registration_error": float(result.get("registration_error", 0.0) or 0.0),
            }
        )

        snapshot = {
            "success": True,
            "mode": "simulated",
            "runtime": {
                "mode": "simulated",
                "frame_index": self.frame_index,
                "updated_at": datetime.now().isoformat(),
                "isolation": "standalone_isolated_plugin",
                "description": (
                    "独立仿真链路仅使用 standalone 内部实例推演，不写回真实插件对象、总 UI 或工程部署链路。"
                ),
            },
            "scenario": {
                "id": spec.scenario_id,
                "name": spec.name,
                "summary": spec.summary,
                "station_area": spec.station_area,
                "voltage_level": spec.voltage_level,
                "focus": spec.focus,
            },
            "status": status,
            "latest_result": self._normalize_result(result),
            "devices": status.get("devices", {}),
            "subsidence": status.get("subsidence", {}),
            "alerts": result.get("subsidence_alerts", []),
            "point_cloud": {
                "sensor_points": self._sample_points(sensor_points, max_points=900),
            },
            "map_preview": self._build_map_preview(spec),
            "sensor_track": list(spec.sensor_track),
            "current_sensor_pose": pose,
            "history": list(self.history),
        }

        self.latest_snapshot = snapshot
        self.frame_index += 1
        return snapshot

    def _build_scenarios(self) -> Dict[str, ScenarioSpec]:
        transformer_track = (
            {"x": -14.0, "y": -8.0, "z": 2.4, "yaw": 0.15},
            {"x": -9.0, "y": -5.5, "z": 2.4, "yaw": 0.12},
            {"x": -4.0, "y": -3.0, "z": 2.4, "yaw": 0.05},
            {"x": 2.0, "y": -1.0, "z": 2.4, "yaw": 0.0},
            {"x": 8.0, "y": 1.5, "z": 2.4, "yaw": -0.08},
            {"x": 12.0, "y": 4.5, "z": 2.4, "yaw": -0.25},
            {"x": 8.0, "y": 8.0, "z": 2.4, "yaw": -0.75},
            {"x": 1.0, "y": 8.5, "z": 2.4, "yaw": -1.25},
            {"x": -6.5, "y": 6.0, "z": 2.4, "yaw": -1.75},
            {"x": -12.0, "y": 1.0, "z": 2.4, "yaw": -2.35},
        )

        gis_track = (
            {"x": -10.5, "y": -4.5, "z": 2.0, "yaw": 0.1},
            {"x": -7.0, "y": -2.6, "z": 2.0, "yaw": 0.08},
            {"x": -3.5, "y": -1.5, "z": 2.0, "yaw": 0.04},
            {"x": 0.5, "y": -0.5, "z": 2.0, "yaw": 0.0},
            {"x": 4.5, "y": 0.4, "z": 2.0, "yaw": -0.03},
            {"x": 8.0, "y": 1.6, "z": 2.0, "yaw": -0.1},
            {"x": 10.8, "y": 3.0, "z": 2.0, "yaw": -0.18},
            {"x": 8.5, "y": 4.8, "z": 2.0, "yaw": -1.9},
            {"x": 2.5, "y": 4.6, "z": 2.0, "yaw": -3.0},
            {"x": -4.0, "y": 3.5, "z": 2.0, "yaw": 2.85},
        )

        return {
            "transformer_bay_220kv": ScenarioSpec(
                scenario_id="transformer_bay_220kv",
                name="220kV主变场区巡检",
                summary="围绕主变、断路器、隔离开关与CT布置点云，验证建图与设备定位是否直观。",
                station_area="主变场区",
                voltage_level="220kV",
                focus="查看设备聚类、占据格扩张和低幅沉降监测的可视化是否够清楚。",
                bounds=(-18.0, 18.0, -12.0, 12.0),
                sensor_track=transformer_track,
                structures=(
                    DeviceSpec("T1", "transformer", "1号主变", (0.0, -1.0, 2.1), (4.8, 3.8, 4.2)),
                    DeviceSpec("BR1", "breaker", "主变断路器", (7.0, 4.2, 1.5), (1.6, 1.6, 3.0)),
                    DeviceSpec("DS1", "disconnector", "刀闸", (11.0, 4.2, 1.7), (1.4, 1.0, 2.4)),
                    DeviceSpec("CT1", "current_transformer", "电流互感器", (6.2, -4.5, 1.4), (1.2, 1.2, 2.6)),
                    DeviceSpec("FENCE", "barrier", "围栏", (-14.0, 0.0, 1.2), (0.4, 18.0, 2.4)),
                ),
                monitors=(
                    MonitorSpec("SUB-A", (2.5, -6.0), 0.0, 4.0),
                    MonitorSpec("SUB-B", (-7.0, 5.0), 0.0, 2.5),
                ),
                overhead_lines=(
                    LineSpec(start=(-10.0, 6.2, 6.0), end=(12.5, 6.2, 6.0), density=220),
                    LineSpec(start=(-8.5, -6.2, 5.0), end=(8.0, -6.2, 5.0), density=180),
                ),
                obstacle_size=(1.6, 0.9, 1.2),
                seed_offset=0,
            ),
            "gis_aisle_subsidence": ScenarioSpec(
                scenario_id="gis_aisle_subsidence",
                name="GIS走廊沉降复盘",
                summary="在GIS柜列与电缆沟附近叠加较明显沉降，便于先在仿真页面观察告警层级。",
                station_area="GIS走廊",
                voltage_level="110kV",
                focus="看沉降告警、站区走廊占据图和局部点云剖面是否足以支持后续算法修订。",
                bounds=(-14.0, 14.0, -8.0, 8.0),
                sensor_track=gis_track,
                structures=(
                    DeviceSpec("GIS-A", "gis_cabinet", "GIS柜A", (-6.0, 0.8, 1.6), (1.8, 3.2, 3.2)),
                    DeviceSpec("GIS-B", "gis_cabinet", "GIS柜B", (0.0, 0.6, 1.6), (1.8, 3.2, 3.2)),
                    DeviceSpec("GIS-C", "gis_cabinet", "GIS柜C", (6.0, 0.8, 1.6), (1.8, 3.2, 3.2)),
                    DeviceSpec("CABLE", "cable_trench", "电缆沟", (0.0, -3.8, 0.2), (11.0, 1.0, 0.4)),
                    DeviceSpec("WALL", "wall_panel", "围护墙", (0.0, 6.2, 1.8), (22.0, 0.4, 3.6)),
                ),
                monitors=(
                    MonitorSpec("SUB-G1", (1.0, -2.8), 0.0, 11.5),
                    MonitorSpec("SUB-G2", (-4.5, -2.8), 0.0, 6.5),
                ),
                overhead_lines=(
                    LineSpec(start=(-10.0, 2.5, 4.8), end=(10.0, 2.5, 4.8), density=220),
                ),
                obstacle_size=(1.2, 1.2, 1.4),
                seed_offset=100,
            ),
        }

    def _compose_world_scene(self, spec: ScenarioSpec, frame_index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + spec.seed_offset + frame_index)
        points = [self._sample_ground(spec, frame_index)]

        for structure in spec.structures:
            points.append(
                self._sample_box_surface(
                    np.array(structure.location),
                    np.array(structure.size),
                    count=260 if structure.device_type != "barrier" else 180,
                    rng=rng,
                )
            )

        for line in spec.overhead_lines:
            points.append(
                self._sample_line(
                    np.array(line.start),
                    np.array(line.end),
                    density=line.density,
                    rng=rng,
                    jitter=0.05,
                )
            )

        points.append(self._sample_dynamic_obstacle(spec, frame_index, rng))
        return np.vstack(points)

    def _sample_ground(self, spec: ScenarioSpec, frame_index: int) -> np.ndarray:
        xmin, xmax, ymin, ymax = spec.bounds
        xs = np.linspace(xmin, xmax, 54)
        ys = np.linspace(ymin, ymax, 36)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.zeros_like(xx)

        progress = min(1.0, frame_index / max(len(spec.sensor_track) * 0.7, 1))
        for monitor in spec.monitors:
            depth = (monitor.settlement_mm / 1000.0) * progress
            dist2 = (xx - monitor.location[0]) ** 2 + (yy - monitor.location[1]) ** 2
            zz -= depth * np.exp(-dist2 / 4.0)

        ground = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        ground += np.random.default_rng(self.seed + spec.seed_offset + frame_index + 500).normal(
            scale=[0.03, 0.03, 0.01],
            size=ground.shape,
        )
        return ground

    def _sample_box_surface(
        self,
        center: np.ndarray,
        size: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        half = size / 2.0
        per_face = [count // 6] * 6
        per_face[0] += count - sum(per_face)
        faces = []
        face_index = 0

        for axis in range(3):
            for sign in (-1.0, 1.0):
                num = per_face[face_index]
                local = rng.uniform(-1.0, 1.0, size=(num, 3)) * half
                local[:, axis] = sign * half[axis]
                faces.append(center + local)
                face_index += 1

        return np.vstack(faces)

    def _sample_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        density: int,
        rng: np.random.Generator,
        jitter: float,
    ) -> np.ndarray:
        t = np.linspace(0.0, 1.0, density)[:, None]
        line = start + (end - start) * t
        line += rng.normal(scale=jitter, size=line.shape)
        return line

    def _sample_dynamic_obstacle(
        self,
        spec: ScenarioSpec,
        frame_index: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        phase = (frame_index % 20) / 20.0
        xmin, xmax, ymin, ymax = spec.bounds
        center = np.array(
            [
                xmin * 0.2 + (xmax - xmin) * 0.45 * phase,
                ymax * 0.35 - 1.5 * np.sin(phase * np.pi * 2.0),
                spec.obstacle_size[2] / 2.0,
            ]
        )
        return self._sample_box_surface(center, np.array(spec.obstacle_size), 120, rng)

    def _world_to_sensor_frame(self, points: np.ndarray, pose: Dict[str, float]) -> np.ndarray:
        translation = np.array([pose["x"], pose["y"], pose["z"]], dtype=float)
        yaw = float(pose.get("yaw", 0.0))
        shifted = points - translation
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        sensor_points = np.column_stack(
            [
                shifted[:, 0] * cos_yaw + shifted[:, 1] * sin_yaw,
                -shifted[:, 0] * sin_yaw + shifted[:, 1] * cos_yaw,
                shifted[:, 2],
            ]
        )
        sensor_points += np.random.default_rng(
            self.seed + self.current_scenario.seed_offset + self.frame_index + 900
        ).normal(scale=self.current_scenario.base_noise, size=sensor_points.shape)
        return sensor_points.astype(float)

    def _build_map_preview(self, spec: ScenarioSpec) -> Dict[str, Any]:
        map_data = self.sim_plugin.get_map_data("2d")
        grid = np.array(map_data["data"], dtype=float)
        sampled = grid[::3, ::3]
        sampled = np.clip(sampled, 0.0, 1.0)
        xmin, xmax, ymin, ymax = spec.bounds
        return {
            "grid": np.round(sampled, 3).tolist(),
            "origin": map_data["origin"],
            "resolution": map_data["resolution"] * 3,
            "world_bounds": {
                "min_x": xmin,
                "max_x": xmax,
                "min_y": ymin,
                "max_y": ymax,
            },
        }

    def _sample_points(self, points: np.ndarray, max_points: int) -> List[List[float]]:
        if len(points) <= max_points:
            selected = points
        else:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            selected = points[indices]
        return np.round(selected[:, :3], 3).tolist()

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frame_id": result.get("frame_id", 0),
            "processed_points": result.get("processed_points", 0),
            "ground_points": result.get("ground_points", 0),
            "object_points": result.get("object_points", 0),
            "detected_objects": result.get("detected_objects", 0),
            "registration_error": result.get("registration_error"),
            "success": result.get("success", False),
        }


def register_simulation_routes(app) -> StandaloneSLAMSimulator:
    """Register standalone-only simulation routes on a runner app."""
    simulator = StandaloneSLAMSimulator(seed=42)

    @app.get("/api/simulator/scenarios")
    async def list_scenarios():
        return JSONResponse({"success": True, "scenarios": simulator.list_scenarios()})

    @app.post("/api/simulator/load")
    async def load_scenario(scenario_id: str = Query(...)):
        try:
            return JSONResponse(simulator.load_scenario(scenario_id))
        except KeyError:
            return JSONResponse(
                {"success": False, "error": f"场景 '{scenario_id}' 不存在"},
                status_code=404,
            )

    @app.post("/api/simulator/step")
    async def step_simulation():
        return JSONResponse(simulator.step())

    @app.get("/api/simulator/status")
    async def get_simulation_status():
        return JSONResponse(simulator.status())

    @app.post("/api/simulator/reset")
    async def reset_simulation():
        return JSONResponse(simulator.reset())

    return simulator
