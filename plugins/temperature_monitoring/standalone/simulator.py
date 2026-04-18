"""Standalone temperature simulation routes and scene generator.

The simulator is scoped to the standalone dashboard only. It generates
substation-like thermal/sensor inputs and feeds them into a dedicated plugin
instance so demo playback never mutates the runner plugin's production state.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import HTTPException


@dataclass(frozen=True)
class SimulationEquipment:
    label: str
    zone_id: str
    zone_name: str
    center: Tuple[float, float]
    sigma: Tuple[float, float]
    target_temp: float
    oscillation: float
    role: str


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    name: str
    description: str
    input_mode: str
    expected_status: str
    ambient_temp: float
    load_pct: int
    cycle_hint: str
    guidance: str
    equipments: Tuple[SimulationEquipment, ...]


class StandaloneTemperatureSimulator:
    """Generate deterministic standalone demo scenes for the dashboard."""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        sensor_cfg = config.get("sensor", {})
        resolution = sensor_cfg.get("resolution", [8, 6])
        upsample_size = sensor_cfg.get("upsample_size", [64, 48])

        self._config = config
        self._zones = config.get("zones", [])
        self._sensor_cols = int(resolution[0])
        self._sensor_rows = int(resolution[1])
        self._heatmap_width = int(upsample_size[0])
        self._heatmap_height = int(upsample_size[1])
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._scenarios = self._build_scenarios(config)
        self._current_scenario_id = next(iter(self._scenarios))

    def list_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = []
        for scenario in self._scenarios.values():
            scenarios.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "name": scenario.name,
                    "description": scenario.description,
                    "input_mode": scenario.input_mode,
                    "expected_status": scenario.expected_status,
                    "ambient_temp": scenario.ambient_temp,
                    "load_pct": scenario.load_pct,
                    "cycle_hint": scenario.cycle_hint,
                }
            )
        return scenarios

    def current_scenario(self) -> Dict[str, Any]:
        scenario = self._scenarios[self._current_scenario_id]
        return {
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "description": scenario.description,
            "input_mode": scenario.input_mode,
            "expected_status": scenario.expected_status,
            "ambient_temp": scenario.ambient_temp,
            "load_pct": scenario.load_pct,
            "cycle_hint": scenario.cycle_hint,
            "guidance": scenario.guidance,
        }

    def load_scenario(self, scenario_id: str) -> bool:
        if scenario_id not in self._scenarios:
            return False
        self._current_scenario_id = scenario_id
        self._step_count = 0
        return True

    def step(self) -> Dict[str, Any]:
        self._step_count += 1
        scenario = self._scenarios[self._current_scenario_id]
        thermal_frame = self._generate_heatmap(scenario)
        sensor_readings = self._sample_sensor_grid(thermal_frame)
        equipment_states = self._build_equipment_states(scenario, thermal_frame)

        scene = {
            "ambient_temp": round(float(scenario.ambient_temp), 1),
            "load_pct": scenario.load_pct,
            "expected_status": scenario.expected_status,
            "description": scenario.description,
            "guidance": scenario.guidance,
            "zones": self._serialize_zones(),
            "equipment_states": equipment_states,
        }

        return {
            "thermal_frame": thermal_frame,
            "sensor_readings": sensor_readings,
            "sensor_snapshot": equipment_states,
            "scene": scene,
            "simulation": {
                "mode": "simulation",
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "input_mode": scenario.input_mode,
                "expected_status": scenario.expected_status,
                "cycle_hint": scenario.cycle_hint,
                "tick": self._step_count,
                "detector_fallback": False,
            },
        }

    def _generate_heatmap(self, scenario: SimulationScenario) -> np.ndarray:
        x_axis = np.linspace(0.0, 1.0, self._heatmap_width, dtype=np.float32)
        y_axis = np.linspace(0.0, 1.0, self._heatmap_height, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)

        tick = float(self._step_count)
        base = np.full((self._heatmap_height, self._heatmap_width), scenario.ambient_temp, dtype=np.float32)
        base += 0.8 * np.sin((x_grid * math.pi * 4.0) + tick / 3.5).astype(np.float32)
        base += 0.6 * np.cos((y_grid * math.pi * 3.0) + tick / 5.0).astype(np.float32)
        base += (0.5 - np.abs(x_grid - 0.5)) * 1.3
        base += self._rng.normal(0.0, 0.18, base.shape).astype(np.float32)

        # Cable trench stays slightly warmer near the bottom to mimic cabinet airflow.
        base += np.clip((y_grid - 0.62) * 12.0, 0.0, 3.5).astype(np.float32)

        for index, equipment in enumerate(scenario.equipments):
            cx, cy = equipment.center
            sx, sy = equipment.sigma
            envelope = np.exp(
                -(
                    (((x_grid - cx) ** 2) / max(sx ** 2, 1e-4))
                    + (((y_grid - cy) ** 2) / max(sy ** 2, 1e-4))
                )
            )
            pulse = math.sin((tick / 3.0) + index * 0.7) * equipment.oscillation
            amplitude = max(equipment.target_temp - scenario.ambient_temp + pulse, 0.0)
            base += (amplitude * envelope).astype(np.float32)

        return np.clip(base, 18.0, 110.0).astype(np.float32)

    def _sample_sensor_grid(self, heatmap: np.ndarray) -> List[Dict[str, Any]]:
        readings: List[Dict[str, Any]] = []
        ts = time.time()

        for row in range(self._sensor_rows):
            for col in range(self._sensor_cols):
                x_pos = round((col / max(self._sensor_cols - 1, 1)) * (self._heatmap_width - 1))
                y_pos = round((row / max(self._sensor_rows - 1, 1)) * (self._heatmap_height - 1))
                temp = float(heatmap[y_pos, x_pos] + self._rng.normal(0.0, 0.12))
                cx = x_pos / max(self._heatmap_width - 1, 1)
                cy = y_pos / max(self._heatmap_height - 1, 1)
                zone_id, zone_name = self._match_zone(cx, cy)
                readings.append(
                    {
                        "sensor_id": f"sensor_{row}_{col}",
                        "position": {"row": row, "col": col},
                        "temperature": round(temp, 1),
                        "timestamp": ts,
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                    }
                )

        return readings

    def _build_equipment_states(
        self, scenario: SimulationScenario, heatmap: np.ndarray
    ) -> List[Dict[str, Any]]:
        states = []
        for equipment in scenario.equipments:
            temp = self._sample_temperature(heatmap, equipment.center)
            states.append(
                {
                    "label": equipment.label,
                    "zone_id": equipment.zone_id,
                    "zone_name": equipment.zone_name,
                    "role": equipment.role,
                    "temperature": round(temp, 1),
                    "status": self._classify_temperature(temp),
                }
            )
        return states

    def _serialize_zones(self) -> List[Dict[str, Any]]:
        serialized = []
        for zone in self._zones:
            region = zone.get("region", [0.0, 0.0, 1.0, 1.0])
            if len(region) != 4:
                continue
            serialized.append(
                {
                    "zone_id": zone.get("zone_id", ""),
                    "zone_name": zone.get("zone_name", zone.get("zone_id", "")),
                    "region": [float(value) for value in region],
                }
            )
        return serialized

    def _sample_temperature(self, heatmap: np.ndarray, center: Tuple[float, float]) -> float:
        x = min(self._heatmap_width - 1, max(0, round(center[0] * (self._heatmap_width - 1))))
        y = min(self._heatmap_height - 1, max(0, round(center[1] * (self._heatmap_height - 1))))
        return float(heatmap[y, x])

    def _classify_temperature(self, temp: float) -> str:
        thresholds = self._config.get("thresholds", {})
        if temp >= thresholds.get("critical", 85):
            return "critical"
        if temp >= thresholds.get("alarm", 70):
            return "alarm"
        if temp >= thresholds.get("warning", 55):
            return "warning"
        return "normal"

    def _match_zone(self, cx: float, cy: float) -> Tuple[str, str]:
        for zone in self._zones:
            region = zone.get("region", [0.0, 0.0, 1.0, 1.0])
            if len(region) != 4:
                continue
            if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                return zone.get("zone_id", ""), zone.get("zone_name", zone.get("zone_id", ""))
        return "", ""

    @staticmethod
    def _build_scenarios(config: Dict[str, Any]) -> Dict[str, SimulationScenario]:
        thresholds = config.get("thresholds", {})
        normal_max = float(thresholds.get("normal_max", 45))
        warning = float(thresholds.get("warning", 55))
        alarm = float(thresholds.get("alarm", 70))
        critical = float(thresholds.get("critical", 85))

        cabinet_top = ("zone_cabinet_top", "机柜顶部")
        cable_zone = ("zone_cable", "电缆区")

        return {
            "night_patrol": SimulationScenario(
                scenario_id="night_patrol",
                name="夜间常规巡检",
                description="站内负荷平稳，机柜顶部与电缆区保持常态温升，用于检查页面基础刷新与趋势稳定性。",
                input_mode="thermal_frame",
                expected_status="normal",
                ambient_temp=28.0,
                load_pct=46,
                cycle_hint="低负荷 / 热成像巡检",
                guidance="适合验证独立页默认启动、热力图绘制与正常态趋势。",
                equipments=(
                    SimulationEquipment("A相母排接头", cabinet_top[0], cabinet_top[1], (0.34, 0.18), (0.08, 0.05), normal_max - 3.0, 0.7, "母排"),
                    SimulationEquipment("B相母排接头", cabinet_top[0], cabinet_top[1], (0.50, 0.16), (0.09, 0.05), normal_max - 2.0, 0.6, "母排"),
                    SimulationEquipment("电缆桥架入口", cable_zone[0], cable_zone[1], (0.60, 0.82), (0.11, 0.08), normal_max - 6.0, 0.8, "电缆"),
                ),
            ),
            "joint_overheat": SimulationScenario(
                scenario_id="joint_overheat",
                name="母排接头过热",
                description="主变高负荷时，机柜顶部接头温度快速抬升，模拟真实接触电阻异常导致的局部热点。",
                input_mode="thermal_frame",
                expected_status="alarm",
                ambient_temp=31.5,
                load_pct=84,
                cycle_hint="高负荷 / 热成像热点识别",
                guidance="适合评估热点列表、告警渲染与母排区域的可视化直观性。",
                equipments=(
                    SimulationEquipment("A相母排接头", cabinet_top[0], cabinet_top[1], (0.37, 0.18), (0.08, 0.05), warning - 2.0, 1.1, "母排"),
                    SimulationEquipment("B相母排接头", cabinet_top[0], cabinet_top[1], (0.53, 0.16), (0.08, 0.05), alarm + 5.0, 1.8, "母排"),
                    SimulationEquipment("电缆桥架入口", cable_zone[0], cable_zone[1], (0.62, 0.82), (0.10, 0.07), warning - 5.0, 0.9, "电缆"),
                ),
            ),
            "cable_trench_rise": SimulationScenario(
                scenario_id="cable_trench_rise",
                name="电缆夹层温升",
                description="电缆区负荷连续拉升，传感器阵列先于热成像暴露出底部区域的温升趋势。",
                input_mode="sensor_readings",
                expected_status="warning",
                ambient_temp=30.0,
                load_pct=72,
                cycle_hint="持续负荷 / 传感器阵列监测",
                guidance="适合验证传感器插值路径、趋势变化与电缆区区域匹配。",
                equipments=(
                    SimulationEquipment("电缆终端 A", cable_zone[0], cable_zone[1], (0.34, 0.80), (0.10, 0.08), warning + 3.0, 1.4, "电缆"),
                    SimulationEquipment("电缆终端 B", cable_zone[0], cable_zone[1], (0.52, 0.84), (0.12, 0.08), warning + 6.0, 1.5, "电缆"),
                    SimulationEquipment("散热风道回风口", cabinet_top[0], cabinet_top[1], (0.78, 0.24), (0.10, 0.07), normal_max - 7.0, 0.5, "通风"),
                ),
            ),
            "ventilation_failure": SimulationScenario(
                scenario_id="ventilation_failure",
                name="通风失效冲高",
                description="散热风道失效后，机柜顶部与电缆区同时积热，形成接近紧急态的站内热场。",
                input_mode="thermal_frame",
                expected_status="critical",
                ambient_temp=36.0,
                load_pct=93,
                cycle_hint="散热故障 / 站内联动态势",
                guidance="适合评估紧急状态、联动事件和极端热场的视觉表达。",
                equipments=(
                    SimulationEquipment("B相母排接头", cabinet_top[0], cabinet_top[1], (0.46, 0.17), (0.10, 0.05), critical + 3.0, 2.0, "母排"),
                    SimulationEquipment("电缆桥架出口", cable_zone[0], cable_zone[1], (0.58, 0.80), (0.12, 0.08), alarm + 7.0, 1.7, "电缆"),
                    SimulationEquipment("散热风道回风口", cabinet_top[0], cabinet_top[1], (0.76, 0.22), (0.12, 0.07), warning + 5.0, 1.2, "通风"),
                ),
            ),
        }


def build_standalone_routes(plugin: Any) -> List[Dict[str, Any]]:
    """Build standalone-only simulation routes expected by the runner."""
    simulator = StandaloneTemperatureSimulator(plugin.config)
    simulation_plugin = plugin.__class__.create_standalone(config=copy.deepcopy(plugin.config))

    def list_scenarios() -> Dict[str, Any]:
        return {
            "success": True,
            "current_scenario_id": simulator.current_scenario()["scenario_id"],
            "scenarios": simulator.list_scenarios(),
        }

    def get_state() -> Dict[str, Any]:
        current = simulator.current_scenario()
        return {
            "success": True,
            "scenario": current,
            "zones": simulator._serialize_zones(),
        }

    def bootstrap() -> Dict[str, Any]:
        current = simulator.current_scenario()
        return {
            "success": True,
            "scenario": current,
            "zones": simulator._serialize_zones(),
            "current_scenario_id": current["scenario_id"],
            "scenarios": simulator.list_scenarios(),
            "initial_data": step_simulation(),
        }

    def load_scenario(scenario_id: str) -> Dict[str, Any]:
        if not simulator.load_scenario(scenario_id):
            raise HTTPException(status_code=404, detail=f"unknown scenario: {scenario_id}")
        if getattr(simulation_plugin, "_detector", None) is not None:
            simulation_plugin._detector.reset()
        return {"success": True, "scenario": simulator.current_scenario()}

    def step_simulation() -> Dict[str, Any]:
        payload = simulator.step()
        detect_kwargs = {
            "context": {"task_id": f"standalone-sim-{payload['simulation']['scenario_id']}-{payload['simulation']['tick']}"},
        }
        if payload["simulation"]["input_mode"] == "sensor_readings":
            detect_kwargs["sensor_readings"] = payload["sensor_readings"]
        else:
            detect_kwargs["thermal_frame"] = payload["thermal_frame"]

        result = simulation_plugin.detect(**detect_kwargs)
        result["simulation"] = payload["simulation"]
        result["substation_scene"] = payload["scene"]
        result["sensor_snapshot"] = payload["sensor_snapshot"]
        return result

    return [
        {
            "path": "/api/simulation/bootstrap",
            "endpoint": bootstrap,
            "methods": ["GET"],
            "name": "temperature_monitoring_simulation_bootstrap",
        },
        {
            "path": "/api/simulation/scenarios",
            "endpoint": list_scenarios,
            "methods": ["GET"],
            "name": "temperature_monitoring_simulation_scenarios",
        },
        {
            "path": "/api/simulation/state",
            "endpoint": get_state,
            "methods": ["GET"],
            "name": "temperature_monitoring_simulation_state",
        },
        {
            "path": "/api/simulation/load",
            "endpoint": load_scenario,
            "methods": ["POST"],
            "name": "temperature_monitoring_simulation_load",
        },
        {
            "path": "/api/simulation/step",
            "endpoint": step_simulation,
            "methods": ["POST"],
            "name": "temperature_monitoring_simulation_step",
        },
    ]
