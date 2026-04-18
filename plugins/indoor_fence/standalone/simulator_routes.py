"""
Indoor Fence V3.0 - Simulator API Routes
==========================================

FastAPI routes for controlling the multi-sensor simulator.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import APIRouter

from ..adapters.simulator import Simulator
from ..protocols import SensorType


def create_simulator_router(simulator: Optional[Simulator] = None) -> APIRouter:
    """Create FastAPI router for simulator control.

    Args:
        simulator: Optional pre-configured simulator instance.
    """
    router = APIRouter(tags=["simulator"])
    _state: Dict = {"simulator": simulator}

    @router.post("/start")
    def start_simulator(
        num_persons: int = 1,
        sensor_types: Optional[List[str]] = None,
    ) -> Dict:
        sensor_list = sensor_types or ["camera"]
        st_list = [SensorType(s) for s in sensor_list]
        _state["simulator"] = Simulator.default()
        return {"status": "started", "num_persons": num_persons, "sensors": sensor_list}

    @router.post("/step")
    def step_simulator() -> Dict:
        sim = _state.get("simulator")
        if sim is None:
            return {"error": "Simulator not started", "data": {}}
        data = sim.step()
        # Serialize sensor data
        result = {}
        for st, sd in data.items():
            result[st.value] = sd.model_dump()
        return {"data": result}

    @router.get("/scenarios")
    def list_scenarios() -> Dict:
        scenarios_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "scenarios"
        )
        scenarios = []
        if os.path.exists(scenarios_dir):
            for f in os.listdir(scenarios_dir):
                if f.endswith(".json"):
                    scenarios.append(f.replace(".json", ""))
        return {"scenarios": scenarios}

    @router.post("/stop")
    def stop_simulator() -> Dict:
        _state["simulator"] = None
        return {"status": "stopped"}

    return router
