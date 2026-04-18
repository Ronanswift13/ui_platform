#!/usr/bin/env python3
"""
鸟类监控 — Standalone Application

启动独立运行的鸟类监控仪表板，包含:
- 真实检测接口 (通过 StandalonePluginRunner /api/detect，runtime_mode=simulation 时返回 no_bird)
- 仿真演示接口 (通过 BirdMonitoringSimulator，与生产算法完全隔离)

边界:
- 生产 detect 路由 /api/detect 调用 plugin.infer，runtime_mode 由 detector 决定。
- 仿真路由全部挂在 /api/simulator/*，所有响应字段显式标记 runtime_mode="standalone_simulation"。
- 仿真器不依赖 plugin.py / detector.py，也不会被生产推理链调用。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Query
from fastapi.responses import JSONResponse, StreamingResponse

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from darkbreaker_sdk.standalone import StandalonePluginRunner
from plugins.bird_monitoring.plugin import BirdMonitoringPlugin as Plugin
from plugins.bird_monitoring.standalone.bird_simulator import (
    BirdMonitoringSimulator,
    RUNTIME_MODE as SIMULATION_RUNTIME_MODE,
)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False


# 模块级仿真器，仅供 standalone 使用
_simulator = BirdMonitoringSimulator(seed=2026)


def _register_simulation_routes(app) -> None:
    """挂载演示仿真路由 — 与生产 /api/detect 完全隔离。"""

    @app.get("/api/simulator/scenarios")
    async def list_scenarios() -> JSONResponse:
        return JSONResponse({
            "runtime_mode": SIMULATION_RUNTIME_MODE,
            "scenarios": _simulator.list_scenarios(),
        })

    @app.post("/api/simulator/load")
    async def load_scenario(scenario_id: str = Query(...)) -> JSONResponse:
        ok = _simulator.load_scenario(scenario_id)
        if not ok:
            return JSONResponse(
                {"error": f"未知场景: {scenario_id}", "runtime_mode": SIMULATION_RUNTIME_MODE},
                status_code=404,
            )
        return JSONResponse({
            "status": "loaded",
            "runtime_mode": SIMULATION_RUNTIME_MODE,
            "scenario": _simulator.get_current_scenario(),
        })

    @app.post("/api/simulator/step")
    async def step_simulation() -> JSONResponse:
        payload = _simulator.step()
        return JSONResponse(payload)

    @app.get("/api/simulator/status")
    async def simulator_status() -> JSONResponse:
        return JSONResponse({
            "runtime_mode": SIMULATION_RUNTIME_MODE,
            "frame_counter": _simulator._frame_counter,
            "current_scenario": _simulator.get_current_scenario(),
        })

    @app.get("/api/simulator/frame")
    async def simulator_frame() -> Any:
        if not CV2_AVAILABLE:
            return JSONResponse(
                {
                    "error": "cv2 未安装，无法生成仿真帧。",
                    "runtime_mode": SIMULATION_RUNTIME_MODE,
                },
                status_code=503,
            )
        frame = _simulator.render_frame()
        jpeg = _simulator.encode_jpeg(frame)
        if jpeg is None:
            return JSONResponse(
                {"error": "JPEG 编码失败", "runtime_mode": SIMULATION_RUNTIME_MODE},
                status_code=500,
            )
        return StreamingResponse(io.BytesIO(jpeg), media_type="image/jpeg",
                                 headers={"X-Runtime-Mode": SIMULATION_RUNTIME_MODE})


def main() -> None:
    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent / "templates",
        plugin_static_dir=Path(__file__).parent / "static",
    )
    _register_simulation_routes(runner.app)
    runner.run(host="0.0.0.0", port=8092)


if __name__ == "__main__":
    main()
