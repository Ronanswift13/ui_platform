#!/usr/bin/env python3
"""
Busbar Inspection - Standalone Application

启动独立运行的母线巡视仪表板，包含:
- 真实检测接口 (通过 StandalonePluginRunner)
- 仿真演示接口 (通过 BusbarSimulator，与生产算法完全隔离)
"""
import io
import sys
from pathlib import Path

import numpy as np
from fastapi import Query
from fastapi.responses import JSONResponse, StreamingResponse

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from darkbreaker_sdk.standalone import StandalonePluginRunner
from plugins.busbar_inspection.plugin import Plugin
from plugins.busbar_inspection.standalone.busbar_simulator import BusbarSimulator

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# 全局仿真器实例 (仅 standalone 使用)
_simulator = BusbarSimulator(seed=42)


# ============================================================
# 仿真 API 路由 — 与生产检测完全隔离
# ============================================================

def _register_simulation_routes(app) -> None:
    """注册仿真演示相关的 API 路由"""

    @app.get("/api/simulator/scenarios")
    async def list_scenarios():
        """列出可用仿真场景"""
        return JSONResponse({"scenarios": _simulator.list_scenarios()})

    @app.post("/api/simulator/load")
    async def load_scenario(scenario_id: str = Query(...)):
        """加载仿真场景"""
        ok = _simulator.load_scenario(scenario_id)
        if not ok:
            return JSONResponse(
                {"error": f"场景 '{scenario_id}' 不存在"},
                status_code=404,
            )
        return JSONResponse({
            "status": "loaded",
            "scenario": _simulator.get_current_scenario(),
        })

    @app.post("/api/simulator/step")
    async def step_simulation():
        """执行一步仿真检测"""
        result = _simulator.step()
        return JSONResponse(_simulator.result_to_api_format(result))

    @app.get("/api/simulator/frame")
    async def get_simulation_frame():
        """获取当前仿真帧图像 (JPEG)"""
        result = _simulator.get_last_result() or _simulator.step()
        defects = result.defects
        frame = _simulator.render_frame(defects=defects)

        if CV2_AVAILABLE:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return StreamingResponse(
                io.BytesIO(buf.tobytes()),
                media_type="image/jpeg",
            )
        else:
            # 无 cv2: 返回 raw PNG-like 占位
            return JSONResponse({"error": "cv2 not available, frame rendering disabled"})

    @app.get("/api/simulator/frame_with_annotations")
    async def get_annotated_frame():
        """获取带标注的仿真帧 (缺陷框 + 标签)"""
        result = _simulator.get_last_result() or _simulator.step()
        defects = result.defects
        frame = _simulator.render_frame(defects=defects)

        if CV2_AVAILABLE:
            frame = _annotate_frame(frame, result)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return StreamingResponse(
                io.BytesIO(buf.tobytes()),
                media_type="image/jpeg",
            )
        return JSONResponse({"error": "cv2 not available"})

    @app.get("/api/simulator/status")
    async def simulator_status():
        """获取仿真器状态"""
        return JSONResponse({
            "frame_counter": _simulator._frame_counter,
            "current_scenario": _simulator.get_current_scenario(),
            "running": _simulator._running,
        })


def _annotate_frame(frame: np.ndarray, result) -> np.ndarray:
    """在帧上绘制检测结果标注"""
    if cv2 is None:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 缺陷标签颜色
    label_colors = {
        "pin_missing": (0, 0, 255),     # 红
        "crack": (0, 165, 255),          # 橙
        "foreign_object": (0, 255, 255), # 黄
    }

    severity_thickness = {"low": 1, "medium": 2, "high": 3}

    for defect in result.defects:
        bx = int(defect.bbox["x"] * w)
        by = int(defect.bbox["y"] * h)
        bw = int(defect.bbox["width"] * w)
        bh = int(defect.bbox["height"] * h)

        color = label_colors.get(defect.label, (255, 255, 255))
        thickness = severity_thickness.get(defect.severity, 2)

        # 绘制框
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, thickness)

        # 标签背景
        label_text = f"{defect.description} {defect.confidence:.0%}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(annotated, (bx, by - text_size[1] - 6),
                      (bx + text_size[0] + 4, by), color, -1)
        cv2.putText(annotated, label_text, (bx + 2, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 质量门禁状态
    if not result.quality_gate_passed:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        reason = result.quality.failure_reason or "Quality Gate Failed"
        cv2.putText(annotated, f"QUALITY GATE FAILED: {reason}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


# ============================================================
# 入口
# ============================================================

def main():
    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent / "templates",
        plugin_static_dir=Path(__file__).parent / "static",
    )

    # 注册仿真路由 (挂载到 runner.app，不影响真实检测路由)
    _register_simulation_routes(runner.app)

    runner.run(host="0.0.0.0", port=8089)


if __name__ == "__main__":
    main()
