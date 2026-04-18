"""
表计仿真器 - 生成合成表计图像用于演示和测试
支持模拟表(指针)、数字表、LED指示灯三种类型

每个场景定义一组虚拟表计及其随时间变化的读数序列。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class SimMeterType(Enum):
    ANALOG = "analog"
    DIGITAL = "digital"
    LED = "led"


@dataclass
class SimMeter:
    """单个虚拟表计"""
    meter_id: str
    meter_type: SimMeterType
    label: str              # 显示名称 (e.g. "压强表 A")
    unit: str = ""
    range_min: float = 0.0
    range_max: float = 1.0
    current_value: float = 0.0
    # LED 专用
    led_color: str = "green"  # red/green/yellow/off


@dataclass
class SimScenario:
    """仿真场景"""
    scenario_id: str
    name: str
    description: str
    meters: list[SimMeter] = field(default_factory=list)
    step_interval_ms: int = 500     # 每步间隔
    total_steps: int = 200


class MeterSimulator:
    """表计仿真引擎: 管理场景加载、步进、图像渲染"""

    IMG_W = 640
    IMG_H = 480

    def __init__(self, scenarios_dir: Optional[Path] = None):
        self._scenarios_dir = scenarios_dir
        self._scenarios: dict[str, dict] = {}
        self._active_scenario: Optional[SimScenario] = None
        self._step_count = 0
        self._running = False
        self._start_time = 0.0

        if scenarios_dir and scenarios_dir.is_dir():
            self._load_scenarios(scenarios_dir)

    # ── 场景管理 ───────────────────────────────────────────

    def _load_scenarios(self, d: Path) -> None:
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data.setdefault("scenario_id", f.stem)
                self._scenarios[data["scenario_id"]] = data
            except Exception:
                pass

    def list_scenarios(self) -> list[dict[str, Any]]:
        result = []
        for sid, data in self._scenarios.items():
            result.append({
                "id": sid,
                "name": data.get("name", sid),
                "description": data.get("description", ""),
                "meter_count": len(data.get("meters", [])),
            })
        return result

    def load_scenario(self, scenario_id: str) -> bool:
        data = self._scenarios.get(scenario_id)
        if data is None:
            return False

        meters = []
        for m in data.get("meters", []):
            mt = SimMeterType(m.get("type", "analog"))
            meters.append(SimMeter(
                meter_id=m.get("id", f"meter_{len(meters)}"),
                meter_type=mt,
                label=m.get("label", "表计"),
                unit=m.get("unit", ""),
                range_min=m.get("range_min", 0.0),
                range_max=m.get("range_max", 1.0),
                current_value=m.get("initial_value", m.get("range_min", 0.0)),
                led_color=m.get("led_color", "green"),
            ))

        self._active_scenario = SimScenario(
            scenario_id=scenario_id,
            name=data.get("name", scenario_id),
            description=data.get("description", ""),
            meters=meters,
            step_interval_ms=data.get("step_interval_ms", 500),
            total_steps=data.get("total_steps", 200),
        )
        self._step_count = 0
        self._running = True
        self._start_time = time.time()
        self._scenario_raw = data
        return True

    def get_active_scenario_id(self) -> Optional[str]:
        return self._active_scenario.scenario_id if self._active_scenario else None

    def stop(self) -> None:
        self._running = False

    # ── 步进与读数更新 ─────────────────────────────────────

    def step(self) -> dict[str, Any]:
        """推进一步, 返回当前所有表计状态"""
        if self._active_scenario is None:
            return {"error": "no active scenario"}

        self._step_count += 1
        raw = self._scenario_raw
        t = self._step_count

        for i, meter in enumerate(self._active_scenario.meters):
            m_cfg = raw.get("meters", [{}])[i] if i < len(raw.get("meters", [])) else {}
            pattern = m_cfg.get("pattern", "sine")
            speed = m_cfg.get("speed", 1.0)

            if meter.meter_type == SimMeterType.LED:
                cycle = m_cfg.get("cycle", ["green", "green", "red", "yellow"])
                idx = (t // max(1, int(10 / speed))) % len(cycle)
                meter.led_color = cycle[idx]
            else:
                rng = meter.range_max - meter.range_min
                if pattern == "sine":
                    frac = 0.5 + 0.4 * math.sin(t * speed * 0.05)
                elif pattern == "ramp":
                    frac = (t * speed * 0.005) % 1.0
                elif pattern == "spike":
                    base = 0.3
                    frac = base + (0.6 if (t % max(1, int(40 / speed))) < 5 else 0.0)
                elif pattern == "drift":
                    frac = min(1.0, 0.1 + t * speed * 0.002)
                elif pattern == "random":
                    frac = np.random.uniform(0.1, 0.9)
                else:
                    frac = 0.5 + 0.4 * math.sin(t * speed * 0.05)

                frac = max(0.0, min(1.0, frac))
                meter.current_value = round(meter.range_min + frac * rng, 2)

        return self._build_state()

    def get_state(self) -> dict[str, Any]:
        return self._build_state()

    def _build_state(self) -> dict[str, Any]:
        if self._active_scenario is None:
            return {"active": False}
        meters = []
        for m in self._active_scenario.meters:
            meters.append({
                "meter_id": m.meter_id,
                "meter_type": m.meter_type.value,
                "label": m.label,
                "value": m.current_value,
                "unit": m.unit,
                "range_min": m.range_min,
                "range_max": m.range_max,
                "led_color": m.led_color,
            })
        return {
            "active": True,
            "scenario_id": self._active_scenario.scenario_id,
            "scenario_name": self._active_scenario.name,
            "step": self._step_count,
            "total_steps": self._active_scenario.total_steps,
            "meters": meters,
        }

    # ── 图像渲染 ───────────────────────────────────────────

    def render_frame(self) -> np.ndarray:
        """渲染当前所有表计到一张合成帧"""
        if cv2 is None:
            return self._render_placeholder()

        img = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        img[:] = (20, 20, 25)  # 深色背景

        if self._active_scenario is None:
            cv2.putText(img, "No Active Scenario", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            return img

        meters = self._active_scenario.meters
        n = len(meters)
        if n == 0:
            return img

        # 布局: 最多 3 列
        cols = min(n, 3)
        rows = math.ceil(n / cols)
        cell_w = self.IMG_W // cols
        cell_h = self.IMG_H // rows

        for idx, meter in enumerate(meters):
            row, col = divmod(idx, cols)
            x0 = col * cell_w
            y0 = row * cell_h
            cx = x0 + cell_w // 2
            cy = y0 + cell_h // 2

            if meter.meter_type == SimMeterType.ANALOG:
                self._draw_analog(img, cx, cy, min(cell_w, cell_h) // 2 - 20, meter)
            elif meter.meter_type == SimMeterType.DIGITAL:
                self._draw_digital(img, x0 + 10, cy - 30, cell_w - 20, 60, meter)
            elif meter.meter_type == SimMeterType.LED:
                self._draw_led(img, cx, cy, min(cell_w, cell_h) // 4, meter)

        # 信息栏
        info = f"SIM | {self._active_scenario.name} | Step {self._step_count}"
        cv2.putText(img, info, (10, self.IMG_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 120), 1)

        return img

    def _draw_analog(self, img: np.ndarray, cx: int, cy: int, r: int, meter: SimMeter) -> None:
        """绘制模拟表盘: 刻度圆弧 + 指针 + 读数"""
        if cv2 is None:
            return
        r = max(r, 40)

        # 表盘底色
        cv2.circle(img, (cx, cy), r, (40, 40, 45), -1)
        cv2.circle(img, (cx, cy), r, (80, 80, 90), 2)

        # 刻度弧 (-135° ~ 135°, OpenCV角度体系: 0=右, 逆时针正)
        angle_min = -135.0
        angle_max = 135.0
        num_ticks = 10
        for i in range(num_ticks + 1):
            frac = i / num_ticks
            angle_deg = angle_min + frac * (angle_max - angle_min)
            # 转换为数学角度 (从正右方顺时针)
            rad = math.radians(180 + angle_deg)  # 从底部开始
            tx = int(cx + (r - 10) * math.cos(rad))
            ty = int(cy - (r - 10) * math.sin(rad))
            tx2 = int(cx + (r - 3) * math.cos(rad))
            ty2 = int(cy - (r - 3) * math.sin(rad))
            thickness = 2 if i % 5 == 0 else 1
            cv2.line(img, (tx, ty), (tx2, ty2), (160, 160, 170), thickness)

        # 指针角度
        rng = meter.range_max - meter.range_min
        if rng > 0:
            frac = (meter.current_value - meter.range_min) / rng
        else:
            frac = 0.5
        frac = max(0.0, min(1.0, frac))
        pointer_deg = angle_min + frac * (angle_max - angle_min)
        ptr_rad = math.radians(180 + pointer_deg)

        # 绘制指针
        ptr_len = int(r * 0.75)
        px = int(cx + ptr_len * math.cos(ptr_rad))
        py = int(cy - ptr_len * math.sin(ptr_rad))
        cv2.line(img, (cx, cy), (px, py), (50, 50, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, (50, 50, 255), -1)

        # 量程标注
        cv2.putText(img, f"{meter.range_min}", (cx - r + 5, cy + r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 130), 1)
        cv2.putText(img, f"{meter.range_max}", (cx + r - 35, cy + r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 130), 1)

        # 读数和标签
        value_text = f"{meter.current_value:.1f} {meter.unit}"
        cv2.putText(img, value_text, (cx - 35, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 230), 1)
        cv2.putText(img, meter.label, (cx - 30, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 180, 140), 1)

    def _draw_digital(self, img: np.ndarray, x: int, y: int, w: int, h: int, meter: SimMeter) -> None:
        """绘制数字显示屏"""
        if cv2 is None:
            return

        # 显示屏背景
        cv2.rectangle(img, (x, y), (x + w, y + h), (10, 30, 10), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 100, 60), 1)

        # 数值
        val_str = f"{meter.current_value:.2f}"
        cv2.putText(img, val_str, (x + 10, y + h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 255, 50), 2)

        # 单位
        cv2.putText(img, meter.unit, (x + w - 50, y + h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 180, 30), 1)

        # 标签
        cv2.putText(img, meter.label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 180, 140), 1)

    def _draw_led(self, img: np.ndarray, cx: int, cy: int, r: int, meter: SimMeter) -> None:
        """绘制 LED 指示灯"""
        if cv2 is None:
            return
        r = max(r, 15)

        color_map = {
            "red": (40, 40, 255),
            "green": (40, 255, 40),
            "yellow": (40, 230, 255),
            "off": (30, 30, 30),
        }
        bgr = color_map.get(meter.led_color, (30, 30, 30))

        # 发光效果
        if meter.led_color != "off":
            glow_r = int(r * 1.5)
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), glow_r, bgr, -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        cv2.circle(img, (cx, cy), r, bgr, -1)
        cv2.circle(img, (cx, cy), r, (80, 80, 90), 2)

        # 标签和状态
        cv2.putText(img, meter.label, (cx - 30, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 180, 140), 1)
        cv2.putText(img, meter.led_color.upper(), (cx - 20, cy + r + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)

    def _render_placeholder(self) -> np.ndarray:
        """无 OpenCV 时的占位帧"""
        img = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        img[:] = (30, 30, 35)
        return img
