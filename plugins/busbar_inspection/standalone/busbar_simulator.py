"""
母线巡视仿真引擎 — 仅用于 standalone 演示，不影响生产检测逻辑

功能:
- 生成合成母线巡视场景图像（含绝缘子、母线、金具）
- 注入可控缺陷（销钉缺失、裂纹、异物）
- 模拟环境干扰（过曝、模糊、遮挡、雨雾）
- 输出仿真检测结果（标签、置信度、bbox、原因码）

设计原则:
- 本模块不依赖 detector_enhanced.py / plugin.py
- 仿真结果格式与真实检测输出一致，便于 UI 复用
- 所有随机数使用可控种子，确保可复现
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


# ============================================================
# 仿真场景定义
# ============================================================

@dataclass
class SimDefect:
    """仿真缺陷"""
    label: str                          # pin_missing / crack / foreign_object
    bbox: Dict[str, float]              # 归一化 x,y,width,height
    confidence: float                   # 模拟置信度
    severity: str = "medium"            # low / medium / high
    description: str = ""


@dataclass
class SimQuality:
    """仿真图像质量"""
    status: str = "pass"
    quality_gate_status: str = "pass"
    clarity_score: float = 0.85
    is_overexposed: bool = False
    is_low_contrast: bool = False
    is_occluded: bool = False
    is_blurred: bool = False
    edge_energy: float = 120.0
    reason_code: Optional[int] = None
    failure_reason: Optional[str] = None


@dataclass
class SimZoom:
    """仿真变焦建议"""
    suggested_zoom: float = 1.0
    suggested_action: str = "NONE"
    min_object_size_px: int = 18
    target_size_px: int = 90


@dataclass
class SimDetectionResult:
    """单次仿真检测输出"""
    defects: List[SimDefect] = field(default_factory=list)
    quality: SimQuality = field(default_factory=SimQuality)
    zoom: SimZoom = field(default_factory=SimZoom)
    quality_gate_passed: bool = True
    quality_gate_status: str = "pass"
    processing_time_ms: float = 0.0
    frame_id: int = 0
    scenario_id: str = ""
    scenario_name: str = ""
    expected_defects: List[str] = field(default_factory=list)


@dataclass
class SimScenario:
    """仿真场景"""
    scenario_id: str
    name: str
    description: str
    weather: str = "sunny"              # sunny / cloudy / rainy / foggy
    lighting: str = "normal"            # normal / overexposed / underexposed / backlit
    defect_types: List[str] = field(default_factory=list)
    defect_count_range: tuple = (0, 3)
    quality_issue: Optional[str] = None  # None / blur / overexpose / occlude / low_contrast
    zoom_needed: bool = False
    expected_quality_gate_status: str = "pass"
    expected_quality_status: str = "pass"
    expected_reason_code: Optional[int] = None


# 预定义场景
BUILTIN_SCENARIOS: List[SimScenario] = [
    SimScenario(
        scenario_id="normal_clear",
        name="正常巡视 - 晴天",
        description="晴天条件下母线正常巡视，无缺陷",
        weather="sunny",
        lighting="normal",
        defect_types=[],
        defect_count_range=(0, 0),
    ),
    SimScenario(
        scenario_id="pin_missing_single",
        name="销钉缺失 - 单处",
        description="发现单处销钉缺失缺陷",
        weather="sunny",
        lighting="normal",
        defect_types=["pin_missing"],
        defect_count_range=(1, 1),
    ),
    SimScenario(
        scenario_id="pin_missing_multi",
        name="销钉缺失 - 多处",
        description="发现多处销钉缺失，需紧急处置",
        weather="cloudy",
        lighting="normal",
        defect_types=["pin_missing"],
        defect_count_range=(2, 4),
    ),
    SimScenario(
        scenario_id="crack_detected",
        name="裂纹检测",
        description="母线表面发现裂纹缺陷",
        weather="sunny",
        lighting="normal",
        defect_types=["crack"],
        defect_count_range=(1, 2),
    ),
    SimScenario(
        scenario_id="foreign_object",
        name="异物悬挂",
        description="检测到异物悬挂在母线设备上",
        weather="sunny",
        lighting="normal",
        defect_types=["foreign_object"],
        defect_count_range=(1, 2),
    ),
    SimScenario(
        scenario_id="mixed_defects",
        name="混合缺陷场景",
        description="多种缺陷同时出现的复杂场景",
        weather="cloudy",
        lighting="normal",
        defect_types=["pin_missing", "crack", "foreign_object"],
        defect_count_range=(2, 5),
    ),
    SimScenario(
        scenario_id="quality_blur",
        name="质量门禁 - 模糊",
        description="图像模糊导致质量门禁失败",
        weather="sunny",
        lighting="normal",
        defect_types=[],
        quality_issue="blur",
        expected_quality_gate_status="hard_fail",
        expected_quality_status="fail_blur",
        expected_reason_code=103,
    ),
    SimScenario(
        scenario_id="quality_overexpose",
        name="质量门禁 - 过曝",
        description="逆光/过曝导致质量门禁失败",
        weather="sunny",
        lighting="overexposed",
        defect_types=[],
        quality_issue="overexpose",
        expected_quality_gate_status="hard_fail",
        expected_quality_status="fail_overexposed",
        expected_reason_code=101,
    ),
    SimScenario(
        scenario_id="quality_occlude",
        name="质量门禁 - 遮挡",
        description="大面积遮挡导致质量门禁失败",
        weather="rainy",
        lighting="normal",
        defect_types=[],
        quality_issue="occlude",
        expected_quality_gate_status="hard_fail",
        expected_quality_status="fail_occluded",
        expected_reason_code=102,
    ),
    SimScenario(
        scenario_id="small_target_zoom",
        name="远距小目标 - 需变焦",
        description="目标过小需要变焦放大确认",
        weather="sunny",
        lighting="normal",
        defect_types=["pin_missing"],
        defect_count_range=(1, 2),
        zoom_needed=True,
    ),
    SimScenario(
        scenario_id="rainy_inspection",
        name="雨天巡视",
        description="雨天条件下巡视，图像质量下降",
        weather="rainy",
        lighting="normal",
        defect_types=["pin_missing", "crack"],
        defect_count_range=(0, 2),
        quality_issue="low_contrast",
        expected_quality_gate_status="soft_fail",
        expected_quality_status="fail_low_contrast",
        expected_reason_code=101,
    ),
]


# ============================================================
# 仿真图像渲染器
# ============================================================

class BusbarSceneRenderer:
    """合成母线场景图像渲染器"""

    # 颜色常量 (BGR)
    SKY_COLOR = (200, 160, 120)           # 天空
    STEEL_COLOR = (140, 140, 150)         # 钢结构
    BUSBAR_COLOR = (180, 180, 190)        # 母线
    INSULATOR_COLOR = (60, 80, 140)       # 绝缘子(棕色)
    INSULATOR_GLAZE = (80, 100, 160)      # 绝缘子釉面
    PIN_COLOR = (80, 80, 90)              # 销钉
    GROUND_COLOR = (100, 110, 90)         # 地面
    CRACK_COLOR = (30, 30, 40)            # 裂纹
    FOREIGN_COLOR = (50, 120, 60)         # 异物(绿色/植物)

    def __init__(self, width: int = 1280, height: int = 720, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

    def render_scene(
        self,
        scenario: SimScenario,
        defects: List[SimDefect],
        frame_id: int = 0,
    ) -> np.ndarray:
        """渲染完整场景图像"""
        if not CV2_AVAILABLE:
            return self._render_fallback(scenario, frame_id)

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. 渲染天空背景
        self._draw_sky(img, scenario.weather)

        # 2. 渲染变电站结构
        self._draw_substation_structure(img)

        # 3. 渲染母线
        busbar_y = int(self.height * 0.30)
        self._draw_busbar(img, busbar_y)

        # 4. 渲染绝缘子
        insulator_positions = self._draw_insulators(img, busbar_y)

        # 5. 渲染销钉
        self._draw_pins(img, insulator_positions)

        # 6. 注入缺陷
        for defect in defects:
            self._draw_defect(img, defect)

        # 7. 应用环境效果
        img = self._apply_environment(img, scenario)

        # 8. 绘制 HUD 信息
        self._draw_hud(img, scenario, frame_id)

        return img

    def _render_fallback(self, scenario: SimScenario, frame_id: int) -> np.ndarray:
        """无 cv2 时的降级渲染"""
        img = np.full((self.height, self.width, 3), 80, dtype=np.uint8)
        # 简单的颜色分区
        img[:self.height // 3, :] = [200, 160, 120]  # 天空
        img[self.height // 3:2 * self.height // 3, :] = [140, 140, 150]  # 结构
        img[2 * self.height // 3:, :] = [100, 110, 90]  # 地面
        return img

    def _draw_sky(self, img: np.ndarray, weather: str):
        """绘制天空"""
        h = self.height
        w = self.width

        if weather == "sunny":
            for y in range(int(h * 0.55)):
                ratio = y / (h * 0.55)
                b = int(200 - ratio * 60)
                g = int(160 - ratio * 40)
                r = int(120 - ratio * 30)
                img[y, :] = [max(b, 0), max(g, 0), max(r, 0)]
        elif weather == "cloudy":
            for y in range(int(h * 0.55)):
                ratio = y / (h * 0.55)
                v = int(170 - ratio * 40)
                img[y, :] = [v, v - 5, v - 10]
        elif weather == "rainy":
            for y in range(int(h * 0.55)):
                ratio = y / (h * 0.55)
                v = int(130 - ratio * 30)
                img[y, :] = [v + 10, v, v - 5]
        elif weather == "foggy":
            for y in range(int(h * 0.55)):
                v = int(180 - (y / (h * 0.55)) * 20)
                img[y, :] = [v, v, v]

    def _draw_substation_structure(self, img: np.ndarray):
        """绘制变电站钢结构"""
        h, w = self.height, self.width

        # 地面
        img[int(h * 0.75):, :] = self.GROUND_COLOR

        # 砂石地面纹理
        for _ in range(500):
            px = self.rng.randint(0, w - 1)
            py = self.rng.randint(int(h * 0.76), h - 1)
            v = self.rng.randint(-15, 15)
            base = list(self.GROUND_COLOR)
            img[py, px] = [max(0, min(255, base[0] + v)),
                           max(0, min(255, base[1] + v)),
                           max(0, min(255, base[2] + v))]

        # 立柱
        pillar_positions = [int(w * 0.15), int(w * 0.5), int(w * 0.85)]
        for px in pillar_positions:
            pw = 20
            cv2.rectangle(img, (px - pw // 2, int(h * 0.25)),
                          (px + pw // 2, int(h * 0.75)), self.STEEL_COLOR, -1)
            # 立柱高光
            cv2.line(img, (px - pw // 4, int(h * 0.25)),
                     (px - pw // 4, int(h * 0.75)), (160, 160, 170), 1)

        # 横梁
        cv2.rectangle(img, (pillar_positions[0] - 10, int(h * 0.23)),
                      (pillar_positions[-1] + 10, int(h * 0.27)), self.STEEL_COLOR, -1)

        # 底部基础
        for px in pillar_positions:
            cv2.rectangle(img, (px - 25, int(h * 0.73)),
                          (px + 25, int(h * 0.77)), (120, 120, 130), -1)

    def _draw_busbar(self, img: np.ndarray, busbar_y: int):
        """绘制母线"""
        w = self.width
        thickness = 12

        # 三相母线 (A/B/C)
        colors = [(45, 45, 180), (55, 160, 55), (180, 120, 45)]  # 红/绿/蓝(BGR)
        for i, color in enumerate(colors):
            y = busbar_y + i * 40
            cv2.rectangle(img, (int(w * 0.1), y - thickness // 2),
                          (int(w * 0.9), y + thickness // 2), color, -1)
            # 高光线
            cv2.line(img, (int(w * 0.1), y - thickness // 2 + 2),
                     (int(w * 0.9), y - thickness // 2 + 2),
                     tuple(min(255, c + 40) for c in color), 1)

    def _draw_insulators(self, img: np.ndarray, busbar_y: int) -> List[tuple]:
        """绘制绝缘子串, 返回位置列表"""
        w = self.width
        positions = []

        insulator_xs = [int(w * p) for p in [0.2, 0.35, 0.5, 0.65, 0.8]]

        for ix in insulator_xs:
            # 绝缘子从母线向下延伸
            for disc in range(5):
                dy = busbar_y + 120 + disc * 22
                disc_w = 28 - disc * 2
                cv2.ellipse(img, (ix, dy), (disc_w, 8), 0, 0, 360,
                            self.INSULATOR_COLOR, -1)
                cv2.ellipse(img, (ix, dy), (disc_w, 8), 0, 0, 180,
                            self.INSULATOR_GLAZE, 2)

            positions.append((ix, busbar_y + 120))

        return positions

    def _draw_pins(self, img: np.ndarray, insulator_positions: List[tuple]):
        """绘制销钉"""
        for (ix, iy) in insulator_positions:
            # 上方连接销钉
            cv2.circle(img, (ix, iy - 15), 5, self.PIN_COLOR, -1)
            cv2.circle(img, (ix, iy - 15), 5, (100, 100, 110), 1)
            # 下方连接销钉
            cv2.circle(img, (ix, iy + 120), 5, self.PIN_COLOR, -1)
            cv2.circle(img, (ix, iy + 120), 5, (100, 100, 110), 1)

    def _draw_defect(self, img: np.ndarray, defect: SimDefect):
        """在图像上绘制缺陷"""
        h, w = self.height, self.width
        bx = int(defect.bbox["x"] * w)
        by = int(defect.bbox["y"] * h)
        bw = int(defect.bbox["width"] * w)
        bh = int(defect.bbox["height"] * h)

        if defect.label == "pin_missing":
            # 销钉缺失 — 用黑色圆洞表示
            cx, cy = bx + bw // 2, by + bh // 2
            radius = max(bw, bh) // 2
            cv2.circle(img, (cx, cy), radius, (20, 20, 25), -1)
            cv2.circle(img, (cx, cy), radius + 1, (10, 10, 15), 2)

        elif defect.label == "crack":
            # 裂纹 — 锯齿线
            pts = []
            steps = 8
            for i in range(steps + 1):
                px = bx + int(bw * i / steps)
                jitter = self.rng.randint(-3, 3)
                py = by + bh // 2 + jitter
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts_arr], False, self.CRACK_COLOR, 2)
            # 裂纹阴影
            cv2.polylines(img, [pts_arr + 1], False, (40, 40, 50), 1)

        elif defect.label == "foreign_object":
            # 异物 — 不规则形状
            cx, cy = bx + bw // 2, by + bh // 2
            pts = []
            n_pts = 7
            for i in range(n_pts):
                angle = 2 * math.pi * i / n_pts
                r = max(bw, bh) // 2 * (0.6 + self.rng.random() * 0.4)
                px = int(cx + r * math.cos(angle))
                py = int(cy + r * math.sin(angle))
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(img, [pts_arr], self.FOREIGN_COLOR)
            cv2.polylines(img, [pts_arr], True, (30, 80, 40), 2)

    def _apply_environment(self, img: np.ndarray, scenario: SimScenario) -> np.ndarray:
        """应用环境效果"""
        if scenario.quality_issue == "blur":
            img = cv2.GaussianBlur(img, (21, 21), 8)
        elif scenario.quality_issue == "overexpose":
            img = cv2.convertScaleAbs(img, alpha=1.8, beta=60)
        elif scenario.quality_issue == "occlude":
            h, w = img.shape[:2]
            overlay = img.copy()
            # 大面积暗区模拟遮挡
            cv2.rectangle(overlay, (0, int(h * 0.3)),
                          (int(w * 0.6), int(h * 0.8)), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        elif scenario.quality_issue == "low_contrast":
            img = cv2.convertScaleAbs(img, alpha=0.4, beta=80)

        # 雨天效果
        if scenario.weather == "rainy":
            rain = np.zeros_like(img)
            for _ in range(300):
                x = self.rng.randint(0, img.shape[1] - 1)
                y = self.rng.randint(0, img.shape[0] - 1)
                length = self.rng.randint(10, 25)
                cv2.line(rain, (x, y), (x + 1, y + length), (180, 180, 180), 1)
            cv2.addWeighted(img, 0.9, rain, 0.1, 0, img)

        # 雾天效果
        if scenario.weather == "foggy":
            fog = np.full_like(img, 200)
            cv2.addWeighted(img, 0.5, fog, 0.5, 0, img)

        return img

    def _draw_hud(self, img: np.ndarray, scenario: SimScenario, frame_id: int):
        """绘制 HUD 信息覆盖层"""
        h, w = img.shape[:2]

        # 左上角: 仿真标识
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (320, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(img, "SIM", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2)
        cv2.putText(img, f"Frame #{frame_id}", (60, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, scenario.name, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # 右上角: 天气/光照
        weather_text = f"{scenario.weather.upper()} | {scenario.lighting.upper()}"
        text_size = cv2.getTextSize(weather_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(overlay, (w - text_size[0] - 20, 0),
                      (w, 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, weather_text, (w - text_size[0] - 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


# ============================================================
# 仿真检测引擎
# ============================================================

class BusbarSimulator:
    """
    母线巡视仿真引擎

    生成合成场景图像和对应的仿真检测结果。
    完全独立于 detector_enhanced.py，不调用任何生产检测逻辑。
    """

    # 原因码映射 (与 02_algorithm_contract.md 一致)
    QUALITY_REASON_MAP = {
        "blur": (103, "模糊/失焦"),
        "overexpose": (101, "逆光/过曝/低对比"),
        "occlude": (102, "遮挡/不可见"),
        "low_contrast": (101, "逆光/过曝/低对比"),
    }

    QUALITY_STATUS_MAP = {
        "blur": ("hard_fail", "fail_blur"),
        "overexpose": ("hard_fail", "fail_overexposed"),
        "occlude": ("hard_fail", "fail_occluded"),
        "low_contrast": ("soft_fail", "fail_low_contrast"),
    }

    DEFECT_DESCRIPTIONS = {
        "pin_missing": "销钉缺失",
        "crack": "裂纹",
        "foreign_object": "异物悬挂",
    }

    ALARM_LEVELS = {
        "pin_missing": "ERROR",
        "crack": "WARNING",
        "foreign_object": "WARNING",
    }

    def __init__(self, seed: Optional[int] = None):
        self._scenarios = {s.scenario_id: s for s in BUILTIN_SCENARIOS}
        self._current_scenario: Optional[SimScenario] = None
        self._renderer = BusbarSceneRenderer(seed=seed)
        self._frame_counter = 0
        self._rng = random.Random(seed)
        self._running = False
        self._step_interval_ms = 2000
        self._last_result: Optional[SimDetectionResult] = None

    # -------------------- 场景管理 --------------------

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """列出可用场景"""
        return [
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "description": s.description,
                "weather": s.weather,
                "lighting": s.lighting,
                "has_defects": len(s.defect_types) > 0,
                "quality_issue": s.quality_issue,
                "zoom_needed": s.zoom_needed,
                "expected_quality_gate_status": s.expected_quality_gate_status,
                "expected_quality_status": s.expected_quality_status,
                "expected_reason_code": s.expected_reason_code,
            }
            for s in self._scenarios.values()
        ]

    def load_scenario(self, scenario_id: str) -> bool:
        """加载场景"""
        if scenario_id not in self._scenarios:
            return False
        self._current_scenario = self._scenarios[scenario_id]
        self._frame_counter = 0
        self._last_result = None
        return True

    def get_current_scenario(self) -> Optional[Dict[str, Any]]:
        """获取当前场景信息"""
        if self._current_scenario is None:
            return None
        s = self._current_scenario
        return {
            "scenario_id": s.scenario_id,
            "name": s.name,
            "description": s.description,
            "expected_quality_gate_status": s.expected_quality_gate_status,
            "expected_quality_status": s.expected_quality_status,
            "expected_reason_code": s.expected_reason_code,
        }

    # -------------------- 仿真执行 --------------------

    def step(self) -> SimDetectionResult:
        """执行一步仿真，返回检测结果"""
        start = time.perf_counter()
        self._frame_counter += 1

        scenario = self._current_scenario
        if scenario is None:
            # 默认使用正常场景
            scenario = self._scenarios["normal_clear"]

        # 1. 生成缺陷
        defects = self._generate_defects(scenario)

        # 2. 质量门禁仿真
        quality = self._simulate_quality(scenario)
        quality_passed = quality.quality_gate_status != "hard_fail"

        # 3. 变焦建议仿真
        zoom = self._simulate_zoom(scenario, defects)

        elapsed = (time.perf_counter() - start) * 1000

        result = SimDetectionResult(
            defects=defects if quality_passed else [],
            quality=quality,
            zoom=zoom,
            quality_gate_passed=quality_passed,
            quality_gate_status=quality.quality_gate_status,
            processing_time_ms=elapsed,
            frame_id=self._frame_counter,
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            expected_defects=[defect.label for defect in defects],
        )
        self._last_result = result
        return result

    def render_frame(self, scenario: Optional[SimScenario] = None,
                     defects: Optional[List[SimDefect]] = None) -> np.ndarray:
        """渲染当前帧图像"""
        sc = scenario or self._current_scenario or self._scenarios["normal_clear"]
        if defects is not None:
            df = defects
        elif self._last_result is not None:
            df = self._last_result.defects
        else:
            df = []
        return self._renderer.render_scene(sc, df, self._frame_counter)

    def get_last_result(self) -> Optional[SimDetectionResult]:
        """获取最近一次仿真结果。"""
        return self._last_result

    # -------------------- 结果转换(兼容 UI 格式) --------------------

    def result_to_api_format(self, result: SimDetectionResult) -> Dict[str, Any]:
        """将仿真结果转换为与真实检测一致的 API 格式"""
        detections = []
        alarms = []

        if result.quality_gate_status == "hard_fail":
            # HARD_FAIL: 阻断输出
            detections.append({
                "detection_id": f"sim-qg-{result.frame_id:04d}",
                "roi_id": "simulated_full_frame",
                "label": "quality_failed",
                "pred_label": "quality_failed",
                "runtime_mode": "standalone_simulation",
                "quality_gate_status": "hard_fail",
                "confidence": 1.0,
                "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
                "candidate_labels": ["quality_failed"],
                "source": "simulator",
                "review_status": "blocked",
                "reason_code": result.quality.reason_code,
                "suggested_action": result.zoom.suggested_action,
                "suggested_zoom": result.zoom.suggested_zoom,
                "failure_reason": str(result.quality.reason_code),
                "metadata": {
                    "runtime_mode": "standalone_simulation",
                    "quality_gate_status": "hard_fail",
                    "status": result.quality.status,
                    "reason": result.quality.failure_reason,
                    "suggested_action": result.zoom.suggested_action,
                    "quality_suggested_action": result.zoom.suggested_action,
                    "zoom_suggested_action": "NONE",
                    "suggested_zoom": result.zoom.suggested_zoom,
                    "quality": {
                        "quality_gate_status": result.quality.quality_gate_status,
                        "status": result.quality.status,
                        "failure_reason": result.quality.failure_reason,
                        "suggested_action": result.zoom.suggested_action,
                        "clarity_score": result.quality.clarity_score,
                        "is_overexposed": result.quality.is_overexposed,
                        "is_low_contrast": result.quality.is_low_contrast,
                        "is_occluded": result.quality.is_occluded,
                        "edge_energy": result.quality.edge_energy,
                    },
                },
            })
            alarms.append({
                "level": "WARNING",
                "title": f"质量门禁未通过: {result.quality.failure_reason}",
                "message": f"图像质量不满足检测条件，建议: {result.zoom.suggested_action}",
            })
        else:
            for index, defect in enumerate(result.defects):
                detections.append({
                    "detection_id": f"sim-{result.frame_id:04d}-{index:03d}",
                    "roi_id": "simulated_full_frame",
                    "label": defect.label,
                    "pred_label": defect.label,
                    "runtime_mode": "standalone_simulation",
                    "quality_gate_status": result.quality.quality_gate_status,
                    "confidence": defect.confidence,
                    "bbox": defect.bbox,
                    "candidate_labels": [defect.label],
                    "source": "simulator",
                    "review_status": "review_required" if result.quality.quality_gate_status == "soft_fail" else "confirmed",
                    "reason_code": result.quality.reason_code,
                    "suggested_action": result.zoom.suggested_action,
                    "suggested_zoom": result.zoom.suggested_zoom,
                    "severity": defect.severity,
                    "description": defect.description,
                    "metadata": {
                        "runtime_mode": "standalone_simulation",
                        "quality_gate_status": result.quality.quality_gate_status,
                        "status": result.quality.status,
                        "review_reason": (
                            f"质量门禁软失败: {result.quality.failure_reason}"
                            if result.quality.quality_gate_status == "soft_fail"
                            else ""
                        ),
                        "quality_suggested_action": result.zoom.suggested_action,
                        "zoom_suggested_action": "NONE",
                        "suggested_zoom": result.zoom.suggested_zoom,
                        "suggested_action": result.zoom.suggested_action,
                        "quality": {
                            "quality_gate_status": result.quality.quality_gate_status,
                            "status": result.quality.status,
                            "failure_reason": result.quality.failure_reason,
                            "suggested_action": result.zoom.suggested_action,
                            "clarity_score": result.quality.clarity_score,
                            "is_overexposed": result.quality.is_overexposed,
                            "is_low_contrast": result.quality.is_low_contrast,
                            "is_occluded": result.quality.is_occluded,
                            "edge_energy": result.quality.edge_energy,
                        },
                    },
                })

                level = self.ALARM_LEVELS.get(defect.label, "WARNING")
                desc = self.DEFECT_DESCRIPTIONS.get(defect.label, defect.label)
                alarms.append({
                    "level": level,
                    "title": f"检测到{desc}",
                    "message": (
                        f"检测到{desc}，置信度: {defect.confidence:.2f}，"
                        f"严重程度: {defect.severity}"
                    ),
                })

            if result.quality_gate_status == "soft_fail" and not detections:
                detections.append({
                    "detection_id": f"sim-qgsoft-{result.frame_id:04d}",
                    "roi_id": "simulated_full_frame",
                    "label": "quality_failed",
                    "pred_label": "quality_failed",
                    "runtime_mode": "standalone_simulation",
                    "quality_gate_status": "soft_fail",
                    "confidence": 1.0,
                    "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "candidate_labels": ["quality_failed"],
                    "source": "simulator",
                    "review_status": "review_required",
                    "reason_code": result.quality.reason_code,
                    "suggested_action": result.zoom.suggested_action,
                    "suggested_zoom": result.zoom.suggested_zoom,
                    "failure_reason": str(result.quality.reason_code),
                    "metadata": {
                        "runtime_mode": "standalone_simulation",
                        "quality_gate_status": "soft_fail",
                        "status": result.quality.status,
                        "review_reason": "质量门禁软失败且未形成稳定检测，需人工复核",
                        "reason": result.quality.failure_reason,
                        "quality_suggested_action": result.zoom.suggested_action,
                        "zoom_suggested_action": "NONE",
                        "suggested_action": result.zoom.suggested_action,
                        "suggested_zoom": result.zoom.suggested_zoom,
                        "quality": {
                            "quality_gate_status": result.quality.quality_gate_status,
                            "status": result.quality.status,
                            "failure_reason": result.quality.failure_reason,
                            "suggested_action": result.zoom.suggested_action,
                            "clarity_score": result.quality.clarity_score,
                            "is_overexposed": result.quality.is_overexposed,
                            "is_low_contrast": result.quality.is_low_contrast,
                            "is_occluded": result.quality.is_occluded,
                            "edge_energy": result.quality.edge_energy,
                        },
                    },
                })

        summary = self._build_summary(detections, result)
        scene_comparison = self._build_scene_comparison(result, detections)

        return {
            "runtime_mode": "standalone_simulation",
            "quality_gate_status": result.quality_gate_status,
            "detections": detections,
            "alarms": alarms,
            "quality_gate_passed": result.quality_gate_passed,
            "processing_time_ms": result.processing_time_ms,
            "frame_id": result.frame_id,
            "scenario_id": result.scenario_id,
            "scenario_name": result.scenario_name,
            "summary": summary,
            "scene_comparison": scene_comparison,
            "stats": {
                "total_defects": len(result.defects),
                "quality": {
                    "quality_gate_status": result.quality.quality_gate_status,
                    "status": result.quality.status,
                    "clarity_score": result.quality.clarity_score,
                    "is_overexposed": result.quality.is_overexposed,
                    "is_low_contrast": result.quality.is_low_contrast,
                    "is_occluded": result.quality.is_occluded,
                },
                "zoom": {
                    "suggested_zoom": result.zoom.suggested_zoom,
                    "suggested_action": result.zoom.suggested_action,
                },
            },
        }

    def _build_summary(
        self,
        detections: List[Dict[str, Any]],
        result: SimDetectionResult,
    ) -> Dict[str, Any]:
        """构建图像级 summary。"""
        counts_by_label: Dict[str, int] = {}
        review_required_count = 0
        for det in detections:
            label = det.get("pred_label", det.get("label", "unknown"))
            counts_by_label[label] = counts_by_label.get(label, 0) + 1
            if det.get("review_status") == "review_required":
                review_required_count += 1

        return {
            "scenario_name": result.scenario_name,
            "total_detections": len(detections),
            "counts_by_label": counts_by_label,
            "review_required_count": review_required_count,
            "quality": {
                "quality_gate_status": result.quality.quality_gate_status,
                "status": result.quality.status,
                "reason_code": result.quality.reason_code,
                "failure_reason": result.quality.failure_reason,
                "clarity_score": result.quality.clarity_score,
                "is_overexposed": result.quality.is_overexposed,
                "is_low_contrast": result.quality.is_low_contrast,
                "is_occluded": result.quality.is_occluded,
                "edge_energy": result.quality.edge_energy,
            },
            "zoom": {
                "suggested_zoom": result.zoom.suggested_zoom,
                "suggested_action": result.zoom.suggested_action,
            },
        }

    def _build_scene_comparison(
        self,
        result: SimDetectionResult,
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """构建仿真预期/实际对照信息。"""
        scenario = self._scenarios.get(result.scenario_id)
        actual_labels = [
            det.get("pred_label", det.get("label", "unknown"))
            for det in detections
            if det.get("label") != "quality_failed"
        ]
        conflicts = [
            {
                "detection_id": det.get("detection_id"),
                "pred_label": det.get("pred_label"),
                "candidate_labels": det.get("candidate_labels", []),
                "review_status": det.get("review_status"),
            }
            for det in detections
            if det.get("review_status") == "review_required"
        ]
        return {
            "scenario_name": result.scenario_name,
            "expected_defects": list(result.expected_defects),
            "expected_quality_gate_status": scenario.expected_quality_gate_status if scenario else "pass",
            "expected_quality_status": scenario.expected_quality_status if scenario else "pass",
            "expected_reason_code": scenario.expected_reason_code if scenario else None,
            "actual_quality_gate_status": result.quality.quality_gate_status,
            "actual_quality_status": result.quality.status,
            "actual_reason_code": result.quality.reason_code,
            "quality_alignment": {
                "gate_status_match": (
                    result.quality.quality_gate_status
                    == (scenario.expected_quality_gate_status if scenario else "pass")
                ),
                "status_match": (
                    result.quality.status
                    == (scenario.expected_quality_status if scenario else "pass")
                ),
                "reason_code_match": (
                    result.quality.reason_code
                    == (scenario.expected_reason_code if scenario else None)
                ),
            },
            "actual_detections": actual_labels,
            "false_positives": self._multiset_difference(actual_labels, result.expected_defects),
            "false_negatives": self._multiset_difference(result.expected_defects, actual_labels),
            "conflicts": conflicts,
        }

    def _multiset_difference(
        self,
        left: List[str],
        right: List[str],
    ) -> List[str]:
        """计算标签多重集合差集。"""
        remaining: Dict[str, int] = {}
        for item in right:
            remaining[item] = remaining.get(item, 0) + 1

        diff: List[str] = []
        for item in left:
            if remaining.get(item, 0) > 0:
                remaining[item] -= 1
            else:
                diff.append(item)
        return diff

    # -------------------- 内部方法 --------------------

    def _generate_defects(self, scenario: SimScenario) -> List[SimDefect]:
        """根据场景生成缺陷"""
        if not scenario.defect_types:
            return []

        count = self._rng.randint(*scenario.defect_count_range)
        defects = []

        # 绝缘子位置区域 (归一化)
        anchor_positions = [
            (0.18, 0.35), (0.33, 0.35), (0.48, 0.35),
            (0.63, 0.35), (0.78, 0.35),
        ]

        for i in range(count):
            label = self._rng.choice(scenario.defect_types)

            # 在锚点附近生成 bbox
            anchor = anchor_positions[i % len(anchor_positions)]
            jitter_x = self._rng.uniform(-0.03, 0.03)
            jitter_y = self._rng.uniform(-0.02, 0.02)

            if label == "pin_missing":
                w_size, h_size = 0.025, 0.035
                conf = self._rng.uniform(0.72, 0.95)
                severity = "high" if conf > 0.85 else "medium"
            elif label == "crack":
                w_size, h_size = 0.06, 0.015
                conf = self._rng.uniform(0.55, 0.88)
                severity = "high" if conf > 0.75 else "medium"
            else:  # foreign_object
                w_size, h_size = 0.04, 0.05
                conf = self._rng.uniform(0.60, 0.90)
                severity = "medium"

            # 帧间微扰动
            frame_jitter_x = math.sin(self._frame_counter * 0.1 + i) * 0.005
            frame_jitter_y = math.cos(self._frame_counter * 0.07 + i) * 0.003

            defects.append(SimDefect(
                label=label,
                bbox={
                    "x": max(0.01, min(0.95, anchor[0] + jitter_x + frame_jitter_x)),
                    "y": max(0.01, min(0.95, anchor[1] + jitter_y + frame_jitter_y)),
                    "width": w_size,
                    "height": h_size,
                },
                confidence=round(conf, 3),
                severity=severity,
                description=self.DEFECT_DESCRIPTIONS.get(label, label),
            ))

        return defects

    def _simulate_quality(self, scenario: SimScenario) -> SimQuality:
        """仿真质量门禁"""
        if scenario.quality_issue is None:
            return SimQuality(
                status="pass",
                quality_gate_status="pass",
                clarity_score=round(self._rng.uniform(0.65, 0.95), 3),
                edge_energy=round(self._rng.uniform(80, 200), 1),
            )

        code, desc = self.QUALITY_REASON_MAP.get(
            scenario.quality_issue, (103, "未知质量问题")
        )
        quality_gate_status, status = self.QUALITY_STATUS_MAP.get(
            scenario.quality_issue,
            ("hard_fail", "fail_blur"),
        )

        if scenario.quality_issue == "blur":
            return SimQuality(
                status=status,
                quality_gate_status=quality_gate_status,
                clarity_score=round(self._rng.uniform(0.08, 0.25), 3),
                is_blurred=True,
                edge_energy=round(self._rng.uniform(5, 20), 1),
                reason_code=code,
                failure_reason=desc,
            )
        elif scenario.quality_issue == "overexpose":
            return SimQuality(
                status=status,
                quality_gate_status=quality_gate_status,
                clarity_score=round(self._rng.uniform(0.40, 0.60), 3),
                is_overexposed=True,
                edge_energy=round(self._rng.uniform(30, 60), 1),
                reason_code=code,
                failure_reason=desc,
            )
        elif scenario.quality_issue == "occlude":
            return SimQuality(
                status=status,
                quality_gate_status=quality_gate_status,
                clarity_score=round(self._rng.uniform(0.30, 0.50), 3),
                is_occluded=True,
                edge_energy=round(self._rng.uniform(15, 40), 1),
                reason_code=code,
                failure_reason=desc,
            )
        else:  # low_contrast
            return SimQuality(
                status=status,
                quality_gate_status=quality_gate_status,
                clarity_score=round(self._rng.uniform(0.35, 0.55), 3),
                is_low_contrast=True,
                edge_energy=round(self._rng.uniform(20, 50), 1),
                reason_code=code,
                failure_reason=desc,
            )

    def _simulate_zoom(self, scenario: SimScenario,
                       defects: List[SimDefect]) -> SimZoom:
        """仿真变焦建议"""
        if scenario.zoom_needed and defects:
            # 计算最小目标尺寸
            min_size = min(
                max(d.bbox["width"], d.bbox["height"]) for d in defects
            )
            # 模拟小目标 → 需要变焦
            target_px = 90
            current_px = min_size * 1280  # 近似像素尺寸
            zoom = round(min(12.0, max(1.0, target_px / max(current_px, 1))), 1)
            action = "ZOOM_IN" if zoom > 1.5 else "NONE"
            return SimZoom(
                suggested_zoom=zoom,
                suggested_action=action,
                min_object_size_px=18,
                target_size_px=target_px,
            )

        if scenario.quality_issue == "blur":
            return SimZoom(suggested_action="REFOCUS")
        if scenario.quality_issue == "overexpose":
            return SimZoom(suggested_action="RECAPTURE")
        if scenario.quality_issue == "occlude":
            return SimZoom(suggested_action="CHANGE_VIEW")
        if scenario.quality_issue == "low_contrast":
            return SimZoom(suggested_action="RECAPTURE")

        return SimZoom()
