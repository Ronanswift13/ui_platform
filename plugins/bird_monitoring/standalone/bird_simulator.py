"""
鸟类监控 standalone 仿真器 — 仅用于演示，绝不污染生产检测链路。

边界契约:
- 本模块输出的任何字段都显式标记 runtime_mode="standalone_simulation"。
- 本模块不依赖 plugin.py / detector.py，也不会被 plugin.infer 调用。
- 预置场景覆盖 9 个 label 中的主要路径: no_bird / review_required /
  unknown_bird / bird_safe / bird_warning / bird_danger / bird_critical /
  quality_failed，便于在无模型时仍可演示工程契约。
- 渲染帧为合成 BGR 图像，若无 cv2 则退化为纯色占位图。

设计原则与 busbar_inspection.standalone.busbar_simulator 对齐。
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
except ImportError:  # pragma: no cover - cv2 optional
    cv2 = None
    CV2_AVAILABLE = False


RUNTIME_MODE = "standalone_simulation"


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SimBird:
    """单只仿真鸟"""
    species_id: str              # sparrow / magpie / crow / eagle / egret / unknown_bird
    species_name: str            # 中文显示名
    wingspan_cm: int
    bbox: Dict[str, float]       # 归一化 x,y,width,height
    confidence: float            # 检测置信度
    species_confidence: float    # 物种识别置信度
    distance_m: float
    risk_level: str              # safe / warning / danger / critical / review
    review_required: bool = False
    review_reason: str = ""


@dataclass
class SimQuality:
    is_valid: bool = True
    clarity_score: float = 0.85
    brightness_score: float = 0.70
    issues: List[str] = field(default_factory=list)
    overall_score: float = 0.88


@dataclass
class SimScenario:
    """预置仿真场景"""
    scenario_id: str
    name: str
    description: str
    expected_label: str
    weather: str = "clear"           # clear / cloudy / dusk / rainy
    birds: List[SimBird] = field(default_factory=list)
    quality: SimQuality = field(default_factory=SimQuality)
    alarm_level: str = "INFO"
    notes: str = ""


# ============================================================
# 预置场景
# ============================================================

def _bird(species_id: str, species_name: str, wingspan: int,
          bbox: tuple, confidence: float, species_confidence: float,
          distance: float, risk_level: str,
          review_required: bool = False, review_reason: str = "") -> SimBird:
    return SimBird(
        species_id=species_id,
        species_name=species_name,
        wingspan_cm=wingspan,
        bbox={"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]},
        confidence=confidence,
        species_confidence=species_confidence,
        distance_m=distance,
        risk_level=risk_level,
        review_required=review_required,
        review_reason=review_reason,
    )


BUILTIN_SCENARIOS: List[SimScenario] = [
    SimScenario(
        scenario_id="no_bird_clear",
        name="空线路 - 无鸟",
        description="输电线路正常巡视，未检测到鸟类",
        expected_label="no_bird",
        weather="clear",
        birds=[],
        alarm_level="INFO",
        notes="验证空检测契约: simulation 不合成任何 bird bbox",
    ),
    SimScenario(
        scenario_id="sparrow_safe",
        name="麻雀 - 安全距离",
        description="远距离麻雀，体型小，不触发告警",
        expected_label="bird_safe",
        weather="clear",
        birds=[
            _bird("sparrow", "麻雀", 22, (0.38, 0.22, 0.06, 0.08),
                  confidence=0.82, species_confidence=0.78,
                  distance=18.0, risk_level="safe"),
        ],
        alarm_level="INFO",
    ),
    SimScenario(
        scenario_id="magpie_warning",
        name="喜鹊 - 接近线路",
        description="喜鹊接近线路，中等风险，生成 WARNING 告警",
        expected_label="bird_warning",
        weather="cloudy",
        birds=[
            _bird("magpie", "喜鹊", 56, (0.32, 0.30, 0.12, 0.14),
                  confidence=0.74, species_confidence=0.70,
                  distance=9.0, risk_level="warning"),
        ],
        alarm_level="WARNING",
    ),
    SimScenario(
        scenario_id="eagle_danger",
        name="老鹰 - 危险距离",
        description="老鹰大翼展接近相间，危险告警",
        expected_label="bird_danger",
        weather="clear",
        birds=[
            _bird("eagle", "老鹰", 200, (0.28, 0.28, 0.28, 0.32),
                  confidence=0.88, species_confidence=0.84,
                  distance=4.0, risk_level="danger"),
        ],
        alarm_level="ERROR",
    ),
    SimScenario(
        scenario_id="eagle_critical",
        name="老鹰 - 临界触电",
        description="老鹰极近距离栖停，临界告警",
        expected_label="bird_critical",
        weather="dusk",
        birds=[
            _bird("eagle", "老鹰", 200, (0.25, 0.32, 0.36, 0.36),
                  confidence=0.91, species_confidence=0.88,
                  distance=1.5, risk_level="critical"),
        ],
        alarm_level="CRITICAL",
    ),
    SimScenario(
        scenario_id="low_confidence_review",
        name="低置信度 - 待复核",
        description="检测置信度低于复核阈值，生成 review_required",
        expected_label="review_required",
        weather="cloudy",
        birds=[
            _bird("magpie", "喜鹊", 56, (0.40, 0.28, 0.10, 0.12),
                  confidence=0.42, species_confidence=0.36,
                  distance=8.0, risk_level="review",
                  review_required=True,
                  review_reason="检测置信度(0.42)低于阈值(0.50)"),
        ],
        alarm_level="WARNING",
    ),
    SimScenario(
        scenario_id="unknown_species",
        name="未知物种 - 数据库缺失",
        description="检测到鸟但物种映射未知（不进入复核路径）",
        expected_label="unknown_bird",
        weather="clear",
        birds=[
            _bird("unknown_bird", "未知鸟类", 40, (0.45, 0.26, 0.10, 0.12),
                  confidence=0.78, species_confidence=0.72,
                  distance=11.0, risk_level="warning"),
        ],
        alarm_level="WARNING",
    ),
    SimScenario(
        scenario_id="quality_dark",
        name="质量门禁 - 过暗帧",
        description="夜间无补光，亮度不足，质量门阻断",
        expected_label="quality_failed",
        weather="dusk",
        birds=[],
        quality=SimQuality(
            is_valid=False,
            clarity_score=0.22,
            brightness_score=0.12,
            issues=["图像亮度不足", "清晰度低于阈值"],
            overall_score=0.18,
        ),
        alarm_level="WARNING",
        notes="验证 quality_failed 契约，不应出现任何 bird bbox",
    ),
    SimScenario(
        scenario_id="quality_tiny",
        name="质量门禁 - 尺寸过小",
        description="输入分辨率低于 min_dimension，质量门阻断",
        expected_label="quality_failed",
        weather="clear",
        birds=[],
        quality=SimQuality(
            is_valid=False,
            clarity_score=0.40,
            brightness_score=0.55,
            issues=["图像尺寸过小(32x32)"],
            overall_score=0.30,
        ),
        alarm_level="WARNING",
    ),
]


# ============================================================
# 渲染器
# ============================================================

class BirdSceneRenderer:
    """合成线路巡视场景渲染器。"""

    SKY_CLEAR = (210, 180, 130)
    SKY_CLOUDY = (180, 170, 150)
    SKY_DUSK = (90, 70, 120)
    SKY_RAINY = (120, 115, 110)
    TOWER = (90, 90, 100)
    LINE = (210, 210, 220)
    GROUND = (80, 95, 70)
    BIRD_BODY = (40, 40, 40)

    def __init__(self, width: int = 1280, height: int = 720, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

    def render(self, scenario: SimScenario, frame_id: int = 0) -> np.ndarray:
        if not CV2_AVAILABLE:
            return self._fallback(scenario)

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_sky(img, scenario.weather)
        self._draw_ground(img)
        self._draw_tower(img)
        self._draw_power_lines(img)

        for bird in scenario.birds:
            self._draw_bird(img, bird)

        if not scenario.quality.is_valid:
            self._apply_quality_issue(img, scenario)

        self._draw_hud(img, scenario, frame_id)
        self._draw_simulation_watermark(img)
        return img

    def _fallback(self, scenario: SimScenario) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 80, dtype=np.uint8)
        img[: self.height // 3, :] = [200, 160, 120]
        img[self.height // 3 : 2 * self.height // 3, :] = [140, 140, 150]
        img[2 * self.height // 3 :, :] = [100, 110, 90]
        return img

    def _draw_sky(self, img: np.ndarray, weather: str) -> None:
        color = {
            "clear": self.SKY_CLEAR,
            "cloudy": self.SKY_CLOUDY,
            "dusk": self.SKY_DUSK,
            "rainy": self.SKY_RAINY,
        }.get(weather, self.SKY_CLEAR)
        sky_h = int(self.height * 0.60)
        img[:sky_h, :] = color

    def _draw_ground(self, img: np.ndarray) -> None:
        ground_h = int(self.height * 0.60)
        img[ground_h:, :] = self.GROUND

    def _draw_tower(self, img: np.ndarray) -> None:
        w, h = self.width, self.height
        base_x = int(w * 0.82)
        pts_left = np.array([
            [base_x - 60, int(h * 0.95)],
            [base_x - 20, int(h * 0.18)],
        ], dtype=np.int32)
        pts_right = np.array([
            [base_x + 60, int(h * 0.95)],
            [base_x + 20, int(h * 0.18)],
        ], dtype=np.int32)
        cv2.line(img, tuple(pts_left[0]), tuple(pts_left[1]), self.TOWER, 6)
        cv2.line(img, tuple(pts_right[0]), tuple(pts_right[1]), self.TOWER, 6)
        # 横担
        for y_ratio in (0.22, 0.30, 0.38):
            y = int(h * y_ratio)
            cv2.line(img, (base_x - 140, y), (base_x + 140, y), self.TOWER, 3)

    def _draw_power_lines(self, img: np.ndarray) -> None:
        w, h = self.width, self.height
        for y_ratio in (0.24, 0.32, 0.40):
            y = int(h * y_ratio)
            cv2.line(img, (0, y + 15), (w, y), self.LINE, 2)

    def _draw_bird(self, img: np.ndarray, bird: SimBird) -> None:
        w, h = self.width, self.height
        bx = int(bird.bbox["x"] * w)
        by = int(bird.bbox["y"] * h)
        bw = int(bird.bbox["width"] * w)
        bh = int(bird.bbox["height"] * h)
        # 椭圆近似鸟身
        cx, cy = bx + bw // 2, by + bh // 2
        cv2.ellipse(img, (cx, cy), (max(bw // 2, 6), max(bh // 2, 4)),
                    0, 0, 360, self.BIRD_BODY, -1)
        # 翅膀痕迹
        cv2.line(img, (bx, cy), (bx - bw // 2, cy - bh // 3), self.BIRD_BODY, 2)
        cv2.line(img, (bx + bw, cy), (bx + bw + bw // 2, cy - bh // 3),
                 self.BIRD_BODY, 2)
        # 框与标签
        color = {
            "safe": (80, 180, 80),
            "warning": (0, 180, 230),
            "danger": (0, 80, 230),
            "critical": (0, 0, 230),
            "review": (200, 120, 50),
        }.get(bird.risk_level, (200, 200, 200))
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
        label_txt = f"{bird.species_name} {bird.confidence:.0%} [{bird.risk_level}]"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (bx, by - th - 6),
                      (bx + tw + 6, by), color, -1)
        cv2.putText(img, label_txt, (bx + 3, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    def _apply_quality_issue(self, img: np.ndarray, scenario: SimScenario) -> None:
        if "过暗" in "".join(scenario.quality.issues) or "亮度" in "".join(scenario.quality.issues):
            img[:] = (img.astype(np.int32) * 0.15).clip(0, 255).astype(np.uint8)
        if "尺寸过小" in "".join(scenario.quality.issues):
            small = cv2.resize(img, (80, 45))
            img[:] = cv2.resize(small, (self.width, self.height),
                                interpolation=cv2.INTER_NEAREST)

        # 红底警示条
        h = self.height
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 60), (self.width, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        reason = "; ".join(scenario.quality.issues) or "quality_failed"
        cv2.putText(img, f"QUALITY GATE FAILED: {reason}",
                    (16, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

    def _draw_hud(self, img: np.ndarray, scenario: SimScenario, frame_id: int) -> None:
        text_lines = [
            f"SCENARIO  {scenario.name}",
            f"EXPECTED  {scenario.expected_label}",
            f"FRAME     #{frame_id}",
            f"WEATHER   {scenario.weather}",
        ]
        for i, line in enumerate(text_lines):
            cv2.putText(img, line, (18, 34 + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    def _draw_simulation_watermark(self, img: np.ndarray) -> None:
        """显式水印，防止把仿真帧误认为真实检测。"""
        txt = "STANDALONE SIMULATION - NOT REAL DETECTION"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = self.width - tw - 20
        y = 40
        cv2.rectangle(img, (x - 10, y - th - 8), (x + tw + 10, y + 10),
                      (0, 0, 0), -1)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 220, 255), 2)


# ============================================================
# Simulator
# ============================================================

class BirdMonitoringSimulator:
    """
    standalone 仿真器主体。完全独立于 plugin.py / detector.py；所有输出带
    runtime_mode="standalone_simulation"，绝不进入生产推理链。
    """

    ALARM_LEVELS = {
        "bird_safe": "INFO",
        "bird_warning": "WARNING",
        "bird_danger": "ERROR",
        "bird_critical": "CRITICAL",
        "review_required": "WARNING",
        "unknown_bird": "WARNING",
        "quality_failed": "WARNING",
        "no_bird": "INFO",
    }

    def __init__(self, seed: Optional[int] = None):
        self._scenarios: Dict[str, SimScenario] = {
            s.scenario_id: s for s in BUILTIN_SCENARIOS
        }
        self._current: Optional[SimScenario] = None
        self._frame_counter = 0
        self._rng = random.Random(seed)
        self._renderer = BirdSceneRenderer(seed=seed)
        self._last_payload: Optional[Dict[str, Any]] = None

    # -------------------- 场景管理 --------------------

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "description": s.description,
                "expected_label": s.expected_label,
                "alarm_level": s.alarm_level,
                "weather": s.weather,
                "has_birds": bool(s.birds),
                "quality_failure": not s.quality.is_valid,
            }
            for s in self._scenarios.values()
        ]

    def load_scenario(self, scenario_id: str) -> bool:
        if scenario_id not in self._scenarios:
            return False
        self._current = self._scenarios[scenario_id]
        self._frame_counter = 0
        self._last_payload = None
        return True

    def get_current_scenario(self) -> Optional[Dict[str, Any]]:
        if self._current is None:
            return None
        s = self._current
        return {
            "scenario_id": s.scenario_id,
            "name": s.name,
            "description": s.description,
            "expected_label": s.expected_label,
            "alarm_level": s.alarm_level,
            "weather": s.weather,
            "notes": s.notes,
        }

    # -------------------- 执行一步 --------------------

    def step(self) -> Dict[str, Any]:
        """推进一帧，返回与真实插件接近但显式标记 simulation 的 payload。"""
        start = time.perf_counter()
        self._frame_counter += 1
        scenario = self._current or self._scenarios["no_bird_clear"]

        detections = self._build_detections(scenario)
        alarms = self._build_alarms(scenario, detections)

        payload = {
            "runtime_mode": RUNTIME_MODE,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "scenario_notes": scenario.notes,
            "expected_label": scenario.expected_label,
            "frame_id": self._frame_counter,
            "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "weather": scenario.weather,
            "quality": {
                "is_valid": scenario.quality.is_valid,
                "clarity_score": scenario.quality.clarity_score,
                "brightness_score": scenario.quality.brightness_score,
                "issues": list(scenario.quality.issues),
                "overall_score": scenario.quality.overall_score,
            },
            "detections": detections,
            "alarms": alarms,
            "summary": self._build_summary(scenario, detections),
        }
        self._last_payload = payload
        return payload

    # -------------------- 渲染 --------------------

    def render_frame(self, scenario: Optional[SimScenario] = None) -> np.ndarray:
        sc = scenario or self._current or self._scenarios["no_bird_clear"]
        return self._renderer.render(sc, frame_id=self._frame_counter)

    def encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        if not CV2_AVAILABLE:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        return buf.tobytes()

    def get_last_payload(self) -> Optional[Dict[str, Any]]:
        return self._last_payload

    # -------------------- 内部构造 --------------------

    def _build_detections(self, scenario: SimScenario) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        if not scenario.quality.is_valid:
            detections.append({
                "detection_id": f"sim-qg-{self._frame_counter:04d}",
                "roi_id": "simulated_full_frame",
                "label": "quality_failed",
                "runtime_mode": RUNTIME_MODE,
                "confidence": 0.0,
                "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
                "failure_reason": "9003",
                "metadata": {
                    "runtime_mode": RUNTIME_MODE,
                    "source": "simulator",
                    "input_quality": {
                        "is_valid": False,
                        "issues": list(scenario.quality.issues),
                        "clarity_score": scenario.quality.clarity_score,
                        "brightness_score": scenario.quality.brightness_score,
                        "overall_score": scenario.quality.overall_score,
                    },
                    "training_placeholders": {
                        "hard_negative_candidate": True,
                        "suggested_label_for_dataset": "quality_failed",
                        "annotation_status": "not_annotated",
                        "model_placeholder": "standalone_simulation",
                    },
                    "suggested_action": "请确认图像清晰度和亮度后重新采集",
                },
            })
            return detections

        if not scenario.birds:
            detections.append({
                "detection_id": f"sim-nobird-{self._frame_counter:04d}",
                "roi_id": "simulated_full_frame",
                "label": "no_bird",
                "runtime_mode": RUNTIME_MODE,
                "confidence": 1.0,
                "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
                "metadata": {
                    "runtime_mode": RUNTIME_MODE,
                    "source": "simulator",
                    "message": "未检测到鸟类",
                    "training_placeholders": {
                        "hard_negative_candidate": True,
                        "suggested_label_for_dataset": "no_bird",
                        "annotation_status": "not_annotated",
                        "model_placeholder": "standalone_simulation",
                    },
                },
            })
            return detections

        for index, bird in enumerate(scenario.birds):
            if bird.species_id == "unknown_bird":
                label = "unknown_bird"
            elif bird.review_required:
                label = "review_required"
            else:
                label = f"bird_{bird.risk_level}"

            detections.append({
                "detection_id": f"sim-{self._frame_counter:04d}-{index:03d}",
                "roi_id": "simulated_full_frame",
                "label": label,
                "runtime_mode": RUNTIME_MODE,
                "confidence": bird.confidence,
                "bbox": dict(bird.bbox),
                "metadata": {
                    "runtime_mode": RUNTIME_MODE,
                    "source": "simulator",
                    "species_id": bird.species_id,
                    "species_name": bird.species_name,
                    "species_confidence": bird.species_confidence,
                    "wingspan_cm": bird.wingspan_cm,
                    "distance_m": bird.distance_m,
                    "risk_level": bird.risk_level,
                    "review_required": bird.review_required,
                    "review_reason": bird.review_reason,
                    "deterrent_suggestion": self._build_deterrent_suggestion(bird),
                    "training_placeholders": {
                        "hard_positive_candidate": bird.risk_level in ("danger", "critical"),
                        "suggested_label_for_dataset": label,
                        "annotation_status": "not_annotated",
                        "model_placeholder": "standalone_simulation",
                    },
                },
            })

        return detections

    def _build_alarms(self, scenario: SimScenario,
                      detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        alarms: List[Dict[str, Any]] = []
        for det in detections:
            level = self.ALARM_LEVELS.get(det["label"], "INFO")
            if level == "INFO":
                continue
            alarms.append({
                "level": level,
                "title": f"[仿真] {det['label']}",
                "message": (
                    det["metadata"].get("review_reason")
                    or det["metadata"].get("message")
                    or det["metadata"].get("suggested_action")
                    or scenario.description
                ),
                "runtime_mode": RUNTIME_MODE,
            })
        return alarms

    def _build_summary(self, scenario: SimScenario,
                       detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for det in detections:
            counts[det["label"]] = counts.get(det["label"], 0) + 1
        return {
            "total_detections": len(detections),
            "labels_seen": counts,
            "quality_gate_passed": scenario.quality.is_valid,
            "expected_label": scenario.expected_label,
            "matches_expected": any(
                det["label"] == scenario.expected_label for det in detections
            ) if detections else scenario.expected_label == "no_bird",
        }

    @staticmethod
    def _build_deterrent_suggestion(bird: SimBird) -> Dict[str, Any]:
        if bird.risk_level in ("danger", "critical"):
            methods = ["acoustic", "laser"]
            action = "trigger_suggested"
        elif bird.risk_level == "warning":
            methods = ["acoustic"]
            action = "monitor"
        else:
            methods = []
            action = "observe_only"
        return {
            "action": action,
            "methods": methods,
            "reason": f"风险等级={bird.risk_level}, 距离={bird.distance_m:.1f}m",
            "runtime_mode": RUNTIME_MODE,
            "hardware_control": "blocked",
        }
