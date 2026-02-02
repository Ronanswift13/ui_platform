"""
动物入侵检测器 - 核心引擎
===========================

功能:
1. YOLOv8小动物目标检测（多尺度）
2. 热成像融合活体判定
3. IoU跟踪与停留时间计算
4. 动物风险评估
5. 驱离设备联动
6. 入侵统计

版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnimalDetection:
    species: str
    bbox: Dict[str, float]
    confidence: float
    is_alive: bool = True
    track_id: int = -1
    duration_sec: float = 0.0
    zone_id: str = ""
    zone_name: str = ""
    risk_level: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterrentAction:
    action_type: str        # ultrasonic | strobe | audio
    target_track_id: int
    zone_id: str
    timestamp: float
    duration_sec: float = 5.0
    result: str = "pending"  # pending | success | failed


class AnimalTracker:
    """动物目标跟踪器"""

    def __init__(self, max_age: int = 50, min_confirm: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_confirm = min_confirm
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, Dict] = {}
        self._next_id = 1
        self._frame_count = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        self._frame_count += 1
        tracked = []
        matched_tracks = set()

        for det in detections:
            best_iou, best_tid = 0, None
            for tid, track in self._tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                matched_tracks.add(best_tid)
                t = self._tracks[best_tid]
                t["bbox"] = det["bbox"].copy()
                t["confidence"] = max(t["confidence"], det.get("confidence", 0))
                t["last_seen"] = self._frame_count
                t["hit_count"] += 1
                t["species"] = det.get("species", t["species"])

                det["track_id"] = best_tid
                det["confirmed"] = t["hit_count"] >= self.min_confirm
                det["duration_sec"] = (self._frame_count - t["first_seen"]) / 15.0  # ~15fps
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "bbox": det["bbox"].copy(),
                    "species": det.get("species", "unknown"),
                    "confidence": det.get("confidence", 0),
                    "first_seen": self._frame_count,
                    "last_seen": self._frame_count,
                    "hit_count": 1,
                }
                det["track_id"] = tid
                det["confirmed"] = False
                det["duration_sec"] = 0.0

            tracked.append(det)

        # 清理
        expired = [tid for tid, t in self._tracks.items() if (self._frame_count - t["last_seen"]) > self.max_age]
        for tid in expired:
            del self._tracks[tid]

        return tracked

    @staticmethod
    def _iou(b1: Dict, b2: Dict) -> float:
        x1 = max(b1["x"], b2["x"])
        y1 = max(b1["y"], b2["y"])
        x2 = min(b1["x"] + b1["width"], b2["x"] + b2["width"])
        y2 = min(b1["y"] + b1["height"], b2["y"] + b2["height"])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = b1["width"] * b1["height"] + b2["width"] * b2["height"] - inter
        return inter / union if union > 0 else 0.0

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0


class AnimalDetector:
    """动物入侵检测器"""

    CLASSES = ["rat", "mouse", "cat", "snake", "bird", "insect", "weasel", "unknown_animal"]

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        model_cfg = config.get("model", {})
        self.model_path = model_cfg.get("path", "")
        self.device = model_cfg.get("device", "cpu")
        self.input_size = tuple(model_cfg.get("input_size", [640, 640]))

        det_cfg = config.get("detection", {})
        self.confidence_threshold = det_cfg.get("confidence_threshold", 0.40)
        self.nms_threshold = det_cfg.get("nms_threshold", 0.45)
        self.dwell_alarm_sec = det_cfg.get("dwell_alarm_seconds", 30)

        self.species_cfg = config.get("species", {})
        self.risk_map = self.species_cfg.get("risk_levels", {})

        thermal_cfg = config.get("thermal_fusion", {})
        self.thermal_enabled = thermal_cfg.get("enabled", True)
        self.alive_temp_diff = thermal_cfg.get("alive_temp_diff", 3.0)
        self.bg_temp = thermal_cfg.get("background_temp", 25.0)

        deterrent_cfg = config.get("deterrent", {})
        self.deterrent_enabled = deterrent_cfg.get("enabled", True)
        self.deterrent_auto = deterrent_cfg.get("auto_trigger", True)
        self.deterrent_conf = deterrent_cfg.get("trigger_confidence", 0.6)
        self.deterrent_cooldown = deterrent_cfg.get("cooldown_seconds", 60)

        self.zones = config.get("zones", [])

        self._session = None
        self._tracker = AnimalTracker(
            max_age=det_cfg.get("max_track_age", 50),
            min_confirm=det_cfg.get("min_confirm_frames", 2),
        )

        # 统计
        self._daily_counts: Dict[str, int] = defaultdict(int)
        self._deterrent_history: List[DeterrentAction] = []
        self._last_deterrent_time: float = 0
        self._inference_count = 0
        self._initialized = False

    def initialize(self) -> bool:
        try:
            path = Path(self.model_path)
            if path.exists() and ort is not None:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(str(path), providers=providers)
                logger.info(f"[AnimalDetector] 模型加载成功: {path}")
            else:
                logger.warning("[AnimalDetector] 模型不存在或onnxruntime不可用，使用模拟模式")
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"[AnimalDetector] 初始化失败: {e}")
            return False

    def detect(
        self,
        frame: np.ndarray,
        thermal_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """执行完整动物检测"""
        start = time.time()
        self._inference_count += 1

        # 1. 视觉检测
        raw = self._visual_detect(frame)

        # 2. 热成像活体判定
        for det in raw:
            det["is_alive"] = self._check_alive(det["bbox"], thermal_frame)

        # 3. 跟踪
        tracked = self._tracker.update(raw)
        confirmed = [d for d in tracked if d.get("confirmed", False)]

        # 4. 构建结果
        detections = []
        for d in confirmed:
            species = d.get("species", "unknown_animal")
            zone_id, zone_name = self._match_zone(d["bbox"])
            risk = self._assess_risk(species, d.get("duration_sec", 0), zone_id)

            self._daily_counts[species] += 1

            detections.append(AnimalDetection(
                species=species,
                bbox=d["bbox"],
                confidence=d.get("confidence", 0),
                is_alive=d.get("is_alive", True),
                track_id=d.get("track_id", -1),
                duration_sec=d.get("duration_sec", 0),
                zone_id=zone_id,
                zone_name=zone_name,
                risk_level=risk,
            ))

        # 5. 驱离判定
        deterrent_actions = self._evaluate_deterrent(detections)

        elapsed_ms = (time.time() - start) * 1000

        return {
            "detections": detections,
            "deterrent_actions": deterrent_actions,
            "inference_time_ms": round(elapsed_ms, 2),
            "total_tracked": len(self._tracker._tracks),
        }

    def _visual_detect(self, frame: np.ndarray) -> List[Dict]:
        if self._session is not None:
            return self._yolo_inference(frame)
        return self._simulate_detection(frame)

    def _yolo_inference(self, frame: np.ndarray) -> List[Dict]:
        try:
            if cv2 is None:
                return []
            img = cv2.resize(frame, self.input_size)
            blob = img.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis]

            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: blob})
            return self._parse_yolo_output(outputs[0])
        except Exception as e:
            logger.error(f"[AnimalDetector] 推理失败: {e}")
            return []

    def _parse_yolo_output(self, output: np.ndarray) -> List[Dict]:
        dets = []
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T

        for row in output:
            if len(row) < 4 + len(self.CLASSES):
                continue
            cx, cy, w, h = row[:4]
            scores = row[4:4 + len(self.CLASSES)]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < self.confidence_threshold:
                continue

            dets.append({
                "bbox": {
                    "x": float(max(0, (cx - w / 2) / self.input_size[0])),
                    "y": float(max(0, (cy - h / 2) / self.input_size[1])),
                    "width": float(min(1, w / self.input_size[0])),
                    "height": float(min(1, h / self.input_size[1])),
                },
                "species": self.CLASSES[cid],
                "confidence": conf,
            })
        return dets

    def _simulate_detection(self, frame: np.ndarray) -> List[Dict]:
        """基于运动检测的模拟（无模型时）"""
        if cv2 is None or frame is None:
            return []

        # 简单：检测小的高对比度区域
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        dets = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w * h * 0.0005) or area > (w * h * 0.05):  # 小目标范围
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-6)
            if 0.2 < aspect < 5.0:
                conf = min(0.7, 0.2 + area / (w * h) * 20)
                dets.append({
                    "bbox": {"x": x / w, "y": y / h, "width": bw / w, "height": bh / h},
                    "species": "unknown_animal",
                    "confidence": conf,
                })

        return dets[:5]

    def _check_alive(self, bbox: Dict, thermal: Optional[np.ndarray]) -> bool:
        if thermal is None or not self.thermal_enabled:
            return True  # 无热成像默认活体

        try:
            h, w = thermal.shape[:2] if thermal.ndim >= 2 else (1, 1)
            x1 = int(bbox["x"] * w)
            y1 = int(bbox["y"] * h)
            x2 = int((bbox["x"] + bbox["width"]) * w)
            y2 = int((bbox["y"] + bbox["height"]) * h)

            roi = thermal[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size == 0:
                return True

            if roi.ndim > 2:
                roi = roi[:, :, 0]

            roi_mean = float(np.mean(roi))
            diff = abs(roi_mean - self.bg_temp)
            return diff >= self.alive_temp_diff
        except Exception:
            return True

    def _match_zone(self, bbox: Dict) -> Tuple[str, str]:
        cx = bbox["x"] + bbox["width"] / 2
        cy = bbox["y"] + bbox["height"] / 2
        for z in self.zones:
            poly = z.get("polygon", [])
            if self._point_in_poly(cx, cy, poly):
                return z["zone_id"], z.get("zone_name", "")
        return "", ""

    @staticmethod
    def _point_in_poly(x, y, poly):
        n = len(poly)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            j = i
        return inside

    def _assess_risk(self, species: str, duration: float, zone_id: str) -> str:
        base = self.risk_map.get(species, "medium")
        level_order = ["low", "medium", "high", "critical"]
        idx = level_order.index(base) if base in level_order else 1

        # 长时间停留升级
        if duration > self.dwell_alarm_sec * 2:
            idx = min(len(level_order) - 1, idx + 1)

        # 高风险区域升级
        for z in self.zones:
            if z.get("zone_id") == zone_id and z.get("sensitivity") == "high":
                idx = min(len(level_order) - 1, idx + 1)
                break

        return level_order[idx]

    def _evaluate_deterrent(self, detections: List[AnimalDetection]) -> List[Dict]:
        if not self.deterrent_enabled or not self.deterrent_auto:
            return []

        now = time.time()
        if (now - self._last_deterrent_time) < self.deterrent_cooldown:
            return []

        actions = []
        for d in detections:
            if d.confidence >= self.deterrent_conf and d.risk_level in ("high", "critical"):
                action = {
                    "action": "activate_deterrent",
                    "type": "ultrasonic" if d.species in ("rat", "mouse", "weasel") else "strobe",
                    "target_track_id": d.track_id,
                    "zone_id": d.zone_id,
                    "species": d.species,
                }
                actions.append(action)
                self._last_deterrent_time = now

        return actions

    def get_statistics(self) -> Dict:
        return {
            "daily_counts": dict(self._daily_counts),
            "total_today": sum(self._daily_counts.values()),
            "active_tracks": len(self._tracker._tracks),
            "deterrent_actions": len(self._deterrent_history),
            "inference_count": self._inference_count,
        }

    def reset_daily_stats(self):
        self._daily_counts.clear()

    def reset(self):
        self._tracker.reset()
        self._daily_counts.clear()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def stats(self) -> Dict:
        return {
            "inference_count": self._inference_count,
            "active_tracks": len(self._tracker._tracks),
            "model_loaded": self._session is not None,
            "daily_intrusions": sum(self._daily_counts.values()),
        }
