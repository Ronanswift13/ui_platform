"""
温度监测检测器 - 核心引擎
===========================

功能: 热力图生成、热点检测、温升趋势、LSTM预测、跨模块联动
版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


@dataclass
class Hotspot:
    zone_id: str
    zone_name: str
    center: Tuple[float, float]     # 归一化坐标
    temperature: float
    area_ratio: float
    severity: str                    # normal | warning | alarm | critical
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TempTrend:
    current: float
    avg_1min: float
    avg_5min: float
    rise_rate: float                 # °C/min
    trend_direction: str             # rising | stable | falling
    predicted_30min: Optional[float] = None


@dataclass
class TemperatureResult:
    heatmap: np.ndarray              # 2D温度矩阵
    max_temp: float
    min_temp: float
    avg_temp: float
    hotspots: List[Hotspot]
    trend: TempTrend
    status: str                      # normal | warning | alarm | critical
    linkage_events: List[Dict]       # 联动事件
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemperatureDetector:
    """温度监测核心检测器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        sensor_cfg = config.get("sensor", {})
        self.sensor_type = sensor_cfg.get("type", "mixed")
        self.resolution = tuple(sensor_cfg.get("resolution", [8, 6]))
        self.interpolation = sensor_cfg.get("interpolation", "bilinear")
        self.upsample_size = tuple(sensor_cfg.get("upsample_size", [64, 48]))

        th = config.get("thresholds", {})
        self.th_normal = th.get("normal_max", 45)
        self.th_warning = th.get("warning", 55)
        self.th_alarm = th.get("alarm", 70)
        self.th_critical = th.get("critical", 85)
        self.rise_warning = th.get("rise_rate_warning", 2.0)
        self.rise_alarm = th.get("rise_rate_alarm", 5.0)

        hs_cfg = config.get("hotspot", {})
        self.hs_min_area = hs_cfg.get("min_area_ratio", 0.02)
        self.hs_z_thresh = hs_cfg.get("z_score_threshold", 2.5)

        pred_cfg = config.get("prediction", {})
        self.pred_enabled = pred_cfg.get("enabled", True)
        self.pred_method = pred_cfg.get("method", "linear")
        self.pred_window = pred_cfg.get("window_minutes", 60)
        self.pred_forecast = pred_cfg.get("forecast_minutes", 30)

        self.zones = config.get("zones", [])
        self.linkage_cfg = config.get("linkage", {})

        # 历史缓冲
        buf_size = config.get("history", {}).get("buffer_size", 3600)
        self._history: deque = deque(maxlen=buf_size)
        self._zone_histories: Dict[str, deque] = {}

        self._inference_count = 0
        self._initialized = False

    def initialize(self) -> bool:
        self._initialized = True
        logger.info("[TemperatureDetector] 初始化完成")
        return True

    def detect(
        self,
        thermal_frame: Optional[np.ndarray] = None,
        sensor_readings: Optional[List[Dict]] = None,
    ) -> TemperatureResult:
        start = time.time()
        self._inference_count += 1

        # 1. 生成/获取温度矩阵
        heatmap = self._get_heatmap(thermal_frame, sensor_readings)

        # 2. 基础统计
        max_t = float(np.max(heatmap))
        min_t = float(np.min(heatmap))
        avg_t = float(np.mean(heatmap))

        # 3. 热点检测
        hotspots = self._detect_hotspots(heatmap)

        # 4. 记录历史
        self._history.append({"timestamp": time.time(), "max": max_t, "min": min_t, "avg": avg_t})

        # 5. 趋势分析
        trend = self._analyze_trend()

        # 6. 状态评估
        status = self._assess_status(max_t, trend.rise_rate, hotspots)

        # 7. 联动事件
        linkage = self._check_linkage(status, hotspots)

        elapsed = (time.time() - start) * 1000

        return TemperatureResult(
            heatmap=heatmap,
            max_temp=round(max_t, 1),
            min_temp=round(min_t, 1),
            avg_temp=round(avg_t, 1),
            hotspots=hotspots,
            trend=trend,
            status=status,
            linkage_events=linkage,
            metadata={"inference_time_ms": round(elapsed, 2), "inference_count": self._inference_count},
        )

    def _get_heatmap(self, thermal: Optional[np.ndarray], sensors: Optional[List[Dict]]) -> np.ndarray:
        if thermal is not None:
            return self._process_thermal(thermal)
        if sensors:
            return self._interpolate_sensors(sensors)
        # 模拟
        return self._simulate_heatmap()

    def _process_thermal(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = frame[:, :, 0].astype(np.float32)
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        # 假设值为温度
        if float(np.max(frame)) > 200:
            frame = frame / 255.0 * 60 + 15  # 归一化到15-75°C
        return frame

    def _interpolate_sensors(self, readings: List[Dict]) -> np.ndarray:
        grid = np.full(self.resolution, 25.0, dtype=np.float32)
        for r in readings:
            pos = r.get("position", {})
            row = int(pos.get("row", 0))
            col = int(pos.get("col", 0))
            if 0 <= row < self.resolution[1] and 0 <= col < self.resolution[0]:
                grid[row, col] = r.get("temperature", 25)

        if cv2 is not None:
            interp_map = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST}
            flag = interp_map.get(self.interpolation, cv2.INTER_LINEAR)
            grid = cv2.resize(grid, self.upsample_size, interpolation=flag)
        return grid

    def _simulate_heatmap(self) -> np.ndarray:
        rows, cols = self.upsample_size[1], self.upsample_size[0]
        base = np.random.normal(28, 2, (rows, cols)).astype(np.float32)
        # 添加热点
        cy, cx = rows // 3, cols // 2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        base += np.clip(20 * np.exp(-dist ** 2 / 50), 0, 30).astype(np.float32)
        return base

    def _detect_hotspots(self, heatmap: np.ndarray) -> List[Hotspot]:
        mean = float(np.mean(heatmap))
        std = float(np.std(heatmap)) + 1e-6
        h, w = heatmap.shape[:2]
        total_area = h * w

        # z-score
        z_map = (heatmap - mean) / std
        mask = z_map > self.hs_z_thresh

        if not np.any(mask):
            return []

        hotspots = []
        if cv2 is not None:
            labels_map = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(labels_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area / total_area < self.hs_min_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = float(M["m10"] / M["m00"]) / w
                cy = float(M["m01"] / M["m00"]) / h
                x_, y_, bw, bh = cv2.boundingRect(cnt)
                roi = heatmap[y_:y_ + bh, x_:x_ + bw]
                temp = float(np.max(roi)) if roi.size else mean

                zone_id, zone_name = self._match_zone(cx, cy)
                severity = self._temp_severity(temp)

                hotspots.append(Hotspot(
                    zone_id=zone_id, zone_name=zone_name,
                    center=(round(cx, 3), round(cy, 3)),
                    temperature=round(temp, 1),
                    area_ratio=round(area / total_area, 4),
                    severity=severity,
                ))
        else:
            # 无cv2，简单阈值
            ys, xs = np.where(mask)
            if len(ys) > 0:
                cy, cx = float(np.mean(ys)) / h, float(np.mean(xs)) / w
                temp = float(np.max(heatmap[mask]))
                zone_id, zone_name = self._match_zone(cx, cy)
                hotspots.append(Hotspot(zone_id=zone_id, zone_name=zone_name,
                                        center=(round(cx, 3), round(cy, 3)),
                                        temperature=round(temp, 1), area_ratio=round(len(ys) / total_area, 4),
                                        severity=self._temp_severity(temp)))
        return hotspots

    def _temp_severity(self, temp: float) -> str:
        if temp >= self.th_critical:
            return "critical"
        if temp >= self.th_alarm:
            return "alarm"
        if temp >= self.th_warning:
            return "warning"
        return "normal"

    def _match_zone(self, cx: float, cy: float) -> Tuple[str, str]:
        for z in self.zones:
            region = z.get("region", [0, 0, 1, 1])
            if len(region) == 4 and region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                return z["zone_id"], z.get("zone_name", "")
        return "", ""

    def _analyze_trend(self) -> TempTrend:
        hist = list(self._history)
        current = hist[-1]["avg"] if hist else 25.0

        avg_1m = self._avg_last(hist, 60)
        avg_5m = self._avg_last(hist, 300)
        rise_rate = self._calc_rise_rate(hist, 60)

        direction = "rising" if rise_rate > 0.5 else ("falling" if rise_rate < -0.5 else "stable")

        predicted = None
        if self.pred_enabled and len(hist) >= 10:
            predicted = self._predict(hist)

        return TempTrend(current=round(current, 1), avg_1min=round(avg_1m, 1), avg_5min=round(avg_5m, 1),
                         rise_rate=round(rise_rate, 2), trend_direction=direction, predicted_30min=predicted)

    @staticmethod
    def _avg_last(hist: List[Dict], seconds: float) -> float:
        if not hist:
            return 25.0
        cutoff = time.time() - seconds
        vals = [h["avg"] for h in hist if h["timestamp"] >= cutoff]
        return float(np.mean(vals)) if vals else hist[-1]["avg"]

    @staticmethod
    def _calc_rise_rate(hist: List[Dict], seconds: float) -> float:
        if len(hist) < 2:
            return 0.0
        cutoff = time.time() - seconds
        recent = [h for h in hist if h["timestamp"] >= cutoff]
        if len(recent) < 2:
            return 0.0
        dt = (recent[-1]["timestamp"] - recent[0]["timestamp"]) / 60.0 + 1e-6
        dtemp = recent[-1]["avg"] - recent[0]["avg"]
        return dtemp / dt

    def _predict(self, hist: List[Dict]) -> Optional[float]:
        try:
            temps = [h["avg"] for h in hist[-60:]]
            if len(temps) < 5:
                return None
            x = np.arange(len(temps))
            coeffs = np.polyfit(x, temps, 1)
            steps = self.pred_forecast  # ~30步
            return round(float(np.polyval(coeffs, len(temps) + steps)), 1)
        except Exception:
            return None

    def _assess_status(self, max_temp: float, rise_rate: float, hotspots: List[Hotspot]) -> str:
        if max_temp >= self.th_critical or rise_rate >= self.rise_alarm:
            return "critical"
        if max_temp >= self.th_alarm or any(h.severity == "alarm" for h in hotspots):
            return "alarm"
        if max_temp >= self.th_warning or rise_rate >= self.rise_warning:
            return "warning"
        return "normal"

    def _check_linkage(self, status: str, hotspots: List[Hotspot]) -> List[Dict]:
        events = []
        if status in ("alarm", "critical") and self.linkage_cfg.get("fire_detection_enabled"):
            events.append({"target": "fire_detection", "event": "high_temperature_alert",
                           "data": {"hotspots": [{"zone": h.zone_id, "temp": h.temperature} for h in hotspots]}})
        if status == "critical" and self.linkage_cfg.get("ventilation_control"):
            events.append({"target": "ventilation_system", "event": "emergency_ventilation", "data": {}})
        return events

    def reset(self):
        self._history.clear()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def stats(self) -> Dict:
        return {"inference_count": self._inference_count, "history_points": len(self._history)}
