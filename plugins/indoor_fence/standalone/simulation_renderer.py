"""
仿真俯视图渲染器 V3.0
========================================

将 Simulator 的人员世界坐标渲染为工程级俯视图帧。
不依赖 adapters/ 或 core/，可独立测试。

渲染元素:
- 带刻度坐标网格 (每米一格)
- 黄色警戒线 + 区域色块
- 机柜区域矩形
- 人员标记 (实心圆 + 方向箭头 + ID 标签)
- 运动轨迹 (渐变尾迹)
- 实时距离标注 (人与围栏、人与人)
- 速度矢量显示
- 轨迹预测线
- 比例尺 + 信息栏
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


# 颜色常量 (BGR)
_C_BG = (20, 18, 15)
_C_GRID = (50, 45, 40)
_C_GRID_MAJOR = (70, 65, 55)
_C_GRID_LABEL = (140, 140, 130)
_C_YELLOW_LINE = (0, 200, 255)
_C_SAFE_ZONE = (80, 60, 0)        # 深蓝绿 (安全区覆盖)
_C_WARN_ZONE = (0, 80, 100)       # 深黄 (警告区覆盖)
_C_DANGER_ZONE = (0, 30, 80)      # 深红 (危险区覆盖)
_C_CABINET = (180, 140, 0)        # 橙色 (机柜边框)
_C_PERSON_SAFE = (0, 220, 100)    # 绿
_C_PERSON_WARN = (0, 200, 255)    # 黄
_C_PERSON_DANGER = (60, 60, 255)  # 红
_C_TRAIL = (180, 160, 60)         # 青蓝色
_C_WHITE = (255, 255, 255)
_C_INFO_BG = (40, 35, 30)
_C_SCALE = (200, 200, 190)
_C_DISTANCE_LINE = (200, 200, 100)  # 距离标注线
_C_PREDICTION = (150, 100, 200)     # 预测轨迹
_C_VELOCITY = (100, 200, 150)       # 速度矢量
_C_ALERT_BG = (0, 0, 120)           # 告警背景

# 默认机柜配置 (世界坐标: x, y, w, h, name)
_DEFAULT_CABINETS = [
    (1.5, 4.0, 1.2, 0.8, "Cabinet A"),
    (4.5, 4.5, 1.2, 0.8, "Cabinet B"),
    (7.5, 4.0, 1.2, 0.8, "Cabinet C"),
]


class SimulationRenderer:
    """将人员世界坐标渲染为工程级俯视图帧，支持实时距离和轨迹显示。"""

    def __init__(
        self,
        scene_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 6.0),
        resolution: Tuple[int, int] = (640, 480),
        yellow_line_y: float = 3.0,
        cabinets: Optional[List[Tuple]] = None,
        scenario_name: str = "",
        show_distances: bool = True,
        show_predictions: bool = True,
        show_velocities: bool = True,
    ):
        self._bounds = scene_bounds          # (x_min, y_min, x_max, y_max) 米
        self._w, self._h = resolution        # 像素
        self._yellow_line_y = yellow_line_y
        self._cabinets = cabinets or _DEFAULT_CABINETS
        self._scenario_name = scenario_name

        # 显示选项
        self._show_distances = show_distances
        self._show_predictions = show_predictions
        self._show_velocities = show_velocities

        # 坐标转换参数 (留边距)
        self._margin = 40  # 像素
        x_min, y_min, x_max, y_max = scene_bounds
        self._scene_w = x_max - x_min
        self._scene_h = y_max - y_min
        draw_w = self._w - 2 * self._margin
        draw_h = self._h - 2 * self._margin
        self._scale_x = draw_w / self._scene_w if self._scene_w > 0 else 1.0
        self._scale_y = draw_h / self._scene_h if self._scene_h > 0 else 1.0

        # 人员轨迹缓冲
        self._trails: Dict[int, deque] = {}
        self._max_trail = 60

        # 上一帧人员位置 (用于方向和速度计算)
        self._prev_pos: Dict[int, Tuple[float, float]] = {}
        self._prev_time: Dict[int, float] = {}

        # 速度缓存
        self._velocities: Dict[int, Tuple[float, float]] = {}

        # 告警缓存
        self._alerts: List[Dict[str, Any]] = []

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    @scenario_name.setter
    def scenario_name(self, name: str):
        self._scenario_name = name

    def set_display_options(
        self,
        show_distances: Optional[bool] = None,
        show_predictions: Optional[bool] = None,
        show_velocities: Optional[bool] = None,
    ) -> None:
        """设置显示选项"""
        if show_distances is not None:
            self._show_distances = show_distances
        if show_predictions is not None:
            self._show_predictions = show_predictions
        if show_velocities is not None:
            self._show_velocities = show_velocities

    def set_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """设置告警列表（用于渲染告警指示）"""
        self._alerts = alerts

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        """世界坐标 (米) → 像素坐标。"""
        x_min, y_min = self._bounds[0], self._bounds[1]
        px = int(self._margin + (wx - x_min) * self._scale_x)
        py = int(self._margin + (wy - y_min) * self._scale_y)
        return (px, py)

    def render(self, persons: List[Dict[str, Any]], step_count: int = 0) -> np.ndarray:
        """
        渲染完整仿真帧。

        Args:
            persons: [{"id": int, "x": float, "y": float, "z": float,
                      "vx": float, "vy": float (可选)}, ...]
            step_count: 当前仿真步数

        Returns:
            BGR numpy 数组 (h, w, 3)
        """
        frame = np.full((self._h, self._w, 3), _C_BG, dtype=np.uint8)
        current_time = time.time()

        if not CV2_AVAILABLE:
            return frame

        # 1. 区域色块 (半透明)
        self._draw_zone_overlay(frame)

        # 2. 坐标网格
        self._draw_grid(frame)

        # 3. 黄色警戒线
        self._draw_yellow_line(frame)

        # 4. 机柜区域
        self._draw_cabinets(frame)

        # 5. 计算速度并更新缓存
        for p in persons:
            pid = p.get("id", 0)
            px, py = p.get("x", 0.0), p.get("y", 0.0)

            # 计算速度
            if pid in self._prev_pos and pid in self._prev_time:
                dt = current_time - self._prev_time[pid]
                if dt > 0.01:
                    prev_x, prev_y = self._prev_pos[pid]
                    vx = (px - prev_x) / dt
                    vy = (py - prev_y) / dt
                    self._velocities[pid] = (vx, vy)
            elif "vx" in p and "vy" in p:
                self._velocities[pid] = (p["vx"], p["vy"])

            self._prev_time[pid] = current_time

        # 6. 绘制人与人之间的距离线
        if self._show_distances and len(persons) > 1:
            self._draw_person_distances(frame, persons)

        # 7. 人员轨迹 + 标记 + 距离标注
        for p in persons:
            pid = p.get("id", 0)
            px, py = p.get("x", 0.0), p.get("y", 0.0)

            # 更新轨迹
            if pid not in self._trails:
                self._trails[pid] = deque(maxlen=self._max_trail)
            self._trails[pid].append((px, py))

            # 绘制轨迹
            self._draw_trail(frame, pid)

            # 绘制预测轨迹
            if self._show_predictions and pid in self._velocities:
                self._draw_prediction(frame, px, py, self._velocities[pid])

            # 绘制人员标记
            self._draw_person(frame, pid, px, py)

            # 绘制到围栏的距离
            if self._show_distances:
                self._draw_fence_distance(frame, pid, px, py)

            # 绘制速度矢量
            if self._show_velocities and pid in self._velocities:
                self._draw_velocity_vector(frame, px, py, self._velocities[pid])

            # 保存位置用于下一帧方向计算
            self._prev_pos[pid] = (px, py)

        # 8. 比例尺
        self._draw_scale_bar(frame)

        # 9. 信息栏
        self._draw_info_bar(frame, len(persons), step_count)

        # 10. 告警面板
        if self._alerts:
            self._draw_alert_panel(frame)

        return frame

    # ── 内部绘制方法 ──────────────────────────────────────

    def _draw_grid(self, frame: np.ndarray):
        """绘制带刻度坐标网格。"""
        x_min, y_min, x_max, y_max = self._bounds

        # 纵线 (每米一条)
        for mx in range(int(math.floor(x_min)), int(math.ceil(x_max)) + 1):
            px, _ = self.world_to_pixel(float(mx), y_min)
            color = _C_GRID_MAJOR if mx % 2 == 0 else _C_GRID
            cv2.line(frame, (px, self._margin), (px, self._h - self._margin), color, 1)
            # 底部刻度标签
            if mx % 2 == 0:
                cv2.putText(frame, f"{mx}m", (px - 10, self._h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, _C_GRID_LABEL, 1)

        # 横线 (每米一条)
        for my in range(int(math.floor(y_min)), int(math.ceil(y_max)) + 1):
            _, py = self.world_to_pixel(x_min, float(my))
            color = _C_GRID_MAJOR if my % 2 == 0 else _C_GRID
            cv2.line(frame, (self._margin, py), (self._w - self._margin, py), color, 1)
            # 左侧刻度标签
            if my % 2 == 0:
                cv2.putText(frame, f"{my}m", (4, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, _C_GRID_LABEL, 1)

    def _draw_zone_overlay(self, frame: np.ndarray):
        """绘制安全/警告/危险区域色块 (半透明)。"""
        x_min, y_min, x_max, y_max = self._bounds
        warn_margin = 0.5  # 警告区距离黄线的宽度 (米)

        overlay = frame.copy()

        # 安全区 (黄线以下)
        safe_top = self.world_to_pixel(x_min, y_min)
        safe_bot = self.world_to_pixel(x_max, max(y_min, self._yellow_line_y - warn_margin))
        cv2.rectangle(overlay, safe_top, safe_bot, _C_SAFE_ZONE, -1)

        # 警告区 (黄线附近)
        warn_top = self.world_to_pixel(x_min, self._yellow_line_y - warn_margin)
        warn_bot = self.world_to_pixel(x_max, self._yellow_line_y + warn_margin)
        cv2.rectangle(overlay, warn_top, warn_bot, _C_WARN_ZONE, -1)

        # 危险区 (黄线以上)
        danger_top = self.world_to_pixel(x_min, self._yellow_line_y + warn_margin)
        danger_bot = self.world_to_pixel(x_max, y_max)
        cv2.rectangle(overlay, danger_top, danger_bot, _C_DANGER_ZONE, -1)

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    def _draw_yellow_line(self, frame: np.ndarray):
        """绘制黄色警戒线 (虚线 + 标签)。"""
        x_min, x_max = self._bounds[0], self._bounds[2]
        p1 = self.world_to_pixel(x_min, self._yellow_line_y)
        p2 = self.world_to_pixel(x_max, self._yellow_line_y)

        # 虚线绘制
        dash_len = 12
        gap_len = 8
        x_start, y_val = p1
        x_end = p2[0]
        x = x_start
        while x < x_end:
            seg_end = min(x + dash_len, x_end)
            cv2.line(frame, (x, y_val), (seg_end, y_val), _C_YELLOW_LINE, 2)
            x = seg_end + gap_len

        # 标签
        label_x = x_end - 120
        cv2.putText(frame, "WARNING LINE", (max(label_x, self._margin), y_val - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _C_YELLOW_LINE, 1)

    def _draw_cabinets(self, frame: np.ndarray):
        """绘制机柜区域。"""
        for (cx, cy, cw, ch, name) in self._cabinets:
            p1 = self.world_to_pixel(cx - cw / 2, cy - ch / 2)
            p2 = self.world_to_pixel(cx + cw / 2, cy + ch / 2)

            # 虚线矩形
            self._draw_dashed_rect(frame, p1, p2, _C_CABINET, 1)

            # 标签
            label_pt = (p1[0] + 4, p1[1] - 6)
            cv2.putText(frame, name, label_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _C_CABINET, 1)

    def _draw_dashed_rect(self, frame, p1, p2, color, thickness):
        """绘制虚线矩形。"""
        x1, y1 = p1
        x2, y2 = p2
        dash = 8
        gap = 5
        # 四条边
        for (sx, sy, ex, ey) in [(x1, y1, x2, y1), (x2, y1, x2, y2),
                                  (x2, y2, x1, y2), (x1, y2, x1, y1)]:
            length = max(abs(ex - sx), abs(ey - sy))
            if length == 0:
                continue
            dx = (ex - sx) / length
            dy = (ey - sy) / length
            d = 0
            while d < length:
                seg_end = min(d + dash, length)
                cv2.line(frame,
                         (int(sx + dx * d), int(sy + dy * d)),
                         (int(sx + dx * seg_end), int(sy + dy * seg_end)),
                         color, thickness)
                d = seg_end + gap

    def _draw_trail(self, frame: np.ndarray, person_id: int):
        """绘制人员运动轨迹 (渐变尾迹)。"""
        trail = self._trails.get(person_id)
        if not trail or len(trail) < 2:
            return

        points = list(trail)
        n = len(points)
        for i in range(1, n):
            alpha = i / n  # 0→1 渐变
            b = int(_C_TRAIL[0] * alpha)
            g = int(_C_TRAIL[1] * alpha)
            r = int(_C_TRAIL[2] * alpha)
            p1 = self.world_to_pixel(points[i - 1][0], points[i - 1][1])
            p2 = self.world_to_pixel(points[i][0], points[i][1])
            cv2.line(frame, p1, p2, (b, g, r), 1, cv2.LINE_AA)

    def _draw_person(self, frame: np.ndarray, person_id: int,
                     wx: float, wy: float):
        """绘制人员标记 (圆 + 方向箭头 + 标签)。"""
        px, py = self.world_to_pixel(wx, wy)

        # 根据区域选择颜色
        color = self._zone_color(wy)

        # 实心圆
        cv2.circle(frame, (px, py), 10, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 10, _C_WHITE, 1, cv2.LINE_AA)

        # 方向箭头
        prev = self._prev_pos.get(person_id)
        if prev is not None:
            dx = wx - prev[0]
            dy = wy - prev[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0.01:
                ndx = dx / dist
                ndy = dy / dist
                arrow_len = 18
                ax = int(px + ndx * arrow_len)
                ay = int(py + ndy * arrow_len)
                cv2.arrowedLine(frame, (px, py), (ax, ay), color, 2,
                                cv2.LINE_AA, tipLength=0.4)

        # ID 标签
        label = f"P{person_id}"
        cv2.putText(frame, label, (px + 14, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _zone_color(self, wy: float) -> Tuple[int, int, int]:
        """根据 y 坐标返回区域颜色。"""
        warn_margin = 0.5
        if wy > self._yellow_line_y + warn_margin:
            return _C_PERSON_DANGER
        elif wy > self._yellow_line_y - warn_margin:
            return _C_PERSON_WARN
        return _C_PERSON_SAFE

    def _draw_scale_bar(self, frame: np.ndarray):
        """绘制比例尺 (右下角)。"""
        one_meter_px = int(self._scale_x * 1.0)
        x_end = self._w - self._margin
        y_pos = self._h - 18
        x_start = x_end - one_meter_px

        cv2.line(frame, (x_start, y_pos), (x_end, y_pos), _C_SCALE, 2)
        cv2.line(frame, (x_start, y_pos - 4), (x_start, y_pos + 4), _C_SCALE, 1)
        cv2.line(frame, (x_end, y_pos - 4), (x_end, y_pos + 4), _C_SCALE, 1)
        cv2.putText(frame, "1m", (x_start + one_meter_px // 2 - 8, y_pos - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _C_SCALE, 1)

    def _draw_info_bar(self, frame: np.ndarray, num_persons: int, step_count: int):
        """绘制顶部信息栏。"""
        bar_h = 28
        cv2.rectangle(frame, (0, 0), (self._w, bar_h), _C_INFO_BG, -1)

        # SIM 徽章
        cv2.rectangle(frame, (8, 4), (48, 24), (0, 140, 200), -1)
        cv2.putText(frame, "SIM", (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _C_WHITE, 1)

        # 信息文本
        scenario_label = self._scenario_name or "default"
        info = (f"Scenario: {scenario_label}  |  "
                f"Step: {step_count}  |  "
                f"Persons: {num_persons}")
        cv2.putText(frame, info, (56, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 190), 1)

        # 时间戳
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts, (self._w - 70, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 150), 1)

    def _draw_person_distances(self, frame: np.ndarray, persons: List[Dict[str, Any]]):
        """绘制人与人之间的距离线和标注。"""
        n = len(persons)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = persons[i]
                p2 = persons[j]
                x1, y1 = p1.get("x", 0.0), p1.get("y", 0.0)
                x2, y2 = p2.get("x", 0.0), p2.get("y", 0.0)

                # 计算距离
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # 像素坐标
                px1, py1 = self.world_to_pixel(x1, y1)
                px2, py2 = self.world_to_pixel(x2, y2)

                # 根据距离选择颜色
                if dist < 0.8:
                    color = _C_PERSON_DANGER
                elif dist < 1.5:
                    color = _C_PERSON_WARN
                else:
                    color = _C_DISTANCE_LINE

                # 绘制虚线
                self._draw_dashed_line(frame, (px1, py1), (px2, py2), color, 1)

                # 中点标注距离
                mx, my = (px1 + px2) // 2, (py1 + py2) // 2
                label = f"{dist:.2f}m"
                cv2.putText(frame, label, (mx - 15, my - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def _draw_dashed_line(self, frame: np.ndarray, p1: Tuple[int, int],
                          p2: Tuple[int, int], color: Tuple[int, int, int],
                          thickness: int = 1):
        """绘制虚线。"""
        x1, y1 = p1
        x2, y2 = p2
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 1:
            return

        dash = 6
        gap = 4
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length

        d = 0
        while d < length:
            seg_end = min(d + dash, length)
            cv2.line(frame,
                     (int(x1 + dx * d), int(y1 + dy * d)),
                     (int(x1 + dx * seg_end), int(y1 + dy * seg_end)),
                     color, thickness)
            d = seg_end + gap

    def _draw_fence_distance(self, frame: np.ndarray, person_id: int,
                             wx: float, wy: float):
        """绘制人员到围栏线的距离标注。"""
        dist = self._yellow_line_y - wy

        # 只在接近围栏时显示
        if dist > 1.5:
            return

        px, py = self.world_to_pixel(wx, wy)
        fence_px, fence_py = self.world_to_pixel(wx, self._yellow_line_y)

        # 根据距离选择颜色
        if dist <= 0:
            color = _C_PERSON_DANGER
        elif dist < 0.2:
            color = _C_PERSON_DANGER
        elif dist < 0.5:
            color = _C_PERSON_WARN
        else:
            color = _C_DISTANCE_LINE

        # 绘制垂直虚线到围栏
        self._draw_dashed_line(frame, (px, py), (fence_px, fence_py), color, 1)

        # 标注距离
        label_y = (py + fence_py) // 2
        label = f"{abs(dist):.2f}m"
        cv2.putText(frame, label, (px + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    def _draw_prediction(self, frame: np.ndarray, wx: float, wy: float,
                         velocity: Tuple[float, float]):
        """绘制轨迹预测线。"""
        vx, vy = velocity
        speed = math.sqrt(vx * vx + vy * vy)
        if speed < 0.1:
            return

        # 预测 1 秒后的位置
        pred_x = wx + vx * 1.0
        pred_y = wy + vy * 1.0

        px, py = self.world_to_pixel(wx, wy)
        pred_px, pred_py = self.world_to_pixel(pred_x, pred_y)

        # 绘制预测轨迹（虚线 + 圆圈）
        self._draw_dashed_line(frame, (px, py), (pred_px, pred_py), _C_PREDICTION, 1)
        cv2.circle(frame, (pred_px, pred_py), 5, _C_PREDICTION, 1, cv2.LINE_AA)

        # 标注预测位置
        cv2.putText(frame, "1s", (pred_px + 6, pred_py + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, _C_PREDICTION, 1, cv2.LINE_AA)

    def _draw_velocity_vector(self, frame: np.ndarray, wx: float, wy: float,
                              velocity: Tuple[float, float]):
        """绘制速度矢量箭头。"""
        vx, vy = velocity
        speed = math.sqrt(vx * vx + vy * vy)
        if speed < 0.05:
            return

        px, py = self.world_to_pixel(wx, wy)

        # 速度矢量缩放（1 m/s = 30 像素）
        scale = 30.0
        arrow_dx = int(vx * scale)
        arrow_dy = int(vy * scale)

        end_x = px + arrow_dx
        end_y = py + arrow_dy

        cv2.arrowedLine(frame, (px, py), (end_x, end_y), _C_VELOCITY, 2,
                        cv2.LINE_AA, tipLength=0.3)

        # 标注速度值
        speed_label = f"{speed:.1f}m/s"
        cv2.putText(frame, speed_label, (end_x + 3, end_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _C_VELOCITY, 1, cv2.LINE_AA)

    def _draw_alert_panel(self, frame: np.ndarray):
        """绘制告警面板。"""
        if not self._alerts:
            return

        # 面板位置（右上角）
        panel_w = 180
        panel_h = min(20 + len(self._alerts) * 18, 100)
        panel_x = self._w - panel_w - 10
        panel_y = 35

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), _C_ALERT_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 边框
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), _C_PERSON_DANGER, 1)

        # 标题
        cv2.putText(frame, "ALERTS", (panel_x + 5, panel_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _C_WHITE, 1)

        # 告警列表（最多显示 4 条）
        for i, alert in enumerate(self._alerts[:4]):
            y_pos = panel_y + 30 + i * 18
            level = alert.get("level", "warning")
            color = _C_PERSON_DANGER if level == "critical" else _C_PERSON_WARN

            msg = alert.get("message", "")[:22]  # 截断长消息
            cv2.putText(frame, msg, (panel_x + 5, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    def reset_trails(self):
        """清空所有轨迹缓冲。"""
        self._trails.clear()
        self._prev_pos.clear()
        self._prev_time.clear()
        self._velocities.clear()
        self._alerts.clear()
