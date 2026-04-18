"""
Indoor Fence V3.0 - Realtime Multi-Person Tracker
===================================================

实时多人轨迹跟踪管理器，提供：
- 轨迹历史记录（位置、速度、时间戳）
- 距离计算（人与人、人与围栏线）
- 轨迹预测（基于速度外推）
- 轨迹可视化数据输出

不依赖 adapters/ 或 standalone/，可独立测试。
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .hungarian import linear_assignment


class TrackStatus(str, Enum):
    """轨迹状态"""
    TENTATIVE = "tentative"  # 初始化中，未确认
    CONFIRMED = "confirmed"  # 已确认
    LOST = "lost"            # 丢失中
    DELETED = "deleted"      # 已删除


class DistanceLevel(str, Enum):
    """距离等级"""
    SAFE = "safe"            # 安全距离
    WARNING = "warning"      # 警告距离
    DANGER = "danger"        # 危险距离
    CRITICAL = "critical"    # 临界距离


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    timestamp: float = 0.0
    confidence: float = 1.0


@dataclass
class PersonTrack:
    """人员轨迹"""
    track_id: int
    status: TrackStatus = TrackStatus.TENTATIVE

    # 当前状态
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    confidence: float = 0.5

    # 轨迹历史
    history: deque = field(default_factory=lambda: deque(maxlen=120))  # 12秒@10Hz

    # 跟踪统计
    hits: int = 0
    age: int = 0
    time_since_update: int = 0

    # 距离信息
    distance_to_fence: float = float('inf')
    distance_level: DistanceLevel = DistanceLevel.SAFE

    # 预测轨迹
    predicted_positions: List[Tuple[float, float, float]] = field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistanceInfo:
    """距离信息"""
    from_track_id: int
    to_track_id: Optional[int] = None  # None 表示到围栏线
    distance: float = 0.0
    level: DistanceLevel = DistanceLevel.SAFE
    description: str = ""


@dataclass
class RealtimeTrackingResult:
    """实时跟踪结果"""
    timestamp: float
    tracks: List[PersonTrack]
    distances: List[DistanceInfo]
    alerts: List[Dict[str, Any]]
    stats: Dict[str, Any]


class RealtimeMultiPersonTracker:
    """
    实时多人轨迹跟踪管理器

    功能：
    1. 管理多人轨迹生命周期
    2. 计算人与人、人与围栏的距离
    3. 预测未来轨迹
    4. 生成告警
    """

    # 距离阈值（米）
    FENCE_WARNING_DISTANCE = 0.5
    FENCE_DANGER_DISTANCE = 0.2
    FENCE_CRITICAL_DISTANCE = 0.0

    PERSON_WARNING_DISTANCE = 1.5
    PERSON_DANGER_DISTANCE = 0.8

    def __init__(
        self,
        fence_line_y: float = 3.0,
        max_age: int = 30,
        min_hits: int = 3,
        prediction_horizon: int = 10,  # 预测帧数
        history_length: int = 120,     # 历史长度
    ):
        self._fence_line_y = fence_line_y
        self._max_age = max_age
        self._min_hits = min_hits
        self._prediction_horizon = prediction_horizon
        self._history_length = history_length

        self._tracks: Dict[int, PersonTrack] = {}
        self._next_id: int = 0

        # 统计
        self._frame_count = 0
        self._total_alerts = 0

    @property
    def fence_line_y(self) -> float:
        return self._fence_line_y

    @fence_line_y.setter
    def fence_line_y(self, value: float):
        self._fence_line_y = value

    def update(
        self,
        detections: List[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> RealtimeTrackingResult:
        """
        更新跟踪器

        Args:
            detections: 检测结果列表，每个包含 position, velocity, confidence
            timestamp: 时间戳，默认使用当前时间

        Returns:
            RealtimeTrackingResult: 跟踪结果
        """
        ts = timestamp if timestamp is not None else time.time()
        self._frame_count += 1

        # 1. 预测所有轨迹
        self._predict_all(ts)

        # 2. 关联检测与轨迹
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        # 3. 更新匹配的轨迹
        for det_idx, track_id in matched:
            self._update_track(track_id, detections[det_idx], ts)

        # 4. 创建新轨迹
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], ts)

        # 5. 处理未匹配轨迹
        for track_id in unmatched_tracks:
            self._mark_missed(track_id)

        # 6. 删除过期轨迹
        self._prune_tracks()

        # 7. 计算距离
        distances = self._compute_distances()

        # 8. 生成预测轨迹
        self._generate_predictions()

        # 9. 生成告警
        alerts = self._generate_alerts(distances)

        # 10. 构建结果
        return RealtimeTrackingResult(
            timestamp=ts,
            tracks=list(self._tracks.values()),
            distances=distances,
            alerts=alerts,
            stats={
                "frame_count": self._frame_count,
                "active_tracks": len([t for t in self._tracks.values()
                                     if t.status == TrackStatus.CONFIRMED]),
                "total_tracks": len(self._tracks),
                "total_alerts": self._total_alerts,
            }
        )

    def _predict_all(self, ts: float) -> None:
        """预测所有轨迹位置"""
        for track in self._tracks.values():
            track.age += 1
            track.time_since_update += 1

            # 使用速度预测位置
            if track.history:
                last_point = track.history[-1]
                dt = ts - last_point.timestamp
                if dt > 0 and dt < 1.0:  # 最多预测1秒
                    vx, vy, vz = track.velocity
                    x, y, z = track.position
                    track.position = (
                        x + vx * dt,
                        y + vy * dt,
                        z + vz * dt,
                    )

    def _associate(
        self,
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """关联检测与轨迹（Hungarian 最优分配 + 3D 距离）"""
        if not detections or not self._tracks:
            return [], list(range(len(detections))), list(self._tracks.keys())

        track_ids = list(self._tracks.keys())
        n_dets = len(detections)
        n_tracks = len(track_ids)

        # 构建 3D 距离代价矩阵
        cost_matrix = np.zeros((n_dets, n_tracks))
        for i, det in enumerate(detections):
            det_pos = det.get("position", (0, 0, 0))
            for j, tid in enumerate(track_ids):
                track = self._tracks[tid]
                dx = det_pos[0] - track.position[0]
                dy = det_pos[1] - track.position[1]
                dz = det_pos[2] - track.position[2] if len(det_pos) >= 3 else 0.0
                cost_matrix[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz)

        max_distance = 2.0

        # 使用 Hungarian 算法求解最优分配
        # linear_assignment 期望 (rows, cols) = (detections, tracks)
        assignments = linear_assignment(cost_matrix)

        matched = []
        matched_det_set: set = set()
        matched_track_set: set = set()

        for det_idx, track_idx in assignments:
            if cost_matrix[det_idx, track_idx] <= max_distance:
                tid = track_ids[track_idx]
                matched.append((det_idx, tid))
                matched_det_set.add(det_idx)
                matched_track_set.add(tid)

        unmatched_dets = [i for i in range(n_dets) if i not in matched_det_set]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_track_set]

        return matched, unmatched_dets, unmatched_tracks

    def _update_track(
        self,
        track_id: int,
        detection: Dict[str, Any],
        ts: float,
    ) -> None:
        """更新轨迹"""
        track = self._tracks[track_id]

        pos = detection.get("position", (0, 0, 0))
        vel = detection.get("velocity", (0, 0, 0))
        conf = detection.get("confidence", 0.8)

        # 计算速度（如果检测未提供）
        if vel == (0, 0, 0) and track.history:
            last = track.history[-1]
            dt = ts - last.timestamp
            if dt > 0.01:
                vel = (
                    (pos[0] - last.x) / dt,
                    (pos[1] - last.y) / dt,
                    (pos[2] - last.z) / dt,
                )

        # 更新状态
        track.position = pos
        track.velocity = vel
        track.confidence = conf
        track.hits += 1
        track.time_since_update = 0

        # 确认轨迹
        if track.hits >= self._min_hits:
            track.status = TrackStatus.CONFIRMED

        # 添加历史点
        track.history.append(TrajectoryPoint(
            x=pos[0], y=pos[1], z=pos[2],
            vx=vel[0], vy=vel[1], vz=vel[2],
            timestamp=ts,
            confidence=conf,
        ))

        # 计算到围栏距离
        track.distance_to_fence = self._fence_line_y - pos[1]
        track.distance_level = self._classify_fence_distance(track.distance_to_fence)

    def _create_track(self, detection: Dict[str, Any], ts: float) -> PersonTrack:
        """创建新轨迹"""
        track_id = self._next_id
        self._next_id += 1

        pos = detection.get("position", (0, 0, 0))
        vel = detection.get("velocity", (0, 0, 0))
        conf = detection.get("confidence", 0.5)

        track = PersonTrack(
            track_id=track_id,
            status=TrackStatus.TENTATIVE,
            position=pos,
            velocity=vel,
            confidence=conf,
            hits=1,
            age=1,
            time_since_update=0,
            history=deque(maxlen=self._history_length),
        )

        track.history.append(TrajectoryPoint(
            x=pos[0], y=pos[1], z=pos[2],
            vx=vel[0], vy=vel[1], vz=vel[2],
            timestamp=ts,
            confidence=conf,
        ))

        track.distance_to_fence = self._fence_line_y - pos[1]
        track.distance_level = self._classify_fence_distance(track.distance_to_fence)

        self._tracks[track_id] = track
        return track

    def _mark_missed(self, track_id: int) -> None:
        """标记轨迹丢失"""
        track = self._tracks[track_id]
        if track.status == TrackStatus.CONFIRMED:
            track.status = TrackStatus.LOST

    def _prune_tracks(self) -> None:
        """删除过期轨迹"""
        to_delete = []
        for tid, track in self._tracks.items():
            if track.time_since_update > self._max_age:
                track.status = TrackStatus.DELETED
                to_delete.append(tid)
            elif track.status == TrackStatus.TENTATIVE and track.time_since_update > 3:
                to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

    def _compute_distances(self) -> List[DistanceInfo]:
        """计算所有距离"""
        distances = []
        confirmed_tracks = [t for t in self._tracks.values()
                          if t.status == TrackStatus.CONFIRMED]

        # 人与围栏距离
        for track in confirmed_tracks:
            dist = track.distance_to_fence
            level = track.distance_level
            distances.append(DistanceInfo(
                from_track_id=track.track_id,
                to_track_id=None,
                distance=dist,
                level=level,
                description=f"P{track.track_id} 距围栏 {dist:.2f}m",
            ))

        # 人与人距离
        n = len(confirmed_tracks)
        for i in range(n):
            for j in range(i + 1, n):
                t1 = confirmed_tracks[i]
                t2 = confirmed_tracks[j]
                dx = t1.position[0] - t2.position[0]
                dy = t1.position[1] - t2.position[1]
                dist = math.sqrt(dx * dx + dy * dy)
                level = self._classify_person_distance(dist)
                distances.append(DistanceInfo(
                    from_track_id=t1.track_id,
                    to_track_id=t2.track_id,
                    distance=dist,
                    level=level,
                    description=f"P{t1.track_id}-P{t2.track_id}: {dist:.2f}m",
                ))

        return distances

    def _classify_fence_distance(self, dist: float) -> DistanceLevel:
        """分类围栏距离等级"""
        if dist <= self.FENCE_CRITICAL_DISTANCE:
            return DistanceLevel.CRITICAL
        elif dist <= self.FENCE_DANGER_DISTANCE:
            return DistanceLevel.DANGER
        elif dist <= self.FENCE_WARNING_DISTANCE:
            return DistanceLevel.WARNING
        return DistanceLevel.SAFE

    def _classify_person_distance(self, dist: float) -> DistanceLevel:
        """分类人员间距离等级"""
        if dist <= self.PERSON_DANGER_DISTANCE:
            return DistanceLevel.DANGER
        elif dist <= self.PERSON_WARNING_DISTANCE:
            return DistanceLevel.WARNING
        return DistanceLevel.SAFE

    def _generate_predictions(self) -> None:
        """生成预测轨迹"""
        dt = 0.1  # 预测步长
        for track in self._tracks.values():
            if track.status != TrackStatus.CONFIRMED:
                track.predicted_positions = []
                continue

            predictions = []
            x, y, z = track.position
            vx, vy, vz = track.velocity

            for i in range(1, self._prediction_horizon + 1):
                px = x + vx * dt * i
                py = y + vy * dt * i
                pz = z + vz * dt * i
                predictions.append((px, py, pz))

            track.predicted_positions = predictions

    def _generate_alerts(self, distances: List[DistanceInfo]) -> List[Dict[str, Any]]:
        """生成告警"""
        alerts = []

        for dist_info in distances:
            if dist_info.level in (DistanceLevel.DANGER, DistanceLevel.CRITICAL):
                alert_type = "fence_breach" if dist_info.to_track_id is None else "proximity"
                level = "critical" if dist_info.level == DistanceLevel.CRITICAL else "warning"

                alert = {
                    "type": alert_type,
                    "level": level,
                    "track_id": dist_info.from_track_id,
                    "target_id": dist_info.to_track_id,
                    "distance": dist_info.distance,
                    "message": dist_info.description,
                    "timestamp": time.time(),
                }
                alerts.append(alert)
                self._total_alerts += 1

        return alerts

    def get_track(self, track_id: int) -> Optional[PersonTrack]:
        """获取指定轨迹"""
        return self._tracks.get(track_id)

    def get_confirmed_tracks(self) -> List[PersonTrack]:
        """获取所有已确认轨迹"""
        return [t for t in self._tracks.values() if t.status == TrackStatus.CONFIRMED]

    def get_all_tracks(self) -> List[PersonTrack]:
        """获取所有轨迹"""
        return list(self._tracks.values())

    def get_trajectory_data(self, track_id: int) -> List[Dict[str, Any]]:
        """获取轨迹历史数据（用于可视化）"""
        track = self._tracks.get(track_id)
        if not track:
            return []

        return [
            {
                "x": p.x, "y": p.y, "z": p.z,
                "vx": p.vx, "vy": p.vy, "vz": p.vz,
                "timestamp": p.timestamp,
                "confidence": p.confidence,
            }
            for p in track.history
        ]

    def reset(self) -> None:
        """重置跟踪器"""
        self._tracks.clear()
        self._next_id = 0
        self._frame_count = 0
        self._total_alerts = 0


__all__ = [
    'TrackStatus',
    'DistanceLevel',
    'TrajectoryPoint',
    'PersonTrack',
    'DistanceInfo',
    'RealtimeTrackingResult',
    'RealtimeMultiPersonTracker',
]
