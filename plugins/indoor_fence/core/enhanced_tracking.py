#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版多目标跟踪模块
==========================================

基于UPDATE.md优化方案实现:
- 集成DeepSORT深度特征关联
- 改进的卡尔曼滤波预测
- 级联匹配策略
- 遮挡处理和重识别
- 长期跟踪支持

作者: G组 | 版本: 3.0.0
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import numpy as np

from .geometry import Point2D

# 尝试导入深度学习跟踪器
try:
    from ai_models.deep_learning.deep_sort_tracker import DeepSORTTracker, TrackerConfig, Track as DeepTrack
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """单帧检测结果"""
    detection_id: str           # 临时检测ID
    position: Point2D           # 位置
    confidence: float = 1.0     # 置信度
    bbox: Optional[Dict] = None # 边界框 {x, y, w, h}
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"     # 来源: visual, lidar, fused
    feature: Optional[np.ndarray] = None  # 深度特征向量
    metadata: Dict = field(default_factory=dict)


@dataclass
class Track:
    """跟踪轨迹"""
    track_id: str               # 稳定的轨迹ID
    position: Point2D           # 当前位置
    velocity: Point2D           # 速度 (米/秒)
    confidence: float = 1.0     # 跟踪置信度

    # 卡尔曼滤波状态
    state: Optional[np.ndarray] = None          # [x, y, vx, vy]
    covariance: Optional[np.ndarray] = None     # 协方差矩阵

    # 历史数据
    position_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # 跟踪状态
    age: int = 0                # 跟踪帧数
    hits: int = 0               # 匹配次数
    misses: int = 0             # 连续未匹配次数
    is_confirmed: bool = False  # 是否已确认

    # 时间戳
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # 关联的检测
    last_detection: Optional[Detection] = None

    # 深度特征 (用于重识别)
    features: List[np.ndarray] = field(default_factory=list)

    # 遮挡状态
    is_occluded: bool = False
    occlusion_count: int = 0

    # 元数据
    metadata: Dict = field(default_factory=dict)

    def get_feature_vector(self) -> Optional[np.ndarray]:
        """获取平均特征向量"""
        if not self.features:
            return None
        return np.mean(self.features[-10:], axis=0)


class KalmanFilter2D:
    """
    2D卡尔曼滤波器 (增强版)

    状态向量: [x, y, vx, vy, ax, ay]
    观测向量: [x, y]

    改进:
    - 支持加速度建模
    - 自适应噪声估计
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
        use_acceleration: bool = False
    ):
        """
        Args:
            dt: 时间步长 (秒)
            process_noise: 过程噪声
            measurement_noise: 测量噪声
            use_acceleration: 是否使用加速度模型
        """
        self.dt = dt
        self.use_acceleration = use_acceleration

        if use_acceleration:
            self.dim_x = 6  # [x, y, vx, vy, ax, ay]
            self.dim_z = 2

            # 状态转移矩阵 (恒加速度模型)
            self.F = np.array([
                [1, 0, dt, 0, 0.5*dt**2, 0],
                [0, 1, 0, dt, 0, 0.5*dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float64)

            # 观测矩阵
            self.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ], dtype=np.float64)

            # 过程噪声协方差
            q = process_noise
            self.Q = np.eye(6, dtype=np.float64) * q
            self.Q[4, 4] *= 10  # 加速度噪声更大
            self.Q[5, 5] *= 10

            # 初始协方差
            self.P0 = np.eye(6, dtype=np.float64) * 10
            self.P0[4:, 4:] *= 100
        else:
            self.dim_x = 4  # [x, y, vx, vy]
            self.dim_z = 2

            # 状态转移矩阵 F
            self.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)

            # 观测矩阵 H
            self.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float64)

            # 过程噪声协方差 Q
            q = process_noise
            self.Q = np.array([
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ], dtype=np.float64) * q

            # 初始协方差
            self.P0 = np.eye(4, dtype=np.float64) * 10

        # 测量噪声协方差 R
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # 自适应噪声参数
        self._innovation_history: deque = deque(maxlen=10)

    def init_state(self, x: float, y: float, vx: float = 0, vy: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """初始化状态"""
        if self.use_acceleration:
            state = np.array([x, y, vx, vy, 0, 0], dtype=np.float64)
        else:
            state = np.array([x, y, vx, vy], dtype=np.float64)
        covariance = self.P0.copy()
        return state, covariance

    def predict(self, state: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测步骤"""
        state_pred = self.F @ state
        cov_pred = self.F @ covariance @ self.F.T + self.Q
        return state_pred, cov_pred

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """更新步骤"""
        # 创新
        y = measurement - self.H @ state

        # 记录创新用于自适应噪声
        self._innovation_history.append(y)

        # 创新协方差
        S = self.H @ covariance @ self.H.T + self.R

        # 卡尔曼增益
        K = covariance @ self.H.T @ np.linalg.inv(S)

        # 更新状态
        state_new = state + K @ y

        # 更新协方差 (Joseph形式,数值更稳定)
        I = np.eye(self.dim_x)
        IKH = I - K @ self.H
        cov_new = IKH @ covariance @ IKH.T + K @ self.R @ K.T

        return state_new, cov_new

    def adapt_noise(self):
        """自适应噪声估计"""
        if len(self._innovation_history) < 5:
            return

        innovations = np.array(self._innovation_history)
        # 估计测量噪声
        estimated_R = np.cov(innovations.T)
        # 平滑更新
        self.R = 0.9 * self.R + 0.1 * estimated_R


class EnhancedMultiTargetTracker:
    """
    增强版多目标跟踪器

    新增特性:
    1. 可选DeepSORT深度特征关联
    2. 级联匹配策略
    3. 遮挡处理
    4. 长期轨迹重识别
    5. 自适应参数调整
    """

    def __init__(
        self,
        max_age: int = 30,              # 最大未匹配帧数
        min_hits: int = 3,              # 确认所需最小匹配数
        iou_threshold: float = 0.3,     # IoU匹配阈值
        distance_threshold: float = 1.0, # 距离匹配阈值 (米)
        use_kalman: bool = True,        # 是否使用卡尔曼滤波
        use_deep_features: bool = True, # 是否使用深度特征
        feature_threshold: float = 0.5, # 特征匹配阈值
        use_acceleration: bool = False, # 是否使用加速度模型
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.use_kalman = use_kalman
        self.use_deep_features = use_deep_features and DEEPSORT_AVAILABLE
        self.feature_threshold = feature_threshold

        # 跟踪列表
        self.tracks: Dict[str, Track] = {}

        # 卡尔曼滤波器
        self.kf = KalmanFilter2D(
            dt=0.1,
            use_acceleration=use_acceleration
        ) if use_kalman else None

        # DeepSORT跟踪器 (可选)
        self._deep_tracker: Optional[DeepSORTTracker] = None
        if self.use_deep_features and DEEPSORT_AVAILABLE:
            try:
                config = TrackerConfig(
                    max_age=max_age,
                    min_hits=min_hits,
                    feature_threshold=feature_threshold,
                    use_deep_features=True
                )
                self._deep_tracker = DeepSORTTracker(config)
                logger.info("[Tracker] DeepSORT已启用")
            except Exception as e:
                logger.warning(f"[Tracker] DeepSORT初始化失败: {e}")
                self._deep_tracker = None

        # ID计数器
        self._next_id = 1

        # 移动平均窗口
        self._smooth_window = 5

        # 已删除轨迹的特征库 (用于重识别)
        self._deleted_features: Dict[str, List[np.ndarray]] = {}
        self._deleted_features_max = 50  # 最多保留50个

        # 统计
        self._total_tracks = 0
        self._reidentified_count = 0

        logger.info(f"[Tracker] 增强版多目标跟踪器初始化: max_age={max_age}, "
                   f"min_hits={min_hits}, deep_features={self.use_deep_features}")

    def _generate_track_id(self) -> str:
        """生成新的轨迹ID"""
        track_id = f"T{self._next_id:04d}"
        self._next_id += 1
        self._total_tracks += 1
        return track_id

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        更新跟踪器

        Args:
            detections: 当前帧检测结果列表

        Returns:
            List[Track]: 已确认的轨迹列表
        """
        current_time = time.time()

        # 如果有DeepSORT且检测有特征,使用DeepSORT
        if self._deep_tracker and detections and detections[0].feature is not None:
            return self._update_with_deep_sort(detections)

        # 否则使用增强的传统方法
        return self._update_traditional(detections)

    def _update_with_deep_sort(self, detections: List[Detection]) -> List[Track]:
        """使用DeepSORT更新"""
        # 转换检测格式
        det_list = []
        features = []

        for det in detections:
            bbox = det.bbox or {"x": det.position.x, "y": det.position.y, "width": 0.5, "height": 1.5}
            det_list.append({
                "bbox": bbox,
                "confidence": det.confidence,
                "class_id": 0,
                "class_name": "person"
            })
            if det.feature is not None:
                features.append(det.feature)

        features_array = np.array(features) if features else None

        # DeepSORT更新
        deep_tracks = self._deep_tracker.update(det_list, features_array)

        # 转换回Track格式并同步
        confirmed = []
        for dt in deep_tracks:
            track_id = f"T{dt.track_id:04d}"

            if track_id not in self.tracks:
                # 新轨迹
                self.tracks[track_id] = Track(
                    track_id=track_id,
                    position=Point2D(dt.bbox["x"], dt.bbox["y"]),
                    velocity=Point2D(dt.velocity[0] if dt.velocity else 0,
                                    dt.velocity[1] if dt.velocity else 0),
                    confidence=1.0,
                    is_confirmed=True,
                    first_seen=current_time,
                    last_seen=current_time,
                )

            track = self.tracks[track_id]
            track.position = Point2D(dt.bbox["x"], dt.bbox["y"])
            track.velocity = Point2D(
                dt.velocity[0] if dt.velocity else 0,
                dt.velocity[1] if dt.velocity else 0
            )
            track.hits = dt.hits
            track.age = dt.age
            track.misses = dt.time_since_update
            track.last_seen = current_time
            track.is_confirmed = True

            # 保存特征
            if dt.features:
                track.features = dt.features[-10:]

            confirmed.append(track)

        return confirmed

    def _update_traditional(self, detections: List[Detection]) -> List[Track]:
        """传统方法更新"""
        current_time = time.time()

        # 1. 预测所有轨迹
        for track in self.tracks.values():
            self._predict_track(track)

        # 2. 级联匹配
        matched, unmatched_dets, unmatched_tracks = self._cascade_associate(detections)

        # 3. 更新已匹配的轨迹
        for track_id, det_idx in matched:
            track = self.tracks[track_id]
            detection = detections[det_idx]
            self._update_track(track, detection)

        # 4. 处理未匹配的轨迹 (可能被遮挡)
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.misses += 1
            track.age += 1
            track.is_occluded = track.misses > 2

        # 5. 尝试重识别或创建新轨迹
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            reident_id = self._try_reidentify(detection)
            if reident_id:
                self._reidentify_track(reident_id, detection)
            else:
                self._create_track(detection)

        # 6. 删除过期轨迹
        self._remove_stale_tracks()

        # 7. 返回已确认的轨迹
        return [t for t in self.tracks.values() if t.is_confirmed]

    def _predict_track(self, track: Track):
        """预测轨迹位置"""
        if self.use_kalman and track.state is not None:
            track.state, track.covariance = self.kf.predict(track.state, track.covariance)
            track.position = Point2D(track.state[0], track.state[1])
            track.velocity = Point2D(track.state[2], track.state[3])

    def _cascade_associate(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[str, int]], List[int], List[str]]:
        """
        级联匹配策略

        1. 首先匹配最近更新的轨迹
        2. 然后匹配较旧的轨迹
        3. 使用特征和距离综合打分
        """
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        if not detections or not self.tracks:
            return matched, unmatched_dets, unmatched_tracks

        # 按最近更新时间排序轨迹
        sorted_tracks = sorted(
            self.tracks.keys(),
            key=lambda tid: self.tracks[tid].last_seen,
            reverse=True
        )

        # 级联匹配 (按age分组)
        for max_miss in range(3):  # 0, 1, 2
            if not unmatched_dets:
                break

            track_ids_at_level = [
                tid for tid in sorted_tracks
                if tid in unmatched_tracks and self.tracks[tid].misses == max_miss
            ]

            if not track_ids_at_level:
                continue

            # 构建代价矩阵
            cost_matrix = self._build_cost_matrix(
                track_ids_at_level,
                [detections[i] for i in unmatched_dets]
            )

            # 匈牙利匹配
            row_indices, col_indices = self._hungarian_match(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < self.distance_threshold:
                    track_id = track_ids_at_level[row]
                    det_idx = unmatched_dets[col]
                    matched.append((track_id, det_idx))
                    unmatched_tracks.remove(track_id)

            # 更新未匹配检测
            matched_cols = [col for _, col in zip(row_indices, col_indices)
                          if cost_matrix[_, col] < self.distance_threshold]
            unmatched_dets = [
                unmatched_dets[i] for i in range(len(unmatched_dets))
                if i not in matched_cols
            ]

        return matched, unmatched_dets, unmatched_tracks

    def _build_cost_matrix(
        self,
        track_ids: List[str],
        detections: List[Detection]
    ) -> np.ndarray:
        """构建代价矩阵"""
        n_tracks = len(track_ids)
        n_dets = len(detections)
        cost_matrix = np.ones((n_tracks, n_dets)) * 1e5

        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            for j, det in enumerate(detections):
                # 距离代价
                dist = track.position.distance_to(det.position)
                distance_cost = dist

                # 特征代价
                feature_cost = 1.0
                if self.use_deep_features and det.feature is not None:
                    track_feat = track.get_feature_vector()
                    if track_feat is not None:
                        similarity = np.dot(track_feat, det.feature) / (
                            np.linalg.norm(track_feat) * np.linalg.norm(det.feature) + 1e-10
                        )
                        feature_cost = 1 - similarity

                # 综合代价
                cost_matrix[i, j] = 0.7 * distance_cost + 0.3 * feature_cost

        return cost_matrix

    def _hungarian_match(
        self,
        cost_matrix: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """匈牙利算法匹配 (简化版贪心实现)"""
        rows, cols = cost_matrix.shape
        matched_rows = []
        matched_cols = []

        cost_copy = cost_matrix.copy()

        for _ in range(min(rows, cols)):
            if cost_copy.size == 0:
                break

            min_idx = np.argmin(cost_copy)
            row, col = divmod(min_idx, cost_copy.shape[1])

            if cost_copy[row, col] < 1e4:
                matched_rows.append(row)
                matched_cols.append(col)

                # 标记已匹配
                cost_copy[row, :] = 1e5
                cost_copy[:, col] = 1e5
            else:
                break

        return matched_rows, matched_cols

    def _update_track(self, track: Track, detection: Detection):
        """更新轨迹"""
        current_time = time.time()

        # 卡尔曼更新
        if self.use_kalman and track.state is not None:
            measurement = np.array([detection.position.x, detection.position.y])
            track.state, track.covariance = self.kf.update(
                track.state, track.covariance, measurement
            )
            track.position = Point2D(track.state[0], track.state[1])
            track.velocity = Point2D(track.state[2], track.state[3])
        else:
            # 移动平均平滑
            track.position = self._smooth_position(track, detection.position)

        # 更新历史
        track.position_history.append(track.position)

        # 更新状态
        track.hits += 1
        track.misses = 0
        track.age += 1
        track.last_seen = current_time
        track.last_detection = detection
        track.confidence = detection.confidence
        track.is_occluded = False
        track.occlusion_count = 0

        # 保存特征
        if detection.feature is not None:
            track.features.append(detection.feature)
            if len(track.features) > 20:
                track.features = track.features[-20:]

        # 确认轨迹
        if track.hits >= self.min_hits:
            track.is_confirmed = True

    def _smooth_position(self, track: Track, new_pos: Point2D) -> Point2D:
        """移动平均平滑位置"""
        history = list(track.position_history)
        history.append(new_pos)

        if len(history) < 2:
            return new_pos

        # 取最近N个点的平均
        recent = history[-self._smooth_window:]
        avg_x = sum(p.x for p in recent) / len(recent)
        avg_y = sum(p.y for p in recent) / len(recent)

        return Point2D(avg_x, avg_y)

    def _create_track(self, detection: Detection):
        """创建新轨迹"""
        track_id = self._generate_track_id()

        track = Track(
            track_id=track_id,
            position=detection.position,
            velocity=Point2D(0, 0),
            confidence=detection.confidence,
            last_detection=detection,
        )

        # 初始化卡尔曼状态
        if self.use_kalman:
            track.state, track.covariance = self.kf.init_state(
                detection.position.x, detection.position.y
            )

        track.position_history.append(detection.position)
        track.hits = 1
        track.age = 1

        # 保存特征
        if detection.feature is not None:
            track.features.append(detection.feature)

        self.tracks[track_id] = track
        logger.debug(f"创建新轨迹: {track_id}")

    def _try_reidentify(self, detection: Detection) -> Optional[str]:
        """尝试重识别"""
        if not self.use_deep_features or detection.feature is None:
            return None

        best_match = None
        best_similarity = self.feature_threshold

        for track_id, features in self._deleted_features.items():
            if not features:
                continue

            avg_feat = np.mean(features, axis=0)
            similarity = np.dot(avg_feat, detection.feature) / (
                np.linalg.norm(avg_feat) * np.linalg.norm(detection.feature) + 1e-10
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = track_id

        if best_match:
            self._reidentified_count += 1
            logger.info(f"重识别成功: {best_match}, 相似度: {best_similarity:.3f}")

        return best_match

    def _reidentify_track(self, old_track_id: str, detection: Detection):
        """恢复已删除的轨迹"""
        track = Track(
            track_id=old_track_id,
            position=detection.position,
            velocity=Point2D(0, 0),
            confidence=detection.confidence,
            last_detection=detection,
            is_confirmed=True,  # 直接确认
        )

        if self.use_kalman:
            track.state, track.covariance = self.kf.init_state(
                detection.position.x, detection.position.y
            )

        track.position_history.append(detection.position)
        track.hits = self.min_hits
        track.age = 1

        if detection.feature is not None:
            track.features.append(detection.feature)

        self.tracks[old_track_id] = track

        # 从已删除列表中移除
        if old_track_id in self._deleted_features:
            del self._deleted_features[old_track_id]

    def _remove_stale_tracks(self):
        """删除过期轨迹"""
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if track.misses > self.max_age
        ]

        for tid in stale_ids:
            track = self.tracks[tid]

            # 保存特征用于重识别
            if track.features and self.use_deep_features:
                if len(self._deleted_features) >= self._deleted_features_max:
                    # 删除最旧的
                    oldest = next(iter(self._deleted_features))
                    del self._deleted_features[oldest]
                self._deleted_features[tid] = track.features[-5:]

            del self.tracks[tid]
            logger.debug(f"删除过期轨迹: {tid}")

    def get_track(self, track_id: str) -> Optional[Track]:
        """获取指定轨迹"""
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[Track]:
        """获取所有轨迹"""
        return list(self.tracks.values())

    def get_confirmed_tracks(self) -> List[Track]:
        """获取已确认的轨迹"""
        return [t for t in self.tracks.values() if t.is_confirmed]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        confirmed = [t for t in self.tracks.values() if t.is_confirmed]
        occluded = [t for t in self.tracks.values() if t.is_occluded]

        return {
            "total_tracks_created": self._total_tracks,
            "active_tracks": len(self.tracks),
            "confirmed_tracks": len(confirmed),
            "occluded_tracks": len(occluded),
            "reidentified_count": self._reidentified_count,
            "deep_sort_enabled": self._deep_tracker is not None,
            "features_cached": len(self._deleted_features),
        }

    def reset(self):
        """重置跟踪器"""
        self.tracks.clear()
        self._deleted_features.clear()
        self._next_id = 1
        self._total_tracks = 0
        self._reidentified_count = 0

        if self._deep_tracker:
            self._deep_tracker.reset()

        logger.info("跟踪器已重置")


# 向后兼容别名
MultiTargetTracker = EnhancedMultiTargetTracker

__all__ = ['Detection', 'Track', 'KalmanFilter2D', 'MultiTargetTracker', 'EnhancedMultiTargetTracker']
