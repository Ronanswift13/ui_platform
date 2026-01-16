"""
DeepSORT 多目标跟踪器
=====================

基于UPDATE.md优化方案实现:
- 深度特征关联
- 卡尔曼滤波预测
- 级联匹配策略
- 遮挡处理

适用于: 室内电子围栏人员跟踪、鸟类轨迹分析

参考文献:
- Simple Online and Realtime Tracking with a Deep Association Metric
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """跟踪器配置"""
    max_age: int = 30                    # 目标消失最大帧数
    min_hits: int = 3                    # 确认目标最小帧数
    iou_threshold: float = 0.3           # IOU匹配阈值
    feature_threshold: float = 0.5       # 特征匹配阈值
    max_feature_distance: float = 0.7    # 最大特征距离
    use_deep_features: bool = True       # 使用深度特征
    feature_dim: int = 128               # 特征维度
    n_init: int = 3                      # 初始化帧数
    max_iou_distance: float = 0.7


@dataclass
class TrackState:
    """跟踪状态"""
    TENTATIVE = 1                        # 暂定
    CONFIRMED = 2                        # 确认
    DELETED = 3                          # 删除


@dataclass
class Track:
    """跟踪目标"""
    track_id: int
    bbox: Dict[str, float]               # {x, y, width, height}
    state: int
    hits: int
    age: int
    time_since_update: int
    features: List[np.ndarray]
    class_id: int = 0
    class_name: str = "person"
    velocity: Optional[Tuple[float, float]] = None
    predicted_bbox: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class KalmanFilter:
    """
    卡尔曼滤波器

    用于预测目标运动轨迹
    """

    def __init__(self):
        # 状态向量: [x, y, w, h, vx, vy, vw, vh]
        self.dim_x = 8
        self.dim_z = 4

        # 状态转移矩阵
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh

        # 观测矩阵
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # 过程噪声
        self.Q = np.eye(self.dim_x) * 0.01
        self.Q[4:, 4:] *= 10  # 速度噪声更大

        # 观测噪声
        self.R = np.eye(self.dim_z) * 1

        # 初始协方差
        self.P = np.eye(self.dim_x) * 10
        self.P[4:, 4:] *= 100

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """初始化状态"""
        mean = np.zeros(self.dim_x)
        mean[:self.dim_z] = measurement

        covariance = self.P.copy()
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测下一状态"""
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """更新状态"""
        # 卡尔曼增益
        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)

        # 更新
        y = measurement - self.H @ mean
        mean = mean + K @ y
        covariance = (np.eye(self.dim_x) - K @ self.H) @ covariance

        return mean, covariance


class DeepSORTTracker:
    """
    DeepSORT 多目标跟踪器

    核心特点:
    1. 深度特征关联 - 使用外观特征提高匹配准确性
    2. 卡尔曼滤波 - 预测目标运动轨迹
    3. 级联匹配 - 优先匹配最近检测到的目标
    4. 遮挡处理 - 短暂遮挡后可恢复跟踪

    使用示例:
    ```python
    config = TrackerConfig(max_age=30, min_hits=3)
    tracker = DeepSORTTracker(config)

    # 更新跟踪
    for frame, detections in video:
        tracks = tracker.update(detections)
        for track in tracks:
            print(f"Track {track.track_id}: {track.bbox}")
    ```
    """

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._tracks: List[Track] = []
        self._next_id = 1
        self._kf = KalmanFilter()
        self._track_states: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        logger.info("[DeepSORT] 跟踪器初始化完成")

    def update(
        self,
        detections: List[Dict[str, Any]],
        features: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        更新跟踪

        Args:
            detections: 检测结果列表 [{bbox, confidence, class_id, class_name}, ...]
            features: 可选的深度特征 (N, feature_dim)

        Returns:
            当前活跃的跟踪列表
        """
        # 预测现有跟踪
        for track in self._tracks:
            self._predict(track)

        # 匹配检测和跟踪
        matches, unmatched_detections, unmatched_tracks = self._match(detections, features)

        # 更新匹配的跟踪
        for track_idx, det_idx in matches:
            self._update_track(self._tracks[track_idx], detections[det_idx], features, det_idx)

        # 处理未匹配的跟踪
        for track_idx in unmatched_tracks:
            self._mark_missed(self._tracks[track_idx])

        # 创建新跟踪
        for det_idx in unmatched_detections:
            self._initiate_track(detections[det_idx], features, det_idx)

        # 删除过期跟踪
        self._tracks = [t for t in self._tracks if t.state != TrackState.DELETED]

        # 返回确认的跟踪
        return [t for t in self._tracks if t.state == TrackState.CONFIRMED]

    def get_tracks(self) -> List[Track]:
        """获取所有活跃跟踪"""
        return [t for t in self._tracks if t.state != TrackState.DELETED]

    def get_track(self, track_id: int) -> Optional[Track]:
        """获取指定ID的跟踪"""
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def reset(self):
        """重置跟踪器"""
        self._tracks.clear()
        self._track_states.clear()
        self._next_id = 1

    # ==========================================================================
    # 内部方法
    # ==========================================================================

    def _predict(self, track: Track):
        """预测跟踪位置"""
        if track.track_id not in self._track_states:
            return

        mean, covariance = self._track_states[track.track_id]
        mean, covariance = self._kf.predict(mean, covariance)
        self._track_states[track.track_id] = (mean, covariance)

        # 更新预测bbox
        track.predicted_bbox = {
            "x": float(mean[0]),
            "y": float(mean[1]),
            "width": float(mean[2]),
            "height": float(mean[3]),
        }

        # 更新速度
        track.velocity = (float(mean[4]), float(mean[5]))

        track.age += 1
        track.time_since_update += 1

    def _match(
        self,
        detections: List[Dict[str, Any]],
        features: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """匹配检测和跟踪"""
        if not self._tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self._tracks)))

        # 计算代价矩阵
        cost_matrix = self._compute_cost_matrix(detections, features)

        # 使用贪心匹配
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self._tracks)))

        # 按代价排序匹配
        rows, cols = cost_matrix.shape
        for _ in range(min(rows, cols)):
            if cost_matrix.size == 0:
                break

            min_idx = np.argmin(cost_matrix)
            row, col = divmod(min_idx, cost_matrix.shape[1])

            if cost_matrix[row, col] < self.config.max_iou_distance:
                # 找到原始索引
                track_idx = unmatched_tracks[row]
                det_idx = unmatched_detections[col]
                matches.append((track_idx, det_idx))

                # 从待匹配列表移除
                unmatched_tracks.remove(track_idx)
                unmatched_detections.remove(det_idx)

                # 从代价矩阵移除
                cost_matrix = np.delete(cost_matrix, row, axis=0)
                cost_matrix = np.delete(cost_matrix, col, axis=1)
            else:
                break

        return matches, unmatched_detections, unmatched_tracks

    def _compute_cost_matrix(
        self,
        detections: List[Dict[str, Any]],
        features: Optional[np.ndarray],
    ) -> np.ndarray:
        """计算代价矩阵"""
        n_tracks = len(self._tracks)
        n_dets = len(detections)
        cost_matrix = np.ones((n_tracks, n_dets))

        for i, track in enumerate(self._tracks):
            for j, det in enumerate(detections):
                # IOU代价
                iou = self._iou(track.predicted_bbox or track.bbox, det["bbox"])
                iou_cost = 1 - iou

                # 特征代价
                feature_cost = 1.0
                if self.config.use_deep_features and features is not None and track.features:
                    det_feature = features[j] if j < len(features) else None
                    if det_feature is not None and len(track.features) > 0:
                        # 余弦距离
                        track_feature = np.mean(track.features[-5:], axis=0)  # 最近5个特征的均值
                        similarity = np.dot(track_feature, det_feature) / (
                            np.linalg.norm(track_feature) * np.linalg.norm(det_feature) + 1e-10
                        )
                        feature_cost = 1 - similarity

                # 综合代价
                cost_matrix[i, j] = 0.7 * iou_cost + 0.3 * feature_cost

        return cost_matrix

    def _update_track(
        self,
        track: Track,
        detection: Dict[str, Any],
        features: Optional[np.ndarray],
        det_idx: int,
    ):
        """更新跟踪"""
        # 更新卡尔曼滤波
        bbox = detection["bbox"]
        measurement = np.array([bbox["x"], bbox["y"], bbox["width"], bbox["height"]])

        if track.track_id in self._track_states:
            mean, covariance = self._track_states[track.track_id]
            mean, covariance = self._kf.update(mean, covariance, measurement)
            self._track_states[track.track_id] = (mean, covariance)

        # 更新bbox
        track.bbox = detection["bbox"]
        track.hits += 1
        track.time_since_update = 0

        # 更新特征
        if features is not None and det_idx < len(features):
            track.features.append(features[det_idx])
            # 保留最近的特征
            if len(track.features) > 100:
                track.features = track.features[-50:]

        # 更新状态
        if track.state == TrackState.TENTATIVE and track.hits >= self.config.min_hits:
            track.state = TrackState.CONFIRMED

    def _mark_missed(self, track: Track):
        """标记跟踪丢失"""
        if track.state == TrackState.TENTATIVE:
            track.state = TrackState.DELETED
        elif track.time_since_update > self.config.max_age:
            track.state = TrackState.DELETED

    def _initiate_track(
        self,
        detection: Dict[str, Any],
        features: Optional[np.ndarray],
        det_idx: int,
    ):
        """创建新跟踪"""
        bbox = detection["bbox"]
        measurement = np.array([bbox["x"], bbox["y"], bbox["width"], bbox["height"]])

        # 初始化卡尔曼滤波
        mean, covariance = self._kf.initiate(measurement)

        # 创建跟踪
        track = Track(
            track_id=self._next_id,
            bbox=bbox,
            state=TrackState.TENTATIVE,
            hits=1,
            age=1,
            time_since_update=0,
            features=[],
            class_id=detection.get("class_id", 0),
            class_name=detection.get("class_name", "person"),
        )

        # 添加特征
        if features is not None and det_idx < len(features):
            track.features.append(features[det_idx])

        self._tracks.append(track)
        self._track_states[self._next_id] = (mean, covariance)
        self._next_id += 1

    def _iou(self, box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """计算IOU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0


# =============================================================================
# 便捷函数
# =============================================================================

def create_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    use_deep_features: bool = True,
) -> DeepSORTTracker:
    """创建跟踪器"""
    config = TrackerConfig(
        max_age=max_age,
        min_hits=min_hits,
        use_deep_features=use_deep_features,
    )
    return DeepSORTTracker(config)
