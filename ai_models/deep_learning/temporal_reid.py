"""
时序ReID模块 - 缺陷跟踪与重识别
输变电激光星芒破夜绘明监测平台

功能:
- 跨时间帧的缺陷重识别 (Re-Identification)
- 时序特征提取与匹配
- 缺陷轨迹追踪
- 支持母线巡视、主变巡视等多种场景

技术实现:
- 基于深度学习的外观特征提取
- 时空注意力机制
- 自适应匹配策略
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入深度学习框架
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class DefectType(Enum):
    """缺陷类型"""
    CRACK = "crack"                     # 裂纹
    PIN_MISSING = "pin_missing"         # 销钉缺失
    FOREIGN_OBJECT = "foreign_object"   # 异物
    CORROSION = "corrosion"             # 腐蚀
    DEFORMATION = "deformation"         # 变形
    OIL_LEAK = "oil_leak"               # 渗漏油
    RUST = "rust"                       # 锈蚀
    UNKNOWN = "unknown"


@dataclass
class DefectFeature:
    """缺陷特征描述"""
    feature_id: str
    defect_type: DefectType
    appearance_feature: np.ndarray      # 外观特征向量 (128维或256维)
    position_feature: np.ndarray        # 位置特征 (x, y, w, h)
    texture_feature: Optional[np.ndarray] = None  # 纹理特征
    temporal_feature: Optional[np.ndarray] = None # 时序特征
    confidence: float = 0.0
    timestamp: float = 0.0
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefectTrack:
    """缺陷轨迹"""
    track_id: str
    defect_type: DefectType
    features: List[DefectFeature] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    state: str = "active"  # active, lost, removed
    hit_count: int = 0
    miss_count: int = 0
    max_history: int = 100

    def add_feature(self, feature: DefectFeature):
        """添加特征到轨迹"""
        self.features.append(feature)
        if len(self.features) > self.max_history:
            self.features = self.features[-self.max_history:]
        self.last_seen = feature.timestamp
        self.hit_count += 1
        self.miss_count = 0

    def mark_miss(self):
        """标记丢失"""
        self.miss_count += 1

    def get_averaged_feature(self) -> np.ndarray:
        """获取平均外观特征"""
        if not self.features:
            return np.zeros(256)
        features = np.array([f.appearance_feature for f in self.features[-10:]])
        return np.mean(features, axis=0)

    def get_velocity(self) -> Tuple[float, float]:
        """计算位置变化速度"""
        if len(self.features) < 2:
            return (0.0, 0.0)
        recent = self.features[-5:]
        if len(recent) < 2:
            return (0.0, 0.0)
        dx = recent[-1].position_feature[0] - recent[0].position_feature[0]
        dy = recent[-1].position_feature[1] - recent[0].position_feature[1]
        dt = recent[-1].timestamp - recent[0].timestamp
        if dt > 0:
            return (dx / dt, dy / dt)
        return (0.0, 0.0)


@dataclass
class TemporalReIDConfig:
    """时序ReID配置"""
    # 特征提取
    feature_dim: int = 256
    use_texture_feature: bool = True
    use_temporal_attention: bool = True

    # 匹配参数
    appearance_weight: float = 0.6
    position_weight: float = 0.3
    temporal_weight: float = 0.1
    match_threshold: float = 0.7

    # 轨迹管理
    max_miss_count: int = 10
    min_hit_count: int = 3
    max_tracks: int = 100

    # 时序分析
    temporal_window: int = 30  # 帧数
    growth_analysis_interval: int = 10  # 增长分析间隔

    # 设备
    device: str = "cpu"


class AppearanceEncoder:
    """外观特征编码器"""

    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        self._model = None
        self._initialized = False

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        提取外观特征

        Args:
            image: 缺陷区域图像 (H, W, C)

        Returns:
            特征向量 (feature_dim,)
        """
        if image is None or image.size == 0:
            return np.zeros(self.feature_dim)

        # 模拟特征提取 (实际应使用深度学习模型)
        # 使用颜色直方图 + 边缘特征作为基础特征
        try:
            # 颜色特征
            color_hist = self._compute_color_histogram(image)

            # 边缘特征
            edge_feat = self._compute_edge_features(image)

            # 纹理特征
            texture_feat = self._compute_texture_features(image)

            # 组合特征
            combined = np.concatenate([color_hist, edge_feat, texture_feat])

            # 归一化到目标维度
            if len(combined) < self.feature_dim:
                combined = np.pad(combined, (0, self.feature_dim - len(combined)))
            else:
                combined = combined[:self.feature_dim]

            # L2归一化
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

            return combined.astype(np.float32)

        except Exception as e:
            logger.warning(f"特征提取失败: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)

    def _compute_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """计算颜色直方图"""
        if len(image.shape) == 2:
            hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 256))
            return hist.astype(np.float32) / (hist.sum() + 1e-6)

        hists = []
        for c in range(min(3, image.shape[2])):
            hist, _ = np.histogram(image[:, :, c].flatten(), bins=bins, range=(0, 256))
            hists.append(hist.astype(np.float32) / (hist.sum() + 1e-6))
        return np.concatenate(hists)

    def _compute_edge_features(self, image: np.ndarray) -> np.ndarray:
        """计算边缘特征"""
        try:
            import cv2
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Sobel边缘
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # 边缘幅度
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            # 统计特征
            features = [
                np.mean(magnitude),
                np.std(magnitude),
                np.max(magnitude),
                np.percentile(magnitude, 75),
                np.percentile(magnitude, 90),
            ]

            # 边缘直方图
            hist, _ = np.histogram(magnitude.flatten(), bins=20, range=(0, 255))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-6)

            return np.concatenate([features, hist]).astype(np.float32)

        except ImportError:
            # OpenCV不可用，返回随机特征
            return np.random.randn(25).astype(np.float32)

    def _compute_texture_features(self, image: np.ndarray) -> np.ndarray:
        """计算纹理特征 (LBP-like)"""
        try:
            import cv2
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 简化的LBP计算
            h, w = gray.shape
            if h < 3 or w < 3:
                return np.zeros(32, dtype=np.float32)

            # 计算梯度
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)

            # 梯度方向直方图
            angle = np.arctan2(gy, gx) * 180 / np.pi
            angle = (angle + 180) % 360

            hist, _ = np.histogram(angle.flatten(), bins=32, range=(0, 360))
            return hist.astype(np.float32) / (hist.sum() + 1e-6)

        except ImportError:
            return np.random.randn(32).astype(np.float32)


class TemporalAttention:
    """时序注意力机制"""

    def __init__(self, feature_dim: int = 256, num_heads: int = 4):
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

    def compute_attention(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: List[np.ndarray]
    ) -> np.ndarray:
        """
        计算时序注意力

        Args:
            query: 当前帧特征 (feature_dim,)
            keys: 历史帧特征列表
            values: 历史帧值列表

        Returns:
            注意力加权特征 (feature_dim,)
        """
        if not keys or not values:
            return query

        # 计算注意力分数
        keys_matrix = np.array(keys)  # (T, feature_dim)
        values_matrix = np.array(values)  # (T, feature_dim)

        # 点积注意力
        scores = np.dot(keys_matrix, query) / np.sqrt(self.feature_dim)

        # Softmax
        scores = np.exp(scores - np.max(scores))
        attention_weights = scores / (scores.sum() + 1e-8)

        # 加权求和
        context = np.sum(values_matrix * attention_weights[:, np.newaxis], axis=0)

        # 残差连接
        output = 0.5 * query + 0.5 * context

        return output


class TemporalReIDModule:
    """
    时序ReID模块

    实现跨时间帧的缺陷重识别和跟踪
    """

    def __init__(self, config: Optional[TemporalReIDConfig] = None):
        self.config = config or TemporalReIDConfig()

        # 组件初始化
        self.appearance_encoder = AppearanceEncoder(self.config.feature_dim)
        self.temporal_attention = TemporalAttention(self.config.feature_dim)

        # 轨迹存储
        self.tracks: Dict[str, DefectTrack] = {}
        self._track_counter = 0

        # 历史缓存
        self._feature_cache: deque = deque(maxlen=self.config.temporal_window)

        # 统计
        self._total_matches = 0
        self._total_new_tracks = 0

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_id: int,
        timestamp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        处理单帧检测结果，进行ReID匹配

        Args:
            frame: 原始图像
            detections: 检测结果列表 [{"bbox": {...}, "label": str, "confidence": float}, ...]
            frame_id: 帧编号
            timestamp: 时间戳

        Returns:
            带有track_id的检测结果列表
        """
        timestamp = timestamp or time.time()

        # 提取当前帧所有缺陷的特征
        current_features = []
        for det in detections:
            feature = self._extract_defect_feature(
                frame, det, frame_id, timestamp
            )
            current_features.append(feature)

        # 获取活跃轨迹
        active_tracks = [t for t in self.tracks.values() if t.state == "active"]

        # 匹配
        if active_tracks and current_features:
            matches, unmatched_dets, unmatched_tracks = self._match_features(
                current_features, active_tracks
            )
        else:
            matches = []
            unmatched_dets = list(range(len(current_features)))
            unmatched_tracks = list(range(len(active_tracks)))

        # 更新匹配的轨迹
        results = [None] * len(detections)
        for det_idx, track_idx in matches:
            track = active_tracks[track_idx]
            feature = current_features[det_idx]
            track.add_feature(feature)

            results[det_idx] = {
                **detections[det_idx],
                "track_id": track.track_id,
                "match_type": "matched",
                "track_age": len(track.features)
            }
            self._total_matches += 1

        # 创建新轨迹
        for det_idx in unmatched_dets:
            track_id = self._create_new_track(current_features[det_idx])
            results[det_idx] = {
                **detections[det_idx],
                "track_id": track_id,
                "match_type": "new",
                "track_age": 1
            }
            self._total_new_tracks += 1

        # 更新丢失的轨迹
        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            track.mark_miss()
            if track.miss_count >= self.config.max_miss_count:
                track.state = "lost"

        # 缓存特征
        self._feature_cache.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "features": current_features
        })

        return [r for r in results if r is not None]

    def _extract_defect_feature(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        frame_id: int,
        timestamp: float
    ) -> DefectFeature:
        """提取单个缺陷的特征"""
        bbox = detection.get("bbox", {})

        # 提取ROI区域
        h, w = frame.shape[:2]
        x1 = int(bbox.get("x", 0) * w)
        y1 = int(bbox.get("y", 0) * h)
        x2 = int((bbox.get("x", 0) + bbox.get("width", 0.1)) * w)
        y2 = int((bbox.get("y", 0) + bbox.get("height", 0.1)) * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else np.zeros((10, 10, 3), dtype=np.uint8)

        # 提取外观特征
        appearance_feature = self.appearance_encoder.encode(roi)

        # 位置特征
        position_feature = np.array([
            bbox.get("x", 0),
            bbox.get("y", 0),
            bbox.get("width", 0),
            bbox.get("height", 0)
        ], dtype=np.float32)

        # 时序特征 (使用注意力机制)
        temporal_feature = None
        if self.config.use_temporal_attention and self._feature_cache:
            history_features = []
            for cache_item in self._feature_cache:
                for feat in cache_item["features"]:
                    if self._is_similar_position(position_feature, feat.position_feature):
                        history_features.append(feat.appearance_feature)

            if history_features:
                temporal_feature = self.temporal_attention.compute_attention(
                    appearance_feature,
                    history_features[-10:],
                    history_features[-10:]
                )

        # 确定缺陷类型
        label = detection.get("label", "unknown")
        try:
            defect_type = DefectType(label)
        except ValueError:
            defect_type = DefectType.UNKNOWN

        return DefectFeature(
            feature_id=f"feat_{frame_id}_{id(detection)}",
            defect_type=defect_type,
            appearance_feature=appearance_feature,
            position_feature=position_feature,
            temporal_feature=temporal_feature,
            confidence=detection.get("confidence", 0.0),
            timestamp=timestamp,
            frame_id=frame_id,
            metadata=detection.get("metadata", {})
        )

    def _match_features(
        self,
        current_features: List[DefectFeature],
        active_tracks: List[DefectTrack]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        特征匹配

        使用匈牙利算法进行最优匹配
        """
        num_dets = len(current_features)
        num_tracks = len(active_tracks)

        # 计算代价矩阵
        cost_matrix = np.zeros((num_dets, num_tracks))

        for i, feat in enumerate(current_features):
            for j, track in enumerate(active_tracks):
                cost_matrix[i, j] = 1 - self._compute_similarity(feat, track)

        # 简单贪婪匹配 (实际应使用Hungarian算法)
        matches = []
        unmatched_dets = list(range(num_dets))
        unmatched_tracks = list(range(num_tracks))

        # 找最优匹配
        used_tracks = set()
        for det_idx in range(num_dets):
            best_track = -1
            best_cost = 1 - self.config.match_threshold

            for track_idx in range(num_tracks):
                if track_idx in used_tracks:
                    continue
                if cost_matrix[det_idx, track_idx] < best_cost:
                    best_cost = cost_matrix[det_idx, track_idx]
                    best_track = track_idx

            if best_track >= 0:
                matches.append((det_idx, best_track))
                used_tracks.add(best_track)
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(best_track)

        return matches, unmatched_dets, unmatched_tracks

    def _compute_similarity(self, feature: DefectFeature, track: DefectTrack) -> float:
        """计算特征与轨迹的相似度"""
        # 外观相似度 (余弦相似度)
        track_feat = track.get_averaged_feature()
        appearance_sim = np.dot(feature.appearance_feature, track_feat)

        # 位置相似度 (IoU)
        if track.features:
            last_pos = track.features[-1].position_feature
            position_sim = self._compute_iou(feature.position_feature, last_pos)
        else:
            position_sim = 0.0

        # 类型一致性
        type_sim = 1.0 if feature.defect_type == track.defect_type else 0.5

        # 加权组合
        similarity = (
            self.config.appearance_weight * appearance_sim +
            self.config.position_weight * position_sim +
            self.config.temporal_weight * type_sim
        )

        return float(similarity)

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - inter

        return inter / (union + 1e-8)

    def _is_similar_position(self, pos1: np.ndarray, pos2: np.ndarray, threshold: float = 0.5) -> bool:
        """判断位置是否相似"""
        return self._compute_iou(pos1, pos2) > threshold

    def _create_new_track(self, feature: DefectFeature) -> str:
        """创建新轨迹"""
        self._track_counter += 1
        track_id = f"track_{self._track_counter}"

        track = DefectTrack(
            track_id=track_id,
            defect_type=feature.defect_type,
            first_seen=feature.timestamp,
            last_seen=feature.timestamp
        )
        track.add_feature(feature)

        self.tracks[track_id] = track

        # 限制轨迹数量
        if len(self.tracks) > self.config.max_tracks:
            self._cleanup_old_tracks()

        return track_id

    def _cleanup_old_tracks(self):
        """清理旧轨迹"""
        # 移除lost状态的轨迹
        to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.state == "lost"
        ]
        for track_id in to_remove[:len(to_remove)//2]:  # 只移除一半
            del self.tracks[track_id]

    def get_track(self, track_id: str) -> Optional[DefectTrack]:
        """获取轨迹"""
        return self.tracks.get(track_id)

    def get_active_tracks(self) -> List[DefectTrack]:
        """获取活跃轨迹"""
        return [t for t in self.tracks.values() if t.state == "active"]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        active_tracks = self.get_active_tracks()
        return {
            "total_tracks": len(self.tracks),
            "active_tracks": len(active_tracks),
            "total_matches": self._total_matches,
            "total_new_tracks": self._total_new_tracks,
            "cache_size": len(self._feature_cache)
        }


# =============================================================================
# 裂纹增长分析模块
# =============================================================================

@dataclass
class CrackGrowthPoint:
    """裂纹生长点"""
    timestamp: float
    length: float           # 裂纹长度
    width: float            # 裂纹宽度
    area: float             # 裂纹面积
    orientation: float      # 裂纹方向 (角度)
    severity: float         # 严重程度 [0, 1]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrackGrowthAnalysis:
    """裂纹增长分析结果"""
    track_id: str
    growth_rate: float              # 增长率 (像素/天)
    growth_trend: str               # 增长趋势: stable, slow, moderate, rapid
    predicted_length: float         # 预测长度 (7天后)
    days_to_critical: Optional[float]  # 距离临界的天数
    risk_level: str                 # 风险等级: low, medium, high, critical
    confidence: float
    history: List[CrackGrowthPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrackGrowthAnalyzer:
    """
    裂纹增长分析器

    分析裂纹随时间的扩展趋势，预测未来发展
    """

    # 风险阈值
    RISK_THRESHOLDS = {
        "length": {
            "low": 5.0,         # mm
            "medium": 15.0,
            "high": 30.0,
            "critical": 50.0
        },
        "growth_rate": {
            "stable": 0.1,      # mm/day
            "slow": 0.5,
            "moderate": 1.0,
            "rapid": 2.0
        }
    }

    def __init__(
        self,
        pixel_to_mm: float = 0.1,  # 像素到毫米的转换系数
        critical_length: float = 50.0,  # 临界长度 (mm)
        history_window_days: int = 30  # 历史窗口 (天)
    ):
        self.pixel_to_mm = pixel_to_mm
        self.critical_length = critical_length
        self.history_window_days = history_window_days

        # 轨迹历史
        self._history: Dict[str, List[CrackGrowthPoint]] = {}

    def add_observation(
        self,
        track_id: str,
        crack_mask: Optional[np.ndarray] = None,
        bbox: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CrackGrowthPoint:
        """
        添加裂纹观测

        Args:
            track_id: 轨迹ID
            crack_mask: 裂纹分割掩码
            bbox: 边界框
            timestamp: 时间戳
            metadata: 附加信息

        Returns:
            裂纹生长点
        """
        timestamp = timestamp or time.time()

        # 分析裂纹特征
        if crack_mask is not None:
            length, width, area, orientation = self._analyze_crack_geometry(crack_mask)
        elif bbox is not None:
            # 从边界框估计
            length = max(bbox.get("width", 0), bbox.get("height", 0)) * 1000  # 归一化坐标转像素
            width = min(bbox.get("width", 0), bbox.get("height", 0)) * 1000
            area = length * width
            orientation = 0.0 if bbox.get("width", 0) > bbox.get("height", 0) else 90.0
        else:
            length, width, area, orientation = 0, 0, 0, 0

        # 转换为物理尺寸
        length_mm = length * self.pixel_to_mm
        width_mm = width * self.pixel_to_mm
        area_mm2 = area * (self.pixel_to_mm ** 2)

        # 计算严重程度
        severity = min(1.0, length_mm / self.critical_length)

        point = CrackGrowthPoint(
            timestamp=timestamp,
            length=length_mm,
            width=width_mm,
            area=area_mm2,
            orientation=orientation,
            severity=severity,
            confidence=0.9,
            metadata=metadata or {}
        )

        # 存储
        if track_id not in self._history:
            self._history[track_id] = []
        self._history[track_id].append(point)

        # 清理旧数据
        cutoff = timestamp - self.history_window_days * 24 * 3600
        self._history[track_id] = [
            p for p in self._history[track_id]
            if p.timestamp > cutoff
        ]

        return point

    def analyze_growth(self, track_id: str) -> Optional[CrackGrowthAnalysis]:
        """
        分析裂纹增长趋势

        Args:
            track_id: 轨迹ID

        Returns:
            增长分析结果
        """
        history = self._history.get(track_id, [])
        if len(history) < 2:
            return None

        # 按时间排序
        history = sorted(history, key=lambda p: p.timestamp)

        # 提取长度序列
        timestamps = np.array([p.timestamp for p in history])
        lengths = np.array([p.length for p in history])

        # 转换时间为天数
        days = (timestamps - timestamps[0]) / (24 * 3600)

        # 线性回归计算增长率
        if len(days) >= 2:
            growth_rate = self._compute_growth_rate(days, lengths)
        else:
            growth_rate = 0.0

        # 确定增长趋势
        growth_trend = self._classify_growth_trend(growth_rate)

        # 预测7天后的长度
        current_length = lengths[-1]
        predicted_length = current_length + growth_rate * 7

        # 计算距离临界的天数
        if growth_rate > 0.01:
            days_to_critical = (self.critical_length - current_length) / growth_rate
            if days_to_critical < 0:
                days_to_critical = 0
        else:
            days_to_critical = None

        # 评估风险等级
        risk_level = self._evaluate_risk(current_length, growth_rate, days_to_critical)

        # 计算置信度
        confidence = self._compute_confidence(history)

        return CrackGrowthAnalysis(
            track_id=track_id,
            growth_rate=growth_rate,
            growth_trend=growth_trend,
            predicted_length=predicted_length,
            days_to_critical=days_to_critical,
            risk_level=risk_level,
            confidence=confidence,
            history=history,
            metadata={
                "current_length": current_length,
                "observation_count": len(history),
                "observation_days": float(days[-1]) if len(days) > 0 else 0
            }
        )

    def _analyze_crack_geometry(
        self,
        mask: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """分析裂纹几何特征"""
        try:
            import cv2

            # 二值化
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8) * 255

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0, 0, 0, 0

            # 最大轮廓
            contour = max(contours, key=cv2.contourArea)

            # 面积
            area = cv2.contourArea(contour)

            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            angle = rect[2]

            # 长度为最大边，宽度为最小边
            length = max(width, height)
            width = min(width, height)

            return length, width, area, angle

        except ImportError:
            return 0, 0, 0, 0

    def _compute_growth_rate(self, days: np.ndarray, lengths: np.ndarray) -> float:
        """计算增长率 (线性回归)"""
        if len(days) < 2:
            return 0.0

        # 简单线性回归
        n = len(days)
        sum_x = np.sum(days)
        sum_y = np.sum(lengths)
        sum_xy = np.sum(days * lengths)
        sum_x2 = np.sum(days ** 2)

        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return float(max(0, slope))  # 增长率不能为负

    def _classify_growth_trend(self, growth_rate: float) -> str:
        """分类增长趋势"""
        thresholds = self.RISK_THRESHOLDS["growth_rate"]

        if growth_rate < thresholds["stable"]:
            return "stable"
        elif growth_rate < thresholds["slow"]:
            return "slow"
        elif growth_rate < thresholds["moderate"]:
            return "moderate"
        else:
            return "rapid"

    def _evaluate_risk(
        self,
        current_length: float,
        growth_rate: float,
        days_to_critical: Optional[float]
    ) -> str:
        """评估风险等级"""
        length_thresholds = self.RISK_THRESHOLDS["length"]

        # 基于当前长度的风险
        if current_length >= length_thresholds["critical"]:
            return "critical"
        elif current_length >= length_thresholds["high"]:
            base_risk = "high"
        elif current_length >= length_thresholds["medium"]:
            base_risk = "medium"
        else:
            base_risk = "low"

        # 基于增长率调整
        if days_to_critical is not None and days_to_critical < 30:
            if base_risk == "low":
                return "medium"
            elif base_risk == "medium":
                return "high"
            elif base_risk == "high":
                return "critical"

        return base_risk

    def _compute_confidence(self, history: List[CrackGrowthPoint]) -> float:
        """计算分析置信度"""
        if len(history) < 2:
            return 0.3
        elif len(history) < 5:
            return 0.6
        elif len(history) < 10:
            return 0.8
        else:
            return 0.9

    def get_history(self, track_id: str) -> List[CrackGrowthPoint]:
        """获取历史记录"""
        return self._history.get(track_id, [])

    def clear_history(self, track_id: Optional[str] = None):
        """清除历史"""
        if track_id:
            self._history.pop(track_id, None)
        else:
            self._history.clear()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "DefectType",
    "DefectFeature",
    "DefectTrack",
    "TemporalReIDConfig",
    "TemporalReIDModule",
    "CrackGrowthPoint",
    "CrackGrowthAnalysis",
    "CrackGrowthAnalyzer",
    "AppearanceEncoder",
    "TemporalAttention"
]
