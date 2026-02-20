#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义SLAM建图插件 V2.0
======================

基于迭代方案升级:
- 点云语义分割 (RandLA-Net / PointNet++)
- 环境变化检测
- 与电子围栏联动
- 三维语义地图

作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 类型定义
# =============================================================================
class SemanticLabel(Enum):
    """语义标签"""
    GROUND = 0          # 地面
    WALL = 1            # 墙壁
    CABINET = 2         # 机柜
    DOOR = 3            # 门
    PILLAR = 4          # 柱子
    EQUIPMENT = 5       # 设备
    LADDER = 6          # 梯子
    CABLE = 7           # 电缆
    PERSON = 8          # 人员 (动态)
    VEHICLE = 9         # 车辆 (动态)
    OBSTACLE = 10       # 障碍物
    UNKNOWN = 255       # 未知


class ChangeType(Enum):
    """变化类型"""
    NO_CHANGE = "no_change"         # 无变化
    NEW_OBJECT = "new_object"       # 新增物体
    REMOVED_OBJECT = "removed"      # 物体移除
    MOVED_OBJECT = "moved"          # 物体移动
    INTRUDER = "intruder"           # 入侵物
    STRUCTURAL = "structural"       # 结构变化


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class SemanticPoint:
    """语义点"""
    x: float
    y: float
    z: float
    label: SemanticLabel
    confidence: float = 1.0
    color: Optional[Tuple[int, int, int]] = None
    intensity: float = 0.0


@dataclass
class PointCloud:
    """点云数据"""
    points: np.ndarray              # (N, 3) xyz坐标
    labels: Optional[np.ndarray] = None  # (N,) 语义标签
    confidences: Optional[np.ndarray] = None  # (N,) 置信度
    colors: Optional[np.ndarray] = None  # (N, 3) RGB
    intensities: Optional[np.ndarray] = None  # (N,) 强度
    timestamp: float = field(default_factory=time.time)
    frame_id: str = ""
    
    @property
    def size(self) -> int:
        return len(self.points) if self.points is not None else 0
    
    def filter_by_label(self, labels: List[SemanticLabel]) -> 'PointCloud':
        """按标签过滤"""
        if self.labels is None:
            return self
        
        label_values = [l.value for l in labels]
        mask = np.isin(self.labels, label_values)
        
        return PointCloud(
            points=self.points[mask],
            labels=self.labels[mask] if self.labels is not None else None,
            confidences=self.confidences[mask] if self.confidences is not None else None,
            colors=self.colors[mask] if self.colors is not None else None,
            intensities=self.intensities[mask] if self.intensities is not None else None,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
        )
    
    def filter_dynamic(self) -> 'PointCloud':
        """过滤动态物体"""
        dynamic_labels = [SemanticLabel.PERSON, SemanticLabel.VEHICLE]
        if self.labels is None:
            return self
        
        label_values = [l.value for l in dynamic_labels]
        mask = ~np.isin(self.labels, label_values)
        
        return PointCloud(
            points=self.points[mask],
            labels=self.labels[mask] if self.labels is not None else None,
            confidences=self.confidences[mask] if self.confidences is not None else None,
            colors=self.colors[mask] if self.colors is not None else None,
            intensities=self.intensities[mask] if self.intensities is not None else None,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'num_points': self.size,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'has_labels': self.labels is not None,
            'has_colors': self.colors is not None,
        }


@dataclass
class DetectedChange:
    """检测到的变化"""
    change_id: str
    change_type: ChangeType
    location: Tuple[float, float, float]  # (x, y, z)
    bounding_box: Dict[str, float]        # {min_x, max_x, min_y, max_y, min_z, max_z}
    volume: float                          # 体积 (m³)
    semantic_label: Optional[SemanticLabel] = None
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'change_id': self.change_id,
            'change_type': self.change_type.value,
            'location': {'x': self.location[0], 'y': self.location[1], 'z': self.location[2]},
            'bounding_box': self.bounding_box,
            'volume': self.volume,
            'semantic_label': self.semantic_label.name if self.semantic_label else None,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'details': self.details,
        }


@dataclass
class SemanticMapStats:
    """语义地图统计"""
    total_points: int = 0
    labeled_points: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    bounding_box: Optional[Dict[str, float]] = None
    last_update: float = field(default_factory=time.time)
    resolution: float = 0.05  # 体素分辨率


# =============================================================================
# 点云语义分割器
# =============================================================================
class SemanticSegmenter:
    """
    点云语义分割器
    
    基于RandLA-Net或PointNet++进行语义分割
    """
    
    # 标签颜色映射
    LABEL_COLORS = {
        SemanticLabel.GROUND: (128, 128, 128),      # 灰色
        SemanticLabel.WALL: (165, 42, 42),          # 棕色
        SemanticLabel.CABINET: (0, 0, 255),         # 蓝色
        SemanticLabel.DOOR: (0, 255, 255),          # 青色
        SemanticLabel.PILLAR: (255, 165, 0),        # 橙色
        SemanticLabel.EQUIPMENT: (255, 0, 255),     # 紫色
        SemanticLabel.LADDER: (255, 255, 0),        # 黄色
        SemanticLabel.CABLE: (0, 128, 0),           # 绿色
        SemanticLabel.PERSON: (255, 0, 0),          # 红色
        SemanticLabel.VEHICLE: (0, 255, 0),         # 亮绿
        SemanticLabel.OBSTACLE: (128, 0, 128),      # 暗紫
        SemanticLabel.UNKNOWN: (200, 200, 200),     # 浅灰
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 12,
        num_points: int = 8192,
        use_gpu: bool = False,
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.num_points = num_points
        self.use_gpu = use_gpu
        
        self._model = None
        self._initialized = False
        
        self._load_model()
    
    def _load_model(self):
        """加载语义分割模型"""
        try:
            if self.model_path and Path(self.model_path).exists():
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if self.use_gpu else ['CPUExecutionProvider']
                self._model = ort.InferenceSession(self.model_path, providers=providers)
                self._initialized = True
                logger.info(f"[SemanticSegmenter] 模型加载成功: {self.model_path}")
            else:
                logger.info("[SemanticSegmenter] 未指定模型，使用规则分割")
                self._initialized = True
        except Exception as e:
            logger.warning(f"[SemanticSegmenter] 模型加载失败: {e}")
            self._initialized = True
    
    def segment(self, point_cloud: PointCloud) -> PointCloud:
        """
        语义分割
        
        Args:
            point_cloud: 输入点云
            
        Returns:
            带语义标签的点云
        """
        if not self._initialized or point_cloud.size == 0:
            return point_cloud
        
        try:
            if self._model is not None:
                return self._segment_with_model(point_cloud)
            else:
                return self._segment_with_rules(point_cloud)
        except Exception as e:
            logger.error(f"[SemanticSegmenter] 分割失败: {e}")
            return point_cloud
    
    def _segment_with_model(self, point_cloud: PointCloud) -> PointCloud:
        """使用模型分割"""
        points = point_cloud.points
        n = len(points)
        
        # 采样到固定数量
        if n > self.num_points:
            indices = np.random.choice(n, self.num_points, replace=False)
            sampled = points[indices]
        else:
            # 重复填充
            indices = np.arange(n)
            repeat = (self.num_points // n) + 1
            indices = np.tile(indices, repeat)[:self.num_points]
            sampled = points[indices]
        
        # 预处理
        # 中心化
        centroid = np.mean(sampled, axis=0)
        sampled = sampled - centroid
        # 归一化
        max_dist = np.max(np.sqrt(np.sum(sampled**2, axis=1)))
        if max_dist > 0:
            sampled = sampled / max_dist
        
        # 推理
        inputs = {self._model.get_inputs()[0].name: sampled.astype(np.float32)[np.newaxis, ...]}
        outputs = self._model.run(None, inputs)
        logits = outputs[0][0]  # (N, num_classes)
        
        # 获取标签
        pred_labels = np.argmax(logits, axis=1)
        pred_conf = np.max(logits, axis=1)
        
        # 映射回原始点
        full_labels = np.full(n, SemanticLabel.UNKNOWN.value, dtype=np.int32)
        full_conf = np.zeros(n, dtype=np.float32)
        
        # 使用最近邻插值
        if n <= self.num_points:
            full_labels = pred_labels[:n]
            full_conf = pred_conf[:n]
        else:
            from scipy.spatial import cKDTree
            tree = cKDTree(points[indices[:self.num_points]])
            _, nearest_idx = tree.query(points, k=1)
            full_labels = pred_labels[nearest_idx]
            full_conf = pred_conf[nearest_idx]
        
        return PointCloud(
            points=point_cloud.points,
            labels=full_labels,
            confidences=full_conf,
            colors=point_cloud.colors,
            intensities=point_cloud.intensities,
            timestamp=point_cloud.timestamp,
            frame_id=point_cloud.frame_id,
        )
    
    def _segment_with_rules(self, point_cloud: PointCloud) -> PointCloud:
        """使用规则分割 (简化版)"""
        points = point_cloud.points
        n = len(points)
        
        labels = np.full(n, SemanticLabel.UNKNOWN.value, dtype=np.int32)
        confidences = np.full(n, 0.5, dtype=np.float32)
        
        # 基于高度的简单分割
        z = points[:, 2]
        
        # 地面: z < 0.1m
        ground_mask = z < 0.1
        labels[ground_mask] = SemanticLabel.GROUND.value
        confidences[ground_mask] = 0.9
        
        # 低位障碍: 0.1 < z < 1.0
        low_mask = (z >= 0.1) & (z < 1.0)
        labels[low_mask] = SemanticLabel.OBSTACLE.value
        confidences[low_mask] = 0.6
        
        # 中位物体: 1.0 < z < 2.5 (可能是机柜、设备)
        mid_mask = (z >= 1.0) & (z < 2.5)
        labels[mid_mask] = SemanticLabel.EQUIPMENT.value
        confidences[mid_mask] = 0.5
        
        # 高位: z >= 2.5 (墙壁、天花板)
        high_mask = z >= 2.5
        labels[high_mask] = SemanticLabel.WALL.value
        confidences[high_mask] = 0.7
        
        return PointCloud(
            points=points,
            labels=labels,
            confidences=confidences,
            colors=point_cloud.colors,
            intensities=point_cloud.intensities,
            timestamp=point_cloud.timestamp,
            frame_id=point_cloud.frame_id,
        )
    
    def get_label_color(self, label: SemanticLabel) -> Tuple[int, int, int]:
        """获取标签颜色"""
        return self.LABEL_COLORS.get(label, (200, 200, 200))


# =============================================================================
# 变化检测器
# =============================================================================
class ChangeDetector:
    """
    环境变化检测器
    
    比较当前点云与基准地图，检测变化
    """
    
    def __init__(
        self,
        voxel_size: float = 0.1,            # 体素大小 (m)
        distance_threshold: float = 0.2,    # 距离阈值 (m)
        min_change_volume: float = 0.01,    # 最小变化体积 (m³)
        min_cluster_points: int = 10,       # 最小聚类点数
    ):
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.min_change_volume = min_change_volume
        self.min_cluster_points = min_cluster_points
        
        self._baseline_map: Optional[np.ndarray] = None
        self._baseline_labels: Optional[np.ndarray] = None
        self._change_count = 0
    
    def set_baseline(self, point_cloud: PointCloud):
        """
        设置基准地图
        
        Args:
            point_cloud: 基准点云 (应该是空环境或初始状态)
        """
        # 过滤动态物体
        static_pc = point_cloud.filter_dynamic()
        
        self._baseline_map = static_pc.points.copy()
        self._baseline_labels = static_pc.labels.copy() if static_pc.labels is not None else None
        
        logger.info(f"[ChangeDetector] 基准地图设置完成: {len(self._baseline_map)} 点")
    
    def detect_changes(
        self,
        current_cloud: PointCloud,
        person_positions: Optional[List[Tuple[float, float]]] = None,
    ) -> List[DetectedChange]:
        """
        检测环境变化
        
        Args:
            current_cloud: 当前点云
            person_positions: 人员位置列表 (用于过滤)
            
        Returns:
            检测到的变化列表
        """
        if self._baseline_map is None:
            logger.warning("[ChangeDetector] 基准地图未设置")
            return []
        
        # 过滤动态物体
        static_current = current_cloud.filter_dynamic()
        
        # 过滤人员周围的点
        if person_positions:
            static_current = self._filter_near_persons(static_current, person_positions)
        
        # 体素化
        current_voxels = self._voxelize(static_current.points)
        baseline_voxels = self._voxelize(self._baseline_map)
        
        # 找到差异
        current_set = set(map(tuple, current_voxels))
        baseline_set = set(map(tuple, baseline_voxels))
        
        # 新增体素
        new_voxels = current_set - baseline_set
        # 移除体素
        removed_voxels = baseline_set - current_set
        
        changes = []
        
        # 聚类新增区域
        if new_voxels:
            new_array = np.array(list(new_voxels))
            clusters = self._cluster_voxels(new_array)
            for cluster in clusters:
                change = self._create_change(cluster, ChangeType.NEW_OBJECT, static_current)
                if change:
                    changes.append(change)
        
        # 聚类移除区域
        if removed_voxels:
            removed_array = np.array(list(removed_voxels))
            clusters = self._cluster_voxels(removed_array)
            for cluster in clusters:
                change = self._create_change(cluster, ChangeType.REMOVED_OBJECT, static_current)
                if change:
                    changes.append(change)
        
        return changes
    
    def _filter_near_persons(
        self,
        point_cloud: PointCloud,
        person_positions: List[Tuple[float, float]],
        radius: float = 1.0
    ) -> PointCloud:
        """过滤人员附近的点"""
        if len(person_positions) == 0:
            return point_cloud
        
        points = point_cloud.points
        mask = np.ones(len(points), dtype=bool)
        
        for px, py in person_positions:
            # 计算xy平面距离
            dist_xy = np.sqrt((points[:, 0] - px)**2 + (points[:, 1] - py)**2)
            mask &= dist_xy > radius
        
        return PointCloud(
            points=points[mask],
            labels=point_cloud.labels[mask] if point_cloud.labels is not None else None,
            confidences=point_cloud.confidences[mask] if point_cloud.confidences is not None else None,
            timestamp=point_cloud.timestamp,
            frame_id=point_cloud.frame_id,
        )
    
    def _voxelize(self, points: np.ndarray) -> np.ndarray:
        """体素化点云"""
        return np.floor(points / self.voxel_size).astype(np.int32)
    
    def _cluster_voxels(self, voxels: np.ndarray) -> List[np.ndarray]:
        """聚类体素"""
        if len(voxels) == 0:
            return []
        
        # 使用简单的连通区域聚类
        from scipy.ndimage import label as nd_label
        
        # 创建稀疏网格
        min_v = voxels.min(axis=0)
        max_v = voxels.max(axis=0)
        
        # 限制网格大小
        size = max_v - min_v + 1
        if np.prod(size) > 1e7:  # 太大则简化
            return [voxels]  # 返回整体作为一个聚类
        
        grid = np.zeros(size, dtype=bool)
        shifted = voxels - min_v
        grid[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = True
        
        # 连通区域标记
        labeled, num_features = nd_label(grid)
        
        clusters = []
        for i in range(1, num_features + 1):
            cluster_mask = labeled == i
            cluster_voxels = np.argwhere(cluster_mask) + min_v
            
            if len(cluster_voxels) >= self.min_cluster_points:
                # 转换回真实坐标
                real_coords = cluster_voxels.astype(np.float32) * self.voxel_size
                clusters.append(real_coords)
        
        return clusters
    
    def _create_change(
        self,
        cluster: np.ndarray,
        change_type: ChangeType,
        point_cloud: PointCloud
    ) -> Optional[DetectedChange]:
        """创建变化对象"""
        if len(cluster) < self.min_cluster_points:
            return None
        
        # 计算边界框
        min_coords = cluster.min(axis=0)
        max_coords = cluster.max(axis=0)
        
        # 计算体积
        dims = max_coords - min_coords
        volume = dims[0] * dims[1] * dims[2]
        
        if volume < self.min_change_volume:
            return None
        
        # 中心位置
        center = (min_coords + max_coords) / 2
        
        self._change_count += 1
        
        return DetectedChange(
            change_id=f"CH{self._change_count:06d}",
            change_type=change_type,
            location=tuple(center),
            bounding_box={
                'min_x': float(min_coords[0]),
                'max_x': float(max_coords[0]),
                'min_y': float(min_coords[1]),
                'max_y': float(max_coords[1]),
                'min_z': float(min_coords[2]),
                'max_z': float(max_coords[2]),
            },
            volume=float(volume),
            confidence=0.8,
            details={
                'num_voxels': len(cluster),
                'dimensions': dims.tolist(),
            }
        )
    
    def clear_baseline(self):
        """清除基准地图"""
        self._baseline_map = None
        self._baseline_labels = None


# =============================================================================
# 语义地图管理器
# =============================================================================
class SemanticMapManager:
    """
    语义地图管理器
    
    维护和更新三维语义地图
    """
    
    def __init__(
        self,
        voxel_size: float = 0.05,
        max_points: int = 1000000,
    ):
        self.voxel_size = voxel_size
        self.max_points = max_points
        
        # 地图数据
        self._points: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._colors: Optional[np.ndarray] = None
        
        # 体素哈希表
        self._voxel_map: Dict[Tuple[int, int, int], Dict] = {}
        
        # 统计
        self._stats = SemanticMapStats()
        self._update_count = 0
    
    def update(self, point_cloud: PointCloud, pose: Optional[np.ndarray] = None):
        """
        更新地图
        
        Args:
            point_cloud: 新的点云帧
            pose: 传感器位姿 (4x4变换矩阵)
        """
        if point_cloud.size == 0:
            return
        
        points = point_cloud.points.copy()
        
        # 应用位姿变换
        if pose is not None:
            points_h = np.hstack([points, np.ones((len(points), 1))])
            points = (pose @ points_h.T).T[:, :3]
        
        # 体素化并更新
        for i, point in enumerate(points):
            voxel_key = self._point_to_voxel(point)
            
            if voxel_key not in self._voxel_map:
                self._voxel_map[voxel_key] = {
                    'count': 0,
                    'sum_pos': np.zeros(3),
                    'label_votes': {},
                    'color_sum': np.zeros(3),
                }
            
            voxel = self._voxel_map[voxel_key]
            voxel['count'] += 1
            voxel['sum_pos'] += point
            
            # 更新标签投票
            if point_cloud.labels is not None:
                label = int(point_cloud.labels[i])
                voxel['label_votes'][label] = voxel['label_votes'].get(label, 0) + 1
            
            # 更新颜色
            if point_cloud.colors is not None:
                voxel['color_sum'] += point_cloud.colors[i]
        
        # 更新统计
        self._update_stats()
        self._update_count += 1
    
    def _point_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        """点转体素索引"""
        return tuple(np.floor(point / self.voxel_size).astype(int))
    
    def _update_stats(self):
        """更新统计信息"""
        if not self._voxel_map:
            return
        
        self._stats.total_points = sum(v['count'] for v in self._voxel_map.values())
        self._stats.last_update = time.time()
        
        # 标签分布
        label_dist = {}
        for voxel in self._voxel_map.values():
            if voxel['label_votes']:
                dominant_label = max(voxel['label_votes'], key=voxel['label_votes'].get)
                label_name = SemanticLabel(dominant_label).name if dominant_label < 255 else 'UNKNOWN'
                label_dist[label_name] = label_dist.get(label_name, 0) + 1
        self._stats.label_distribution = label_dist
        self._stats.labeled_points = sum(label_dist.values())
        
        # 边界框
        if self._voxel_map:
            keys = np.array(list(self._voxel_map.keys()))
            min_v = keys.min(axis=0) * self.voxel_size
            max_v = keys.max(axis=0) * self.voxel_size
            self._stats.bounding_box = {
                'min_x': float(min_v[0]), 'max_x': float(max_v[0]),
                'min_y': float(min_v[1]), 'max_y': float(max_v[1]),
                'min_z': float(min_v[2]), 'max_z': float(max_v[2]),
            }
    
    def get_point_cloud(self) -> PointCloud:
        """获取当前地图点云"""
        if not self._voxel_map:
            return PointCloud(points=np.empty((0, 3)))
        
        n = len(self._voxel_map)
        points = np.zeros((n, 3))
        labels = np.zeros(n, dtype=np.int32)
        colors = np.zeros((n, 3), dtype=np.uint8)
        
        for i, (key, voxel) in enumerate(self._voxel_map.items()):
            points[i] = voxel['sum_pos'] / voxel['count']
            
            if voxel['label_votes']:
                labels[i] = max(voxel['label_votes'], key=voxel['label_votes'].get)
            else:
                labels[i] = SemanticLabel.UNKNOWN.value
            
            if voxel['count'] > 0:
                colors[i] = (voxel['color_sum'] / voxel['count']).astype(np.uint8)
        
        return PointCloud(
            points=points,
            labels=labels,
            colors=colors,
            timestamp=time.time(),
        )
    
    def get_statistics(self) -> SemanticMapStats:
        """获取统计信息"""
        return self._stats
    
    def clear(self):
        """清空地图"""
        self._voxel_map.clear()
        self._stats = SemanticMapStats()
        self._update_count = 0


# =============================================================================
# 语义SLAM建图插件
# =============================================================================
class SemanticSLAMPlugin:
    """
    语义SLAM建图插件 V2.0
    
    功能:
    - 点云语义分割
    - 环境变化检测
    - 语义地图构建与维护
    - 与电子围栏联动
    """
    
    VERSION = "2.0.0"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        voxel_size: float = 0.05,
        use_gpu: bool = False,
    ):
        # 初始化组件
        self._segmenter = SemanticSegmenter(
            model_path=model_path,
            use_gpu=use_gpu,
        )
        self._change_detector = ChangeDetector(voxel_size=voxel_size * 2)
        self._map_manager = SemanticMapManager(voxel_size=voxel_size)
        
        # 状态
        self._frame_count = 0
        self._baseline_set = False
        
        # 历史
        self._change_history: deque = deque(maxlen=100)
        
        self._initialized = True
        logger.info(f"[SemanticSLAMPlugin] 初始化完成 V{self.VERSION}")
    
    def process_point_cloud(
        self,
        points: np.ndarray,
        sensor_pose: Optional[np.ndarray] = None,
        person_positions: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        处理点云帧
        
        Args:
            points: 点云数组 (N, 3) 或 (N, 4)
            sensor_pose: 传感器位姿 (4x4)
            person_positions: 人员位置 (用于动态过滤)
            
        Returns:
            处理结果
        """
        self._frame_count += 1
        
        result = {
            'success': True,
            'frame_id': self._frame_count,
            'timestamp': time.time(),
            'num_points': 0,
            'semantic_labels': {},
            'changes': [],
            'map_stats': None,
        }
        
        try:
            # 转换为PointCloud对象
            if points.shape[1] == 4:
                pc = PointCloud(
                    points=points[:, :3],
                    intensities=points[:, 3],
                    frame_id=f"frame_{self._frame_count}",
                )
            else:
                pc = PointCloud(
                    points=points,
                    frame_id=f"frame_{self._frame_count}",
                )
            
            result['num_points'] = pc.size
            
            # 1. 语义分割
            segmented_pc = self._segmenter.segment(pc)
            
            # 统计标签分布
            if segmented_pc.labels is not None:
                unique, counts = np.unique(segmented_pc.labels, return_counts=True)
                label_dist = {}
                for u, c in zip(unique, counts):
                    try:
                        label_name = SemanticLabel(int(u)).name
                    except:
                        label_name = 'UNKNOWN'
                    label_dist[label_name] = int(c)
                result['semantic_labels'] = label_dist
            
            # 2. 更新地图
            self._map_manager.update(segmented_pc, sensor_pose)
            
            # 3. 变化检测
            if self._baseline_set:
                changes = self._change_detector.detect_changes(
                    segmented_pc, person_positions
                )
                result['changes'] = [c.to_dict() for c in changes]
                
                for change in changes:
                    self._change_history.append(change)
            
            # 4. 地图统计
            result['map_stats'] = {
                'total_points': self._map_manager._stats.total_points,
                'labeled_points': self._map_manager._stats.labeled_points,
                'label_distribution': self._map_manager._stats.label_distribution,
                'bounding_box': self._map_manager._stats.bounding_box,
            }
            
        except Exception as e:
            logger.error(f"[SemanticSLAMPlugin] 处理失败: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def set_baseline(self, points: Optional[np.ndarray] = None):
        """
        设置基准地图
        
        Args:
            points: 基准点云 (None则使用当前地图)
        """
        if points is not None:
            pc = PointCloud(points=points)
            segmented = self._segmenter.segment(pc)
            self._change_detector.set_baseline(segmented)
        else:
            current_map = self._map_manager.get_point_cloud()
            self._change_detector.set_baseline(current_map)
        
        self._baseline_set = True
        logger.info("[SemanticSLAMPlugin] 基准地图已设置")
    
    def get_semantic_map(self) -> PointCloud:
        """获取语义地图"""
        return self._map_manager.get_point_cloud()
    
    def get_map_statistics(self) -> Dict[str, Any]:
        """获取地图统计"""
        stats = self._map_manager.get_statistics()
        return {
            'total_points': stats.total_points,
            'labeled_points': stats.labeled_points,
            'label_distribution': stats.label_distribution,
            'bounding_box': stats.bounding_box,
            'resolution': stats.resolution,
            'last_update': stats.last_update,
            'frame_count': self._frame_count,
        }
    
    def get_recent_changes(self, limit: int = 20) -> List[Dict]:
        """获取最近的变化"""
        return [c.to_dict() for c in list(self._change_history)[-limit:]]
    
    def reset(self):
        """重置插件"""
        self._map_manager.clear()
        self._change_detector.clear_baseline()
        self._frame_count = 0
        self._baseline_set = False
        self._change_history.clear()
        logger.info("[SemanticSLAMPlugin] 已重置")


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    # 类型
    'SemanticLabel',
    'ChangeType',
    # 数据类
    'SemanticPoint',
    'PointCloud',
    'DetectedChange',
    'SemanticMapStats',
    # 组件
    'SemanticSegmenter',
    'ChangeDetector',
    'SemanticMapManager',
    # 插件
    'SemanticSLAMPlugin',
]
