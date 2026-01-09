"""
SLAM三维建图插件
SLAM and 3D Mapping Plugin

功能：
- LiDAR/雷达点云处理
- 三维地图构建与更新
- 设备定位与追踪
- 路径规划
- 地面沉降监测
- 障碍物检测
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json

# 导入插件状态混入类
try:
    from platform_core.plugin_mixin import PluginStatusMixin, PluginStatus
except ImportError:
    from enum import Enum

    class PluginStatus(str, Enum):
        UNLOADED = "unloaded"
        LOADING = "loading"
        READY = "ready"
        RUNNING = "running"
        ERROR = "error"
        DISABLED = "disabled"

    class PluginStatusMixin:
        def __init_status__(self):
            self._status = PluginStatus.UNLOADED
            self._last_error = ""

        @property
        def status(self):
            return getattr(self, '_status', PluginStatus.UNLOADED)

        @status.setter
        def status(self, value):
            self._status = value

        def set_status(self, status, error: str = ""):
            if isinstance(status, str):
                try:
                    status = PluginStatus(status)
                except ValueError:
                    status = PluginStatus.ERROR
            self._status = status
            if error:
                self._last_error = error

        def get_plugin_status(self) -> dict:
            return {
                'plugin_id': getattr(self, 'PLUGIN_ID', 'unknown'),
                'name': getattr(self, 'PLUGIN_NAME', 'Unknown'),
                'version': getattr(self, 'PLUGIN_VERSION', '0.0.0'),
                'status': self._status.value if hasattr(self, '_status') else 'unknown',
                'initialized': getattr(self, '_is_initialized', False),
                'last_error': getattr(self, '_last_error', '')
            }

logger = logging.getLogger(__name__)


@dataclass
class Point3D:
    """三维点"""
    x: float
    y: float
    z: float
    intensity: float = 0.0
    label: int = 0  # 0: unknown, 1: ground, 2: building, 3: equipment, 4: vegetation


@dataclass
class Pose:
    """位姿（位置+姿态）"""
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0


@dataclass
class MapCell:
    """地图单元格"""
    occupied: bool = False
    height: float = 0.0
    variance: float = 0.0
    update_count: int = 0
    last_update: float = 0.0
    semantic_label: int = 0


@dataclass
class SubsidencePoint:
    """沉降监测点"""
    location: Tuple[float, float]
    initial_height: float
    current_height: float
    measurements: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, height)
    

class PointCloudProcessor:
    """点云处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.voxel_size = config.get('voxel_size', 0.1)
        self.ground_threshold = config.get('ground_threshold', 0.3)
        self.cluster_tolerance = config.get('cluster_tolerance', 0.5)
        self.min_cluster_size = config.get('min_cluster_size', 10)
        
    def preprocess(self, points: np.ndarray) -> np.ndarray:
        """
        预处理点云
        
        Args:
            points: Nx3或Nx4数组（x,y,z,[intensity]）
            
        Returns:
            处理后的点云
        """
        if len(points) == 0:
            return points
            
        # 移除NaN和无穷值
        valid_mask = np.all(np.isfinite(points[:, :3]), axis=1)
        points = points[valid_mask]
        
        # 距离过滤
        max_range = self.config.get('max_range', 100.0)
        min_range = self.config.get('min_range', 0.5)
        distances = np.linalg.norm(points[:, :3], axis=1)
        range_mask = (distances >= min_range) & (distances <= max_range)
        points = points[range_mask]
        
        # 体素下采样
        points = self._voxel_downsample(points)
        
        return points
    
    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """体素下采样"""
        if len(points) == 0:
            return points
            
        # 计算体素索引
        voxel_indices = np.floor(points[:, :3] / self.voxel_size).astype(int)
        
        # 使用字典聚合每个体素中的点
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[i])
        
        # 计算每个体素的质心
        downsampled = []
        for pts in voxel_dict.values():
            pts = np.array(pts)
            centroid = np.mean(pts, axis=0)
            downsampled.append(centroid)
            
        return np.array(downsampled) if downsampled else points
    
    def segment_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        地面分割（RANSAC平面拟合）
        
        Returns:
            ground_points, non_ground_points
        """
        if len(points) < 10:
            return np.array([]), points
            
        # 简化的RANSAC地面分割
        best_inliers = []
        best_plane = None
        n_iterations = self.config.get('ransac_iterations', 100)
        
        for _ in range(n_iterations):
            # 随机选择3个点
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample = points[sample_idx, :3]
            
            # 计算平面参数 ax + by + cz + d = 0
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -np.dot(normal, sample[0])
            
            # 检查是否近似水平（法向量接近z轴）
            if abs(normal[2]) < 0.8:
                continue
            
            # 计算所有点到平面的距离
            distances = np.abs(np.dot(points[:, :3], normal) + d)
            inliers = np.where(distances < self.ground_threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, d)
        
        if len(best_inliers) == 0:
            return np.array([]), points
            
        ground_mask = np.zeros(len(points), dtype=bool)
        ground_mask[best_inliers] = True
        
        return points[ground_mask], points[~ground_mask]
    
    def cluster_objects(self, points: np.ndarray) -> List[np.ndarray]:
        """
        欧氏聚类提取物体
        
        Returns:
            聚类列表
        """
        if len(points) < self.min_cluster_size:
            return []
            
        # 简化的聚类实现（基于KD树的区域生长）
        clusters = []
        visited = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if visited[i]:
                continue
                
            # BFS区域生长
            cluster_indices = []
            queue = [i]
            
            while queue:
                idx = queue.pop(0)
                if visited[idx]:
                    continue
                visited[idx] = True
                cluster_indices.append(idx)
                
                # 找到邻近点
                distances = np.linalg.norm(points[:, :3] - points[idx, :3], axis=1)
                neighbors = np.where((distances < self.cluster_tolerance) & (~visited))[0]
                queue.extend(neighbors.tolist())
            
            if len(cluster_indices) >= self.min_cluster_size:
                clusters.append(points[cluster_indices])
        
        return clusters
    
    def extract_features(self, points: np.ndarray) -> Dict[str, Any]:
        """提取点云特征"""
        if len(points) == 0:
            return {}
            
        features = {
            'centroid': np.mean(points[:, :3], axis=0).tolist(),
            'min_bound': np.min(points[:, :3], axis=0).tolist(),
            'max_bound': np.max(points[:, :3], axis=0).tolist(),
            'point_count': len(points),
            'std': np.std(points[:, :3], axis=0).tolist()
        }
        
        # 计算协方差矩阵的特征值（形状特征）
        centered = points[:, :3] - features['centroid']
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # 形状描述子
        if eigenvalues[0] > 1e-6:
            features['linearity'] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
            features['planarity'] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
            features['sphericity'] = eigenvalues[2] / eigenvalues[0]
        
        return features


class OccupancyGridMap:
    """占据栅格地图"""
    
    def __init__(self, config: Dict[str, Any]):
        self.resolution = config.get('resolution', 0.5)  # meters per cell
        self.origin = np.array(config.get('origin', [0, 0, 0]))
        self.size = config.get('size', [200, 200, 20])  # cells
        
        # 初始化地图
        self.grid = np.zeros(self.size, dtype=np.float32)  # log-odds
        self.height_map = np.full(self.size[:2], np.nan)
        self.semantic_map = np.zeros(self.size[:2], dtype=np.uint8)
        
        # 更新参数
        self.log_odds_hit = np.log(0.9 / 0.1)
        self.log_odds_miss = np.log(0.3 / 0.7)
        self.log_odds_max = 5.0
        self.log_odds_min = -5.0
        
    def world_to_grid(self, world_coords: np.ndarray) -> np.ndarray:
        """世界坐标转栅格坐标"""
        grid_coords = (world_coords - self.origin) / self.resolution
        return grid_coords.astype(int)
    
    def grid_to_world(self, grid_coords: np.ndarray) -> np.ndarray:
        """栅格坐标转世界坐标"""
        return grid_coords * self.resolution + self.origin
    
    def update(self, points: np.ndarray, sensor_origin: np.ndarray):
        """
        使用点云更新地图
        
        Args:
            points: 点云数据
            sensor_origin: 传感器位置
        """
        if len(points) == 0:
            return
            
        # 转换到栅格坐标
        grid_points = self.world_to_grid(points[:, :3])
        grid_origin = self.world_to_grid(sensor_origin)
        
        # 对每个点进行射线追踪更新
        for point, grid_pt in zip(points, grid_points):
            # 检查边界
            if not self._in_bounds(grid_pt):
                continue
                
            # 标记占据
            self._update_cell(grid_pt, hit=True)
            
            # 更新高度图
            x, y = grid_pt[0], grid_pt[1]
            if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                if np.isnan(self.height_map[x, y]):
                    self.height_map[x, y] = point[2]
                else:
                    # 滑动平均
                    self.height_map[x, y] = 0.9 * self.height_map[x, y] + 0.1 * point[2]
            
            # 射线追踪标记空闲（简化实现）
            ray_cells = self._bresenham_3d(grid_origin, grid_pt)
            for cell in ray_cells[:-1]:  # 排除终点
                if self._in_bounds(cell):
                    self._update_cell(cell, hit=False)
    
    def _in_bounds(self, grid_coords: np.ndarray) -> bool:
        """检查是否在地图边界内"""
        return all(0 <= grid_coords[i] < self.size[i] for i in range(3))
    
    def _update_cell(self, coords: np.ndarray, hit: bool):
        """更新单个单元格"""
        x, y, z = coords
        if hit:
            self.grid[x, y, z] = min(self.grid[x, y, z] + self.log_odds_hit, self.log_odds_max)
        else:
            self.grid[x, y, z] = max(self.grid[x, y, z] + self.log_odds_miss, self.log_odds_min)
    
    def _bresenham_3d(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        """3D Bresenham线段算法"""
        cells = []
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        dz = abs(end[2] - start[2])
        
        sx = 1 if end[0] > start[0] else -1
        sy = 1 if end[1] > start[1] else -1
        sz = 1 if end[2] > start[2] else -1
        
        dm = max(dx, dy, dz)
        if dm == 0:
            return [start.copy()]
            
        x, y, z = start.copy()
        
        for _ in range(dm + 1):
            cells.append(np.array([int(x), int(y), int(z)]))
            x += dx / dm * sx
            y += dy / dm * sy
            z += dz / dm * sz
            
        return cells
    
    def get_occupancy_probability(self, coords: np.ndarray) -> float:
        """获取占据概率"""
        grid_coords = self.world_to_grid(coords)
        if not self._in_bounds(grid_coords):
            return 0.5
        x, y, z = grid_coords
        return 1.0 / (1.0 + np.exp(-self.grid[x, y, z]))
    
    def get_2d_map(self, z_min: float = 0, z_max: float = 3) -> np.ndarray:
        """获取2D投影地图"""
        z_min_idx = max(0, int((z_min - self.origin[2]) / self.resolution))
        z_max_idx = min(self.size[2], int((z_max - self.origin[2]) / self.resolution))
        
        # 取z方向最大值
        map_2d = np.max(self.grid[:, :, z_min_idx:z_max_idx], axis=2)
        return 1.0 / (1.0 + np.exp(-map_2d))


class ICPMatcher:
    """ICP点云配准"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_iterations = config.get('icp_max_iterations', 50)
        self.tolerance = config.get('icp_tolerance', 1e-5)
        self.max_correspondence_distance = config.get('max_correspondence_distance', 1.0)
        
    def align(self, source: np.ndarray, target: np.ndarray, 
              initial_transform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        ICP配准
        
        Args:
            source: 源点云
            target: 目标点云
            initial_transform: 初始变换矩阵
            
        Returns:
            变换矩阵, 配准误差
        """
        if len(source) == 0 or len(target) == 0:
            return np.eye(4), float('inf')
            
        if initial_transform is None:
            initial_transform = np.eye(4)
            
        transform = initial_transform.copy()
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # 应用当前变换
            source_transformed = self._apply_transform(source[:, :3], transform)
            
            # 找最近点对应
            correspondences, distances = self._find_correspondences(
                source_transformed, target[:, :3])
            
            if len(correspondences) == 0:
                break
                
            # 计算当前误差
            error = np.mean(distances)
            
            # 检查收敛
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error
            
            # 计算最优变换
            src_pts = source_transformed[correspondences[:, 0]]
            tgt_pts = target[correspondences[:, 1], :3]
            delta_transform = self._compute_transform(src_pts, tgt_pts)
            
            # 更新变换
            transform = delta_transform @ transform
            
        return transform, prev_error
    
    def _apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """应用变换"""
        ones = np.ones((len(points), 1))
        points_h = np.hstack([points, ones])
        transformed = (transform @ points_h.T).T
        return transformed[:, :3]
    
    def _find_correspondences(self, source: np.ndarray, 
                              target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """找最近点对应"""
        correspondences = []
        distances = []
        
        for i, src_pt in enumerate(source):
            dists = np.linalg.norm(target - src_pt, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            
            if min_dist < self.max_correspondence_distance:
                correspondences.append([i, min_idx])
                distances.append(min_dist)
                
        return np.array(correspondences), np.array(distances)
    
    def _compute_transform(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """计算最优刚体变换（SVD方法）"""
        # 计算质心
        src_centroid = np.mean(source, axis=0)
        tgt_centroid = np.mean(target, axis=0)
        
        # 去中心化
        src_centered = source - src_centroid
        tgt_centered = target - tgt_centroid
        
        # SVD
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 确保旋转矩阵（行列式为1）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        # 计算平移
        t = tgt_centroid - R @ src_centroid
        
        # 构建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform


class PathPlanner:
    """路径规划器"""
    
    def __init__(self, occupancy_map: OccupancyGridMap):
        self.map = occupancy_map
        self.occupancy_threshold = 0.6
        
    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """
        A*路径规划
        
        Args:
            start: 起点世界坐标
            goal: 终点世界坐标
            
        Returns:
            路径点列表（世界坐标）
        """
        start_grid = self.map.world_to_grid(start)[:2]
        goal_grid = self.map.world_to_grid(goal)[:2]
        
        # 检查起点和终点
        if not self._is_free(start_grid) or not self._is_free(goal_grid):
            return []
            
        # A*搜索
        open_set = {tuple(start_grid)}
        came_from = {}
        
        g_score = {tuple(start_grid): 0}
        f_score = {tuple(start_grid): self._heuristic(start_grid, goal_grid)}
        
        while open_set:
            # 找f值最小的节点
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if np.allclose(current, goal_grid):
                # 重建路径
                return self._reconstruct_path(came_from, current)
                
            open_set.remove(current)
            
            # 扩展邻居
            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + self._distance(current, neighbor)
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    open_set.add(neighbor)
                    
        return []  # 无解
    
    def _is_free(self, grid_coords: Tuple[int, int]) -> bool:
        """检查栅格是否可通行"""
        x, y = grid_coords
        if not (0 <= x < self.map.size[0] and 0 <= y < self.map.size[1]):
            return False
        # 检查2D投影地图
        map_2d = self.map.get_2d_map()
        return map_2d[x, y] < self.occupancy_threshold
    
    def _get_neighbors(self, coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取8邻域"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = coords[0] + dx, coords[1] + dy
                if self._is_free((nx, ny)):
                    neighbors.append((nx, ny))
        return neighbors
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """启发函数（欧氏距离）"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """两点距离"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple) -> List[np.ndarray]:
        """重建路径"""
        path = [np.array([current[0], current[1], 0])]
        while current in came_from:
            current = came_from[current]
            path.append(np.array([current[0], current[1], 0]))
        path.reverse()
        
        # 转换为世界坐标
        world_path = [self.map.grid_to_world(p) for p in path]
        return world_path


class SubsidenceMonitor:
    """地面沉降监测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_points: Dict[str, SubsidencePoint] = {}
        self.alert_threshold = config.get('subsidence_alert_mm', 10.0)
        self.warning_threshold = config.get('subsidence_warning_mm', 5.0)
        
    def add_monitoring_point(self, point_id: str, location: Tuple[float, float], 
                            initial_height: float):
        """添加监测点"""
        self.monitoring_points[point_id] = SubsidencePoint(
            location=location,
            initial_height=initial_height,
            current_height=initial_height,
            measurements=[(datetime.now().timestamp(), initial_height)]
        )
        
    def update_measurement(self, point_id: str, height: float, timestamp: float = None):
        """更新测量值"""
        if point_id not in self.monitoring_points:
            return
            
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        point = self.monitoring_points[point_id]
        point.current_height = height
        point.measurements.append((timestamp, height))
        
        # 保留最近1000个测量值
        if len(point.measurements) > 1000:
            point.measurements = point.measurements[-1000:]
    
    def get_subsidence(self, point_id: str) -> Optional[float]:
        """获取沉降量（mm）"""
        if point_id not in self.monitoring_points:
            return None
            
        point = self.monitoring_points[point_id]
        return (point.initial_height - point.current_height) * 1000  # meters to mm
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """检查告警"""
        alerts = []
        
        for point_id, point in self.monitoring_points.items():
            subsidence = self.get_subsidence(point_id)
            if subsidence is None:
                continue
                
            if abs(subsidence) >= self.alert_threshold:
                alerts.append({
                    'point_id': point_id,
                    'level': 'alert',
                    'subsidence_mm': subsidence,
                    'location': point.location,
                    'message': f'监测点 {point_id} 沉降量 {subsidence:.1f}mm 超过告警阈值'
                })
            elif abs(subsidence) >= self.warning_threshold:
                alerts.append({
                    'point_id': point_id,
                    'level': 'warning',
                    'subsidence_mm': subsidence,
                    'location': point.location,
                    'message': f'监测点 {point_id} 沉降量 {subsidence:.1f}mm 超过预警阈值'
                })
                
        return alerts
    
    def analyze_trend(self, point_id: str) -> Dict[str, Any]:
        """分析沉降趋势"""
        if point_id not in self.monitoring_points:
            return {}
            
        point = self.monitoring_points[point_id]
        measurements = point.measurements
        
        if len(measurements) < 2:
            return {'trend': 'insufficient_data'}
            
        # 线性回归计算沉降速率
        times = np.array([m[0] for m in measurements])
        heights = np.array([m[1] for m in measurements])
        
        # 归一化时间（天）
        times_days = (times - times[0]) / 86400
        
        # 简单线性回归
        n = len(times)
        sum_x = np.sum(times_days)
        sum_y = np.sum(heights)
        sum_xy = np.sum(times_days * heights)
        sum_x2 = np.sum(times_days ** 2)
        
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return {'trend': 'stable', 'rate_mm_per_day': 0.0}
            
        slope = (n * sum_xy - sum_x * sum_y) / denom
        rate_mm_per_day = -slope * 1000  # 沉降为正
        
        # 判断趋势
        if rate_mm_per_day > 0.1:
            trend = 'subsiding'
        elif rate_mm_per_day < -0.1:
            trend = 'rising'
        else:
            trend = 'stable'
            
        return {
            'trend': trend,
            'rate_mm_per_day': rate_mm_per_day,
            'total_subsidence_mm': self.get_subsidence(point_id),
            'measurement_count': len(measurements),
            'time_span_days': times_days[-1]
        }


class SLAMMappingPlugin(PluginStatusMixin):
    """SLAM三维建图插件"""

    PLUGIN_ID = "slam_mapping"
    PLUGIN_NAME = "SLAM建图"
    PLUGIN_VERSION = "1.0.0"

    def __init__(self, manifest=None, plugin_dir=None):
        """
        初始化SLAM建图插件

        Args:
            manifest: 插件清单 (PluginManifest)
            plugin_dir: 插件目录 (Path)
        """
        # 初始化状态管理
        self.__init_status__()

        self.manifest = manifest
        self.plugin_dir = plugin_dir

        # 从 manifest 获取配置或使用默认值
        config = {}
        if manifest and hasattr(manifest, 'config_schema'):
            config = manifest.config_schema or {}
        self.config = config if config else self._default_config()

        self.name = "slam_mapping"
        self.version = "1.0.0"
        self._is_initialized = False
        
        # 初始化组件
        self.point_processor = PointCloudProcessor(self.config)
        self.occupancy_map = OccupancyGridMap(self.config)
        self.icp_matcher = ICPMatcher(self.config)
        self.path_planner = PathPlanner(self.occupancy_map)
        self.subsidence_monitor = SubsidenceMonitor(self.config)
        
        # 状态
        self.current_pose = Pose(0, 0, 0)
        self.pose_history: List[Pose] = []
        self.last_keyframe_points: Optional[np.ndarray] = None
        self.keyframe_interval = self.config.get('keyframe_interval', 10)
        self.frame_count = 0
        
        # 设备位置数据库
        self.device_locations: Dict[str, Dict[str, Any]] = {}
        
        # 模型注册表（深度学习）
        self.model_registry = None
        self.dl_enabled = False
        
        logger.info(f"SLAM建图插件初始化完成: {self.name} v{self.version}")
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'voxel_size': 0.1,
            'ground_threshold': 0.3,
            'cluster_tolerance': 0.5,
            'min_cluster_size': 10,
            'max_range': 100.0,
            'min_range': 0.5,
            'resolution': 0.5,
            'origin': [-50, -50, -2],
            'size': [200, 200, 20],
            'icp_max_iterations': 50,
            'icp_tolerance': 1e-5,
            'max_correspondence_distance': 1.0,
            'ransac_iterations': 100,
            'keyframe_interval': 10,
            'subsidence_alert_mm': 10.0,
            'subsidence_warning_mm': 5.0
        }

    # =========================================================================
    # 关键属性 (兼容 platform_core.plugin_manager)
    # =========================================================================

    @property
    def id(self) -> str:
        """插件ID - 兼容 platform_core.plugin_manager"""
        if self.manifest and hasattr(self.manifest, 'id'):
            return self.manifest.id
        return self.PLUGIN_ID

    @property
    def code_hash(self) -> str:
        """代码哈希"""
        import hashlib
        from pathlib import Path
        h = hashlib.sha256()
        plugin_file = Path(__file__).parent / "plugin.py"
        if plugin_file.exists():
            h.update(plugin_file.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"

    def set_model_registry(self, registry):
        """设置模型注册表"""
        self.model_registry = registry
        self.dl_enabled = True
        logger.info("SLAM插件已连接模型注册表，启用深度学习功能")
    
    def init(self, model_registry=None):
        """初始化插件"""
        if model_registry:
            self.set_model_registry(model_registry)
        self._is_initialized = True
        logger.info("SLAM建图插件启动")
        return True
    
    def shutdown(self):
        """关闭插件"""
        logger.info("SLAM建图插件关闭")
    
    def process_point_cloud(self, points: np.ndarray, 
                           sensor_pose: Optional[Pose] = None) -> Dict[str, Any]:
        """
        处理单帧点云数据
        
        Args:
            points: Nx3或Nx4点云数组
            sensor_pose: 传感器位姿（可选，用于定位）
            
        Returns:
            处理结果
        """
        self.frame_count += 1
        result = {
            'success': True,
            'frame_id': self.frame_count,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 预处理
            processed_points = self.point_processor.preprocess(points)
            result['processed_points'] = len(processed_points)
            
            if len(processed_points) == 0:
                result['success'] = False
                result['error'] = '有效点云为空'
                return result
            
            # 地面分割
            ground, non_ground = self.point_processor.segment_ground(processed_points)
            result['ground_points'] = len(ground)
            result['object_points'] = len(non_ground)
            
            # 位姿估计（ICP配准）
            if self.last_keyframe_points is not None:
                transform, error = self.icp_matcher.align(
                    processed_points, self.last_keyframe_points)
                
                # 更新位姿
                self._update_pose(transform)
                result['registration_error'] = error
            
            # 更新关键帧
            if self.frame_count % self.keyframe_interval == 0:
                self.last_keyframe_points = processed_points.copy()
                self.pose_history.append(Pose(
                    self.current_pose.x, self.current_pose.y, self.current_pose.z,
                    self.current_pose.roll, self.current_pose.pitch, self.current_pose.yaw,
                    datetime.now().timestamp()
                ))
            
            # 更新地图
            sensor_origin = np.array([
                self.current_pose.x, self.current_pose.y, self.current_pose.z])
            self.occupancy_map.update(processed_points, sensor_origin)
            
            # 物体聚类
            clusters = self.point_processor.cluster_objects(non_ground)
            result['detected_objects'] = len(clusters)
            
            # 提取物体特征
            objects = []
            for i, cluster in enumerate(clusters):
                features = self.point_processor.extract_features(cluster)
                features['cluster_id'] = i
                objects.append(features)
            result['objects'] = objects
            
            # 当前位姿
            result['pose'] = {
                'x': self.current_pose.x,
                'y': self.current_pose.y,
                'z': self.current_pose.z,
                'roll': self.current_pose.roll,
                'pitch': self.current_pose.pitch,
                'yaw': self.current_pose.yaw
            }
            
            # 地面沉降检查（如果有监测点）
            if ground.any() and self.subsidence_monitor.monitoring_points:
                self._update_subsidence(ground)
                alerts = self.subsidence_monitor.check_alerts()
                if alerts:
                    result['subsidence_alerts'] = alerts
            
        except Exception as e:
            logger.error(f"点云处理失败: {e}")
            result['success'] = False
            result['error'] = str(e)
            
        return result
    
    def _update_pose(self, transform: np.ndarray):
        """从变换矩阵更新位姿"""
        # 提取平移
        self.current_pose.x += transform[0, 3]
        self.current_pose.y += transform[1, 3]
        self.current_pose.z += transform[2, 3]
        
        # 提取旋转（简化，仅考虑yaw）
        R = transform[:3, :3]
        self.current_pose.yaw += np.arctan2(R[1, 0], R[0, 0])
    
    def _update_subsidence(self, ground_points: np.ndarray):
        """更新沉降监测"""
        for point_id, point in self.subsidence_monitor.monitoring_points.items():
            loc = point.location
            # 找到该位置附近的地面点
            distances = np.sqrt(
                (ground_points[:, 0] - loc[0])**2 + 
                (ground_points[:, 1] - loc[1])**2
            )
            nearby_mask = distances < 1.0  # 1米范围内
            
            if np.any(nearby_mask):
                avg_height = np.mean(ground_points[nearby_mask, 2])
                self.subsidence_monitor.update_measurement(point_id, avg_height)
    
    def register_device(self, device_id: str, location: Dict[str, float], 
                       device_type: str, metadata: Dict = None):
        """
        注册设备位置
        
        Args:
            device_id: 设备ID
            location: 位置 {'x': ..., 'y': ..., 'z': ...}
            device_type: 设备类型
            metadata: 附加信息
        """
        self.device_locations[device_id] = {
            'location': location,
            'type': device_type,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat()
        }
        logger.info(f"已注册设备: {device_id} @ ({location['x']}, {location['y']}, {location['z']})")
    
    def locate_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """获取设备位置"""
        return self.device_locations.get(device_id)
    
    def find_nearest_devices(self, location: Dict[str, float], 
                            radius: float = 10.0,
                            device_type: str = None) -> List[Dict[str, Any]]:
        """
        查找附近设备
        
        Args:
            location: 查询位置
            radius: 搜索半径（米）
            device_type: 设备类型过滤
            
        Returns:
            附近设备列表
        """
        results = []
        query_point = np.array([location['x'], location['y'], location['z']])
        
        for device_id, info in self.device_locations.items():
            if device_type and info['type'] != device_type:
                continue
                
            device_point = np.array([
                info['location']['x'], 
                info['location']['y'], 
                info['location']['z']
            ])
            distance = np.linalg.norm(query_point - device_point)
            
            if distance <= radius:
                results.append({
                    'device_id': device_id,
                    'distance': distance,
                    **info
                })
                
        # 按距离排序
        results.sort(key=lambda x: x['distance'])
        return results
    
    def plan_inspection_path(self, start: Dict[str, float], 
                            device_ids: List[str]) -> Dict[str, Any]:
        """
        规划巡检路径
        
        Args:
            start: 起点位置
            device_ids: 待巡检设备ID列表
            
        Returns:
            规划结果
        """
        if not device_ids:
            return {'success': False, 'error': '设备列表为空'}
            
        # 获取所有设备位置
        waypoints = [start]
        for device_id in device_ids:
            device_info = self.locate_device(device_id)
            if device_info:
                waypoints.append(device_info['location'])
                
        if len(waypoints) < 2:
            return {'success': False, 'error': '有效设备位置不足'}
            
        # 简单的顺序访问（可优化为TSP）
        full_path = []
        total_distance = 0
        
        for i in range(len(waypoints) - 1):
            start_np = np.array([waypoints[i]['x'], waypoints[i]['y'], 0])
            end_np = np.array([waypoints[i+1]['x'], waypoints[i+1]['y'], 0])
            
            segment = self.path_planner.plan_path(start_np, end_np)
            if not segment:
                return {
                    'success': False, 
                    'error': f'无法规划从点{i}到点{i+1}的路径'
                }
                
            full_path.extend(segment)
            
            # 计算距离
            for j in range(len(segment) - 1):
                total_distance += np.linalg.norm(segment[j+1] - segment[j])
        
        return {
            'success': True,
            'path': [{'x': p[0], 'y': p[1], 'z': p[2]} for p in full_path],
            'waypoints': waypoints,
            'device_ids': device_ids,
            'total_distance': total_distance,
            'estimated_time_minutes': total_distance / 1.0  # 假设1m/s步行速度
        }
    
    def add_subsidence_monitor(self, point_id: str, location: Tuple[float, float],
                               initial_height: float = None):
        """添加沉降监测点"""
        if initial_height is None:
            # 从地图获取当前高度
            grid_coords = self.occupancy_map.world_to_grid(
                np.array([location[0], location[1], 0]))
            x, y = grid_coords[0], grid_coords[1]
            if 0 <= x < self.occupancy_map.size[0] and 0 <= y < self.occupancy_map.size[1]:
                initial_height = self.occupancy_map.height_map[x, y]
                if np.isnan(initial_height):
                    initial_height = 0.0
            else:
                initial_height = 0.0
                
        self.subsidence_monitor.add_monitoring_point(point_id, location, initial_height)
        logger.info(f"添加沉降监测点: {point_id} @ {location}, 初始高度: {initial_height:.3f}m")
    
    def get_map_data(self, format: str = '2d') -> Dict[str, Any]:
        """
        获取地图数据
        
        Args:
            format: '2d' 或 '3d'
            
        Returns:
            地图数据
        """
        if format == '2d':
            map_2d = self.occupancy_map.get_2d_map()
            return {
                'format': '2d',
                'resolution': self.occupancy_map.resolution,
                'origin': self.occupancy_map.origin.tolist(),
                'size': self.occupancy_map.size[:2],
                'data': map_2d.tolist(),
                'height_map': np.nan_to_num(
                    self.occupancy_map.height_map, nan=0).tolist()
            }
        else:
            # 3D体素地图
            return {
                'format': '3d',
                'resolution': self.occupancy_map.resolution,
                'origin': self.occupancy_map.origin.tolist(),
                'size': self.occupancy_map.size,
                'occupied_voxels': self._get_occupied_voxels()
            }
    
    def _get_occupied_voxels(self, threshold: float = 0.6) -> List[Dict]:
        """获取占据的体素列表"""
        voxels = []
        prob_map = 1.0 / (1.0 + np.exp(-self.occupancy_map.grid))
        occupied = np.where(prob_map > threshold)
        
        for i in range(len(occupied[0])):
            x, y, z = occupied[0][i], occupied[1][i], occupied[2][i]
            world_coords = self.occupancy_map.grid_to_world(np.array([x, y, z]))
            voxels.append({
                'x': float(world_coords[0]),
                'y': float(world_coords[1]),
                'z': float(world_coords[2]),
                'probability': float(prob_map[x, y, z])
            })
            
        return voxels
    
    def detect_obstacles(self, robot_pose: Dict[str, float], 
                        detection_range: float = 5.0) -> List[Dict[str, Any]]:
        """
        检测障碍物
        
        Args:
            robot_pose: 机器人当前位置
            detection_range: 检测范围（米）
            
        Returns:
            障碍物列表
        """
        obstacles = []
        robot_pos = np.array([robot_pose['x'], robot_pose['y']])
        
        map_2d = self.occupancy_map.get_2d_map()
        
        # 在检测范围内搜索
        grid_range = int(detection_range / self.occupancy_map.resolution)
        robot_grid = self.occupancy_map.world_to_grid(
            np.array([robot_pose['x'], robot_pose['y'], 0]))
        
        for dx in range(-grid_range, grid_range + 1):
            for dy in range(-grid_range, grid_range + 1):
                gx, gy = robot_grid[0] + dx, robot_grid[1] + dy
                
                if not (0 <= gx < self.occupancy_map.size[0] and 
                        0 <= gy < self.occupancy_map.size[1]):
                    continue
                    
                if map_2d[gx, gy] > 0.7:  # 高置信度障碍物
                    world_pos = self.occupancy_map.grid_to_world(np.array([gx, gy, 0]))
                    distance = np.linalg.norm(world_pos[:2] - robot_pos)
                    
                    if distance <= detection_range:
                        obstacles.append({
                            'x': float(world_pos[0]),
                            'y': float(world_pos[1]),
                            'distance': float(distance),
                            'probability': float(map_2d[gx, gy])
                        })
        
        # 按距离排序
        obstacles.sort(key=lambda x: x['distance'])
        return obstacles
    
    def export_map(self, filepath: str, format: str = 'json'):
        """导出地图"""
        map_data = self.get_map_data('2d')
        map_data['devices'] = self.device_locations
        map_data['subsidence_points'] = {
            pid: {
                'location': p.location,
                'initial_height': p.initial_height,
                'current_height': p.current_height,
                'subsidence_mm': self.subsidence_monitor.get_subsidence(pid)
            }
            for pid, p in self.subsidence_monitor.monitoring_points.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(map_data, f, indent=2)
            
        logger.info(f"地图已导出: {filepath}")
    
    def infer(self, frame, rois, context):
        """实现BasePlugin抽象方法 - 执行推理"""
        return []

    def postprocess(self, results, rules):
        """实现BasePlugin抽象方法 - 后处理"""
        return []

    def healthcheck(self):
        """实现BasePlugin抽象方法 - 健康检查"""
        try:
            from platform_core.plugin_manager.base import HealthStatus
            return HealthStatus(healthy=True, message="OK")
        except ImportError:
            return {"healthy": True, "message": "OK"}

    def get_status(self) -> Dict[str, Any]:
        """获取插件状态"""
        return {
            'name': self.name,
            'version': self.version,
            'frame_count': self.frame_count,
            'current_pose': {
                'x': self.current_pose.x,
                'y': self.current_pose.y,
                'z': self.current_pose.z,
                'yaw': self.current_pose.yaw
            },
            'pose_history_length': len(self.pose_history),
            'registered_devices': len(self.device_locations),
            'subsidence_points': len(self.subsidence_monitor.monitoring_points),
            'map_resolution': self.occupancy_map.resolution,
            'map_size': self.occupancy_map.size,
            'dl_enabled': self.dl_enabled
        }


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建插件实例
    plugin = SLAMMappingPlugin()
    plugin.init()
    
    # 模拟点云数据
    np.random.seed(42)
    
    # 生成地面点
    ground_points = np.random.randn(500, 3) * [20, 20, 0.1]
    ground_points[:, 2] = 0  # 地面高度为0
    
    # 生成物体点（模拟变压器）
    obj1 = np.random.randn(100, 3) * [1, 1, 2] + [5, 5, 1]
    obj2 = np.random.randn(80, 3) * [0.5, 0.5, 1.5] + [-3, 8, 0.75]
    
    points = np.vstack([ground_points, obj1, obj2])
    
    # 处理点云
    result = plugin.process_point_cloud(points)
    print(f"处理结果: {json.dumps(result, indent=2, default=str)}")
    
    # 注册设备
    plugin.register_device("T1", {'x': 5.0, 'y': 5.0, 'z': 0.0}, "transformer")
    plugin.register_device("T2", {'x': -3.0, 'y': 8.0, 'z': 0.0}, "transformer")
    plugin.register_device("SW1", {'x': 10.0, 'y': 0.0, 'z': 0.0}, "switch")
    
    # 规划巡检路径
    path_result = plugin.plan_inspection_path(
        {'x': 0, 'y': 0, 'z': 0},
        ['T1', 'T2', 'SW1']
    )
    print(f"\n巡检路径规划: {path_result['success']}, 总距离: {path_result.get('total_distance', 0):.1f}m")
    
    # 添加沉降监测点
    plugin.add_subsidence_monitor("SUB1", (5.0, 5.0), 0.0)
    
    # 获取状态
    status = plugin.get_status()
    print(f"\n插件状态: {json.dumps(status, indent=2)}")
    
    plugin.shutdown()
