"""
SLAM建图与导航插件
输变电站全自动AI巡检方案 - 3D LiDAR SLAM

功能:
1. 点云建图与定位 (LOAM/LIO-SAM风格)
2. 路径规划与导航
3. 障碍物检测与规避
4. 巡检路径生成

依赖:
- 3D LiDAR数据
- IMU数据 (可选)
- 里程计数据 (可选)

版本: 2.0.0
"""

from __future__ import annotations
import os
import time
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

try:
    import open3d as o3d
except ImportError:
    o3d = None

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================
class SLAMState(Enum):
    """SLAM状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    MAPPING = "mapping"


@dataclass
class Pose3D:
    """3D位姿"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0
    
    def to_matrix(self) -> np.ndarray:
        """转换为4x4变换矩阵"""
        # 旋转矩阵
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.x, self.y, self.z]
        
        return T
    
    @classmethod
    def from_matrix(cls, T: np.ndarray, timestamp: float = 0.0) -> 'Pose3D':
        """从变换矩阵创建"""
        x, y, z = T[:3, 3]
        
        # 从旋转矩阵提取欧拉角
        R = T[:3, :3]
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return cls(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, timestamp=timestamp)


@dataclass
class PointCloudFrame:
    """点云帧"""
    points: np.ndarray          # (N, 3) 或 (N, 4) xyz(i)
    timestamp: float = 0.0
    pose: Optional[Pose3D] = None
    features: Optional[np.ndarray] = None
    
    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class MapPoint:
    """地图点"""
    position: np.ndarray        # (3,) xyz
    normal: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    observations: int = 0
    last_seen: float = 0.0


@dataclass
class Waypoint:
    """路径点"""
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    inspection_point: bool = False
    device_id: Optional[str] = None


@dataclass 
class NavigationPath:
    """导航路径"""
    waypoints: List[Waypoint]
    total_distance: float = 0.0
    estimated_time: float = 0.0
    created_time: float = field(default_factory=time.time)


# =============================================================================
# 点云特征提取
# =============================================================================
class PointCloudFeatureExtractor:
    """点云特征提取器 (LOAM风格)"""
    
    def __init__(self, 
                 edge_threshold: float = 0.1,
                 surface_threshold: float = 0.1,
                 scan_lines: int = 16):
        self.edge_threshold = edge_threshold
        self.surface_threshold = surface_threshold
        self.scan_lines = scan_lines
    
    def extract_features(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取边缘和平面特征点
        
        Args:
            points: 点云 (N, 3)
        
        Returns:
            edge_points: 边缘特征点
            surface_points: 平面特征点
        """
        if len(points) < 100:
            return np.array([]), np.array([])
        
        # 计算曲率
        curvatures = self._compute_curvature(points)
        
        # 根据曲率分类
        edge_mask = curvatures > self.edge_threshold
        surface_mask = curvatures < self.surface_threshold
        
        edge_points = points[edge_mask]
        surface_points = points[surface_mask]
        
        return edge_points, surface_points
    
    def _compute_curvature(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """计算每个点的曲率"""
        n = len(points)
        curvatures = np.zeros(n)
        
        # 简化实现: 使用局部平面拟合
        for i in range(k, n - k):
            # 获取邻域点
            neighbors = points[i-k:i+k+1]
            
            # 计算中心点
            center = np.mean(neighbors, axis=0)
            
            # 计算协方差矩阵
            centered = neighbors - center
            cov = np.dot(centered.T, centered) / len(neighbors)
            
            # 特征值分解
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)
            
            # 曲率 = 最小特征值 / 特征值之和
            curvatures[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-8)
        
        return curvatures


# =============================================================================
# 点云配准
# =============================================================================
class PointCloudRegistration:
    """点云配准 (ICP/NDT)"""
    
    def __init__(self, 
                 method: str = "icp",
                 max_iterations: int = 50,
                 tolerance: float = 1e-6,
                 max_correspondence_distance: float = 0.5):
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
    
    def align(self, source: np.ndarray, target: np.ndarray,
              initial_transform: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        配准源点云到目标点云
        
        Args:
            source: 源点云 (N, 3)
            target: 目标点云 (M, 3)
            initial_transform: 初始变换 (4, 4)
        
        Returns:
            transform: 变换矩阵 (4, 4)
            fitness: 配准质量分数
        """
        if o3d is not None:
            return self._align_open3d(source, target, initial_transform)
        else:
            return self._align_simple_icp(source, target, initial_transform)
    
    def _align_open3d(self, source: np.ndarray, target: np.ndarray,
                      initial_transform: np.ndarray) -> Tuple[np.ndarray, float]:
        """使用Open3D进行配准"""
        # 创建点云对象
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        
        # 估计法线
        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # 初始变换
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        # 执行ICP
        if self.method == "icp":
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations
                )
            )
        else:  # NDT-like using colored ICP as approximation
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations
                )
            )
        
        return result.transformation, result.fitness
    
    def _align_simple_icp(self, source: np.ndarray, target: np.ndarray,
                          initial_transform: np.ndarray) -> Tuple[np.ndarray, float]:
        """简单ICP实现(不依赖Open3D)"""
        if initial_transform is None:
            T = np.eye(4)
        else:
            T = initial_transform.copy()
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # 应用当前变换
            source_transformed = self._apply_transform(source, T)
            
            # 找最近邻
            correspondences, distances = self._find_correspondences(
                source_transformed, target
            )
            
            # 过滤远点
            valid_mask = distances < self.max_correspondence_distance
            if np.sum(valid_mask) < 10:
                break
            
            src_valid = source_transformed[valid_mask]
            tgt_valid = target[correspondences[valid_mask]]
            
            # 计算变换
            delta_T = self._compute_transform(src_valid, tgt_valid)
            T = delta_T @ T
            
            # 检查收敛
            error = np.mean(distances[valid_mask])
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error
        
        fitness = np.sum(distances < self.max_correspondence_distance) / len(source)
        return T, fitness
    
    def _apply_transform(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """应用变换"""
        R, t = T[:3, :3], T[:3, 3]
        return (R @ points.T).T + t
    
    def _find_correspondences(self, source: np.ndarray, 
                               target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """找最近邻对应"""
        correspondences = np.zeros(len(source), dtype=int)
        distances = np.zeros(len(source))
        
        for i, src_pt in enumerate(source):
            dists = np.linalg.norm(target - src_pt, axis=1)
            correspondences[i] = np.argmin(dists)
            distances[i] = dists[correspondences[i]]
        
        return correspondences, distances
    
    def _compute_transform(self, source: np.ndarray, 
                           target: np.ndarray) -> np.ndarray:
        """计算刚体变换 (SVD方法)"""
        # 计算质心
        src_centroid = np.mean(source, axis=0)
        tgt_centroid = np.mean(target, axis=0)
        
        # 去中心化
        src_centered = source - src_centroid
        tgt_centered = target - tgt_centroid
        
        # SVD
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        
        # 旋转矩阵
        R = Vt.T @ U.T
        
        # 处理反射
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 平移向量
        t = tgt_centroid - R @ src_centroid
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T


# =============================================================================
# SLAM核心
# =============================================================================
class LiDARSLAM:
    """
    LiDAR SLAM核心算法
    
    实现简化版LOAM/LIO-SAM风格的SLAM
    """
    
    def __init__(self,
                 voxel_size: float = 0.1,
                 keyframe_distance: float = 1.0,
                 keyframe_angle: float = 0.2,
                 map_resolution: float = 0.05):
        # 参数
        self.voxel_size = voxel_size
        self.keyframe_distance = keyframe_distance
        self.keyframe_angle = keyframe_angle
        self.map_resolution = map_resolution
        
        # 组件
        self.feature_extractor = PointCloudFeatureExtractor()
        self.registration = PointCloudRegistration()
        
        # 状态
        self.state = SLAMState.IDLE
        self.current_pose = Pose3D()
        self.poses: List[Pose3D] = []
        self.keyframes: List[PointCloudFrame] = []
        self.global_map: Optional[np.ndarray] = None
        
        # 缓存
        self._last_keyframe_pose: Optional[Pose3D] = None
        self._local_map: Optional[np.ndarray] = None
        self._lock = threading.Lock()
    
    def initialize(self, first_frame: PointCloudFrame) -> bool:
        """初始化SLAM"""
        with self._lock:
            self.state = SLAMState.INITIALIZING
            
            # 预处理点云
            processed = self._preprocess(first_frame.points)
            
            # 设置初始位姿
            self.current_pose = Pose3D(timestamp=first_frame.timestamp)
            first_frame.pose = self.current_pose
            
            # 添加第一个关键帧
            self.keyframes.append(first_frame)
            self._last_keyframe_pose = self.current_pose
            
            # 初始化地图
            self.global_map = processed.copy()
            self._local_map = processed.copy()
            
            self.state = SLAMState.TRACKING
            logger.info("SLAM初始化完成")
            return True
    
    def process_frame(self, frame: PointCloudFrame) -> Tuple[bool, Pose3D]:
        """
        处理点云帧
        
        Args:
            frame: 点云帧
        
        Returns:
            success: 是否成功
            pose: 估计的位姿
        """
        with self._lock:
            if self.state == SLAMState.IDLE:
                return self.initialize(frame), self.current_pose
            
            if self.state == SLAMState.LOST:
                # 尝试重定位
                return self._relocalize(frame)
            
            # 预处理
            processed = self._preprocess(frame.points)
            
            # 提取特征
            edge_pts, surface_pts = self.feature_extractor.extract_features(processed)
            
            # 位姿估计 (scan-to-map matching)
            if self._local_map is not None and len(self._local_map) > 100:
                initial_T = self.current_pose.to_matrix()
                T, fitness = self.registration.align(processed, self._local_map, initial_T)
                
                if fitness < 0.3:  # 配准失败
                    self.state = SLAMState.LOST
                    logger.warning(f"跟踪丢失, fitness={fitness:.3f}")
                    return False, self.current_pose
                
                # 更新位姿
                self.current_pose = Pose3D.from_matrix(T, frame.timestamp)
            
            # 保存位姿
            self.poses.append(self.current_pose)
            frame.pose = self.current_pose
            
            # 检查是否需要添加关键帧
            if self._should_add_keyframe():
                self._add_keyframe(frame, processed)
            
            return True, self.current_pose
    
    def _preprocess(self, points: np.ndarray) -> np.ndarray:
        """预处理点云"""
        # 去除无效点
        valid_mask = ~np.isnan(points).any(axis=1)
        valid_mask &= ~np.isinf(points).any(axis=1)
        points = points[valid_mask]
        
        # 距离过滤
        distances = np.linalg.norm(points[:, :3], axis=1)
        range_mask = (distances > 0.5) & (distances < 100.0)
        points = points[range_mask]
        
        # 体素下采样
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd = pcd.voxel_down_sample(self.voxel_size)
            points = np.asarray(pcd.points)
        else:
            # 简单网格下采样
            points = self._voxel_downsample(points[:, :3], self.voxel_size)
        
        return points
    
    def _voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """简单体素下采样"""
        # 计算体素索引
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # 使用字典去重
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[i])
        
        # 计算每个体素的中心点
        downsampled = []
        for pts in voxel_dict.values():
            downsampled.append(np.mean(pts, axis=0))
        
        return np.array(downsampled)
    
    def _should_add_keyframe(self) -> bool:
        """检查是否应该添加关键帧"""
        if self._last_keyframe_pose is None:
            return True
        
        # 距离检查
        dx = self.current_pose.x - self._last_keyframe_pose.x
        dy = self.current_pose.y - self._last_keyframe_pose.y
        dz = self.current_pose.z - self._last_keyframe_pose.z
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance > self.keyframe_distance:
            return True
        
        # 角度检查
        dyaw = abs(self.current_pose.yaw - self._last_keyframe_pose.yaw)
        if dyaw > self.keyframe_angle:
            return True
        
        return False
    
    def _add_keyframe(self, frame: PointCloudFrame, processed: np.ndarray) -> None:
        """添加关键帧"""
        self.keyframes.append(frame)
        self._last_keyframe_pose = self.current_pose
        
        # 更新局部地图
        T = self.current_pose.to_matrix()
        transformed = (T[:3, :3] @ processed.T).T + T[:3, 3]
        
        if self._local_map is None:
            self._local_map = transformed
        else:
            self._local_map = np.vstack([self._local_map, transformed])
            # 限制局部地图大小
            if len(self._local_map) > 100000:
                self._local_map = self._voxel_downsample(
                    self._local_map, self.voxel_size * 2
                )
        
        # 更新全局地图
        if self.global_map is None:
            self.global_map = transformed
        else:
            self.global_map = np.vstack([self.global_map, transformed])
        
        logger.debug(f"添加关键帧 #{len(self.keyframes)}, 地图点数: {len(self.global_map)}")
    
    def _relocalize(self, frame: PointCloudFrame) -> Tuple[bool, Pose3D]:
        """重定位"""
        if self.global_map is None or len(self.global_map) < 100:
            return False, self.current_pose
        
        processed = self._preprocess(frame.points)
        
        # 尝试全局配准
        T, fitness = self.registration.align(processed, self.global_map)
        
        if fitness > 0.5:
            self.current_pose = Pose3D.from_matrix(T, frame.timestamp)
            self.state = SLAMState.TRACKING
            logger.info(f"重定位成功, fitness={fitness:.3f}")
            return True, self.current_pose
        
        return False, self.current_pose
    
    def get_map(self) -> Optional[np.ndarray]:
        """获取全局地图"""
        return self.global_map
    
    def get_trajectory(self) -> List[Pose3D]:
        """获取轨迹"""
        return self.poses.copy()
    
    def save_map(self, filepath: str) -> bool:
        """保存地图"""
        if self.global_map is None:
            return False
        
        try:
            np.save(filepath, self.global_map)
            logger.info(f"地图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存地图失败: {e}")
            return False
    
    def load_map(self, filepath: str) -> bool:
        """加载地图"""
        try:
            self.global_map = np.load(filepath)
            self._local_map = self.global_map.copy()
            logger.info(f"地图已加载: {filepath}, 点数: {len(self.global_map)}")
            return True
        except Exception as e:
            logger.error(f"加载地图失败: {e}")
            return False


# =============================================================================
# 路径规划
# =============================================================================
class PathPlanner:
    """路径规划器"""
    
    def __init__(self,
                 grid_resolution: float = 0.5,
                 robot_radius: float = 0.5,
                 safety_margin: float = 0.3):
        self.grid_resolution = grid_resolution
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self._occupancy_grid: Optional[np.ndarray] = None
        self._grid_origin: np.ndarray = np.zeros(2)
    
    def build_occupancy_grid(self, point_cloud: np.ndarray,
                             height_range: Tuple[float, float] = (-0.5, 2.0)) -> np.ndarray:
        """
        从点云构建2D占据栅格
        
        Args:
            point_cloud: 点云 (N, 3)
            height_range: 高度范围过滤
        
        Returns:
            occupancy_grid: 占据栅格
        """
        # 高度过滤
        height_mask = (point_cloud[:, 2] > height_range[0]) & \
                      (point_cloud[:, 2] < height_range[1])
        points_2d = point_cloud[height_mask][:, :2]
        
        if len(points_2d) == 0:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # 计算边界
        min_xy = np.min(points_2d, axis=0)
        max_xy = np.max(points_2d, axis=0)
        
        # 添加边界
        margin = self.robot_radius + self.safety_margin
        min_xy -= margin
        max_xy += margin
        
        self._grid_origin = min_xy
        
        # 计算栅格大小
        grid_size = np.ceil((max_xy - min_xy) / self.grid_resolution).astype(int)
        grid_size = np.maximum(grid_size, [10, 10])
        
        # 创建栅格
        self._occupancy_grid = np.zeros(grid_size, dtype=np.uint8)
        
        # 填充障碍物
        grid_coords = np.floor((points_2d - min_xy) / self.grid_resolution).astype(int)
        valid_mask = (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < grid_size[0]) & \
                     (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < grid_size[1])
        
        for coord in grid_coords[valid_mask]:
            self._occupancy_grid[coord[0], coord[1]] = 1
        
        # 膨胀障碍物
        inflate_cells = int(np.ceil((self.robot_radius + self.safety_margin) / self.grid_resolution))
        self._occupancy_grid = self._inflate_obstacles(self._occupancy_grid, inflate_cells)
        
        return self._occupancy_grid
    
    def _inflate_obstacles(self, grid: np.ndarray, radius: int) -> np.ndarray:
        """膨胀障碍物"""
        from scipy import ndimage
        
        # 创建圆形结构元素
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        structure = x*x + y*y <= radius*radius
        
        # 膨胀
        inflated = ndimage.binary_dilation(grid, structure=structure)
        
        return inflated.astype(np.uint8)
    
    def plan_path(self, start: Tuple[float, float],
                  goal: Tuple[float, float]) -> Optional[NavigationPath]:
        """
        A*路径规划
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
        
        Returns:
            path: 导航路径
        """
        if self._occupancy_grid is None:
            logger.error("占据栅格未初始化")
            return None
        
        # 转换为栅格坐标
        start_cell = self._world_to_grid(start)
        goal_cell = self._world_to_grid(goal)
        
        # 检查有效性
        if not self._is_valid_cell(start_cell) or not self._is_valid_cell(goal_cell):
            logger.error("起点或终点无效")
            return None
        
        # A*搜索
        path_cells = self._astar(start_cell, goal_cell)
        
        if path_cells is None:
            logger.warning("找不到路径")
            return None
        
        # 转换为世界坐标
        waypoints = []
        for cell in path_cells:
            world_pos = self._grid_to_world(cell)
            waypoints.append(Waypoint(x=world_pos[0], y=world_pos[1]))
        
        # 路径平滑
        waypoints = self._smooth_path(waypoints)
        
        # 计算总距离
        total_distance = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i-1].x
            dy = waypoints[i].y - waypoints[i-1].y
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        return NavigationPath(
            waypoints=waypoints,
            total_distance=total_distance,
            estimated_time=total_distance / 0.5  # 假设0.5m/s速度
        )
    
    def _world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        grid_x = int((pos[0] - self._grid_origin[0]) / self.grid_resolution)
        grid_y = int((pos[1] - self._grid_origin[1]) / self.grid_resolution)
        return (grid_x, grid_y)
    
    def _grid_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        world_x = cell[0] * self.grid_resolution + self._grid_origin[0]
        world_y = cell[1] * self.grid_resolution + self._grid_origin[1]
        return (world_x, world_y)
    
    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """检查栅格是否有效"""
        if self._occupancy_grid is None:
            return False
        
        rows, cols = self._occupancy_grid.shape
        if cell[0] < 0 or cell[0] >= rows or cell[1] < 0 or cell[1] >= cols:
            return False
        
        return self._occupancy_grid[cell[0], cell[1]] == 0
    
    def _astar(self, start: Tuple[int, int], 
               goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A*算法"""
        import heapq
        
        def heuristic(a, b):
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
        # 8邻域
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid_cell(neighbor):
                    continue
                
                # 对角线移动代价更高
                move_cost = np.sqrt(dx*dx + dy*dy)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _smooth_path(self, waypoints: List[Waypoint],
                     weight_data: float = 0.5,
                     weight_smooth: float = 0.1,
                     tolerance: float = 0.001) -> List[Waypoint]:
        """路径平滑"""
        if len(waypoints) <= 2:
            return waypoints
        
        # 转换为数组
        path = np.array([[wp.x, wp.y] for wp in waypoints])
        smoothed = path.copy()
        
        change = tolerance + 1
        while change > tolerance:
            change = 0
            for i in range(1, len(path) - 1):
                for j in range(2):
                    old = smoothed[i, j]
                    smoothed[i, j] += weight_data * (path[i, j] - smoothed[i, j])
                    smoothed[i, j] += weight_smooth * (
                        smoothed[i-1, j] + smoothed[i+1, j] - 2 * smoothed[i, j]
                    )
                    change += abs(old - smoothed[i, j])
        
        # 转换回Waypoint
        result = []
        for i, pt in enumerate(smoothed):
            wp = Waypoint(x=pt[0], y=pt[1])
            if i > 0:
                dx = pt[0] - smoothed[i-1, 0]
                dy = pt[1] - smoothed[i-1, 1]
                wp.yaw = np.arctan2(dy, dx)
            result.append(wp)
        
        return result


# =============================================================================
# SLAM建图插件
# =============================================================================
class SLAMMappingPlugin:
    """
    SLAM建图与导航插件
    
    提供完整的建图、定位、导航功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # SLAM核心
        self.slam = LiDARSLAM(
            voxel_size=self.config.get("voxel_size", 0.1),
            keyframe_distance=self.config.get("keyframe_distance", 1.0),
            keyframe_angle=self.config.get("keyframe_angle", 0.2)
        )
        
        # 路径规划
        self.planner = PathPlanner(
            grid_resolution=self.config.get("grid_resolution", 0.5),
            robot_radius=self.config.get("robot_radius", 0.5)
        )
        
        # 模型注册中心(可选,用于深度学习增强)
        self._model_registry = None
        
        # 状态
        self._initialized = False
        self._current_path: Optional[NavigationPath] = None
    
    def set_model_registry(self, registry) -> None:
        """设置模型注册中心"""
        self._model_registry = registry
    
    def initialize(self) -> bool:
        """初始化插件"""
        self._initialized = True
        logger.info("SLAM建图插件初始化完成")
        return True
    
    def process_lidar_frame(self, points: np.ndarray, 
                            timestamp: float = None) -> Dict[str, Any]:
        """
        处理LiDAR帧
        
        Args:
            points: 点云数据 (N, 3) 或 (N, 4)
            timestamp: 时间戳
        
        Returns:
            结果字典
        """
        if timestamp is None:
            timestamp = time.time()
        
        frame = PointCloudFrame(
            points=points,
            timestamp=timestamp
        )
        
        success, pose = self.slam.process_frame(frame)
        
        result = {
            "success": success,
            "state": self.slam.state.value,
            "pose": {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "roll": pose.roll,
                "pitch": pose.pitch,
                "yaw": pose.yaw
            },
            "keyframe_count": len(self.slam.keyframes),
            "map_points": len(self.slam.global_map) if self.slam.global_map is not None else 0
        }
        
        return result
    
    def get_current_pose(self) -> Pose3D:
        """获取当前位姿"""
        return self.slam.current_pose
    
    def get_map(self) -> Optional[np.ndarray]:
        """获取地图"""
        return self.slam.get_map()
    
    def plan_inspection_route(self, 
                              inspection_points: List[Tuple[float, float]]) -> Optional[NavigationPath]:
        """
        规划巡检路线
        
        Args:
            inspection_points: 巡检点列表 [(x, y), ...]
        
        Returns:
            导航路径
        """
        if self.slam.global_map is None:
            logger.error("地图未建立")
            return None
        
        # 构建占据栅格
        self.planner.build_occupancy_grid(self.slam.global_map)
        
        # 获取当前位置
        current_pos = (self.slam.current_pose.x, self.slam.current_pose.y)
        
        # TSP求解巡检顺序(简化版: 贪心)
        ordered_points = self._order_inspection_points(current_pos, inspection_points)
        
        # 规划完整路径
        all_waypoints = []
        prev_point = current_pos
        
        for point in ordered_points:
            segment = self.planner.plan_path(prev_point, point)
            if segment:
                all_waypoints.extend(segment.waypoints)
            prev_point = point
        
        if not all_waypoints:
            return None
        
        # 标记巡检点
        for wp in all_waypoints:
            for ip in inspection_points:
                if np.sqrt((wp.x - ip[0])**2 + (wp.y - ip[1])**2) < 0.5:
                    wp.inspection_point = True
                    break
        
        # 计算总距离
        total_distance = 0.0
        for i in range(1, len(all_waypoints)):
            dx = all_waypoints[i].x - all_waypoints[i-1].x
            dy = all_waypoints[i].y - all_waypoints[i-1].y
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        self._current_path = NavigationPath(
            waypoints=all_waypoints,
            total_distance=total_distance,
            estimated_time=total_distance / 0.5
        )
        
        return self._current_path
    
    def _order_inspection_points(self, start: Tuple[float, float],
                                 points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """贪心TSP排序巡检点"""
        remaining = list(points)
        ordered = []
        current = start
        
        while remaining:
            # 找最近的点
            distances = [np.sqrt((p[0]-current[0])**2 + (p[1]-current[1])**2) 
                        for p in remaining]
            nearest_idx = np.argmin(distances)
            
            current = remaining.pop(nearest_idx)
            ordered.append(current)
        
        return ordered
    
    def save_map(self, filepath: str) -> bool:
        """保存地图"""
        return self.slam.save_map(filepath)
    
    def load_map(self, filepath: str) -> bool:
        """加载地图"""
        return self.slam.load_map(filepath)
    
    def inspect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行巡检(插件统一接口)
        
        Args:
            data: 包含点云数据的字典
        
        Returns:
            巡检结果
        """
        points = data.get("point_cloud") or data.get("lidar_data")
        if points is None:
            return {"success": False, "error": "缺少点云数据"}
        
        return self.process_lidar_frame(points, data.get("timestamp"))


# =============================================================================
# 检测器增强版
# =============================================================================
class SLAMDetectorEnhanced:
    """SLAM检测器增强版"""
    
    def __init__(self, config: Dict[str, Any] = None, model_registry=None):
        self.config = config or {}
        self._model_registry = model_registry
        self._use_deep_learning = model_registry is not None
        
        # 核心组件
        self.slam_plugin = SLAMMappingPlugin(config)
        
        # 深度学习模型ID
        self._slam_model_id = "pointcloud_inspection_lidar_slam"
        self._seg_model_id = "pointcloud_inspection_point_seg"
    
    def set_model_registry(self, registry) -> None:
        """设置模型注册中心"""
        self._model_registry = registry
        self._use_deep_learning = registry is not None
        self.slam_plugin.set_model_registry(registry)
    
    def detect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行检测"""
        # 优先尝试深度学习
        if self._use_deep_learning and self._model_registry:
            try:
                result = self._detect_deep_learning(data)
                if result.get("success"):
                    return result
            except Exception as e:
                logger.warning(f"深度学习检测失败,回退到传统方法: {e}")
        
        # 回退到传统SLAM
        return self.slam_plugin.inspect(data)
    
    def _detect_deep_learning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """深度学习检测"""
        points = data.get("point_cloud") or data.get("lidar_data")
        if points is None:
            return {"success": False, "error": "缺少点云数据"}
        
        # 调用点云语义分割模型
        if self._model_registry.is_registered(self._seg_model_id):
            seg_result = self._model_registry.infer(
                self._seg_model_id,
                {"point_cloud": points}
            )
            
            if seg_result.success and seg_result.semantic_labels is not None:
                # 识别场景元素
                labels = seg_result.semantic_labels
                
                return {
                    "success": True,
                    "method": "deep_learning",
                    "semantic_labels": labels,
                    "road_points": np.sum(labels == 0),
                    "obstacle_points": np.sum(labels == 1),
                    "equipment_points": np.sum(labels == 2)
                }
        
        return {"success": False, "error": "模型不可用"}
