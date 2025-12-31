#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图优化SLAM研究模块
回环检测与位姿图优化

功能:
1. 回环检测 (PointNetVLAD/ScanContext)
2. 位姿图优化 (g2o/GTSAM风格)
3. 因子图构建
4. 增量式优化
5. 鲁棒核函数

作者: AI巡检系统
版本: 1.0.0
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class Pose3D:
    """3D位姿"""
    translation: np.ndarray  # [x, y, z]
    rotation: np.ndarray     # 四元数 [w, x, y, z] 或旋转矩阵
    
    def __post_init__(self):
        self.translation = np.asarray(self.translation, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        
        # 如果是旋转矩阵,转换为四元数
        if self.rotation.shape == (3, 3):
            r = Rotation.from_matrix(self.rotation)
            self.rotation = r.as_quat()[[3, 0, 1, 2]]  # scipy是[x,y,z,w],转为[w,x,y,z]
    
    def to_matrix(self) -> np.ndarray:
        """转换为4x4变换矩阵"""
        T = np.eye(4)
        r = Rotation.from_quat(self.rotation[[1, 2, 3, 0]])  # [w,x,y,z] -> [x,y,z,w]
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = self.translation
        return T
    
    @classmethod
    def from_matrix(cls, T: np.ndarray) -> 'Pose3D':
        """从4x4变换矩阵创建"""
        translation = T[:3, 3]
        r = Rotation.from_matrix(T[:3, :3])
        rotation = r.as_quat()[[3, 0, 1, 2]]
        return cls(translation, rotation)
    
    def inverse(self) -> 'Pose3D':
        """计算逆变换"""
        T_inv = np.linalg.inv(self.to_matrix())
        return Pose3D.from_matrix(T_inv)
    
    def compose(self, other: 'Pose3D') -> 'Pose3D':
        """复合变换: self * other"""
        T = self.to_matrix() @ other.to_matrix()
        return Pose3D.from_matrix(T)


@dataclass
class PoseNode:
    """位姿图节点"""
    id: int
    pose: Pose3D
    timestamp: float = 0.0
    fixed: bool = False  # 是否固定不优化
    descriptor: Optional[np.ndarray] = None  # 用于回环检测的描述子


@dataclass
class PoseEdge:
    """位姿图边 (约束)"""
    id_from: int
    id_to: int
    measurement: Pose3D  # 相对位姿测量
    information: np.ndarray  # 6x6信息矩阵 (协方差逆)
    edge_type: str = "odometry"  # odometry, loop_closure
    
    def __post_init__(self):
        if self.information is None:
            self.information = np.eye(6)


# =============================================================================
# 回环检测
# =============================================================================
class ScanContext:
    """
    ScanContext回环检测
    基于点云的全局描述子
    """
    
    def __init__(self, 
                 num_sectors: int = 60,
                 num_rings: int = 20,
                 max_range: float = 80.0):
        self.num_sectors = num_sectors
        self.num_rings = num_rings
        self.max_range = max_range
        
        # 预计算角度和距离边界
        self.sector_angles = np.linspace(0, 2 * np.pi, num_sectors + 1)
        self.ring_distances = np.linspace(0, max_range, num_rings + 1)
    
    def compute_descriptor(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        计算ScanContext描述子
        
        Args:
            point_cloud: (N, 3) 点云
        
        Returns:
            descriptor: (num_rings, num_sectors) 描述子矩阵
        """
        # 过滤范围外的点
        distances = np.linalg.norm(point_cloud[:, :2], axis=1)
        mask = distances < self.max_range
        points = point_cloud[mask]
        distances = distances[mask]
        
        # 计算角度
        angles = np.arctan2(points[:, 1], points[:, 0])
        angles[angles < 0] += 2 * np.pi
        
        # 创建描述子矩阵
        descriptor = np.zeros((self.num_rings, self.num_sectors))
        
        for i in range(len(points)):
            # 找到对应的ring和sector
            ring_idx = np.searchsorted(self.ring_distances, distances[i]) - 1
            sector_idx = np.searchsorted(self.sector_angles, angles[i]) - 1
            
            ring_idx = np.clip(ring_idx, 0, self.num_rings - 1)
            sector_idx = np.clip(sector_idx, 0, self.num_sectors - 1)
            
            # 更新最大高度
            descriptor[ring_idx, sector_idx] = max(
                descriptor[ring_idx, sector_idx],
                points[i, 2]
            )
        
        return descriptor
    
    def compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        计算两个描述子的相似度
        考虑旋转不变性
        """
        max_similarity = 0.0
        
        # 尝试所有可能的旋转
        for shift in range(self.num_sectors):
            desc2_shifted = np.roll(desc2, shift, axis=1)
            
            # 余弦相似度
            norm1 = np.linalg.norm(desc1)
            norm2 = np.linalg.norm(desc2_shifted)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.sum(desc1 * desc2_shifted) / (norm1 * norm2)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def find_loop_candidates(self, 
                             query_desc: np.ndarray,
                             database: List[np.ndarray],
                             threshold: float = 0.7,
                             exclude_recent: int = 50) -> List[Tuple[int, float]]:
        """
        查找回环候选
        
        Args:
            query_desc: 查询描述子
            database: 描述子数据库
            threshold: 相似度阈值
            exclude_recent: 排除最近N帧
        
        Returns:
            候选列表 [(index, similarity), ...]
        """
        candidates = []
        
        for i in range(len(database) - exclude_recent):
            similarity = self.compute_similarity(query_desc, database[i])
            if similarity > threshold:
                candidates.append((i, similarity))
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates


if TORCH_AVAILABLE:
    class PointNetVLADLoopDetector(nn.Module):
        """
        基于PointNetVLAD的深度学习回环检测
        """
        
        def __init__(self, feature_dim: int = 256, num_clusters: int = 64):
            super().__init__()
            
            # PointNet编码器
            self.conv1 = nn.Conv1d(3, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 256, 1)
            self.conv4 = nn.Conv1d(256, feature_dim, 1)
            
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(256)
            self.bn4 = nn.BatchNorm1d(feature_dim)
            
            # NetVLAD层
            self.num_clusters = num_clusters
            self.feature_dim = feature_dim
            
            self.cluster_centers = nn.Parameter(
                torch.randn(num_clusters, feature_dim) * 0.1
            )
            self.fc_soft = nn.Conv1d(feature_dim, num_clusters, 1)
            
            # 最终投影
            self.fc_final = nn.Linear(num_clusters * feature_dim, 256)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, N, 3) 点云
            Returns:
                descriptor: (B, 256) 全局描述子
            """
            x = x.transpose(2, 1)  # (B, 3, N)
            
            # PointNet编码
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))  # (B, D, N)
            
            # NetVLAD聚合
            B, D, N = x.shape
            
            # 软分配
            soft_assign = F.softmax(self.fc_soft(x), dim=1)  # (B, K, N)
            
            # VLAD
            x = x.unsqueeze(1)  # (B, 1, D, N)
            centers = self.cluster_centers.unsqueeze(0).unsqueeze(-1)  # (1, K, D, 1)
            
            residuals = x - centers  # (B, K, D, N)
            weighted_residuals = residuals * soft_assign.unsqueeze(2)
            vlad = weighted_residuals.sum(dim=3)  # (B, K, D)
            
            # L2归一化
            vlad = F.normalize(vlad, p=2, dim=2)
            vlad = vlad.view(B, -1)  # (B, K*D)
            
            # 最终描述子
            descriptor = self.fc_final(vlad)
            descriptor = F.normalize(descriptor, p=2, dim=1)
            
            return descriptor
        
        def compute_similarity_batch(self, 
                                     query: torch.Tensor,
                                     database: torch.Tensor) -> torch.Tensor:
            """批量计算相似度"""
            return torch.mm(query, database.t())


# =============================================================================
# 位姿图优化
# =============================================================================
class PoseGraphOptimizer:
    """
    位姿图优化器
    基于非线性最小二乘的图优化
    """
    
    def __init__(self, 
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-6,
                 robust_kernel: str = "huber",
                 kernel_delta: float = 1.0):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.robust_kernel = robust_kernel
        self.kernel_delta = kernel_delta
        
        self.nodes: Dict[int, PoseNode] = {}
        self.edges: List[PoseEdge] = []
    
    def add_node(self, node: PoseNode):
        """添加节点"""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: PoseEdge):
        """添加边"""
        self.edges.append(edge)
    
    def _pose_to_vector(self, pose: Pose3D) -> np.ndarray:
        """位姿转6维向量 [x, y, z, roll, pitch, yaw]"""
        translation = pose.translation
        r = Rotation.from_quat(pose.rotation[[1, 2, 3, 0]])
        euler = r.as_euler('xyz')
        return np.concatenate([translation, euler])
    
    def _vector_to_pose(self, vec: np.ndarray) -> Pose3D:
        """6维向量转位姿"""
        translation = vec[:3]
        r = Rotation.from_euler('xyz', vec[3:])
        rotation = r.as_quat()[[3, 0, 1, 2]]
        return Pose3D(translation, rotation)
    
    def _compute_error(self, edge: PoseEdge) -> np.ndarray:
        """计算边的误差"""
        pose_i = self.nodes[edge.id_from].pose
        pose_j = self.nodes[edge.id_to].pose
        
        # 预测的相对位姿
        predicted = pose_i.inverse().compose(pose_j)
        
        # 测量的相对位姿
        measured = edge.measurement
        
        # 误差 = log(measured^{-1} * predicted)
        error_pose = measured.inverse().compose(predicted)
        error = self._pose_to_vector(error_pose)
        
        return error
    
    def _robust_weight(self, error: np.ndarray) -> float:
        """计算鲁棒核权重"""
        e_norm = np.linalg.norm(error)
        delta = self.kernel_delta
        
        if self.robust_kernel == "huber":
            if e_norm <= delta:
                return 1.0
            else:
                return delta / e_norm
        elif self.robust_kernel == "cauchy":
            return 1.0 / (1.0 + (e_norm / delta) ** 2)
        else:
            return 1.0
    
    def _compute_jacobian(self, edge: PoseEdge) -> Tuple[np.ndarray, np.ndarray]:
        """计算误差对位姿的雅可比矩阵"""
        # 数值微分
        eps = 1e-6
        
        Ji = np.zeros((6, 6))
        Jj = np.zeros((6, 6))
        
        # 保存原始位姿
        pose_i_orig = self._pose_to_vector(self.nodes[edge.id_from].pose)
        pose_j_orig = self._pose_to_vector(self.nodes[edge.id_to].pose)
        
        error_0 = self._compute_error(edge)
        
        # 对pose_i求导
        for k in range(6):
            pose_i_pert = pose_i_orig.copy()
            pose_i_pert[k] += eps
            self.nodes[edge.id_from].pose = self._vector_to_pose(pose_i_pert)
            
            error_pert = self._compute_error(edge)
            Ji[:, k] = (error_pert - error_0) / eps
            
            self.nodes[edge.id_from].pose = self._vector_to_pose(pose_i_orig)
        
        # 对pose_j求导
        for k in range(6):
            pose_j_pert = pose_j_orig.copy()
            pose_j_pert[k] += eps
            self.nodes[edge.id_to].pose = self._vector_to_pose(pose_j_pert)
            
            error_pert = self._compute_error(edge)
            Jj[:, k] = (error_pert - error_0) / eps
            
            self.nodes[edge.id_to].pose = self._vector_to_pose(pose_j_orig)
        
        return Ji, Jj
    
    def optimize(self) -> Dict[str, Any]:
        """
        执行位姿图优化
        使用Gauss-Newton或Levenberg-Marquardt
        """
        # 获取非固定节点
        free_nodes = [n for n in self.nodes.values() if not n.fixed]
        if not free_nodes:
            logger.warning("没有可优化的节点")
            return {"success": False, "message": "No free nodes"}
        
        # 创建节点索引映射
        node_to_idx = {n.id: i for i, n in enumerate(free_nodes)}
        num_vars = len(free_nodes) * 6
        
        prev_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # 构建线性系统 H * dx = b
            H = np.zeros((num_vars, num_vars))
            b = np.zeros(num_vars)
            total_cost = 0.0
            
            for edge in self.edges:
                # 计算误差
                error = self._compute_error(edge)
                
                # 鲁棒权重
                weight = self._robust_weight(error)
                
                # 计算雅可比
                Ji, Jj = self._compute_jacobian(edge)
                
                # 信息矩阵
                omega = edge.information * weight
                
                # 累加到H和b
                if edge.id_from in node_to_idx:
                    i = node_to_idx[edge.id_from] * 6
                    H[i:i+6, i:i+6] += Ji.T @ omega @ Ji
                    b[i:i+6] += Ji.T @ omega @ error
                
                if edge.id_to in node_to_idx:
                    j = node_to_idx[edge.id_to] * 6
                    H[j:j+6, j:j+6] += Jj.T @ omega @ Jj
                    b[j:j+6] += Jj.T @ omega @ error
                
                if edge.id_from in node_to_idx and edge.id_to in node_to_idx:
                    i = node_to_idx[edge.id_from] * 6
                    j = node_to_idx[edge.id_to] * 6
                    H[i:i+6, j:j+6] += Ji.T @ omega @ Jj
                    H[j:j+6, i:i+6] += Jj.T @ omega @ Ji
                
                total_cost += error.T @ omega @ error
            
            # 添加正则化 (Levenberg-Marquardt)
            lambda_lm = 1e-3
            H += lambda_lm * np.eye(num_vars)
            
            # 求解
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                logger.warning("线性系统奇异,使用伪逆")
                dx = np.linalg.lstsq(H, -b, rcond=None)[0]
            
            # 更新位姿
            for node in free_nodes:
                idx = node_to_idx[node.id] * 6
                pose_vec = self._pose_to_vector(node.pose)
                pose_vec += dx[idx:idx+6]
                node.pose = self._vector_to_pose(pose_vec)
            
            # 检查收敛
            delta_cost = prev_cost - total_cost
            logger.debug(f"Iteration {iteration}: cost={total_cost:.6f}, delta={delta_cost:.6f}")
            
            if abs(delta_cost) < self.convergence_threshold:
                logger.info(f"优化收敛于迭代 {iteration}")
                break
            
            prev_cost = total_cost
        
        return {
            "success": True,
            "iterations": iteration + 1,
            "final_cost": total_cost,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges)
        }
    
    def get_optimized_poses(self) -> Dict[int, Pose3D]:
        """获取优化后的位姿"""
        return {node_id: node.pose for node_id, node in self.nodes.items()}


# =============================================================================
# 增量式SLAM系统
# =============================================================================
class IncrementalSLAM:
    """
    增量式SLAM系统
    集成回环检测和位姿图优化
    """
    
    def __init__(self,
                 loop_detection_method: str = "scan_context",
                 loop_threshold: float = 0.7,
                 min_loop_distance: int = 50,
                 optimize_frequency: int = 10):
        self.loop_detection_method = loop_detection_method
        self.loop_threshold = loop_threshold
        self.min_loop_distance = min_loop_distance
        self.optimize_frequency = optimize_frequency
        
        # 位姿图
        self.pose_graph = PoseGraphOptimizer()
        
        # 回环检测器
        if loop_detection_method == "scan_context":
            self.loop_detector = ScanContext()
        else:
            self.loop_detector = ScanContext()  # 默认
        
        # 描述子数据库
        self.descriptor_database: List[np.ndarray] = []
        
        # 当前帧ID
        self.current_frame_id = 0
        
        # 检测到的回环
        self.detected_loops: List[Tuple[int, int, float]] = []
    
    def add_keyframe(self,
                     pose: Pose3D,
                     point_cloud: np.ndarray,
                     relative_pose: Optional[Pose3D] = None) -> int:
        """
        添加关键帧
        
        Args:
            pose: 当前位姿估计
            point_cloud: 点云数据
            relative_pose: 相对于上一帧的位姿 (里程计约束)
        
        Returns:
            帧ID
        """
        frame_id = self.current_frame_id
        
        # 计算描述子
        descriptor = self.loop_detector.compute_descriptor(point_cloud)
        
        # 创建节点
        node = PoseNode(
            id=frame_id,
            pose=pose,
            descriptor=descriptor,
            fixed=(frame_id == 0)  # 第一帧固定
        )
        self.pose_graph.add_node(node)
        
        # 添加里程计约束
        if frame_id > 0 and relative_pose is not None:
            edge = PoseEdge(
                id_from=frame_id - 1,
                id_to=frame_id,
                measurement=relative_pose,
                information=np.diag([100, 100, 100, 200, 200, 200]),  # 里程计较准确
                edge_type="odometry"
            )
            self.pose_graph.add_edge(edge)
        
        # 回环检测
        if frame_id >= self.min_loop_distance:
            candidates = self.loop_detector.find_loop_candidates(
                descriptor,
                self.descriptor_database,
                threshold=self.loop_threshold,
                exclude_recent=self.min_loop_distance
            )
            
            for loop_idx, similarity in candidates[:3]:  # 最多3个回环
                # 添加回环约束
                # 这里简化处理,实际应该用ICP等方法精确估计相对位姿
                loop_pose = self._estimate_loop_pose(frame_id, loop_idx)
                
                edge = PoseEdge(
                    id_from=loop_idx,
                    id_to=frame_id,
                    measurement=loop_pose,
                    information=np.diag([50, 50, 50, 100, 100, 100]) * similarity,
                    edge_type="loop_closure"
                )
                self.pose_graph.add_edge(edge)
                
                self.detected_loops.append((loop_idx, frame_id, similarity))
                logger.info(f"检测到回环: {loop_idx} -> {frame_id}, 相似度: {similarity:.3f}")
        
        # 更新数据库
        self.descriptor_database.append(descriptor)
        
        # 定期优化
        if frame_id > 0 and frame_id % self.optimize_frequency == 0:
            self.optimize()
        
        self.current_frame_id += 1
        return frame_id
    
    def _estimate_loop_pose(self, current_id: int, loop_id: int) -> Pose3D:
        """估计回环相对位姿"""
        # 简化实现: 使用当前位姿估计的差异
        # 实际应该使用ICP或其他配准方法
        pose_current = self.pose_graph.nodes[current_id].pose
        pose_loop = self.pose_graph.nodes[loop_id].pose
        
        return pose_loop.inverse().compose(pose_current)
    
    def optimize(self) -> Dict[str, Any]:
        """执行全局优化"""
        logger.info("执行位姿图优化...")
        result = self.pose_graph.optimize()
        logger.info(f"优化完成: {result}")
        return result
    
    def get_trajectory(self) -> List[Pose3D]:
        """获取优化后的轨迹"""
        poses = []
        for i in range(self.current_frame_id):
            if i in self.pose_graph.nodes:
                poses.append(self.pose_graph.nodes[i].pose)
        return poses
    
    def get_loop_closures(self) -> List[Tuple[int, int, float]]:
        """获取检测到的回环"""
        return self.detected_loops.copy()
    
    def save_state(self, filepath: str):
        """保存状态"""
        import pickle
        
        state = {
            "nodes": {
                k: {
                    "id": v.id,
                    "pose_t": v.pose.translation.tolist(),
                    "pose_r": v.pose.rotation.tolist(),
                    "fixed": v.fixed
                }
                for k, v in self.pose_graph.nodes.items()
            },
            "edges": [
                {
                    "id_from": e.id_from,
                    "id_to": e.id_to,
                    "measurement_t": e.measurement.translation.tolist(),
                    "measurement_r": e.measurement.rotation.tolist(),
                    "information": e.information.tolist(),
                    "edge_type": e.edge_type
                }
                for e in self.pose_graph.edges
            ],
            "detected_loops": self.detected_loops,
            "current_frame_id": self.current_frame_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"状态保存到: {filepath}")
    
    def load_state(self, filepath: str):
        """加载状态"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # 恢复节点
        for node_data in state["nodes"].values():
            pose = Pose3D(
                np.array(node_data["pose_t"]),
                np.array(node_data["pose_r"])
            )
            node = PoseNode(
                id=node_data["id"],
                pose=pose,
                fixed=node_data["fixed"]
            )
            self.pose_graph.add_node(node)
        
        # 恢复边
        for edge_data in state["edges"]:
            measurement = Pose3D(
                np.array(edge_data["measurement_t"]),
                np.array(edge_data["measurement_r"])
            )
            edge = PoseEdge(
                id_from=edge_data["id_from"],
                id_to=edge_data["id_to"],
                measurement=measurement,
                information=np.array(edge_data["information"]),
                edge_type=edge_data["edge_type"]
            )
            self.pose_graph.add_edge(edge)
        
        self.detected_loops = state["detected_loops"]
        self.current_frame_id = state["current_frame_id"]
        
        logger.info(f"状态加载完成: {len(self.pose_graph.nodes)}个节点, {len(self.pose_graph.edges)}条边")


# =============================================================================
# 兼容封装 (GraphSLAM接口)
# =============================================================================
@dataclass
class GraphSLAMConfig:
    """GraphSLAM配置"""
    loop_detection_method: str = "scan_context"
    loop_closure_threshold: float = 0.7
    min_loop_distance: int = 50
    optimize_frequency: int = 10


class GraphSLAM:
    """GraphSLAM兼容封装，提供integration脚本所需接口"""

    def __init__(self, config: Optional[GraphSLAMConfig] = None):
        self.config = config or GraphSLAMConfig()
        self._slam = IncrementalSLAM(
            loop_detection_method=self.config.loop_detection_method,
            loop_threshold=self.config.loop_closure_threshold,
            min_loop_distance=self.config.min_loop_distance,
            optimize_frequency=self.config.optimize_frequency,
        )
        self._last_pose: Optional[Pose3D] = None
        self._point_clouds: Dict[int, np.ndarray] = {}

    def process_frame(self,
                      pose: "Union[Pose3D, np.ndarray]",
                      point_cloud: np.ndarray,
                      timestamp: float = 0.0) -> Dict[str, Any]:
        """处理单帧数据"""
        pose_obj = self._normalize_pose(pose)

        relative_pose = None
        if self._last_pose is not None:
            relative_pose = self._last_pose.inverse().compose(pose_obj)

        frame_id = self._slam.add_keyframe(pose_obj, point_cloud, relative_pose)
        self._last_pose = pose_obj
        self._point_clouds[frame_id] = point_cloud

        loop_info = None
        if self._slam.detected_loops:
            loop_from, loop_to, score = self._slam.detected_loops[-1]
            if loop_to == frame_id:
                loop_info = {
                    "from_id": loop_from,
                    "to_id": loop_to,
                    "score": score
                }

        return {
            "frame_id": frame_id,
            "loop_closure": loop_info
        }

    def get_trajectory(self) -> List[Pose3D]:
        """获取优化后的轨迹"""
        return self._slam.get_trajectory()

    def get_map(self) -> Optional[np.ndarray]:
        """获取简化全局点云地图"""
        if not self._point_clouds:
            return None
        return np.vstack(list(self._point_clouds.values()))

    def _normalize_pose(self, pose: "Union[Pose3D, np.ndarray]") -> Pose3D:
        """兼容Pose3D或6D向量输入"""
        if isinstance(pose, Pose3D):
            return pose

        pose_arr = np.asarray(pose, dtype=np.float64)
        if pose_arr.shape == (3,):
            return Pose3D(pose_arr, np.array([1.0, 0.0, 0.0, 0.0]))
        if pose_arr.shape == (6,):
            translation = pose_arr[:3]
            roll, pitch, yaw = pose_arr[3:]
            rotation = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
            return Pose3D(translation, rotation[[3, 0, 1, 2]])

        raise ValueError(f"Unsupported pose format: {pose_arr.shape}")


# =============================================================================
# 测试函数
# =============================================================================
def test_graph_slam():
    """测试图优化SLAM"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建SLAM系统
    slam = IncrementalSLAM(
        loop_detection_method="scan_context",
        loop_threshold=0.6,
        optimize_frequency=5
    )
    
    # 模拟一个圆形轨迹
    num_frames = 50
    radius = 10.0
    
    prev_pose = None
    
    for i in range(num_frames):
        # 圆形轨迹
        angle = 2 * np.pi * i / num_frames
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        yaw = angle + np.pi / 2
        
        # 当前位姿
        r = Rotation.from_euler('z', yaw)
        pose = Pose3D(
            np.array([x, y, 0.0]),
            r.as_quat()[[3, 0, 1, 2]]
        )
        
        # 生成模拟点云
        point_cloud = np.random.randn(1000, 3) * 5
        point_cloud[:, 0] += x
        point_cloud[:, 1] += y
        
        # 计算相对位姿
        if prev_pose is not None:
            relative_pose = prev_pose.inverse().compose(pose)
        else:
            relative_pose = None
        
        # 添加关键帧
        slam.add_keyframe(pose, point_cloud, relative_pose)
        
        prev_pose = pose
    
    # 最终优化
    result = slam.optimize()
    
    # 获取轨迹
    trajectory = slam.get_trajectory()
    loops = slam.get_loop_closures()
    
    logger.info(f"轨迹长度: {len(trajectory)}")
    logger.info(f"检测到的回环: {len(loops)}")
    
    return slam


if __name__ == "__main__":
    test_graph_slam()
