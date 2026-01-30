#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间数学库 - SpatialMath
========================

根据《室外监测平台迭代方案》核心基础设施要求实现

功能:
1. SE(3) 李群/李代数运算
2. 像素坐标到世界坐标转换
3. 消失点计算
4. 深度估计支持

版本: 1.0.0
作者: 输变电监测平台开发组
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Point3D:
    """三维点"""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point3D':
        return cls(x=arr[0], y=arr[1], z=arr[2])


@dataclass
class Point2D:
    """二维点"""
    x: float
    y: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point2D':
        return cls(x=arr[0], y=arr[1])


@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float  # 焦距x
    fy: float  # 焦距y
    cx: float  # 主点x
    cy: float  # 主点y
    
    def to_matrix(self) -> np.ndarray:
        """转换为3x3内参矩阵K"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @classmethod
    def from_matrix(cls, K: np.ndarray) -> 'CameraIntrinsics':
        return cls(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])


class SpatialMath:
    """
    空间数学类
    
    提供SE(3)变换、坐标转换、消失点计算等功能
    """
    
    # ==================== SE(3) 李群运算 ====================
    
    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        """
        向量到反对称矩阵 (skew-symmetric matrix)
        
        Args:
            v: 3维向量 [vx, vy, vz]
            
        Returns:
            3x3 反对称矩阵
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=np.float64)
    
    @staticmethod
    def exp_so3(omega: np.ndarray) -> np.ndarray:
        """
        so(3) -> SO(3) 指数映射 (Rodrigues公式)
        
        Args:
            omega: 3维旋转向量 (轴角表示)
            
        Returns:
            3x3 旋转矩阵
        """
        theta = np.linalg.norm(omega)
        if theta < 1e-10:
            return np.eye(3)
        
        k = omega / theta  # 单位轴
        K = SpatialMath.skew(k)
        
        # Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R
    
    @staticmethod
    def log_so3(R: np.ndarray) -> np.ndarray:
        """
        SO(3) -> so(3) 对数映射
        
        Args:
            R: 3x3 旋转矩阵
            
        Returns:
            3维旋转向量
        """
        trace = np.trace(R)
        theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if theta < 1e-10:
            return np.zeros(3)
        
        # k = (R - R^T) / (2*sin(θ))
        k = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))
        
        return theta * k
    
    @staticmethod
    def exp_se3(twist: np.ndarray) -> np.ndarray:
        """
        se(3) -> SE(3) 指数映射
        
        Args:
            twist: 6维扭量 [v; omega]，前3维为平移，后3维为旋转
            
        Returns:
            4x4 变换矩阵
        """
        v = twist[:3]  # 平移部分
        omega = twist[3:]  # 旋转部分
        
        R = SpatialMath.exp_so3(omega)
        theta = np.linalg.norm(omega)
        
        if theta < 1e-10:
            t = v
        else:
            K = SpatialMath.skew(omega / theta)
            # V = I + (1-cos(θ))/θ * K + (θ-sin(θ))/θ * K²
            V = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * (K @ K) + \
                (theta - np.sin(theta)) / (theta**3) * (K @ K @ K)
            t = V @ v
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    @staticmethod
    def log_se3(T: np.ndarray) -> np.ndarray:
        """
        SE(3) -> se(3) 对数映射
        
        Args:
            T: 4x4 变换矩阵
            
        Returns:
            6维扭量
        """
        R = T[:3, :3]
        t = T[:3, 3]
        
        omega = SpatialMath.log_so3(R)
        theta = np.linalg.norm(omega)
        
        if theta < 1e-10:
            v = t
        else:
            K = SpatialMath.skew(omega / theta)
            V_inv = np.eye(3) - 0.5 * K + \
                    (1 / theta**2) * (1 - theta * np.sin(theta) / (2 * (1 - np.cos(theta)))) * (K @ K)
            v = V_inv @ t
        
        return np.concatenate([v, omega])
    
    # ==================== 坐标转换 ====================
    
    @staticmethod
    def pixel2world(
        u: float, 
        v: float, 
        depth: float, 
        K: np.ndarray, 
        T_cw: np.ndarray
    ) -> np.ndarray:
        """
        像素坐标到世界坐标转换
        
        Args:
            u: 像素x坐标
            v: 像素y坐标
            depth: 深度值 (米)
            K: 3x3 相机内参矩阵
            T_cw: 4x4 相机到世界坐标系变换矩阵
            
        Returns:
            世界坐标 [X, Y, Z]
        """
        # 像素坐标 -> 归一化相机坐标
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x_c = (u - cx) * depth / fx
        y_c = (v - cy) * depth / fy
        z_c = depth
        
        # 相机坐标系下的点
        p_c = np.array([x_c, y_c, z_c, 1.0])
        
        # 转换到世界坐标系
        p_w = T_cw @ p_c
        
        return p_w[:3]
    
    @staticmethod
    def world2pixel(
        X: float, 
        Y: float, 
        Z: float, 
        K: np.ndarray, 
        T_wc: np.ndarray
    ) -> Tuple[float, float]:
        """
        世界坐标到像素坐标转换
        
        Args:
            X, Y, Z: 世界坐标
            K: 3x3 相机内参矩阵
            T_wc: 4x4 世界到相机坐标系变换矩阵
            
        Returns:
            像素坐标 (u, v)
        """
        # 世界坐标 -> 相机坐标
        p_w = np.array([X, Y, Z, 1.0])
        p_c = T_wc @ p_w
        
        # 相机坐标 -> 像素坐标
        x_c, y_c, z_c = p_c[:3]
        
        if z_c < 1e-6:
            return (float('inf'), float('inf'))
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        u = fx * x_c / z_c + cx
        v = fy * y_c / z_c + cy
        
        return (u, v)
    
    @staticmethod
    def depth_to_pointcloud(
        depth_map: np.ndarray,
        K: np.ndarray,
        T_cw: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        step: int = 1
    ) -> np.ndarray:
        """
        深度图转换为点云
        
        Args:
            depth_map: HxW 深度图 (米)
            K: 3x3 相机内参矩阵
            T_cw: 4x4 相机到世界变换 (默认为单位阵)
            mask: HxW 有效像素掩码
            step: 采样步长
            
        Returns:
            Nx3 点云数组
        """
        H, W = depth_map.shape
        
        if T_cw is None:
            T_cw = np.eye(4)
        
        if mask is None:
            mask = np.ones((H, W), dtype=bool)
        
        # 生成像素坐标网格
        v_coords, u_coords = np.meshgrid(
            np.arange(0, H, step),
            np.arange(0, W, step),
            indexing='ij'
        )
        
        # 获取有效点
        valid = mask[::step, ::step] & (depth_map[::step, ::step] > 0)
        
        u = u_coords[valid]
        v = v_coords[valid]
        d = depth_map[::step, ::step][valid]
        
        # 批量转换
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x_c = (u - cx) * d / fx
        y_c = (v - cy) * d / fy
        z_c = d
        
        # 相机坐标
        points_c = np.stack([x_c, y_c, z_c, np.ones_like(x_c)], axis=1)
        
        # 转换到世界坐标
        points_w = (T_cw @ points_c.T).T[:, :3]
        
        return points_w
    
    # ==================== 消失点计算 ====================
    
    @staticmethod
    def compute_vanishing_point(lines: List[Tuple[float, float, float, float]]) -> Tuple[float, float]:
        """
        计算多条平行线的消失点
        
        用于精确倾斜角度计算（电容器插件等）
        
        Args:
            lines: 线段列表，每个元素为 (x1, y1, x2, y2)
            
        Returns:
            消失点坐标 (vp_x, vp_y)
        """
        if len(lines) < 2:
            raise ValueError("至少需要2条线段计算消失点")
        
        # 将线段转换为齐次坐标表示
        line_coeffs = []
        for x1, y1, x2, y2 in lines:
            # 两点确定的直线: l = p1 × p2 (叉积)
            p1 = np.array([x1, y1, 1.0])
            p2 = np.array([x2, y2, 1.0])
            l = np.cross(p1, p2)
            
            # 归一化
            norm = np.sqrt(l[0]**2 + l[1]**2)
            if norm > 1e-10:
                l = l / norm
            line_coeffs.append(l)
        
        # 最小二乘法求消失点
        # 所有线应该通过消失点: l · vp = 0
        # 构造方程组 A * [vp_x, vp_y, 1]^T = 0
        A = np.array(line_coeffs)
        
        # SVD分解，最小特征值对应的特征向量为解
        _, _, Vt = np.linalg.svd(A)
        vp = Vt[-1]  # 最后一行
        
        # 归一化为非齐次坐标
        if abs(vp[2]) > 1e-10:
            vp_x = vp[0] / vp[2]
            vp_y = vp[1] / vp[2]
        else:
            # 消失点在无穷远处
            vp_x = float('inf') * np.sign(vp[0])
            vp_y = float('inf') * np.sign(vp[1])
        
        return (vp_x, vp_y)
    
    @staticmethod
    def compute_tilt_angle(
        vanishing_point: Tuple[float, float],
        K: np.ndarray
    ) -> float:
        """
        根据消失点计算倾斜角度
        
        用于电容器巡视等场景的精确倾斜检测
        
        Args:
            vanishing_point: 消失点坐标 (vp_x, vp_y)
            K: 相机内参矩阵
            
        Returns:
            倾斜角度 (弧度)
        """
        vp_x, vp_y = vanishing_point
        cx = K[0, 2]
        f = K[0, 0]  # 使用fx作为焦距
        
        # θ = arctan((vp_x - cx) / f)
        theta = np.arctan((vp_x - cx) / f)
        
        return theta
    
    # ==================== 线段检测辅助 ====================
    
    @staticmethod
    def filter_vertical_lines(
        lines: List[Tuple[float, float, float, float]],
        angle_threshold: float = 15.0
    ) -> List[Tuple[float, float, float, float]]:
        """
        筛选接近垂直的线段
        
        Args:
            lines: 线段列表
            angle_threshold: 与垂直方向的最大偏差角度 (度)
            
        Returns:
            筛选后的垂直线段列表
        """
        threshold_rad = np.radians(angle_threshold)
        vertical_lines = []
        
        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            
            # 计算与垂直方向的夹角
            angle = np.arctan2(abs(dx), abs(dy))
            
            if angle < threshold_rad:
                vertical_lines.append((x1, y1, x2, y2))
        
        return vertical_lines
    
    # ==================== 深度估计支持 ====================
    
    @staticmethod
    def estimate_scale_from_reference(
        depth_map: np.ndarray,
        reference_points: List[Tuple[int, int, float]],
        K: np.ndarray
    ) -> float:
        """
        根据参考点估计深度图的尺度因子
        
        Args:
            depth_map: 相对深度图
            reference_points: [(u, v, real_depth), ...] 已知真实深度的参考点
            K: 相机内参
            
        Returns:
            尺度因子
        """
        if len(reference_points) == 0:
            return 1.0
        
        scales = []
        for u, v, real_depth in reference_points:
            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                pred_depth = depth_map[v, u]
                if pred_depth > 0:
                    scales.append(real_depth / pred_depth)
        
        if len(scales) == 0:
            return 1.0
        
        # 使用中值作为尺度因子（鲁棒估计）
        return np.median(scales)
    
    @staticmethod
    def compute_physical_size(
        bbox_pixels: Tuple[int, int, int, int],
        depth: float,
        K: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算目标的物理尺寸
        
        Args:
            bbox_pixels: 边界框 (x1, y1, x2, y2) 像素坐标
            depth: 目标深度 (米)
            K: 相机内参
            
        Returns:
            物理尺寸 (width_m, height_m) 单位:米
        """
        x1, y1, x2, y2 = bbox_pixels
        fx, fy = K[0, 0], K[1, 1]
        
        # 像素尺寸
        width_px = abs(x2 - x1)
        height_px = abs(y2 - y1)
        
        # 物理尺寸
        width_m = width_px * depth / fx
        height_m = height_px * depth / fy
        
        return (width_m, height_m)


# ==================== 便捷函数导出 ====================

pixel2world = SpatialMath.pixel2world
world2pixel = SpatialMath.world2pixel
compute_vanishing_point = SpatialMath.compute_vanishing_point
compute_tilt_angle = SpatialMath.compute_tilt_angle


if __name__ == "__main__":
    # 测试代码
    print("=== 空间数学库测试 ===")
    
    # 测试SE(3)运算
    twist = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    T = SpatialMath.exp_se3(twist)
    twist_recovered = SpatialMath.log_se3(T)
    print(f"SE(3) 指数/对数映射误差: {np.linalg.norm(twist - twist_recovered):.6f}")
    
    # 测试坐标转换
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    T_cw = np.eye(4)
    
    p_w = SpatialMath.pixel2world(320, 240, 2.0, K, T_cw)
    print(f"像素(320,240) 深度2m -> 世界坐标: {p_w}")
    
    # 测试消失点计算
    lines = [
        (100, 100, 100, 400),
        (200, 80, 200, 420),
        (300, 60, 300, 440),
    ]
    vp = SpatialMath.compute_vanishing_point(lines)
    print(f"消失点: ({vp[0]:.2f}, {vp[1]:.2f})")
    
    print("测试完成!")
