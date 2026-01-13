#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光雷达适配器
==========================================

封装2D激光雷达测距:
- 扫描数据获取
- 点云聚类
- 距离测量

适配器占位符 - 等待真实设备参数

作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
import logging
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .base_adapter import BaseAdapter, AdapterConfig, AdapterStatus

logger = logging.getLogger(__name__)


@dataclass
class LidarConfig(AdapterConfig):
    """雷达配置"""
    # 连接参数
    device_ip: str = "192.168.1.100"
    device_port: int = 2368
    protocol: str = "udp"           # udp/tcp
    
    # 扫描参数
    scan_rate_hz: float = 10.0      # 扫描频率
    angle_min_deg: float = -45.0    # 最小角度
    angle_max_deg: float = 45.0     # 最大角度
    angle_resolution_deg: float = 0.5  # 角度分辨率
    
    # 测距参数
    range_min_m: float = 0.1        # 最小测量距离
    range_max_m: float = 10.0       # 最大测量距离
    
    # 安装参数
    height_m: float = 2.5           # 安装高度
    tilt_deg: float = 20.0          # 向下倾斜角度
    angle_offset_deg: float = 0.0   # 角度偏移
    
    # 聚类参数
    cluster_distance_threshold: float = 0.3  # 聚类距离阈值 (米)
    min_cluster_points: int = 3     # 最小聚类点数


@dataclass
class LidarScan:
    """雷达扫描数据"""
    timestamp: float
    angles: np.ndarray              # 角度数组 (度)
    distances: np.ndarray           # 距离数组 (米)
    intensities: Optional[np.ndarray] = None  # 强度数组 (可选)
    
    @property
    def num_points(self) -> int:
        return len(self.distances)
    
    def to_cartesian(self) -> Tuple[np.ndarray, np.ndarray]:
        """转换为笛卡尔坐标"""
        angles_rad = np.radians(self.angles)
        x = self.distances * np.sin(angles_rad)
        y = self.distances * np.cos(angles_rad)
        return x, y


@dataclass
class LidarCluster:
    """雷达聚类结果"""
    cluster_id: str
    center_distance: float          # 聚类中心距离 (米)
    center_angle: float             # 聚类中心角度 (度)
    min_distance: float             # 最小距离
    max_distance: float             # 最大距离
    point_count: int                # 点数
    confidence: float               # 置信度
    timestamp: float
    
    # 原始点
    points_angles: Optional[np.ndarray] = None
    points_distances: Optional[np.ndarray] = None


class LidarAdapter(BaseAdapter):
    """
    激光雷达适配器
    
    提供统一的测距接口:
    - get_scan() -> LidarScan
    - get_clusters() -> List[LidarCluster]
    
    内部封装雷达通信和点云处理
    """
    
    def __init__(self, config: LidarConfig):
        super().__init__(config)
        self.config: LidarConfig = config
        
        # 通信套接字
        self._socket = None
        
        # 扫描缓冲
        self._scan_buffer: List[LidarScan] = []
        self._buffer_size = 10
        
        # 背景模型 (用于前景提取)
        self._background: Optional[np.ndarray] = None
        
        # 聚类ID计数器
        self._cluster_id_counter = 0
        
        logger.info(f"雷达适配器初始化: {config.device_ip}:{config.device_port}")
    
    def connect(self) -> bool:
        """连接雷达"""
        self._stats['connect_attempts'] += 1
        self._status = AdapterStatus.CONNECTING
        
        try:
            # === 尝试连接真实硬件 ===
            # TODO: 替换为实际的雷达连接代码
            # import socket
            # self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # self._socket.settimeout(self.config.connection_timeout)
            # self._socket.bind(('0.0.0.0', self.config.device_port))
            
            # 目前使用模拟模式
            if self.config.simulate_if_unavailable:
                self._set_simulated()
                self._init_background()
                logger.info("雷达适配器: 使用模拟模式 (硬件不可用)")
                return True
            else:
                self._set_error("雷达连接失败,且未启用模拟模式")
                return False
            
        except Exception as e:
            if self.config.simulate_if_unavailable:
                self._set_simulated()
                self._init_background()
                return True
            else:
                self._set_error(f"雷达连接异常: {e}")
                return False
    
    def disconnect(self):
        """断开雷达"""
        if self._socket:
            # self._socket.close()
            self._socket = None
        
        self._connected = False
        self._simulated = False
        self._status = AdapterStatus.DISCONNECTED
        self._scan_buffer.clear()
        logger.info("雷达适配器已断开")
    
    def healthcheck(self) -> bool:
        """健康检查"""
        if self._simulated:
            return True
        
        # TODO: 检查实际雷达状态
        return self._socket is not None
    
    def _init_background(self):
        """初始化背景模型"""
        # 创建默认背景 (远距离)
        num_points = int((self.config.angle_max_deg - self.config.angle_min_deg) 
                        / self.config.angle_resolution_deg)
        self._background = np.full(num_points, self.config.range_max_m - 1.0)
    
    def get_scan(self) -> Optional[LidarScan]:
        """
        获取一帧扫描数据
        
        Returns:
            LidarScan: 扫描数据 (None表示失败)
        """
        if not self.is_connected:
            logger.warning("雷达未连接")
            return None
        
        current_time = time.time()
        
        # 模拟模式
        if self._simulated:
            return self._simulate_scan(current_time)
        
        # === 真实扫描 ===
        try:
            # TODO: 实际雷达数据读取代码
            # data = self._socket.recv(4096)
            # scan = self._parse_scan_data(data)
            
            self._stats['successful_reads'] += 1
            self._stats['last_read_time'] = current_time
            
            return None  # 占位
            
        except Exception as e:
            self._stats['failed_reads'] += 1
            logger.error(f"扫描读取失败: {e}")
            return None
    
    def get_clusters(self, scan: Optional[LidarScan] = None) -> List[LidarCluster]:
        """
        获取聚类结果
        
        Args:
            scan: 扫描数据 (可选,如果为None则获取新扫描)
            
        Returns:
            List[LidarCluster]: 聚类结果列表
        """
        if scan is None:
            scan = self.get_scan()
        
        if scan is None:
            return []
        
        # 前景提取
        fg_mask = self._extract_foreground(scan)
        
        if not np.any(fg_mask):
            return []
        
        # 聚类
        clusters = self._cluster_points(scan, fg_mask)
        
        return clusters
    
    def _simulate_scan(self, timestamp: float) -> LidarScan:
        """模拟扫描数据"""
        # 生成角度数组
        angles = np.arange(
            self.config.angle_min_deg,
            self.config.angle_max_deg,
            self.config.angle_resolution_deg
        )
        
        # 生成距离 (背景 + 随机前景)
        distances = self._background.copy() if self._background is not None else np.full(len(angles), 8.0)
        
        # 添加噪声
        distances += np.random.normal(0, 0.02, len(distances))
        
        # 模拟1-3个人 (前景目标)
        num_targets = random.randint(1, 3)
        for _ in range(num_targets):
            # 随机位置
            center_angle = random.uniform(-30, 30)
            center_distance = random.uniform(1.5, 4.0)
            width_deg = random.uniform(3, 8)  # 角度宽度
            
            # 找到对应角度范围
            for i, angle in enumerate(angles):
                if abs(angle - center_angle) < width_deg / 2:
                    # 添加目标
                    target_dist = center_distance + random.gauss(0, 0.05)
                    distances[i] = min(distances[i], max(0.2, target_dist))
        
        # 裁剪范围
        distances = np.clip(distances, self.config.range_min_m, self.config.range_max_m)
        
        return LidarScan(
            timestamp=timestamp,
            angles=angles,
            distances=distances,
        )
    
    def _extract_foreground(self, scan: LidarScan) -> np.ndarray:
        """提取前景点"""
        if self._background is None:
            # 没有背景模型,使用固定阈值
            threshold = self.config.range_max_m - 2.0
            return scan.distances < threshold
        
        # 与背景比较
        diff = self._background[:len(scan.distances)] - scan.distances
        
        # 前景判定: 距离明显小于背景
        fg_mask = diff > 0.5
        
        return fg_mask
    
    def _cluster_points(self, scan: LidarScan, fg_mask: np.ndarray) -> List[LidarCluster]:
        """对前景点聚类"""
        clusters = []
        
        fg_indices = np.where(fg_mask)[0]
        if len(fg_indices) == 0:
            return clusters
        
        # 简单的连续性聚类
        current_cluster_indices = []
        
        for i, idx in enumerate(fg_indices):
            if not current_cluster_indices:
                current_cluster_indices.append(idx)
            else:
                # 检查连续性 (角度和距离)
                prev_idx = current_cluster_indices[-1]
                angle_diff = abs(scan.angles[idx] - scan.angles[prev_idx])
                dist_diff = abs(scan.distances[idx] - scan.distances[prev_idx])
                
                if angle_diff < 3 * self.config.angle_resolution_deg and dist_diff < self.config.cluster_distance_threshold:
                    current_cluster_indices.append(idx)
                else:
                    # 保存当前聚类,开始新聚类
                    if len(current_cluster_indices) >= self.config.min_cluster_points:
                        cluster = self._create_cluster(scan, current_cluster_indices)
                        clusters.append(cluster)
                    current_cluster_indices = [idx]
        
        # 处理最后一个聚类
        if len(current_cluster_indices) >= self.config.min_cluster_points:
            cluster = self._create_cluster(scan, current_cluster_indices)
            clusters.append(cluster)
        
        return clusters
    
    def _create_cluster(self, scan: LidarScan, indices: List[int]) -> LidarCluster:
        """创建聚类对象"""
        self._cluster_id_counter += 1
        
        cluster_angles = scan.angles[indices]
        cluster_distances = scan.distances[indices]
        
        return LidarCluster(
            cluster_id=f"LC{self._cluster_id_counter:05d}",
            center_distance=float(np.mean(cluster_distances)),
            center_angle=float(np.mean(cluster_angles)),
            min_distance=float(np.min(cluster_distances)),
            max_distance=float(np.max(cluster_distances)),
            point_count=len(indices),
            confidence=min(1.0, len(indices) / 10.0),
            timestamp=scan.timestamp,
            points_angles=cluster_angles,
            points_distances=cluster_distances,
        )
    
    def update_background(self, scan: LidarScan):
        """更新背景模型"""
        if self._background is None:
            self._background = scan.distances.copy()
        else:
            # 指数移动平均
            alpha = 0.1
            min_len = min(len(self._background), len(scan.distances))
            self._background[:min_len] = (
                alpha * scan.distances[:min_len] + 
                (1 - alpha) * self._background[:min_len]
            )


__all__ = ['LidarConfig', 'LidarScan', 'LidarCluster', 'LidarAdapter']
