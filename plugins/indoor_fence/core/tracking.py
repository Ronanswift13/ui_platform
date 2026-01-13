#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标跟踪模块
==========================================

实现跨帧多目标跟踪:
- 卡尔曼滤波平滑坐标
- 移动平均降噪
- 轨迹ID管理
- 离群值检测与恢复

作者: G组 | 版本: 2.0.0
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
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
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
    
    # 元数据
    metadata: Dict = field(default_factory=dict)


class KalmanFilter2D:
    """
    2D卡尔曼滤波器
    
    状态向量: [x, y, vx, vy]
    观测向量: [x, y]
    """
    
    def __init__(self, dt: float = 0.1, process_noise: float = 1.0, measurement_noise: float = 1.0):
        """
        Args:
            dt: 时间步长 (秒)
            process_noise: 过程噪声
            measurement_noise: 测量噪声
        """
        self.dt = dt
        
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
        
        # 测量噪声协方差 R
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        
        # 初始协方差
        self.P0 = np.eye(4, dtype=np.float64) * 10
    
    def init_state(self, x: float, y: float, vx: float = 0, vy: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """初始化状态"""
        state = np.array([x, y, vx, vy], dtype=np.float64)
        covariance = self.P0.copy()
        return state, covariance
    
    def predict(self, state: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测步骤"""
        state_pred = self.F @ state
        cov_pred = self.F @ covariance @ self.F.T + self.Q
        return state_pred, cov_pred
    
    def update(self, state: np.ndarray, covariance: np.ndarray, 
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """更新步骤"""
        # 创新
        y = measurement - self.H @ state
        
        # 创新协方差
        S = self.H @ covariance @ self.H.T + self.R
        
        # 卡尔曼增益
        K = covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        state_new = state + K @ y
        
        # 更新协方差
        I = np.eye(4)
        cov_new = (I - K @ self.H) @ covariance
        
        return state_new, cov_new


class MultiTargetTracker:
    """
    多目标跟踪器
    
    功能:
    - 维护多个目标的轨迹
    - 使用匈牙利算法进行检测-轨迹关联
    - 卡尔曼滤波平滑轨迹
    - 轨迹创建、确认和删除
    """
    
    def __init__(
        self,
        max_age: int = 30,          # 最大未匹配帧数
        min_hits: int = 3,          # 确认所需最小匹配数
        iou_threshold: float = 0.3,  # IoU匹配阈值
        distance_threshold: float = 1.0,  # 距离匹配阈值 (米)
        use_kalman: bool = True,    # 是否使用卡尔曼滤波
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.use_kalman = use_kalman
        
        # 跟踪列表
        self.tracks: Dict[str, Track] = {}
        
        # 卡尔曼滤波器
        self.kf = KalmanFilter2D(dt=0.1) if use_kalman else None
        
        # ID计数器
        self._next_id = 1
        
        # 移动平均窗口
        self._smooth_window = 5
        
        logger.info(f"多目标跟踪器初始化: max_age={max_age}, min_hits={min_hits}")
    
    def _generate_track_id(self) -> str:
        """生成新的轨迹ID"""
        track_id = f"T{self._next_id:04d}"
        self._next_id += 1
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
        
        # 1. 预测所有轨迹
        for track in self.tracks.values():
            self._predict_track(track)
        
        # 2. 关联检测与轨迹
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)
        
        # 3. 更新已匹配的轨迹
        for track_id, det_idx in matched:
            track = self.tracks[track_id]
            detection = detections[det_idx]
            self._update_track(track, detection)
        
        # 4. 处理未匹配的轨迹
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.misses += 1
            track.age += 1
        
        # 5. 创建新轨迹
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            self._create_track(detection)
        
        # 6. 删除过期轨迹
        self._remove_stale_tracks()
        
        # 7. 返回已确认的轨迹
        confirmed_tracks = [t for t in self.tracks.values() if t.is_confirmed]
        
        return confirmed_tracks
    
    def _predict_track(self, track: Track):
        """预测轨迹位置"""
        if self.use_kalman and track.state is not None:
            track.state, track.covariance = self.kf.predict(track.state, track.covariance)
            track.position = Point2D(track.state[0], track.state[1])
            track.velocity = Point2D(track.state[2], track.state[3])
    
    def _associate(self, detections: List[Detection]) -> Tuple[List, List, List]:
        """
        关联检测与轨迹
        
        使用距离矩阵和贪心匹配 (简化版)
        """
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        if not detections or not self.tracks:
            return matched, unmatched_dets, unmatched_tracks
        
        # 构建距离矩阵
        track_ids = list(self.tracks.keys())
        dist_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            for j, det in enumerate(detections):
                dist_matrix[i, j] = track.position.distance_to(det.position)
        
        # 贪心匹配
        while True:
            if dist_matrix.size == 0:
                break
            
            # 找最小距离
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            min_dist = dist_matrix[min_idx]
            
            if min_dist > self.distance_threshold:
                break
            
            track_idx, det_idx = min_idx
            track_id = track_ids[track_idx]
            
            matched.append((track_id, det_idx))
            
            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)
            if track_id in unmatched_tracks:
                unmatched_tracks.remove(track_id)
            
            # 标记已匹配的行列
            dist_matrix[track_idx, :] = float('inf')
            dist_matrix[:, det_idx] = float('inf')
        
        return matched, unmatched_dets, unmatched_tracks
    
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
        
        self.tracks[track_id] = track
        logger.debug(f"创建新轨迹: {track_id}")
    
    def _remove_stale_tracks(self):
        """删除过期轨迹"""
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if track.misses > self.max_age
        ]
        
        for tid in stale_ids:
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
    
    def reset(self):
        """重置跟踪器"""
        self.tracks.clear()
        self._next_id = 1
        logger.info("跟踪器已重置")


__all__ = ['Detection', 'Track', 'KalmanFilter2D', 'MultiTargetTracker']
