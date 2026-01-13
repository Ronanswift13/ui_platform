#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉-雷达融合模块
==========================================

封装视觉与测距融合逻辑:
- 时间戳同步
- 角度匹配
- 冲突解决: 横向信任视觉,纵向信任测距

作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from .geometry import Point2D, fuse_coordinates, lidar_to_ground, pixel_to_ground
from .tracking import Detection, Track, MultiTargetTracker

logger = logging.getLogger(__name__)


@dataclass
class VisualDetection:
    """视觉检测结果"""
    detection_id: str
    bbox: Dict[str, float]          # {x, y, width, height} 归一化坐标
    foot_pixel: Tuple[float, float] # 脚部像素坐标
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    class_name: str = "person"
    track_id: Optional[str] = None  # 视觉跟踪ID


@dataclass
class LidarDetection:
    """雷达检测结果"""
    cluster_id: str
    center_distance: float          # 距离 (米)
    center_angle: float             # 角度 (度)
    point_count: int = 1
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusedDetection:
    """融合检测结果"""
    fused_id: str
    ground_position: Point2D        # 地面坐标
    
    # 来源数据
    visual: Optional[VisualDetection] = None
    lidar: Optional[LidarDetection] = None
    
    # 融合状态
    fusion_type: str = "fused"      # fused, visual_only, lidar_only
    confidence: float = 1.0
    
    # 位置分解
    visual_position: Optional[Point2D] = None
    lidar_position: Optional[Point2D] = None
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)
    
    # 用于跟踪
    bbox: Optional[Dict] = None
    track_id: Optional[str] = None


class SensorFusion:
    """
    传感器融合器
    
    融合视觉和雷达数据，生成统一的人员位置
    """
    
    def __init__(
        self,
        max_time_diff_ms: float = 100,      # 最大时间差 (毫秒)
        angle_match_threshold_deg: float = 10.0,  # 角度匹配阈值 (度)
        distance_match_threshold_m: float = 0.5,  # 距离匹配阈值 (米)
        image_size: Tuple[int, int] = (640, 480),
        camera_height: float = 3.0,
        camera_tilt_deg: float = 40.0,
        lidar_height: float = 2.5,
        lidar_tilt_deg: float = 20.0,
        homography: Optional[np.ndarray] = None,
    ):
        self.max_time_diff_ms = max_time_diff_ms
        self.angle_match_threshold = angle_match_threshold_deg
        self.distance_match_threshold = distance_match_threshold_m
        
        # 相机参数
        self.image_width, self.image_height = image_size
        self.camera_height = camera_height
        self.camera_tilt_deg = camera_tilt_deg
        self.homography = homography
        
        # 雷达参数
        self.lidar_height = lidar_height
        self.lidar_tilt_deg = lidar_tilt_deg
        
        # 融合ID计数器
        self._next_fused_id = 1
        
        logger.info("传感器融合器初始化完成")
    
    def _generate_fused_id(self) -> str:
        """生成融合ID"""
        fused_id = f"F{self._next_fused_id:04d}"
        self._next_fused_id += 1
        return fused_id
    
    def fuse(
        self,
        visual_detections: List[VisualDetection],
        lidar_detections: List[LidarDetection]
    ) -> List[FusedDetection]:
        """
        融合视觉和雷达检测
        
        Args:
            visual_detections: 视觉检测列表
            lidar_detections: 雷达检测列表
            
        Returns:
            List[FusedDetection]: 融合结果列表
        """
        fused_results = []
        used_lidar = set()
        
        # 1. 为每个视觉检测计算地面坐标
        for vis_det in visual_detections:
            # 计算视觉位置
            visual_pos = self._visual_to_ground(vis_det)
            
            # 2. 尝试匹配雷达检测
            best_lidar = None
            best_score = float('inf')
            
            for i, lid_det in enumerate(lidar_detections):
                if i in used_lidar:
                    continue
                
                # 检查时间同步
                time_diff = abs(vis_det.timestamp - lid_det.timestamp) * 1000
                if time_diff > self.max_time_diff_ms:
                    continue
                
                # 计算雷达位置
                lidar_pos = self._lidar_to_ground(lid_det)
                
                # 计算匹配分数 (距离)
                distance = visual_pos.distance_to(lidar_pos)
                
                if distance < self.distance_match_threshold and distance < best_score:
                    best_lidar = (i, lid_det, lidar_pos)
                    best_score = distance
            
            # 3. 创建融合结果
            if best_lidar:
                idx, lid_det, lidar_pos = best_lidar
                used_lidar.add(idx)
                
                # 融合坐标: 横向信任视觉,纵向信任雷达
                fused_pos = fuse_coordinates(visual_pos, lidar_pos)
                
                fused = FusedDetection(
                    fused_id=self._generate_fused_id(),
                    ground_position=fused_pos,
                    visual=vis_det,
                    lidar=lid_det,
                    fusion_type="fused",
                    confidence=(vis_det.confidence + lid_det.confidence) / 2,
                    visual_position=visual_pos,
                    lidar_position=lidar_pos,
                    bbox=vis_det.bbox,
                    track_id=vis_det.track_id,
                )
            else:
                # 仅视觉
                fused = FusedDetection(
                    fused_id=self._generate_fused_id(),
                    ground_position=visual_pos,
                    visual=vis_det,
                    lidar=None,
                    fusion_type="visual_only",
                    confidence=vis_det.confidence * 0.8,  # 降低置信度
                    visual_position=visual_pos,
                    bbox=vis_det.bbox,
                    track_id=vis_det.track_id,
                )
            
            fused_results.append(fused)
        
        # 4. 处理未匹配的雷达检测
        for i, lid_det in enumerate(lidar_detections):
            if i in used_lidar:
                continue
            
            lidar_pos = self._lidar_to_ground(lid_det)
            
            fused = FusedDetection(
                fused_id=self._generate_fused_id(),
                ground_position=lidar_pos,
                visual=None,
                lidar=lid_det,
                fusion_type="lidar_only",
                confidence=lid_det.confidence * 0.7,  # 降低置信度
                lidar_position=lidar_pos,
            )
            fused_results.append(fused)
        
        logger.debug(f"融合结果: {len(fused_results)}个目标 "
                    f"(融合:{sum(1 for f in fused_results if f.fusion_type=='fused')}, "
                    f"仅视觉:{sum(1 for f in fused_results if f.fusion_type=='visual_only')}, "
                    f"仅雷达:{sum(1 for f in fused_results if f.fusion_type=='lidar_only')})")
        
        return fused_results
    
    def _visual_to_ground(self, vis_det: VisualDetection) -> Point2D:
        """将视觉检测转换为地面坐标"""
        foot_x, foot_y = vis_det.foot_pixel
        
        return pixel_to_ground(
            pixel_x=foot_x,
            pixel_y=foot_y,
            image_width=self.image_width,
            image_height=self.image_height,
            homography=self.homography,
            camera_height=self.camera_height,
            camera_tilt_deg=self.camera_tilt_deg,
        )
    
    def _lidar_to_ground(self, lid_det: LidarDetection) -> Point2D:
        """将雷达检测转换为地面坐标"""
        return lidar_to_ground(
            distance=lid_det.center_distance,
            angle_deg=lid_det.center_angle,
            lidar_height=self.lidar_height,
            lidar_tilt_deg=self.lidar_tilt_deg,
        )


class FusedTracker:
    """
    融合跟踪器
    
    整合传感器融合和多目标跟踪:
    1. 调用适配器获取视觉和雷达数据
    2. 融合数据生成统一检测
    3. 多目标跟踪维护轨迹ID
    4. 输出带有稳定ID的Track列表
    """
    
    def __init__(
        self,
        fusion_config: Optional[Dict] = None,
        tracker_config: Optional[Dict] = None,
    ):
        # 初始化融合器
        fusion_config = fusion_config or {}
        self.fusion = SensorFusion(**fusion_config)
        
        # 初始化跟踪器
        tracker_config = tracker_config or {}
        self.tracker = MultiTargetTracker(**tracker_config)
        
        # 统计
        self._frame_count = 0
        self._total_detections = 0
        
        logger.info("融合跟踪器初始化完成")
    
    def update(
        self,
        visual_detections: List[VisualDetection],
        lidar_detections: List[LidarDetection]
    ) -> List[Track]:
        """
        更新融合跟踪
        
        Args:
            visual_detections: 视觉检测列表
            lidar_detections: 雷达检测列表
            
        Returns:
            List[Track]: 已确认的轨迹列表
        """
        self._frame_count += 1
        
        # 1. 融合传感器数据
        fused_detections = self.fusion.fuse(visual_detections, lidar_detections)
        
        # 2. 转换为跟踪器输入格式
        tracker_detections = []
        for fused in fused_detections:
            det = Detection(
                detection_id=fused.fused_id,
                position=fused.ground_position,
                confidence=fused.confidence,
                bbox=fused.bbox,
                source=fused.fusion_type,
                metadata={
                    'visual': fused.visual,
                    'lidar': fused.lidar,
                    'visual_position': fused.visual_position,
                    'lidar_position': fused.lidar_position,
                }
            )
            tracker_detections.append(det)
        
        self._total_detections += len(tracker_detections)
        
        # 3. 更新跟踪器
        tracks = self.tracker.update(tracker_detections)
        
        return tracks
    
    def get_track_positions(self) -> Dict[str, Point2D]:
        """获取所有轨迹的位置字典"""
        positions = {}
        for track in self.tracker.get_confirmed_tracks():
            positions[track.track_id] = track.position
        return positions
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'frame_count': self._frame_count,
            'total_detections': self._total_detections,
            'active_tracks': len(self.tracker.tracks),
            'confirmed_tracks': len(self.tracker.get_confirmed_tracks()),
        }
    
    def reset(self):
        """重置跟踪器"""
        self.tracker.reset()
        self._frame_count = 0
        self._total_detections = 0
        self.fusion._next_fused_id = 1


__all__ = [
    'VisualDetection', 'LidarDetection', 'FusedDetection',
    'SensorFusion', 'FusedTracker'
]
