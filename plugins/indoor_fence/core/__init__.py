#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内电子围栏核心模块 V3.0
==========================================

基于《变电站多目标操作安全监测系统》方案实现
提供几何计算、区域配置、状态机、跟踪和融合功能

V3.0升级:
- DeepSORT深度特征跟踪
- 姿态估计和行为分析
- 风险意图判定

作者: G组 | 版本: 3.0.0
"""

from .geometry import (
    Point2D,
    LineSegment,
    Polygon,
    point_in_polygon,
    point_to_segment_distance,
    point_to_polygon_distance,
    lidar_to_ground,
    pixel_to_ground,
    fuse_coordinates,
)

from .zone_config import (
    ZoneType,
    AlertLevel,
    ZoneDefinition,
    CabinetDefinition,
    YellowLineDefinition,
    ZoneConfiguration,
    ZoneConfigLoader,
)

from .state_machine import (
    PersonState,
    GlobalAlarmLevel,
    PersonStateResult,
    GlobalStateResult,
    StateMachine,
)

from .tracking import (
    Detection,
    Track,
    KalmanFilter2D,
    MultiTargetTracker,
)

from .fusion import (
    VisualDetection,
    LidarDetection,
    FusedDetection,
    SensorFusion,
    FusedTracker,
)

# V3.0 增强模块
from .tracker_v3 import (
    # 类型
    BehaviorType,
    RiskIntent,
    # 数据类
    Keypoint,
    PoseEstimation,
    EnhancedTrackResult,
    # 组件
    ReIDFeatureExtractor,
    PoseEstimator,
    BehaviorAnalyzer,
    RiskIntentAnalyzer,
    # 跟踪器
    FusedTrackerV3,
)


__version__ = "3.0.0"
__author__ = "G组"

__all__ = [
    # Geometry
    'Point2D', 'LineSegment', 'Polygon',
    'point_in_polygon', 'point_to_segment_distance', 'point_to_polygon_distance',
    'lidar_to_ground', 'pixel_to_ground', 'fuse_coordinates',

    # Zone Config
    'ZoneType', 'AlertLevel',
    'ZoneDefinition', 'CabinetDefinition', 'YellowLineDefinition',
    'ZoneConfiguration', 'ZoneConfigLoader',

    # State Machine
    'PersonState', 'GlobalAlarmLevel',
    'PersonStateResult', 'GlobalStateResult',
    'StateMachine',

    # Tracking
    'Detection', 'Track', 'KalmanFilter2D', 'MultiTargetTracker',

    # Fusion
    'VisualDetection', 'LidarDetection', 'FusedDetection',
    'SensorFusion', 'FusedTracker',

    # V3.0 类型
    'BehaviorType',
    'RiskIntent',
    # V3.0 数据类
    'Keypoint',
    'PoseEstimation',
    'EnhancedTrackResult',
    # V3.0 组件
    'ReIDFeatureExtractor',
    'PoseEstimator',
    'BehaviorAnalyzer',
    'RiskIntentAnalyzer',
    # V3.0 跟踪器
    'FusedTrackerV3',
]
