#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输变电激光星芒破夜绘明监测平台 - 核心模块
==========================================

Version: 4.0.0

核心组件:
1. spatial_math - 空间数学库 (SE3变换、消失点计算)
2. fusion_engine - 融合引擎 (D-S证据理论、物理逻辑检查)

插件:
- busbar_inspection - 母线巡视 (时序ReID、裂纹检测)
- switch_inspection - 开关巡视 (动作轨迹分析)
- transformer_inspection - 变压器巡视 (声纹、热场)
- capacitor_inspection - 电容器巡视 (倾斜检测)
- meter_reading - 表计读数
- bird_monitoring - 鸟类监控
- acoustic_detection - 声学监测
- gas_detection - 气体检测
- hyperspectral_detection - 高光谱检测
- slam_mapping - SLAM建图
- multimodal_fusion - 多模态融合
"""

__version__ = "4.0.0"
__author__ = "输变电监测平台开发组"

from .spatial_math import (
    SpatialMath,
    Point2D,
    Point3D,
    CameraIntrinsics,
    pixel2world,
    world2pixel,
    compute_vanishing_point,
    compute_tilt_angle,
)

from .fusion_engine import (
    FusionEngine,
    SensorReading,
    SensorType,
    FaultType,
    FusionResult,
    DempsterShafer,
    PhysicalLogicChecker,
    LogicConflictError,
    create_fusion_engine,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Spatial Math
    "SpatialMath",
    "Point2D",
    "Point3D",
    "CameraIntrinsics",
    "pixel2world",
    "world2pixel",
    "compute_vanishing_point",
    "compute_tilt_angle",
    
    # Fusion Engine
    "FusionEngine",
    "SensorReading",
    "SensorType",
    "FaultType",
    "FusionResult",
    "DempsterShafer",
    "PhysicalLogicChecker",
    "LogicConflictError",
    "create_fusion_engine",
]
