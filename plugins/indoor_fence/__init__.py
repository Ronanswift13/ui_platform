#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内电子围栏插件 V2.0
===================

多人安全监测系统 - 基于视觉-雷达融合

主要模块:
- plugin: 主插件类 IndoorFencePlugin
- api_integration: API集成层
- core: 核心算法模块 (几何、区域、状态机、跟踪、融合)
- adapters: 硬件适配器 (相机、雷达、灯光)

使用示例:
    from plugins.indoor_fence import IndoorFencePlugin

    # 通过插件管理器加载
    # 或直接使用核心组件

作者: G组 | 版本: 2.0.0
"""

from .plugin import IndoorFencePlugin, AuditLogger

from .core import (
    # 几何
    Point2D, Polygon,
    # 区域配置
    ZoneConfiguration, ZoneConfigLoader, ZoneType, AlertLevel,
    # 状态机
    PersonState, GlobalAlarmLevel, PersonStateResult, GlobalStateResult, StateMachine,
    # 跟踪
    Detection, Track, MultiTargetTracker,
    # 融合
    VisualDetection, LidarDetection, FusedDetection, SensorFusion, FusedTracker,
)

from .adapters import (
    # 基础
    BaseAdapter, AdapterStatus,
    # 相机
    CameraAdapter, CameraConfig, PersonDetection,
    # 雷达
    LidarAdapter, LidarConfig, LidarScan, LidarCluster,
    # 灯光
    LightAdapter, LightColor, LightConfig, LightState,
)

# API 集成 (可选)
try:
    from .api_integration import (
        create_indoor_fence_router,
        integrate_with_platform,
        WebSocketManager,
        RealtimePushService,
    )
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    create_indoor_fence_router = None
    integrate_with_platform = None
    WebSocketManager = None
    RealtimePushService = None

__version__ = "2.0.0"
__author__ = "G组"

__all__ = [
    # 主插件
    'IndoorFencePlugin',
    'AuditLogger',
    # 几何
    'Point2D', 'Polygon',
    # 区域
    'ZoneConfiguration', 'ZoneConfigLoader', 'ZoneType', 'AlertLevel',
    # 状态机
    'PersonState', 'GlobalAlarmLevel', 'PersonStateResult', 'GlobalStateResult', 'StateMachine',
    # 跟踪
    'Detection', 'Track', 'MultiTargetTracker',
    # 融合
    'VisualDetection', 'LidarDetection', 'FusedDetection', 'SensorFusion', 'FusedTracker',
    # 适配器
    'BaseAdapter', 'AdapterStatus',
    'CameraAdapter', 'CameraConfig', 'PersonDetection',
    'LidarAdapter', 'LidarConfig', 'LidarScan', 'LidarCluster',
    'LightAdapter', 'LightColor', 'LightConfig', 'LightState',
    # API 集成
    'create_indoor_fence_router', 'integrate_with_platform',
    'WebSocketManager', 'RealtimePushService',
    'API_AVAILABLE',
]
