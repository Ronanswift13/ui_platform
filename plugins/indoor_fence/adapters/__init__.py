#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内电子围栏 - 传感器适配器
==========================================

封装相机检测+跟踪、LiDAR/ToF测距、灯光控制等硬件
每个适配器输出统一格式数据

基于《变电站多目标操作安全监测系统》方案
阶段2-3：设计适配器并接入真实传感器

作者: G组
版本: 2.0.0
"""

from .base_adapter import BaseAdapter, AdapterConfig, AdapterStatus, AdapterHealth
from .camera_adapter import CameraAdapter, CameraConfig, PersonDetection
from .lidar_adapter import LidarAdapter, LidarConfig, LidarScan, LidarCluster
from .light_adapter import (
    LightAdapter, LightColor, LightConfig, LightMode, LightState,
)

__version__ = "2.0.0"

__all__ = [
    # 基类
    'BaseAdapter',
    'AdapterConfig',
    'AdapterStatus',
    'AdapterHealth',
    # 相机
    'CameraAdapter',
    'CameraConfig',
    'PersonDetection',
    # 雷达
    'LidarAdapter',
    'LidarConfig',
    'LidarScan',
    'LidarCluster',
    # 灯光
    'LightAdapter',
    'LightColor',
    'LightConfig',
    'LightMode',
    'LightState',
]
