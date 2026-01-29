#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消防监测插件 V1.0
==================

提供室内消防监测功能:
- 火焰/烟雾深度学习检测
- 多模态数据融合
- 火势跟踪
- 分级报警

作者: G组 | 版本: 1.0.0
"""

from .plugin import (
    # 类型
    FireType,
    AlarmLevel,
    SensorType,
    # 数据类
    FireDetection,
    SensorReading,
    FireZoneStatus,
    FireEvent,
    # 组件
    FireSmokeDetector,
    MultiModalFusion,
    FireTracker,
    AlarmEvaluator,
    # 插件
    FireDetectionPlugin,
)

__version__ = "1.0.0"
__author__ = "G组"

__all__ = [
    # 类型
    'FireType',
    'AlarmLevel',
    'SensorType',
    # 数据类
    'FireDetection',
    'SensorReading',
    'FireZoneStatus',
    'FireEvent',
    # 组件
    'FireSmokeDetector',
    'MultiModalFusion',
    'FireTracker',
    'AlarmEvaluator',
    # 插件
    'FireDetectionPlugin',
]
