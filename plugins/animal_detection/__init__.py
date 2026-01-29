#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动物入侵检测插件 V1.0
=======================

提供室内小动物入侵检测功能:
- YOLOv8小目标检测
- 热成像二次确认
- 动物跟踪
- 自动驱离

作者: G组 | 版本: 1.0.0
"""

from .plugin import (
    # 类型
    AnimalType,
    ThreatLevel,
    DeterrentType,
    # 数据类
    AnimalDetection,
    DeterrentAction,
    AnimalEvent,
    # 组件
    AnimalDetector,
    ThermalConfirmer,
    AnimalTracker,
    DeterrentController,
    # 插件
    AnimalDetectionPlugin,
)

__version__ = "1.0.0"
__author__ = "G组"

__all__ = [
    # 类型
    'AnimalType',
    'ThreatLevel',
    'DeterrentType',
    # 数据类
    'AnimalDetection',
    'DeterrentAction',
    'AnimalEvent',
    # 组件
    'AnimalDetector',
    'ThermalConfirmer',
    'AnimalTracker',
    'DeterrentController',
    # 插件
    'AnimalDetectionPlugin',
]
