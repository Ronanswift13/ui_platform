#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义SLAM建图插件 V2.0
======================

提供语义SLAM建图功能:
- 点云语义分割
- 环境变化检测
- 语义地图维护
- 与电子围栏联动

作者: G组 | 版本: 2.0.0
"""

from .semantic_slam_plugin import (
    # 类型
    SemanticLabel,
    ChangeType,
    # 数据类
    SemanticPoint,
    PointCloud,
    DetectedChange,
    SemanticMapStats,
    # 组件
    SemanticSegmenter,
    ChangeDetector,
    SemanticMapManager,
    # 插件
    SemanticSLAMPlugin,
)

# 保持向后兼容
try:
    from .plugin import SLAMMappingPlugin
except ImportError:
    SLAMMappingPlugin = SemanticSLAMPlugin

__version__ = "2.0.0"
__author__ = "G组"

__all__ = [
    # 类型
    'SemanticLabel',
    'ChangeType',
    # 数据类
    'SemanticPoint',
    'PointCloud',
    'DetectedChange',
    'SemanticMapStats',
    # 组件
    'SemanticSegmenter',
    'ChangeDetector',
    'SemanticMapManager',
    # 插件
    'SemanticSLAMPlugin',
    'SLAMMappingPlugin',  # 兼容
]
