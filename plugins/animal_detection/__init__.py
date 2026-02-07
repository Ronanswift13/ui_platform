#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测插件 V1.0
===========================

完整闭环: YOLOv8检测 → 热成像校验 → 多目标跟踪 → 声光驱离 → 统计报告

主要模块:
- plugin: 主插件类 AnimalDetectionPlugin
- core: 核心算法 (检测器/热成像/跟踪/驱离/统计/事件契约)
- train: 可复现训练脚本

使用:
    from plugins.animal_detection import AnimalDetectionPlugin

作者: G组 | 版本: 1.0.0
"""

from .plugin import AnimalDetectionPlugin

__version__ = "1.0.0"
__author__ = "G组"

__all__ = ["AnimalDetectionPlugin"]
