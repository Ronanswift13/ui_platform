#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声学监测插件 (Acoustic Monitoring Plugin)
输变电站全自动AI巡检方案 - 丰富模型与插件开发

功能:
1. 局部放电 (Partial Discharge) 检测
2. 电晕放电 (Corona Discharge) 检测  
3. 接触不良 (Poor Contact) 检测
4. 轴承故障 (Bearing Fault) 检测
5. 变压器异常嗡鸣检测
6. 冷却风扇故障检测

作者: AI巡检系统
版本: 1.0.0
"""

from .plugin import AcousticMonitoringPlugin
from .detector import AcousticDetector, AcousticDetectorEnhanced
from .analyzer import AcousticAnalyzer

__all__ = [
    'AcousticMonitoringPlugin',
    'AcousticDetector',
    'AcousticDetectorEnhanced',
    'AcousticAnalyzer'
]

__version__ = '1.0.0'
__author__ = 'AI巡检系统'
