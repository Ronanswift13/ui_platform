#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气体泄漏检测插件 (Gas Leakage Detection Plugin)
输变电站全自动AI巡检方案 - 丰富模型与插件开发

功能:
1. SF6气体浓度监测与泄漏检测
2. H2 (氢气) 浓度监测
3. CO (一氧化碳) 浓度监测
4. C2H2 (乙炔) 浓度监测
5. 气体浓度趋势预测
6. 泄漏告警与定位

作者: AI巡检系统
版本: 1.0.0
"""

from .plugin import GasDetectionPlugin
from .predictor import GasConcentrationPredictor
from .analyzer import GasDataAnalyzer

__all__ = [
    'GasDetectionPlugin',
    'GasConcentrationPredictor',
    'GasDataAnalyzer'
]

__version__ = '1.0.0'
__author__ = 'AI巡检系统'
