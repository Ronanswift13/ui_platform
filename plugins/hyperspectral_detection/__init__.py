"""
高光谱缺陷检测插件
Hyperspectral Defect Detection Plugin

功能：
- 高光谱图像预处理（暗电流校正、平场校正）
- PCA特征提取
- 深度学习缺陷检测（过热、腐蚀、绝缘老化、漏油等）
- 传统方法备选（NIR能量分析、光谱比值）
- 缺陷分割与定位
"""

from .plugin import HyperspectralDetectionPlugin

__all__ = ['HyperspectralDetectionPlugin']
__version__ = '1.0.0'
__author__ = 'Substation AI Team'
