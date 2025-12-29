"""
高光谱/短波红外缺陷检测插件
Hyperspectral/SWIR Defect Detection Plugin

功能:
- 多光谱图像预处理与特征提取
- 光谱特征分析（反射率、吸收特征）
- 缺陷检测（腐蚀、涂层脱落、瓷瓶污染）
- 缺陷定位与严重程度评估
"""

from .plugin import (
    # 枚举类型
    DefectType,
    SeverityLevel,

    # 数据类
    SpectralBand,
    DefectResult,
    HyperspectralConfig,

    # 核心类
    SpectralPreprocessor,
    SpectralFeatureExtractor,
    RuleBasedDefectDetector,
    DeepLearningDefectDetector,

    # 插件接口
    HyperspectralDetectionPlugin,
    HyperspectralDetectorEnhanced,
)

__all__ = [
    'DefectType',
    'SeverityLevel',
    'SpectralBand',
    'DefectResult',
    'HyperspectralConfig',
    'SpectralPreprocessor',
    'SpectralFeatureExtractor',
    'RuleBasedDefectDetector',
    'DeepLearningDefectDetector',
    'HyperspectralDetectionPlugin',
    'HyperspectralDetectorEnhanced',
]

__version__ = '1.0.0'
