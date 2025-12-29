"""
气体泄漏检测与预测插件
Gas Leak Detection and Forecasting Plugin

功能:
- SF6/H2/CO/C2H2等气体浓度监测
- 时序预测与趋势分析
- 泄漏检测与定位
- 多级阈值告警
"""

from .plugin import (
    # 特征提取
    TimeSeriesFeatureExtractor,
    
    # 预测器
    TimeSeriesPredictor,
    
    # 异常检测
    GasAnomalyDetector,
    LeakageAnalyzer,
    
    # 插件接口
    GasDetectionPlugin,
    GasDetectorEnhanced,
)

__all__ = [
    'TimeSeriesFeatureExtractor',
    'TimeSeriesPredictor',
    'GasAnomalyDetector',
    'LeakageAnalyzer',
    'GasDetectionPlugin',
    'GasDetectorEnhanced',
]

__version__ = '1.0.0'
