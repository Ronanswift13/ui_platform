"""
声学异常监测插件
Acoustic Anomaly Monitoring Plugin

功能:
- 局部放电声学检测
- 电晕放电识别
- 机械故障声音分析
- 变压器异常嗡鸣检测
- 冷却风扇故障监测
"""

from .plugin import (
    # 特征提取
    AudioFeatureExtractor,
    
    # 异常检测器
    RuleBasedAnomalyDetector,
    DeepLearningAnomalyDetector,
    
    # 插件接口
    AcousticMonitoringPlugin,
    AcousticDetectorEnhanced,
)

__all__ = [
    'AudioFeatureExtractor',
    'RuleBasedAnomalyDetector',
    'DeepLearningAnomalyDetector',
    'AcousticMonitoringPlugin',
    'AcousticDetectorEnhanced',
]

__version__ = '1.0.0'
