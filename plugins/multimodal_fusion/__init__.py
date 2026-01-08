# -*- coding: utf-8 -*-
"""
多模态融合插件
整合声学、视觉、气体、高光谱等多种传感器数据进行综合诊断
"""

from .plugin import MultimodalFusionPlugin
from .fusion_engine import FusionEngine, AttentionFusion, DecisionFusion

__all__ = [
    'MultimodalFusionPlugin',
    'FusionEngine',
    'AttentionFusion',
    'DecisionFusion'
]

__version__ = '1.0.0'
