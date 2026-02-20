"""
母线自主巡视插件 (C组)

功能范围:
- 远距小目标检测: 4K大视场精准识别微小销钉缺失或裂纹
- 多目标并发处理: 绝缘子串、金具、导线连接
- 环境干扰过滤: 鸟类、飞虫等干扰过滤

版本: 3.6.0
"""

from plugins.busbar_inspection.plugin import BusbarInspectionPlugin
from plugins.busbar_inspection.detector_enhanced import (
    BusbarEnhancedDetector,
    BusbarDefectType,
    BusbarDetection,
    QualityAssessment,
    QualityAssessor,
    TileProcessor,
    NMSProcessor,
    DEFECT_LABELS,
    DEFECT_ALARM_LEVELS,
)

__all__ = [
    "BusbarInspectionPlugin",
    "BusbarEnhancedDetector",
    "BusbarDefectType",
    "BusbarDetection",
    "QualityAssessment",
    "QualityAssessor",
    "TileProcessor",
    "NMSProcessor",
    "DEFECT_LABELS",
    "DEFECT_ALARM_LEVELS",
]
