"""
母线自主巡视插件 (C组)

功能范围:
- 远距小目标检测: 4K大视场精准识别微小销钉缺失或裂纹
- 多目标并发处理: 绝缘子串、金具、导线连接
- 环境干扰过滤: 鸟类、飞虫等干扰过滤

版本: 3.6.0

注意: 延迟导入以避免模块级触发 training → torch 初始化链。
"""

import importlib as _importlib


def __getattr__(name: str):
    """PEP 562 延迟属性加载，仅在实际访问时触发 import。"""
    _plugin_names = {"BusbarInspectionPlugin"}
    _detector_names = {
        "BusbarDetectorEnhanced",
        "BusbarDefectType",
        "BusbarDetection",
        "QualityGateStatus",
        "QualityGateResult",
        "ZoomSuggestion",
        "BusbarInspectionResult",
        "DetectROIResult",
        "create_detector",
    }

    if name in _plugin_names:
        mod = _importlib.import_module("plugins.busbar_inspection.plugin")
        return getattr(mod, name)
    if name in _detector_names:
        mod = _importlib.import_module("plugins.busbar_inspection.detector_enhanced")
        return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BusbarInspectionPlugin",
    "BusbarDetectorEnhanced",
    "BusbarDefectType",
    "BusbarDetection",
    "QualityGateStatus",
    "QualityGateResult",
    "ZoomSuggestion",
    "BusbarInspectionResult",
    "DetectROIResult",
    "create_detector",
]
