"""
电容器自主巡视插件 (D组)

功能范围:
- 结构完整性检测: 倾斜、倒塌、部件缺失
- 区域入侵检测: 人员、车辆、动物入侵告警
- 电容器组状态监控: 三相电容器组排列检测

版本: 3.6.0

注意: 延迟导入以避免模块级触发 training → torch 初始化链。
"""

import importlib as _importlib


def __getattr__(name: str):
    """PEP 562 延迟属性加载，仅在实际访问时触发 import。"""
    _plugin_names = {"CapacitorInspectionPlugin"}
    _detector_names = {
        "CapacitorDetectorEnhanced",
        "CapacitorDefectType",
        "CapacitorDetection",
        "IntrusionType",
        "IntrusionDetection",
        "IntrusionTracker",
        "CapacitorBankStatus",
        "CapacitorInspectionResult",
    }

    if name in _plugin_names:
        mod = _importlib.import_module("plugins.capacitor_inspection.plugin")
        return getattr(mod, name)
    if name in _detector_names:
        mod = _importlib.import_module("plugins.capacitor_inspection.detector_enhanced")
        return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CapacitorInspectionPlugin",
    "CapacitorDetectorEnhanced",
    "CapacitorDefectType",
    "CapacitorDetection",
    "IntrusionType",
    "IntrusionDetection",
    "IntrusionTracker",
    "CapacitorBankStatus",
    "CapacitorInspectionResult",
]
