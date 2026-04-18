"""
开关间隔自主巡视插件 (B组)

功能范围:
- 分合位状态识别: 断路器、隔离开关、接地开关
- 五防逻辑校验: 防止带负荷拉刀闸、防止带电合接地刀
- 图像清晰度评价: 识别结果可信度评估
"""

import importlib as _importlib

__all__ = ["SwitchInspectionPlugin"]


def __getattr__(name: str):
    """PEP 562 延迟加载，避免包级 import 触发 plugin.py → torch 链。"""
    _plugin_names = {"SwitchInspectionPlugin"}
    if name in _plugin_names:
        mod = _importlib.import_module("plugins.switch_inspection.plugin")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
