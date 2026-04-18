"""
主变自主巡视插件 (A组)

功能范围:
- 外观缺陷识别: 破损、锈蚀、渗漏油、异物悬挂
- 状态识别: 呼吸器硅胶变色、阀门开闭状态
- 热成像集成: 红外图像温度提取
"""

import importlib as _importlib

__all__ = ["TransformerInspectionPlugin"]


def __getattr__(name: str):
    """PEP 562 延迟加载，避免包级 import 触发 plugin.py → torch 链。"""
    _plugin_names = {"TransformerInspectionPlugin"}
    if name in _plugin_names:
        mod = _importlib.import_module("plugins.transformer_inspection.plugin")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
