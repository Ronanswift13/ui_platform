#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 数据加载器模块
Training Data Loaders for Multi-Voltage Level Substation Equipment

支持的插件:
- transformer: 主变压器巡检
- switch: 开关间隔检测
- busbar: 母线巡检
- capacitor: 电容器巡检
- meter: 表计读数
"""

from pathlib import Path

# 基础路径
BASE_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")

# 尝试导入各加载器
try:
    from .transformer_loader import (
        TransformerDataLoader,
        TRANSFORMER_CLASSES,
        prepare_all_transformer_data
    )
except ImportError:
    TransformerDataLoader = None
    TRANSFORMER_CLASSES = {}
    prepare_all_transformer_data = None

try:
    from .switch_loader import (
        SwitchDataLoader,
        SWITCH_CLASSES,
        prepare_all_switch_data
    )
except ImportError:
    SwitchDataLoader = None
    SWITCH_CLASSES = {}
    prepare_all_switch_data = None

try:
    from .busbar_loader import (
        BusbarDataLoader,
        BUSBAR_CLASSES,
        prepare_all_busbar_data
    )
except ImportError:
    BusbarDataLoader = None
    BUSBAR_CLASSES = {}
    prepare_all_busbar_data = None

try:
    from .capacitor_meter_loader import (
        CapacitorDataLoader,
        MeterDataLoader,
        CAPACITOR_CLASSES,
        METER_CLASSES,
        prepare_all_capacitor_data,
        prepare_all_meter_data
    )
except ImportError:
    CapacitorDataLoader = None
    MeterDataLoader = None
    CAPACITOR_CLASSES = {}
    METER_CLASSES = {}
    prepare_all_capacitor_data = None
    prepare_all_meter_data = None


# 导出
__all__ = [
    "BASE_PATH",
    "TransformerDataLoader",
    "SwitchDataLoader",
    "BusbarDataLoader",
    "CapacitorDataLoader",
    "MeterDataLoader",
    "TRANSFORMER_CLASSES",
    "SWITCH_CLASSES",
    "BUSBAR_CLASSES",
    "CAPACITOR_CLASSES",
    "METER_CLASSES",
    "prepare_all_transformer_data",
    "prepare_all_switch_data",
    "prepare_all_busbar_data",
    "prepare_all_capacitor_data",
    "prepare_all_meter_data",
]


def get_loader(plugin: str, voltage_level: str):
    """
    获取指定插件和电压等级的数据加载器
    
    Args:
        plugin: 插件名称 (transformer, switch, busbar, capacitor, meter)
        voltage_level: 电压等级 (如 HV_220kV)
    
    Returns:
        数据加载器实例
    """
    loaders = {
        "transformer": TransformerDataLoader,
        "switch": SwitchDataLoader,
        "busbar": BusbarDataLoader,
        "capacitor": CapacitorDataLoader,
        "meter": MeterDataLoader,
    }
    
    loader_class = loaders.get(plugin)
    if loader_class is None:
        raise ValueError(f"未知插件: {plugin}")
    
    return loader_class(voltage_level)


def prepare_all_data():
    """准备所有插件的所有电压等级数据"""
    results = {}
    
    prepare_funcs = {
        "transformer": prepare_all_transformer_data,
        "switch": prepare_all_switch_data,
        "busbar": prepare_all_busbar_data,
        "capacitor": prepare_all_capacitor_data,
        "meter": prepare_all_meter_data,
    }
    
    for plugin, func in prepare_funcs.items():
        if func is not None:
            try:
                results[plugin] = func()
            except Exception as e:
                results[plugin] = {"error": str(e)}
        else:
            results[plugin] = {"error": "加载器未导入"}
    
    return results
