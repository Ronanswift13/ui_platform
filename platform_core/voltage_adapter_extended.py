#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变电站电压等级适配管理系统 - 扩展版
==========================================

功能:
1. 支持全电压等级分类管理:
   - 特高压变电站：交流1000kV及以上、直流±800kV及以上
   - 超高压变电站：交流330kV、500kV；直流±500kV
   - 高压变电站：110kV、220kV
   - 中压变电站：35kV
   - 低压变电站：10kV及以下
2. 自动匹配对应的预训练模型库
3. 支持不同电压等级的设备配置差异
4. 提供统一的适配层接口

使用方法:
    from platform_core.voltage_adapter_extended import VoltageAdapterManager

    # 初始化管理器
    manager = VoltageAdapterManager()

    # 设置电压等级
    manager.set_voltage_level("1000kV_AC")  # 特高压交流
    manager.set_voltage_level("500kV_AC")   # 超高压交流
    manager.set_voltage_level("220kV")      # 高压
    manager.set_voltage_level("35kV")       # 中压
    manager.set_voltage_level("10kV")       # 低压

    # 获取模型路径
    model_path = manager.get_model_path("switch", "state_detection")

作者: 破夜绘明团队
日期: 2025
"""

from __future__ import annotations
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# 电压等级分类定义
# =============================================================================
class VoltageCategory(Enum):
    """变电站电压等级分类"""
    UHV = "特高压"      # Ultra High Voltage: 交流1000kV及以上、直流±800kV及以上
    EHV = "超高压"      # Extra High Voltage: 交流330kV、500kV；直流±500kV
    HV = "高压"         # High Voltage: 110kV、220kV
    MV = "中压"         # Medium Voltage: 35kV
    LV = "低压"         # Low Voltage: 10kV及以下


class VoltageLevel(Enum):
    """具体电压等级"""
    # 特高压 (UHV)
    KV_1000_AC = "1000kV_AC"      # 特高压交流
    KV_1100_AC = "1100kV_AC"      # 特高压交流 (最高运行电压)
    KV_800_DC = "±800kV_DC"       # 特高压直流
    KV_1100_DC = "±1100kV_DC"     # 特高压直流 (昌吉-古泉线)
    
    # 超高压 (EHV)
    KV_500_AC = "500kV_AC"        # 超高压交流
    KV_330_AC = "330kV_AC"        # 超高压交流
    KV_750_AC = "750kV_AC"        # 超高压交流
    KV_500_DC = "±500kV_DC"       # 超高压直流
    KV_660_DC = "±660kV_DC"       # 超高压直流
    
    # 高压 (HV)
    KV_220 = "220kV"
    KV_110 = "110kV"
    
    # 中压 (MV)
    KV_35 = "35kV"
    KV_66 = "66kV"                # 部分地区使用
    
    # 低压 (LV)
    KV_10 = "10kV"
    KV_6 = "6kV"
    V_380 = "380V"


# 电压等级与分类的映射关系
VOLTAGE_CATEGORY_MAPPING = {
    # 特高压
    VoltageLevel.KV_1000_AC: VoltageCategory.UHV,
    VoltageLevel.KV_1100_AC: VoltageCategory.UHV,
    VoltageLevel.KV_800_DC: VoltageCategory.UHV,
    VoltageLevel.KV_1100_DC: VoltageCategory.UHV,
    
    # 超高压
    VoltageLevel.KV_500_AC: VoltageCategory.EHV,
    VoltageLevel.KV_330_AC: VoltageCategory.EHV,
    VoltageLevel.KV_750_AC: VoltageCategory.EHV,
    VoltageLevel.KV_500_DC: VoltageCategory.EHV,
    VoltageLevel.KV_660_DC: VoltageCategory.EHV,
    
    # 高压
    VoltageLevel.KV_220: VoltageCategory.HV,
    VoltageLevel.KV_110: VoltageCategory.HV,
    
    # 中压
    VoltageLevel.KV_35: VoltageCategory.MV,
    VoltageLevel.KV_66: VoltageCategory.MV,
    
    # 低压
    VoltageLevel.KV_10: VoltageCategory.LV,
    VoltageLevel.KV_6: VoltageCategory.LV,
    VoltageLevel.V_380: VoltageCategory.LV,
}


# =============================================================================
# 设备配置定义
# =============================================================================
@dataclass
class TransformerConfig:
    """变压器配置"""
    typical_models: List[str] = field(default_factory=list)
    capacity_range_mva: Tuple[float, float] = (0, 0)
    oil_tank_diameter_range_m: Tuple[float, float] = (0, 0)
    cooling_types: List[str] = field(default_factory=list)
    detection_classes: List[str] = field(default_factory=list)
    thermal_thresholds: Dict[str, float] = field(default_factory=dict)
    special_features: List[str] = field(default_factory=list)


@dataclass
class SwitchConfig:
    """开关设备配置"""
    breaker_types: List[str] = field(default_factory=list)
    isolator_types: List[str] = field(default_factory=list)
    grounding_switch_types: List[str] = field(default_factory=list)
    detection_classes: List[str] = field(default_factory=list)
    angle_reference: Dict[str, Dict[str, float]] = field(default_factory=dict)
    special_features: List[str] = field(default_factory=list)


@dataclass
class BusbarConfig:
    """母线配置"""
    conductor_types: List[str] = field(default_factory=list)
    busbar_height_m: float = 0
    phase_spacing_m: float = 0
    detection_classes: List[str] = field(default_factory=list)
    small_target_min_px: int = 10
    special_features: List[str] = field(default_factory=list)


@dataclass
class CapacitorConfig:
    """电容器配置"""
    typical_models: List[str] = field(default_factory=list)
    bank_capacity_mvar: List[float] = field(default_factory=list)
    detection_classes: List[str] = field(default_factory=list)
    special_features: List[str] = field(default_factory=list)


@dataclass
class MeterConfig:
    """表计配置"""
    types: List[str] = field(default_factory=list)
    sf6_pressure_range_mpa: Tuple[float, float] = (0, 0)
    oil_temp_range_celsius: Tuple[float, float] = (0, 0)
    detection_classes: List[str] = field(default_factory=list)
    special_features: List[str] = field(default_factory=list)


@dataclass
class DCSystemConfig:
    """直流系统配置 (特高压/超高压直流特有)"""
    converter_transformer_types: List[str] = field(default_factory=list)
    converter_valve_types: List[str] = field(default_factory=list)
    smoothing_reactor_types: List[str] = field(default_factory=list)
    dc_filter_types: List[str] = field(default_factory=list)
    detection_classes: List[str] = field(default_factory=list)
    special_features: List[str] = field(default_factory=list)


@dataclass
class GISConfig:
    """GIS组合电器配置"""
    types: List[str] = field(default_factory=list)
    rated_current_a: List[int] = field(default_factory=list)
    rated_breaking_current_ka: List[int] = field(default_factory=list)
    detection_classes: List[str] = field(default_factory=list)
    sf6_monitoring: bool = False
    partial_discharge_monitoring: bool = False
    special_features: List[str] = field(default_factory=list)


@dataclass
class EquipmentConfig:
    """完整设备配置"""
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    switch: SwitchConfig = field(default_factory=SwitchConfig)
    busbar: BusbarConfig = field(default_factory=BusbarConfig)
    capacitor: CapacitorConfig = field(default_factory=CapacitorConfig)
    meter: MeterConfig = field(default_factory=MeterConfig)
    dc_system: Optional[DCSystemConfig] = None
    gis: Optional[GISConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "transformer": asdict(self.transformer),
            "switch": asdict(self.switch),
            "busbar": asdict(self.busbar),
            "capacitor": asdict(self.capacitor),
            "meter": asdict(self.meter),
        }
        if self.dc_system:
            result["dc_system"] = asdict(self.dc_system)
        if self.gis:
            result["gis"] = asdict(self.gis)
        return result


# =============================================================================
# 各电压等级配置定义
# =============================================================================

# -----------------------------------------------------------------------------
# 特高压交流 1000kV 配置
# -----------------------------------------------------------------------------
CONFIG_1000KV_AC = EquipmentConfig(
    transformer=TransformerConfig(
        typical_models=[
            "ODFPS-1000000/1000",   # 单相1000MVA
            "ODFPS-1200000/1000",   # 单相1200MVA
        ],
        capacity_range_mva=(1000, 3000),
        oil_tank_diameter_range_m=(6.0, 10.0),
        cooling_types=["ODAF", "OFWF"],
        detection_classes=[
            "oil_leak", "rust_corrosion", "surface_damage",
            "foreign_object", "silica_gel_blue", "silica_gel_pink",
            "oil_level_normal", "oil_level_abnormal",
            "bushing_crack", "porcelain_contamination",
            "uhv_bushing_defect", "oil_conservator_defect",
            "cooling_fan_abnormal", "oil_pump_abnormal"
        ],
        thermal_thresholds={
            "normal": 70, "warning": 85, "alarm": 100
        },
        special_features=[
            "特高压套管监测", "油色谱在线监测", "局部放电监测",
            "铁芯接地电流监测", "绕组变形检测"
        ]
    ),
    switch=SwitchConfig(
        breaker_types=["1000kV SF6断路器", "1000kV GIS组合电器"],
        isolator_types=["GW4-1000", "GW5-1000D"],
        grounding_switch_types=["JW8-1000"],
        detection_classes=[
            "breaker_open", "breaker_closed",
            "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed",
            "indicator_red", "indicator_green",
            "gis_position", "sf6_density_relay",
            "operating_mechanism_state"
        ],
        angle_reference={
            "breaker": {"open_deg": -65, "closed_deg": 25},
            "isolator": {"open_deg": -75, "closed_deg": 15},
            "grounding": {"open_deg": -85, "closed_deg": 5}
        },
        special_features=[
            "双断口/四断口设计", "合闸电阻", "操作过电压抑制"
        ]
    ),
    busbar=BusbarConfig(
        conductor_types=["LGJ-800/55", "LGJQ-1000/70", "软导线8×LGJ-630/45"],
        busbar_height_m=20,
        phase_spacing_m=15.0,
        detection_classes=[
            "insulator_crack", "insulator_dirty", "insulator_flashover",
            "fitting_loose", "fitting_rust", "wire_damage",
            "foreign_object", "bird", "pin_missing",
            "spacer_damage", "corona_discharge", "ice_coating"
        ],
        small_target_min_px=25,
        special_features=["复合绝缘子", "大吨位绝缘子", "防污闪涂料"]
    ),
    capacitor=CapacitorConfig(
        typical_models=["CW-1000", "TBB-1000"],
        bank_capacity_mvar=[200, 400, 600],
        detection_classes=[
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle",
            "fuse_blown", "oil_leak"
        ],
        special_features=["串联补偿电容器", "可控串补"]
    ),
    meter=MeterConfig(
        types=[
            "SF6压力表", "SF6密度继电器", "油温表", "油位计",
            "气体继电器", "压力释放阀"
        ],
        sf6_pressure_range_mpa=(0.5, 0.7),
        oil_temp_range_celsius=(-30, 120),
        detection_classes=[
            "meter_pointer", "meter_digital", "gauge_normal",
            "gauge_warning", "gauge_alarm"
        ],
        special_features=["在线监测", "智能表计"]
    ),
    gis=GISConfig(
        types=["1000kV GIS", "ZF-1000"],
        rated_current_a=[4000, 5000, 6300],
        rated_breaking_current_ka=[50, 63],
        detection_classes=[
            "gis_position_indicator", "gis_sf6_density",
            "gis_partial_discharge", "gis_temperature"
        ],
        sf6_monitoring=True,
        partial_discharge_monitoring=True,
        special_features=["双/四断口灭弧室", "大容量分合闸电阻"]
    )
)

# -----------------------------------------------------------------------------
# 特高压直流 ±800kV 配置
# -----------------------------------------------------------------------------
CONFIG_800KV_DC = EquipmentConfig(
    transformer=TransformerConfig(
        typical_models=[
            "ZZDFPZ-509300/±800",   # 509.3MVA换流变
            "ZZDFPZ-400000/±800",   # 400MVA换流变
        ],
        capacity_range_mva=(300, 600),
        oil_tank_diameter_range_m=(5.0, 8.0),
        cooling_types=["ODAF", "OFWF"],
        detection_classes=[
            "oil_leak", "rust_corrosion", "surface_damage",
            "converter_valve_cooling", "dc_bushing_defect",
            "oil_level_abnormal", "thermal_anomaly"
        ],
        thermal_thresholds={
            "normal": 70, "warning": 85, "alarm": 100
        },
        special_features=["换流变压器", "阀侧套管", "网侧套管"]
    ),
    switch=SwitchConfig(
        breaker_types=["高速直流开关", "直流隔离开关", "接地开关"],
        isolator_types=["直流隔离开关"],
        grounding_switch_types=["直流接地开关"],
        detection_classes=[
            "dc_breaker_open", "dc_breaker_closed",
            "dc_isolator_state", "grounding_state"
        ],
        angle_reference={},
        special_features=["快速开关", "直流断路器"]
    ),
    busbar=BusbarConfig(
        conductor_types=["直流导线6×LGJQ-1250/70"],
        busbar_height_m=25,
        phase_spacing_m=20.0,
        detection_classes=[
            "dc_insulator_contamination", "dc_fitting_defect",
            "corona_discharge", "ion_flow"
        ],
        small_target_min_px=30,
        special_features=["直流绝缘子", "防离子流设计"]
    ),
    capacitor=CapacitorConfig(
        typical_models=["直流滤波电容器"],
        bank_capacity_mvar=[200, 400],
        detection_classes=[
            "dc_filter_unit", "capacitor_defect"
        ],
        special_features=["直流滤波器", "交流滤波器"]
    ),
    meter=MeterConfig(
        types=[
            "直流电压表", "直流电流表", "功率表"
        ],
        sf6_pressure_range_mpa=(0, 0),
        oil_temp_range_celsius=(-30, 120),
        detection_classes=[
            "dc_meter_reading", "power_meter"
        ],
        special_features=["直流测量"]
    ),
    dc_system=DCSystemConfig(
        converter_transformer_types=[
            "单相双绕组换流变压器",
            "单相三绕组换流变压器"
        ],
        converter_valve_types=[
            "晶闸管换流阀",
            "12脉动换流阀"
        ],
        smoothing_reactor_types=[
            "干式平波电抗器",
            "空心平波电抗器"
        ],
        dc_filter_types=[
            "直流滤波器",
            "高频滤波器"
        ],
        detection_classes=[
            "converter_valve_temperature",
            "converter_valve_cooling",
            "thyristor_status",
            "smoothing_reactor_temperature",
            "dc_filter_status"
        ],
        special_features=[
            "换流阀冷却系统监测",
            "晶闸管状态监测",
            "控制保护系统"
        ]
    )
)

# -----------------------------------------------------------------------------
# 超高压交流 500kV 配置
# -----------------------------------------------------------------------------
CONFIG_500KV_AC = EquipmentConfig(
    transformer=TransformerConfig(
        typical_models=[
            "OSFPSZ-500000/500",   # 500MVA
            "OSFPS-750000/500",    # 750MVA
            "OSFPSZ-1000000/500",  # 1000MVA
        ],
        capacity_range_mva=(500, 1000),
        oil_tank_diameter_range_m=(4.5, 7.0),
        cooling_types=["OFAF", "ODAF"],
        detection_classes=[
            "oil_leak", "rust_corrosion", "surface_damage",
            "foreign_object", "silica_gel_blue", "silica_gel_pink",
            "oil_level_normal", "oil_level_abnormal",
            "bushing_crack", "porcelain_contamination"
        ],
        thermal_thresholds={
            "normal": 65, "warning": 80, "alarm": 95
        },
        special_features=["套管裂纹检测", "油色谱监测", "局放监测"]
    ),
    switch=SwitchConfig(
        breaker_types=["SF6断路器", "GIS组合电器"],
        isolator_types=["GW4-500", "GW5-500D", "GW6-500W"],
        grounding_switch_types=["JW7-500", "JW8-500"],
        detection_classes=[
            "breaker_open", "breaker_closed",
            "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed",
            "indicator_red", "indicator_green",
            "gis_position"
        ],
        angle_reference={
            "breaker": {"open_deg": -60, "closed_deg": 30},
            "isolator": {"open_deg": -70, "closed_deg": 20},
            "grounding": {"open_deg": -80, "closed_deg": 10}
        },
        special_features=["GIS位置检测", "SF6密度监测"]
    ),
    busbar=BusbarConfig(
        conductor_types=["LGJ-630/45", "LGJQ-800/55"],
        busbar_height_m=15,
        phase_spacing_m=9.0,
        detection_classes=[
            "insulator_crack", "insulator_dirty", "insulator_flashover",
            "fitting_loose", "fitting_rust", "wire_damage",
            "foreign_object", "bird", "pin_missing", "spacer_damage"
        ],
        small_target_min_px=20,
        special_features=["间隔棒损坏检测"]
    ),
    capacitor=CapacitorConfig(
        typical_models=["CW1-500", "TBB35-12000"],
        bank_capacity_mvar=[60, 120, 180],
        detection_classes=[
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle", "fuse_blown"
        ],
        special_features=["串联电容器"]
    ),
    meter=MeterConfig(
        types=[
            "SF6压力表", "SF6密度继电器", "油温表", "油位计", "气体继电器"
        ],
        sf6_pressure_range_mpa=(0.45, 0.65),
        oil_temp_range_celsius=(-20, 110),
        detection_classes=[
            "meter_pointer", "meter_digital", "density_relay_indicator"
        ],
        special_features=["密度继电器检测"]
    ),
    gis=GISConfig(
        types=["ZF11-500", "ZF12-500"],
        rated_current_a=[3150, 4000, 5000],
        rated_breaking_current_ka=[40, 50, 63],
        detection_classes=[
            "gis_position_indicator", "gis_sf6_density",
            "gis_partial_discharge"
        ],
        sf6_monitoring=True,
        partial_discharge_monitoring=True,
        special_features=["GIS局放监测"]
    )
)

# -----------------------------------------------------------------------------
# 超高压交流 330kV 配置
# -----------------------------------------------------------------------------
CONFIG_330KV_AC = EquipmentConfig(
    transformer=TransformerConfig(
        typical_models=[
            "SFPSZ-180000/330",   # 180MVA
            "SFPSZ-240000/330",   # 240MVA
            "SFPSZ-360000/330",   # 360MVA
        ],
        capacity_range_mva=(180, 500),
        oil_tank_diameter_range_m=(3.5, 5.5),
        cooling_types=["ONAN", "ONAF", "OFAF"],
        detection_classes=[
            "oil_leak", "rust_corrosion", "surface_damage",
            "foreign_object", "silica_gel_blue", "silica_gel_pink",
            "oil_level_normal", "oil_level_abnormal",
            "bushing_crack"
        ],
        thermal_thresholds={
            "normal": 63, "warning": 78, "alarm": 92
        },
        special_features=["西北电网常用电压等级"]
    ),
    switch=SwitchConfig(
        breaker_types=["SF6断路器", "GIS组合电器"],
        isolator_types=["GW4-330", "GW5-330D"],
        grounding_switch_types=["JW6-330"],
        detection_classes=[
            "breaker_open", "breaker_closed",
            "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed",
            "indicator_red", "indicator_green"
        ],
        angle_reference={
            "breaker": {"open_deg": -58, "closed_deg": 32},
            "isolator": {"open_deg": -68, "closed_deg": 22},
            "grounding": {"open_deg": -78, "closed_deg": 12}
        },
        special_features=[]
    ),
    busbar=BusbarConfig(
        conductor_types=["LGJ-500/45", "LGJ-630/45"],
        busbar_height_m=12,
        phase_spacing_m=7.0,
        detection_classes=[
            "insulator_crack", "insulator_dirty",
            "fitting_loose", "fitting_rust", "wire_damage",
            "foreign_object", "bird", "pin_missing"
        ],
        small_target_min_px=18,
        special_features=[]
    ),
    capacitor=CapacitorConfig(
        typical_models=["CW1-330", "TBB20-8000"],
        bank_capacity_mvar=[40, 80, 120],
        detection_classes=[
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle"
        ],
        special_features=[]
    ),
    meter=MeterConfig(
        types=["SF6压力表", "油温表", "油位计"],
        sf6_pressure_range_mpa=(0.42, 0.62),
        oil_temp_range_celsius=(-20, 105),
        detection_classes=["meter_pointer", "meter_digital"],
        special_features=[]
    )
)

# -----------------------------------------------------------------------------
# 高压 220kV 配置
# -----------------------------------------------------------------------------
CONFIG_220KV = EquipmentConfig(
    transformer=TransformerConfig(
        typical_models=[
            "SZ11-50000/220",      # 50MVA
            "SFSZ11-120000/220",   # 120MVA
            "SFSZ9-180000/220",    # 180MVA
        ],
        capacity_range_mva=(50, 180),
        oil_tank_diameter_range_m=(2.5, 4.0),
        cooling_types=["ONAN", "ONAF", "OFAF"],
        detection_classes=[
            "oil_leak", "rust_corrosion", "surface_damage",
            "foreign_object", "silica_gel_blue", "silica_gel_pink",
            "oil_level_normal", "oil_level_abnormal"
        ],
        thermal_thresholds={
            "normal": 60, "warning": 75, "alarm": 85
        },
        special_features=[]
    ),
    switch=SwitchConfig(
        breaker_types=["SF6断路器", "真空断路器"],
        isolator_types=["GW4-220", "GW5-220D"],
        grounding_switch_types=["JW6-220"],
        detection_classes=[
            "breaker_open", "breaker_closed",
            "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed",
            "indicator_red", "indicator_green"
        ],
        angle_reference={
            "breaker": {"open_deg": -55, "closed_deg": 35},
            "isolator": {"open_deg": -65, "closed_deg": 25},
            "grounding": {"open_deg": -75, "closed_deg": 15}
        },
        special_features=[]
    ),
    busbar=BusbarConfig(
        conductor_types=["LGJ-400/35"],
        busbar_height_m=8,
        phase_spacing_m=4.5,
        detection_classes=[
            "insulator_crack", "insulator_dirty",
            "fitting_loose", "fitting_rust", "wire_damage",
            "foreign_object", "bird", "pin_missing"
        ],
        small_target_min_px=15,
        special_features=[]
    ),
    capacitor=CapacitorConfig(
        typical_models=["CW1-220", "TBB10-6600"],
        bank_capacity_mvar=[30, 60, 90],
        detection_classes=[
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle"
        ],
        special_features=[]
    ),
    meter=MeterConfig(
        types=["SF6压力表", "油温表", "油位计"],
        sf6_pressure_range_mpa=(0.4, 0.6),
        oil_temp_range_celsius=(-20, 100),
        detection_classes=["meter_pointer", "meter_digital"],
        special_features=[]
    )
)

# -----------------------------------------------------------------------------
# 高压 110kV 配置
# -----------------------------------------------------------------------------
CONFIG_110KV = EquipmentConfig(
    transformer=TransformerConfig(
        typical_models=[
            "SZ11-20000/110",    # 20MVA
            "SZ11-31500/110",    # 31.5MVA
            "SZ11-50000/110",    # 50MVA
            "SZ11-63000/110",    # 63MVA
        ],
        capacity_range_mva=(20, 63),
        oil_tank_diameter_range_m=(2.0, 3.5),
        cooling_types=["ONAN", "ONAF"],
        detection_classes=[
            "oil_leak", "rust_corrosion", "surface_damage",
            "foreign_object", "silica_gel_blue", "silica_gel_pink",
            "oil_level_normal", "oil_level_abnormal"
        ],
        thermal_thresholds={
            "normal": 55, "warning": 70, "alarm": 80
        },
        special_features=[]
    ),
    switch=SwitchConfig(
        breaker_types=["SF6断路器", "真空断路器"],
        isolator_types=["GW4-110", "GW5-110D"],
        grounding_switch_types=["JW5-110"],
        detection_classes=[
            "breaker_open", "breaker_closed",
            "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed",
            "indicator_red", "indicator_green"
        ],
        angle_reference={
            "breaker": {"open_deg": -50, "closed_deg": 40},
            "isolator": {"open_deg": -60, "closed_deg": 30},
            "grounding": {"open_deg": -70, "closed_deg": 20}
        },
        special_features=[]
    ),
    busbar=BusbarConfig(
        conductor_types=["LGJ-240/30", "LGJ-300/35"],
        busbar_height_m=6,
        phase_spacing_m=3.0,
        detection_classes=[
            "insulator_crack", "insulator_dirty",
            "fitting_loose", "fitting_rust", "wire_damage",
            "foreign_object", "bird"
        ],
        small_target_min_px=12,
        special_features=[]
    ),
    capacitor=CapacitorConfig(
        typical_models=["TBB6-3300", "TBB10-3300"],
        bank_capacity_mvar=[10, 20, 30],
        detection_classes=[
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle"
        ],
        special_features=[]
    ),
    meter=MeterConfig(
        types=["SF6压力表", "油温表", "油位计"],
        sf6_pressure_range_mpa=(0.38, 0.58),
        oil_temp_range_celsius=(-20, 95),
        detection_classes=["meter_pointer", "meter_digital"],
        special_features=[]
    )
)

# -----------------------------------------------------------------------------
# 中压 35kV 配置
# -----------------------------------------------------------------------------
CONFIG_35KV = EquipmentConfig(
    transformer=TransformerConfig(
        typical_models=[
            "S11-3150/35",    # 3.15MVA
            "S11-6300/35",    # 6.3MVA
            "S11-10000/35",   # 10MVA
            "SZ11-16000/35",  # 16MVA
            "SZ11-20000/35",  # 20MVA
        ],
        capacity_range_mva=(2, 20),
        oil_tank_diameter_range_m=(1.2, 2.5),
        cooling_types=["ONAN"],
        detection_classes=[
            "oil_leak", "rust_corrosion", "surface_damage",
            "foreign_object", "oil_level_normal", "oil_level_abnormal"
        ],
        thermal_thresholds={
            "normal": 50, "warning": 65, "alarm": 75
        },
        special_features=["干式变压器可选"]
    ),
    switch=SwitchConfig(
        breaker_types=["真空断路器", "SF6断路器"],
        isolator_types=["GW4-35", "GN-35"],
        grounding_switch_types=["JW3-35"],
        detection_classes=[
            "breaker_open", "breaker_closed",
            "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed",
            "indicator_red", "indicator_green"
        ],
        angle_reference={
            "breaker": {"open_deg": -45, "closed_deg": 45},
            "isolator": {"open_deg": -55, "closed_deg": 35}
        },
        special_features=["户内开关柜", "箱式变电站"]
    ),
    busbar=BusbarConfig(
        conductor_types=["LGJ-120/20", "LGJ-185/25", "矩形母线"],
        busbar_height_m=4,
        phase_spacing_m=1.5,
        detection_classes=[
            "insulator_crack", "insulator_dirty",
            "fitting_loose", "wire_damage", "foreign_object"
        ],
        small_target_min_px=10,
        special_features=["户内母线", "绝缘母线"]
    ),
    capacitor=CapacitorConfig(
        typical_models=["TBB6-1000", "TBB10-1800"],
        bank_capacity_mvar=[3, 6, 10],
        detection_classes=[
            "capacitor_unit", "capacitor_tilted",
            "capacitor_missing", "person"
        ],
        special_features=["集合式电容器"]
    ),
    meter=MeterConfig(
        types=["SF6压力表", "油温表", "电流表", "电压表"],
        sf6_pressure_range_mpa=(0.35, 0.55),
        oil_temp_range_celsius=(-20, 85),
        detection_classes=["meter_pointer", "meter_digital"],
        special_features=["综合自动化监测"]
    )
)

# -----------------------------------------------------------------------------
# 低压 10kV 配置
# -----------------------------------------------------------------------------
CONFIG_10KV = EquipmentConfig(
    transformer=TransformerConfig(
        typical_models=[
            "S11-315/10",     # 315kVA
            "S11-500/10",     # 500kVA
            "S11-800/10",     # 800kVA
            "S11-1000/10",    # 1000kVA
            "S11-1600/10",    # 1600kVA
            "SCB11-1000/10",  # 干式1000kVA
        ],
        capacity_range_mva=(0.1, 2),
        oil_tank_diameter_range_m=(0.6, 1.5),
        cooling_types=["ONAN", "AN"],  # AN为干式变压器
        detection_classes=[
            "oil_leak", "rust_corrosion", "surface_damage",
            "oil_level_normal", "oil_level_abnormal"
        ],
        thermal_thresholds={
            "normal": 45, "warning": 60, "alarm": 70
        },
        special_features=["干式变压器", "箱式变电站", "环网柜"]
    ),
    switch=SwitchConfig(
        breaker_types=["真空断路器", "负荷开关"],
        isolator_types=["GN-10", "隔离手车"],
        grounding_switch_types=["JN-10"],
        detection_classes=[
            "breaker_open", "breaker_closed",
            "isolator_open", "isolator_closed",
            "indicator_red", "indicator_green",
            "switch_cabinet_door"
        ],
        angle_reference={
            "breaker": {"open_deg": -40, "closed_deg": 50}
        },
        special_features=["开关柜", "环网柜", "充气柜"]
    ),
    busbar=BusbarConfig(
        conductor_types=["矩形母线", "管形母线", "绝缘母线"],
        busbar_height_m=2.5,
        phase_spacing_m=0.3,
        detection_classes=[
            "busbar_temperature", "insulator_dirty",
            "connector_loose", "foreign_object"
        ],
        small_target_min_px=8,
        special_features=["户内母线", "封闭母线"]
    ),
    capacitor=CapacitorConfig(
        typical_models=["BSMJ-0.45-30", "BKMJ-0.45-40"],
        bank_capacity_mvar=[0.3, 0.5, 1.0],
        detection_classes=[
            "capacitor_unit", "capacitor_swelling"
        ],
        special_features=["智能电容器", "自动投切"]
    ),
    meter=MeterConfig(
        types=["电流表", "电压表", "功率表", "电度表", "功率因数表"],
        sf6_pressure_range_mpa=(0, 0),
        oil_temp_range_celsius=(-20, 75),
        detection_classes=[
            "meter_pointer", "meter_digital", "meter_display"
        ],
        special_features=["智能仪表", "多功能电力仪表"]
    )
)


# =============================================================================
# 模型库配置
# =============================================================================
@dataclass
class ModelLibrary:
    """模型库配置"""
    voltage_level: str
    base_path: str
    models: Dict[str, Dict[str, str]] = field(default_factory=dict)


# 各电压等级模型库
MODEL_LIBRARIES = {
    "1000kV_AC": ModelLibrary(
        voltage_level="1000kV_AC",
        base_path="models/uhv/1000kV_AC",
        models={
            "transformer": {
                "defect_detection": "transformer_defect_uhv.onnx",
                "oil_segmentation": "oil_seg_uhv.onnx",
                "thermal_anomaly": "thermal_uhv.onnx",
                "bushing_detection": "bushing_uhv.onnx",
                "pd_detection": "pd_uhv.onnx"
            },
            "switch": {
                "state_detection": "switch_state_uhv.onnx",
                "gis_position": "gis_position_uhv.onnx"
            },
            "busbar": {
                "defect_detection": "busbar_defect_uhv.onnx",
                "corona_detection": "corona_uhv.onnx"
            }
        }
    ),
    "±800kV_DC": ModelLibrary(
        voltage_level="±800kV_DC",
        base_path="models/uhv/800kV_DC",
        models={
            "converter": {
                "valve_monitoring": "converter_valve_uhv.onnx",
                "cooling_system": "cooling_uhv.onnx"
            },
            "transformer": {
                "defect_detection": "converter_transformer_uhv.onnx"
            }
        }
    ),
    "500kV_AC": ModelLibrary(
        voltage_level="500kV_AC",
        base_path="models/ehv/500kV",
        models={
            "transformer": {
                "defect_detection": "transformer_defect_500kv.onnx",
                "oil_segmentation": "oil_seg_500kv.onnx",
                "thermal_anomaly": "thermal_500kv.onnx",
                "bushing_detection": "bushing_500kv.onnx"
            },
            "switch": {
                "state_detection": "switch_state_500kv.onnx",
                "gis_position": "gis_position_500kv.onnx"
            },
            "busbar": {
                "defect_detection": "busbar_defect_500kv.onnx",
                "spacer_detection": "spacer_500kv.onnx"
            },
            "meter": {
                "keypoint_detection": "meter_keypoint_500kv.onnx",
                "density_relay": "density_relay_500kv.onnx"
            }
        }
    ),
    "330kV_AC": ModelLibrary(
        voltage_level="330kV_AC",
        base_path="models/ehv/330kV",
        models={
            "transformer": {
                "defect_detection": "transformer_defect_330kv.onnx",
                "thermal_anomaly": "thermal_330kv.onnx"
            },
            "switch": {
                "state_detection": "switch_state_330kv.onnx"
            },
            "busbar": {
                "defect_detection": "busbar_defect_330kv.onnx"
            }
        }
    ),
    "220kV": ModelLibrary(
        voltage_level="220kV",
        base_path="models/hv/220kV",
        models={
            "transformer": {
                "defect_detection": "transformer_defect_220kv.onnx",
                "oil_segmentation": "oil_seg_220kv.onnx",
                "thermal_anomaly": "thermal_220kv.onnx"
            },
            "switch": {
                "state_detection": "switch_state_220kv.onnx",
                "indicator_ocr": "indicator_ocr_220kv.onnx"
            },
            "busbar": {
                "defect_detection": "busbar_defect_220kv.onnx"
            },
            "meter": {
                "keypoint_detection": "meter_keypoint_220kv.onnx",
                "ocr": "meter_ocr_220kv.onnx"
            }
        }
    ),
    "110kV": ModelLibrary(
        voltage_level="110kV",
        base_path="models/hv/110kV",
        models={
            "transformer": {
                "defect_detection": "transformer_defect_110kv.onnx",
                "thermal_anomaly": "thermal_110kv.onnx"
            },
            "switch": {
                "state_detection": "switch_state_110kv.onnx"
            },
            "busbar": {
                "defect_detection": "busbar_defect_110kv.onnx"
            },
            "meter": {
                "keypoint_detection": "meter_keypoint_110kv.onnx"
            }
        }
    ),
    "35kV": ModelLibrary(
        voltage_level="35kV",
        base_path="models/mv/35kV",
        models={
            "transformer": {
                "defect_detection": "transformer_defect_35kv.onnx",
                "thermal_anomaly": "thermal_35kv.onnx"
            },
            "switch": {
                "state_detection": "switch_state_35kv.onnx",
                "cabinet_detection": "cabinet_35kv.onnx"
            },
            "meter": {
                "keypoint_detection": "meter_keypoint_35kv.onnx",
                "ocr": "meter_ocr_35kv.onnx"
            }
        }
    ),
    "10kV": ModelLibrary(
        voltage_level="10kV",
        base_path="models/lv/10kV",
        models={
            "transformer": {
                "defect_detection": "transformer_defect_10kv.onnx",
                "thermal_anomaly": "thermal_10kv.onnx"
            },
            "switch": {
                "state_detection": "switch_state_10kv.onnx",
                "cabinet_detection": "cabinet_10kv.onnx"
            },
            "meter": {
                "keypoint_detection": "meter_keypoint_10kv.onnx",
                "ocr": "meter_ocr_10kv.onnx",
                "digital_display": "digital_display_10kv.onnx"
            },
            "environment": {
                "sf6_monitoring": "sf6_monitor_10kv.onnx",
                "temperature_humidity": "temp_humidity_10kv.onnx"
            }
        }
    )
}


# =============================================================================
# 配置映射
# =============================================================================
VOLTAGE_CONFIGS = {
    "1000kV_AC": CONFIG_1000KV_AC,
    "1100kV_AC": CONFIG_1000KV_AC,  # 使用相同配置
    "±800kV_DC": CONFIG_800KV_DC,
    "±1100kV_DC": CONFIG_800KV_DC,  # 使用相似配置
    "500kV_AC": CONFIG_500KV_AC,
    "500kV": CONFIG_500KV_AC,       # 别名
    "330kV_AC": CONFIG_330KV_AC,
    "330kV": CONFIG_330KV_AC,       # 别名
    "750kV_AC": CONFIG_500KV_AC,    # 使用500kV配置基础
    "±500kV_DC": CONFIG_500KV_AC,   # 使用500kV配置基础
    "220kV": CONFIG_220KV,
    "110kV": CONFIG_110KV,
    "66kV": CONFIG_35KV,            # 使用35kV配置
    "35kV": CONFIG_35KV,
    "10kV": CONFIG_10KV,
    "6kV": CONFIG_10KV,             # 使用10kV配置
    "380V": CONFIG_10KV,            # 使用10kV配置
}


# =============================================================================
# 插件功能定义
# =============================================================================
@dataclass
class PluginCapability:
    """插件功能定义"""
    name: str
    description: str
    supported_voltage_levels: List[str]
    detection_types: List[str]
    requires_models: List[str]


# 各电压等级支持的插件功能
PLUGIN_CAPABILITIES = {
    # 特高压专有功能
    "uhv_bushing_monitor": PluginCapability(
        name="特高压套管监测",
        description="针对特高压设备套管的专项监测，包括裂纹、污损、局放等",
        supported_voltage_levels=["1000kV_AC", "±800kV_DC"],
        detection_types=["bushing_crack", "porcelain_contamination", "partial_discharge"],
        requires_models=["bushing_uhv.onnx", "pd_uhv.onnx"]
    ),
    "uhv_corona_detection": PluginCapability(
        name="特高压电晕检测",
        description="检测特高压线路的电晕放电现象",
        supported_voltage_levels=["1000kV_AC", "±800kV_DC", "500kV_AC"],
        detection_types=["corona_discharge", "arc_flash"],
        requires_models=["corona_uhv.onnx"]
    ),
    "converter_valve_monitor": PluginCapability(
        name="换流阀监测",
        description="直流特高压换流阀温度、冷却系统监测",
        supported_voltage_levels=["±800kV_DC", "±1100kV_DC", "±500kV_DC"],
        detection_types=["valve_temperature", "cooling_status", "thyristor_status"],
        requires_models=["converter_valve_uhv.onnx", "cooling_uhv.onnx"]
    ),
    
    # 超高压/高压通用功能
    "gis_monitoring": PluginCapability(
        name="GIS设备监测",
        description="GIS组合电器位置状态、SF6密度、局放监测",
        supported_voltage_levels=["1000kV_AC", "500kV_AC", "330kV_AC", "220kV", "110kV"],
        detection_types=["gis_position", "sf6_density", "partial_discharge"],
        requires_models=["gis_position_*.onnx"]
    ),
    "sf6_density_relay": PluginCapability(
        name="SF6密度继电器识别",
        description="识别SF6密度继电器状态",
        supported_voltage_levels=["500kV_AC", "330kV_AC", "220kV", "110kV", "35kV"],
        detection_types=["density_relay_indicator"],
        requires_models=["density_relay_*.onnx"]
    ),
    
    # 通用功能
    "transformer_monitor": PluginCapability(
        name="变压器监测",
        description="变压器油位、油温、渗漏、外观缺陷检测",
        supported_voltage_levels=["all"],
        detection_types=["oil_leak", "oil_level", "thermal_anomaly", "surface_defect"],
        requires_models=["transformer_defect_*.onnx"]
    ),
    "switch_state_detection": PluginCapability(
        name="开关状态检测",
        description="断路器、隔离开关、接地刀闸状态识别",
        supported_voltage_levels=["all"],
        detection_types=["breaker_state", "isolator_state", "grounding_state"],
        requires_models=["switch_state_*.onnx"]
    ),
    "busbar_inspection": PluginCapability(
        name="母线巡检",
        description="绝缘子、金具、导线缺陷检测",
        supported_voltage_levels=["all"],
        detection_types=["insulator_defect", "fitting_defect", "wire_damage"],
        requires_models=["busbar_defect_*.onnx"]
    ),
    "meter_reading": PluginCapability(
        name="表计读数识别",
        description="指针式、数字式表计读数识别",
        supported_voltage_levels=["all"],
        detection_types=["meter_pointer", "meter_digital"],
        requires_models=["meter_keypoint_*.onnx", "meter_ocr_*.onnx"]
    ),
    "thermal_imaging": PluginCapability(
        name="红外热成像分析",
        description="设备温度异常检测",
        supported_voltage_levels=["all"],
        detection_types=["thermal_anomaly", "hot_spot"],
        requires_models=["thermal_*.onnx"]
    ),
    
    # 中低压专有功能
    "cabinet_inspection": PluginCapability(
        name="开关柜巡检",
        description="开关柜状态、门禁、指示灯检测",
        supported_voltage_levels=["35kV", "10kV", "6kV"],
        detection_types=["cabinet_door", "indicator_light", "switch_position"],
        requires_models=["cabinet_*.onnx"]
    ),
    "environment_monitor": PluginCapability(
        name="环境监测",
        description="SF6浓度、温湿度、烟雾监测",
        supported_voltage_levels=["35kV", "10kV"],
        detection_types=["sf6_concentration", "temperature", "humidity", "smoke"],
        requires_models=["sf6_monitor_*.onnx", "temp_humidity_*.onnx"]
    ),
    "smart_meter": PluginCapability(
        name="智能仪表读取",
        description="数字显示屏、多功能电力仪表读取",
        supported_voltage_levels=["35kV", "10kV", "6kV", "380V"],
        detection_types=["digital_display", "multi_function_meter"],
        requires_models=["digital_display_*.onnx"]
    )
}


# =============================================================================
# 电压等级适配管理器
# =============================================================================
class VoltageAdapterManager:
    """
    电压等级适配管理器
    
    管理不同电压等级变电站的模型配置和设备参数
    """
    
    def __init__(self, config_path: str = "configs/voltage_config.yaml"):
        self.config_path = Path(config_path)
        self.current_level: Optional[str] = None
        self.current_category: Optional[VoltageCategory] = None
        self.equipment_config: Optional[EquipmentConfig] = None
        self.model_library: Optional[ModelLibrary] = None
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config and "current_voltage_level" in config:
                        self.set_voltage_level(config["current_voltage_level"])
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")
    
    def _save_config(self):
        """保存配置文件"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "current_voltage_level": self.current_level,
                "current_category": self.current_category.value if self.current_category else None,
                "last_updated": str(Path.cwd()),
            }
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def get_available_voltage_levels(self) -> Dict[str, List[str]]:
        """获取所有可用的电压等级，按分类返回"""
        return {
            VoltageCategory.UHV.value: ["1000kV_AC", "±800kV_DC", "±1100kV_DC"],
            VoltageCategory.EHV.value: ["500kV_AC", "330kV_AC", "750kV_AC", "±500kV_DC"],
            VoltageCategory.HV.value: ["220kV", "110kV"],
            VoltageCategory.MV.value: ["35kV", "66kV"],
            VoltageCategory.LV.value: ["10kV", "6kV", "380V"],
        }
    
    def set_voltage_level(self, level: str) -> bool:
        """
        设置当前电压等级
        
        Args:
            level: 电压等级字符串
        
        Returns:
            是否设置成功
        """
        try:
            # 标准化电压等级名称
            normalized_level = self._normalize_voltage_level(level)
            
            if normalized_level not in VOLTAGE_CONFIGS:
                logger.error(f"不支持的电压等级: {level}")
                return False
            
            self.current_level = normalized_level
            self.equipment_config = VOLTAGE_CONFIGS[normalized_level]
            
            # 确定电压分类
            self.current_category = self._get_voltage_category(normalized_level)
            
            # 获取模型库
            if normalized_level in MODEL_LIBRARIES:
                self.model_library = MODEL_LIBRARIES[normalized_level]
            else:
                # 尝试找最接近的模型库
                self.model_library = self._find_closest_model_library(normalized_level)
            
            self._save_config()
            logger.info(f"电压等级已切换至: {self.current_level} ({self.current_category.value})")
            return True
            
        except Exception as e:
            logger.error(f"设置电压等级失败: {e}")
            return False
    
    def _normalize_voltage_level(self, level: str) -> str:
        """标准化电压等级名称"""
        # 移除空格
        level = level.strip()
        
        # 常见别名映射
        aliases = {
            "500kv": "500kV_AC",
            "500KV": "500kV_AC",
            "220kv": "220kV",
            "220KV": "220kV",
            "110kv": "110kV",
            "110KV": "110kV",
            "35kv": "35kV",
            "35KV": "35kV",
            "10kv": "10kV",
            "10KV": "10kV",
            "1000kv": "1000kV_AC",
            "1000KV": "1000kV_AC",
            "800kv_dc": "±800kV_DC",
            "800KV_DC": "±800kV_DC",
        }
        
        return aliases.get(level, level)
    
    def _get_voltage_category(self, level: str) -> VoltageCategory:
        """根据电压等级获取分类"""
        # 解析电压等级字符串
        level_lower = level.lower()
        
        if "1000" in level or "1100" in level or "800" in level:
            return VoltageCategory.UHV
        elif "500" in level or "330" in level or "750" in level or "660" in level:
            return VoltageCategory.EHV
        elif "220" in level or "110" in level:
            return VoltageCategory.HV
        elif "35" in level or "66" in level:
            return VoltageCategory.MV
        else:
            return VoltageCategory.LV
    
    def _find_closest_model_library(self, level: str) -> Optional[ModelLibrary]:
        """找到最接近的模型库"""
        # 根据电压分类选择
        category = self._get_voltage_category(level)
        
        fallback_mapping = {
            VoltageCategory.UHV: "1000kV_AC",
            VoltageCategory.EHV: "500kV_AC",
            VoltageCategory.HV: "220kV",
            VoltageCategory.MV: "35kV",
            VoltageCategory.LV: "10kV",
        }
        
        fallback_level = fallback_mapping.get(category)
        if fallback_level and fallback_level in MODEL_LIBRARIES:
            return MODEL_LIBRARIES[fallback_level]
        
        return None
    
    def get_current_level(self) -> Optional[str]:
        """获取当前电压等级"""
        return self.current_level
    
    def get_current_category(self) -> Optional[str]:
        """获取当前电压分类"""
        return self.current_category.value if self.current_category else None
    
    def get_equipment_config(self, equipment_type: str) -> Dict[str, Any]:
        """
        获取设备配置
        
        Args:
            equipment_type: transformer, switch, busbar, capacitor, meter, dc_system, gis
        
        Returns:
            设备配置字典
        """
        if not self.equipment_config:
            logger.warning("未设置电压等级，返回空配置")
            return {}
        
        config = getattr(self.equipment_config, equipment_type, None)
        if config is None:
            return {}
        
        return asdict(config) if hasattr(config, '__dataclass_fields__') else config
    
    def get_all_equipment_config(self) -> Dict[str, Any]:
        """获取所有设备配置"""
        if not self.equipment_config:
            return {}
        return self.equipment_config.to_dict()
    
    def get_model_path(self, equipment_type: str, model_name: str) -> Optional[str]:
        """
        获取模型文件路径
        
        Args:
            equipment_type: 设备类型
            model_name: 模型名称
        
        Returns:
            完整模型路径
        """
        if not self.model_library:
            logger.warning("模型库未初始化")
            return None
        
        models = self.model_library.models.get(equipment_type, {})
        model_file = models.get(model_name)
        
        if model_file:
            return f"{self.model_library.base_path}/{equipment_type}/{model_file}"
        
        return None
    
    def get_all_model_paths(self) -> Dict[str, Dict[str, str]]:
        """获取所有模型路径"""
        if not self.model_library:
            return {}
        
        result = {}
        for equipment_type, models in self.model_library.models.items():
            result[equipment_type] = {}
            for model_name, model_file in models.items():
                result[equipment_type][model_name] = (
                    f"{self.model_library.base_path}/{equipment_type}/{model_file}"
                )
        
        return result
    
    def get_detection_classes(self, equipment_type: str) -> List[str]:
        """获取检测类别列表"""
        config = self.get_equipment_config(equipment_type)
        return config.get("detection_classes", [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        """获取热成像阈值"""
        config = self.get_equipment_config("transformer")
        return config.get("thermal_thresholds", {})
    
    def get_angle_reference(self, switch_type: str) -> Dict[str, float]:
        """获取开关角度参考值"""
        config = self.get_equipment_config("switch")
        angle_refs = config.get("angle_reference", {})
        return angle_refs.get(switch_type, {})
    
    def get_supported_plugins(self) -> List[Dict[str, Any]]:
        """获取当前电压等级支持的插件功能"""
        if not self.current_level:
            return []
        
        supported = []
        for plugin_id, capability in PLUGIN_CAPABILITIES.items():
            if (capability.supported_voltage_levels == ["all"] or 
                self.current_level in capability.supported_voltage_levels):
                supported.append({
                    "id": plugin_id,
                    "name": capability.name,
                    "description": capability.description,
                    "detection_types": capability.detection_types,
                    "requires_models": capability.requires_models
                })
        
        return supported
    
    def get_special_features(self) -> List[str]:
        """获取当前电压等级的特殊功能特性"""
        if not self.equipment_config:
            return []
        
        features = []
        for equipment_type in ["transformer", "switch", "busbar", "capacitor", "meter"]:
            config = self.get_equipment_config(equipment_type)
            equipment_features = config.get("special_features", [])
            features.extend(equipment_features)
        
        # GIS 和 DC System 特殊处理
        if self.equipment_config.gis:
            features.extend(self.equipment_config.gis.special_features)
        if self.equipment_config.dc_system:
            features.extend(self.equipment_config.dc_system.special_features)
        
        return list(set(features))  # 去重
    
    def check_model_status(self) -> Dict[str, Dict[str, bool]]:
        """检查模型文件状态"""
        model_paths = self.get_all_model_paths()
        status = {}
        
        for equipment_type, models in model_paths.items():
            status[equipment_type] = {}
            for model_name, model_path in models.items():
                # 检查文件是否存在
                status[equipment_type][model_name] = Path(model_path).exists()
        
        return status
    
    def get_voltage_level_info(self) -> Dict[str, Any]:
        """获取当前电压等级的完整信息"""
        return {
            "voltage_level": self.current_level,
            "category": self.get_current_category(),
            "category_description": self._get_category_description(),
            "equipment_config": self.get_all_equipment_config(),
            "model_library": {
                "base_path": self.model_library.base_path if self.model_library else None,
                "models": self.get_all_model_paths()
            },
            "supported_plugins": self.get_supported_plugins(),
            "special_features": self.get_special_features(),
            "thermal_thresholds": self.get_thermal_thresholds()
        }
    
    def _get_category_description(self) -> str:
        """获取电压分类描述"""
        descriptions = {
            VoltageCategory.UHV: "特高压变电站：交流1000kV及以上、直流±800kV及以上",
            VoltageCategory.EHV: "超高压变电站：交流330kV、500kV、750kV；直流±500kV",
            VoltageCategory.HV: "高压变电站：110kV、220kV",
            VoltageCategory.MV: "中压变电站：35kV、66kV",
            VoltageCategory.LV: "低压变电站：10kV及以下"
        }
        return descriptions.get(self.current_category, "")


# =============================================================================
# 便捷函数
# =============================================================================
def create_voltage_adapter(voltage_level: str) -> VoltageAdapterManager:
    """创建并初始化电压适配器"""
    adapter = VoltageAdapterManager()
    adapter.set_voltage_level(voltage_level)
    return adapter


def get_all_voltage_categories() -> List[Dict[str, Any]]:
    """获取所有电压分类信息"""
    return [
        {
            "category": VoltageCategory.UHV.value,
            "description": "交流1000kV及以上、直流±800kV及以上",
            "levels": ["1000kV_AC", "±800kV_DC", "±1100kV_DC"]
        },
        {
            "category": VoltageCategory.EHV.value,
            "description": "交流330kV、500kV、750kV；直流±500kV",
            "levels": ["500kV_AC", "330kV_AC", "750kV_AC", "±500kV_DC"]
        },
        {
            "category": VoltageCategory.HV.value,
            "description": "110kV、220kV",
            "levels": ["220kV", "110kV"]
        },
        {
            "category": VoltageCategory.MV.value,
            "description": "35kV、66kV",
            "levels": ["35kV", "66kV"]
        },
        {
            "category": VoltageCategory.LV.value,
            "description": "10kV及以下",
            "levels": ["10kV", "6kV", "380V"]
        }
    ]


# =============================================================================
# 命令行接口
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="变电站电压等级适配管理系统")
    parser.add_argument("--set", type=str, help="设置电压等级")
    parser.add_argument("--show", action="store_true", help="显示当前配置")
    parser.add_argument("--list", action="store_true", help="列出所有可用电压等级")
    parser.add_argument("--export", type=str, help="导出配置到指定文件")
    
    args = parser.parse_args()
    
    manager = VoltageAdapterManager()
    
    if args.set:
        success = manager.set_voltage_level(args.set)
        if success:
            print(f"✓ 电压等级已设置为: {manager.get_current_level()}")
            print(f"  分类: {manager.get_current_category()}")
        else:
            print(f"✗ 设置失败: 不支持的电压等级 '{args.set}'")
    
    if args.show:
        info = manager.get_voltage_level_info()
        print("\n当前电压等级配置:")
        print(f"  电压等级: {info['voltage_level']}")
        print(f"  分类: {info['category']}")
        print(f"  描述: {info['category_description']}")
        print(f"\n支持的插件功能:")
        for plugin in info['supported_plugins']:
            print(f"  - {plugin['name']}: {plugin['description']}")
        print(f"\n特殊功能特性:")
        for feature in info['special_features']:
            print(f"  - {feature}")
    
    if args.list:
        print("\n可用的电压等级分类:")
        for category in get_all_voltage_categories():
            print(f"\n{category['category']}:")
            print(f"  描述: {category['description']}")
            print(f"  电压等级: {', '.join(category['levels'])}")
    
    if args.export:
        info = manager.get_voltage_level_info()
        with open(args.export, 'w', encoding='utf-8') as f:
            yaml.dump(info, f, default_flow_style=False, allow_unicode=True)
        print(f"✓ 配置已导出到: {args.export}")
