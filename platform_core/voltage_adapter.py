#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 电压等级适配管理系统
==========================================

功能:
1. 管理员在操作系统中选择 220kV 或 500kV
2. 系统自动匹配对应的预训练模型库
3. 支持不同电压等级的设备配置差异
4. 提供统一的适配层接口

使用方法:
    from platform_core.voltage_adapter import VoltageAdapterManager

    # 初始化管理器
    manager = VoltageAdapterManager()

    # 设置电压等级
    manager.set_voltage_level("500kV")

    # 获取适配后的模型配置
    config = manager.get_model_config("switch")

作者: 破夜绘明团队
日期: 2025
"""

from __future__ import annotations
import sys
import os

# 确保使用标准库 logging (避免与 platform_core/logging 冲突)
_orig_path = sys.path.copy()
sys.path = [p for p in sys.path if 'platform_core' not in p]
import logging  # noqa: E402
sys.path = _orig_path

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# 电压等级定义
# =============================================================================
class VoltageLevel(Enum):
    """变电站电压等级"""
    KV_220 = "220kV"
    KV_500 = "500kV"
    # 未来扩展
    KV_110 = "110kV"
    KV_750 = "750kV"
    KV_1000 = "1000kV"  # 特高压


# =============================================================================
# 设备配置差异定义
# =============================================================================
@dataclass
class EquipmentConfig:
    """设备配置"""
    # 主变参数
    transformer: Dict[str, Any] = field(default_factory=dict)
    # 开关参数
    switch: Dict[str, Any] = field(default_factory=dict)
    # 母线参数
    busbar: Dict[str, Any] = field(default_factory=dict)
    # 电容器参数
    capacitor: Dict[str, Any] = field(default_factory=dict)
    # 表计参数
    meter: Dict[str, Any] = field(default_factory=dict)


# 220kV 变电站配置
CONFIG_220KV = EquipmentConfig(
    transformer={
        "typical_models": [
            "SZ11-50000/220",  # 50MVA
            "SFSZ11-120000/220",  # 120MVA
            "SFSZ9-180000/220",  # 180MVA
        ],
        "oil_tank_diameter_range": [2.5, 4.0],  # 米
        "cooling_type": ["ONAN", "ONAF", "OFAF"],
        "detection_classes": [
            "oil_leak",           # 油泄漏
            "rust_corrosion",     # 锈蚀
            "surface_damage",     # 表面破损
            "foreign_object",     # 异物
            "silica_gel_blue",    # 硅胶蓝色
            "silica_gel_pink",    # 硅胶粉色
            "oil_level_normal",   # 油位正常
            "oil_level_abnormal", # 油位异常
        ],
        "thermal_threshold_celsius": {
            "normal": 60,
            "warning": 75,
            "alarm": 85,
        },
    },
    switch={
        "breaker_types": [
            "SF6断路器",
            "真空断路器",
        ],
        "isolator_types": [
            "GW4-220",
            "GW5-220D",
        ],
        "grounding_switch_types": [
            "JW6-220",
        ],
        "detection_classes": [
            "breaker_open",
            "breaker_closed",
            "isolator_open",
            "isolator_closed",
            "grounding_open",
            "grounding_closed",
            "indicator_red",
            "indicator_green",
        ],
        "angle_reference": {
            "breaker": {"open_deg": -55, "closed_deg": 35},
            "isolator": {"open_deg": -65, "closed_deg": 25},
            "grounding": {"open_deg": -75, "closed_deg": 15},
        },
    },
    busbar={
        "typical_specs": {
            "conductor_type": "LGJ-400/35",  # 铝绞线
            "busbar_height_m": 8,
            "phase_spacing_m": 4.5,
        },
        "detection_classes": [
            "insulator_crack",
            "insulator_dirty",
            "fitting_loose",
            "fitting_rust",
            "wire_damage",
            "foreign_object",
            "bird",
            "pin_missing",
        ],
        "small_target_min_px": 15,  # 最小目标像素
    },
    capacitor={
        "typical_models": [
            "CW1-220",
            "TBB10-6600",
        ],
        "bank_capacity_mvar": [30, 60, 90],
        "detection_classes": [
            "capacitor_unit",
            "capacitor_tilted",
            "capacitor_fallen",
            "capacitor_missing",
            "person",
            "vehicle",
        ],
    },
    meter={
        "types": [
            "SF6压力表",
            "油温表",
            "油位计",
        ],
        "sf6_pressure_range_mpa": [0.4, 0.6],
        "oil_temp_range_celsius": [-20, 100],
    },
)

# 500kV 变电站配置
CONFIG_500KV = EquipmentConfig(
    transformer={
        "typical_models": [
            "OSFPSZ-500000/500",  # 500MVA
            "OSFPS-750000/500",   # 750MVA
            "OSFPSZ-1000000/500", # 1000MVA
        ],
        "oil_tank_diameter_range": [4.5, 7.0],  # 米，更大
        "cooling_type": ["OFAF", "ODAF"],
        "detection_classes": [
            "oil_leak",
            "rust_corrosion",
            "surface_damage",
            "foreign_object",
            "silica_gel_blue",
            "silica_gel_pink",
            "oil_level_normal",
            "oil_level_abnormal",
            "bushing_crack",      # 500kV特有：套管裂纹
            "porcelain_contamination",  # 瓷套污损
        ],
        "thermal_threshold_celsius": {
            "normal": 65,
            "warning": 80,
            "alarm": 95,
        },
    },
    switch={
        "breaker_types": [
            "SF6断路器",
            "GIS组合电器",
        ],
        "isolator_types": [
            "GW4-500",
            "GW5-500D",
            "GW6-500W",
        ],
        "grounding_switch_types": [
            "JW7-500",
            "JW8-500",
        ],
        "detection_classes": [
            "breaker_open",
            "breaker_closed",
            "isolator_open",
            "isolator_closed",
            "grounding_open",
            "grounding_closed",
            "indicator_red",
            "indicator_green",
            "gis_position",       # GIS位置指示
        ],
        "angle_reference": {
            "breaker": {"open_deg": -60, "closed_deg": 30},
            "isolator": {"open_deg": -70, "closed_deg": 20},
            "grounding": {"open_deg": -80, "closed_deg": 10},
        },
    },
    busbar={
        "typical_specs": {
            "conductor_type": "LGJ-630/45",  # 更粗的导线
            "busbar_height_m": 15,  # 更高
            "phase_spacing_m": 9.0,  # 更大相间距
        },
        "detection_classes": [
            "insulator_crack",
            "insulator_dirty",
            "insulator_flashover",  # 500kV特有：闪络痕迹
            "fitting_loose",
            "fitting_rust",
            "wire_damage",
            "foreign_object",
            "bird",
            "pin_missing",
            "spacer_damage",      # 间隔棒损坏
        ],
        "small_target_min_px": 20,  # 更大的最小目标
    },
    capacitor={
        "typical_models": [
            "CW1-500",
            "TBB35-12000",
        ],
        "bank_capacity_mvar": [60, 120, 180],
        "detection_classes": [
            "capacitor_unit",
            "capacitor_tilted",
            "capacitor_fallen",
            "capacitor_missing",
            "person",
            "vehicle",
            "fuse_blown",         # 熔丝熔断
        ],
    },
    meter={
        "types": [
            "SF6压力表",
            "SF6密度继电器",
            "油温表",
            "油位计",
            "气体继电器",
        ],
        "sf6_pressure_range_mpa": [0.45, 0.65],
        "oil_temp_range_celsius": [-20, 110],
    },
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


# 220kV 模型库
MODEL_LIBRARY_220KV = ModelLibrary(
    voltage_level="220kV",
    base_path="models/220kV",
    models={
        "transformer": {
            "defect_detection": "transformer_defect_220kv.onnx",
            "oil_segmentation": "oil_seg_220kv.onnx",
            "silica_classifier": "silica_220kv.onnx",
            "thermal_anomaly": "thermal_220kv.onnx",
        },
        "switch": {
            "state_detection": "switch_state_220kv.onnx",
            "indicator_ocr": "indicator_ocr_220kv.onnx",
        },
        "busbar": {
            "defect_detection": "busbar_defect_220kv.onnx",
            "noise_filter": "noise_filter_220kv.onnx",
        },
        "capacitor": {
            "unit_detection": "capacitor_220kv.onnx",
            "intrusion_detection": "intrusion_220kv.onnx",
        },
        "meter": {
            "keypoint_detection": "meter_keypoint_220kv.onnx",
            "ocr": "meter_ocr_220kv.onnx",
        },
    },
)

# 500kV 模型库
MODEL_LIBRARY_500KV = ModelLibrary(
    voltage_level="500kV",
    base_path="models/500kV",
    models={
        "transformer": {
            "defect_detection": "transformer_defect_500kv.onnx",
            "oil_segmentation": "oil_seg_500kv.onnx",
            "silica_classifier": "silica_500kv.onnx",
            "thermal_anomaly": "thermal_500kv.onnx",
            "bushing_detection": "bushing_500kv.onnx",  # 500kV特有
        },
        "switch": {
            "state_detection": "switch_state_500kv.onnx",
            "indicator_ocr": "indicator_ocr_500kv.onnx",
            "gis_position": "gis_position_500kv.onnx",  # 500kV特有
        },
        "busbar": {
            "defect_detection": "busbar_defect_500kv.onnx",
            "noise_filter": "noise_filter_500kv.onnx",
            "spacer_detection": "spacer_500kv.onnx",  # 500kV特有
        },
        "capacitor": {
            "unit_detection": "capacitor_500kv.onnx",
            "intrusion_detection": "intrusion_500kv.onnx",
        },
        "meter": {
            "keypoint_detection": "meter_keypoint_500kv.onnx",
            "ocr": "meter_ocr_500kv.onnx",
            "density_relay": "density_relay_500kv.onnx",  # 500kV特有
        },
    },
)


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
        self.current_level: Optional[VoltageLevel] = None
        self.equipment_config: Optional[EquipmentConfig] = None
        self.model_library: Optional[ModelLibrary] = None
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config and "current_voltage_level" in config:
                    self.set_voltage_level(config["current_voltage_level"])
    
    def _save_config(self):
        """保存配置文件"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "current_voltage_level": self.current_level.value if self.current_level else None,
            "last_updated": str(Path.cwd()),
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def set_voltage_level(self, level: str) -> bool:
        """
        设置当前电压等级
        
        Args:
            level: "220kV" 或 "500kV"
        
        Returns:
            是否设置成功
        """
        try:
            if level == "220kV" or level == VoltageLevel.KV_220:
                self.current_level = VoltageLevel.KV_220
                self.equipment_config = CONFIG_220KV
                self.model_library = MODEL_LIBRARY_220KV
            elif level == "500kV" or level == VoltageLevel.KV_500:
                self.current_level = VoltageLevel.KV_500
                self.equipment_config = CONFIG_500KV
                self.model_library = MODEL_LIBRARY_500KV
            else:
                logger.error(f"不支持的电压等级: {level}")
                return False
            
            self._save_config()
            logger.info(f"电压等级已切换至: {self.current_level.value}")
            return True
            
        except Exception as e:
            logger.error(f"设置电压等级失败: {e}")
            return False
    
    def get_current_level(self) -> Optional[str]:
        """获取当前电压等级"""
        return self.current_level.value if self.current_level else None
    
    def get_equipment_config(self, equipment_type: str) -> Dict[str, Any]:
        """
        获取设备配置
        
        Args:
            equipment_type: transformer, switch, busbar, capacitor, meter
        
        Returns:
            设备配置字典
        """
        if not self.equipment_config:
            logger.warning("未设置电压等级，返回空配置")
            return {}
        
        config = getattr(self.equipment_config, equipment_type, {})
        return config if isinstance(config, dict) else {}
    
    def get_model_path(self, equipment_type: str, model_name: str) -> Optional[str]:
        """
        获取模型文件路径
        
        Args:
            equipment_type: 设备类型
            model_name: 模型名称
        
        Returns:
            模型文件完整路径
        """
        if not self.model_library:
            logger.warning("未设置电压等级，无法获取模型路径")
            return None
        
        models = self.model_library.models.get(equipment_type, {})
        model_file = models.get(model_name)
        
        if model_file:
            return str(Path(self.model_library.base_path) / equipment_type / model_file)
        
        return None
    
    def get_all_model_paths(self) -> Dict[str, Dict[str, str]]:
        """获取所有模型路径"""
        if not self.model_library:
            return {}
        
        result = {}
        for equipment_type, models in self.model_library.models.items():
            result[equipment_type] = {}
            for model_name, model_file in models.items():
                result[equipment_type][model_name] = str(
                    Path(self.model_library.base_path) / equipment_type / model_file
                )
        
        return result
    
    def get_detection_classes(self, equipment_type: str) -> List[str]:
        """获取检测类别列表"""
        config = self.get_equipment_config(equipment_type)
        return config.get("detection_classes", [])
    
    def get_thermal_thresholds(self) -> Dict[str, int]:
        """获取热成像阈值"""
        config = self.get_equipment_config("transformer")
        return config.get("thermal_threshold_celsius", {})
    
    def get_angle_reference(self, switch_type: str) -> Dict[str, float]:
        """获取开关角度参考值"""
        config = self.get_equipment_config("switch")
        return config.get("angle_reference", {}).get(switch_type, {})
    
    def export_config(self, output_path: str) -> str:
        """
        导出当前配置
        
        Args:
            output_path: 输出文件路径
        
        Returns:
            导出的文件路径
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "voltage_level": self.current_level.value if self.current_level else None,
            "equipment_config": asdict(self.equipment_config) if self.equipment_config else {},
            "model_library": asdict(self.model_library) if self.model_library else {},
        }
        
        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置已导出: {output}")
        return str(output)


# =============================================================================
# 适配层接口 (供插件使用)
# =============================================================================
class VoltageAdaptedPlugin:
    """
    电压等级适配的插件基类
    
    所有插件可继承此类以获得电压等级适配能力
    """
    
    def __init__(self, adapter: VoltageAdapterManager):
        self.adapter = adapter
        self._validate_config()
    
    def _validate_config(self):
        """验证配置"""
        if not self.adapter.current_level:
            logger.warning("电压等级未设置，插件可能无法正常工作")
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置 (子类重写)"""
        raise NotImplementedError
    
    def get_detection_config(self) -> Dict[str, Any]:
        """获取检测配置 (子类重写)"""
        raise NotImplementedError


# =============================================================================
# Web API 接口
# =============================================================================
def create_voltage_api_routes(app, adapter: VoltageAdapterManager):
    """
    创建电压等级管理的 API 路由
    
    供 FastAPI 或 Flask 使用
    """
    
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel
        
        router = APIRouter(prefix="/api/voltage", tags=["voltage"])
        
        class VoltageRequest(BaseModel):
            level: str
        
        @router.get("/current")
        async def get_current_level():
            """获取当前电压等级"""
            return {
                "success": True,
                "voltage_level": adapter.get_current_level(),
            }
        
        @router.post("/set")
        async def set_voltage_level(request: VoltageRequest):
            """设置电压等级"""
            if request.level not in ["220kV", "500kV"]:
                raise HTTPException(status_code=400, detail="不支持的电压等级")
            
            success = adapter.set_voltage_level(request.level)
            return {
                "success": success,
                "voltage_level": adapter.get_current_level(),
            }
        
        @router.get("/config/{equipment_type}")
        async def get_equipment_config(equipment_type: str):
            """获取设备配置"""
            config = adapter.get_equipment_config(equipment_type)
            return {
                "success": True,
                "equipment_type": equipment_type,
                "config": config,
            }
        
        @router.get("/models")
        async def get_all_models():
            """获取所有模型路径"""
            return {
                "success": True,
                "models": adapter.get_all_model_paths(),
            }
        
        @router.get("/classes/{equipment_type}")
        async def get_detection_classes(equipment_type: str):
            """获取检测类别"""
            classes = adapter.get_detection_classes(equipment_type)
            return {
                "success": True,
                "equipment_type": equipment_type,
                "classes": classes,
            }
        
        # 注册路由
        app.include_router(router)
        logger.info("电压等级 API 路由已注册")
        
        return router
        
    except ImportError:
        logger.warning("FastAPI 未安装，跳过 API 路由创建")
        return None


# =============================================================================
# 命令行工具
# =============================================================================
def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="电压等级适配管理工具")
    parser.add_argument("--set", type=str, choices=["220kV", "500kV"],
                       help="设置电压等级")
    parser.add_argument("--show", action="store_true", help="显示当前配置")
    parser.add_argument("--export", type=str, help="导出配置到文件")
    parser.add_argument("--equipment", type=str, 
                       choices=["transformer", "switch", "busbar", "capacitor", "meter"],
                       help="查看指定设备配置")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("破夜绘明激光监测平台 - 电压等级适配管理")
    print("=" * 60)
    
    adapter = VoltageAdapterManager()
    
    if args.set:
        adapter.set_voltage_level(args.set)
        print(f"\n✓ 电压等级已设置为: {args.set}")
    
    if args.show or not any([args.set, args.export, args.equipment]):
        print(f"\n当前电压等级: {adapter.get_current_level() or '未设置'}")
        
        if adapter.current_level:
            print("\n--- 设备配置概览 ---")
            for equipment in ["transformer", "switch", "busbar", "capacitor", "meter"]:
                classes = adapter.get_detection_classes(equipment)
                print(f"  {equipment}: {len(classes)} 个检测类别")
            
            print("\n--- 模型配置 ---")
            models = adapter.get_all_model_paths()
            for equipment, model_dict in models.items():
                print(f"  {equipment}:")
                for name, path in model_dict.items():
                    print(f"    - {name}: {path}")
    
    if args.equipment:
        config = adapter.get_equipment_config(args.equipment)
        print(f"\n{args.equipment.upper()} 配置:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
    
    if args.export:
        adapter.export_config(args.export)
        print(f"\n✓ 配置已导出到: {args.export}")


if __name__ == "__main__":
    main()
