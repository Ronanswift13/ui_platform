#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电压等级管理 API 路由
破夜绘明激光监测平台

功能:
1. 获取/设置当前电压等级
2. 获取设备配置
3. 获取模型路径和状态
4. 切换模型库

集成方式:
    在 apps/api_server.py 中:
    
    from platform_core.voltage_api import create_voltage_router
    
    def create_api_app():
        app = FastAPI(...)
        # ... 其他路由 ...
        app.include_router(create_voltage_router())
        return app
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# 请求/响应模型
# =============================================================================
class VoltageSetRequest(BaseModel):
    """电压等级设置请求"""
    level: str  # "220kV" 或 "500kV"


class VoltageResponse(BaseModel):
    """电压等级响应"""
    success: bool
    voltage_level: Optional[str] = None
    message: str = ""


class ConfigResponse(BaseModel):
    """配置响应"""
    success: bool
    equipment_type: str
    config: Dict[str, Any]


class ModelsResponse(BaseModel):
    """模型列表响应"""
    success: bool
    models: Dict[str, Dict[str, str]]


class ModelStatusResponse(BaseModel):
    """模型状态响应"""
    success: bool
    path: str
    exists: bool
    size_mb: float = 0


# =============================================================================
# 电压等级配置数据
# =============================================================================
# 220kV 配置
CONFIG_220KV = {
    "transformer": {
        "typical_models": ["SZ11-50000/220", "SFSZ11-120000/220", "SFSZ9-180000/220"],
        "oil_tank_diameter_range": [2.5, 4.0],
        "cooling_type": ["ONAN", "ONAF", "OFAF"],
        "detection_classes": [
            "oil_leak", "rust_corrosion", "surface_damage", "foreign_object",
            "silica_gel_blue", "silica_gel_pink", "oil_level_normal", "oil_level_abnormal"
        ],
        "thermal_threshold_celsius": {"normal": 60, "warning": 75, "alarm": 85},
    },
    "switch": {
        "breaker_types": ["SF6断路器", "真空断路器"],
        "isolator_types": ["GW4-220", "GW5-220D"],
        "detection_classes": [
            "breaker_open", "breaker_closed", "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed", "indicator_red", "indicator_green"
        ],
        "angle_reference": {
            "breaker": {"open_deg": -55, "closed_deg": 35},
            "isolator": {"open_deg": -65, "closed_deg": 25},
            "grounding": {"open_deg": -75, "closed_deg": 15},
        },
    },
    "busbar": {
        "typical_specs": {"conductor_type": "LGJ-400/35", "busbar_height_m": 8, "phase_spacing_m": 4.5},
        "detection_classes": [
            "insulator_crack", "insulator_dirty", "fitting_loose", "fitting_rust",
            "wire_damage", "foreign_object", "bird", "pin_missing"
        ],
        "small_target_min_px": 15,
    },
    "capacitor": {
        "typical_models": ["CW1-220", "TBB10-6600"],
        "bank_capacity_mvar": [30, 60, 90],
        "detection_classes": [
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle"
        ],
    },
    "meter": {
        "types": ["SF6压力表", "油温表", "油位计"],
        "sf6_pressure_range_mpa": [0.4, 0.6],
        "oil_temp_range_celsius": [-20, 100],
        "detection_classes": ["meter", "pointer", "scale", "digital_display"],
    },
}

# 500kV 配置
CONFIG_500KV = {
    "transformer": {
        "typical_models": ["OSFPSZ-500000/500", "OSFPS-750000/500", "OSFPSZ-1000000/500"],
        "oil_tank_diameter_range": [4.5, 7.0],
        "cooling_type": ["OFAF", "ODAF"],
        "detection_classes": [
            "oil_leak", "rust_corrosion", "surface_damage", "foreign_object",
            "silica_gel_blue", "silica_gel_pink", "oil_level_normal", "oil_level_abnormal",
            "bushing_crack", "porcelain_contamination"  # 500kV 特有
        ],
        "thermal_threshold_celsius": {"normal": 65, "warning": 80, "alarm": 95},
    },
    "switch": {
        "breaker_types": ["SF6断路器", "GIS组合电器"],
        "isolator_types": ["GW4-500", "GW5-500D", "GW6-500W"],
        "detection_classes": [
            "breaker_open", "breaker_closed", "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed", "indicator_red", "indicator_green",
            "gis_position"  # 500kV 特有
        ],
        "angle_reference": {
            "breaker": {"open_deg": -60, "closed_deg": 30},
            "isolator": {"open_deg": -70, "closed_deg": 20},
            "grounding": {"open_deg": -80, "closed_deg": 10},
        },
    },
    "busbar": {
        "typical_specs": {"conductor_type": "LGJ-630/45", "busbar_height_m": 15, "phase_spacing_m": 9.0},
        "detection_classes": [
            "insulator_crack", "insulator_dirty", "insulator_flashover",  # 500kV 特有
            "fitting_loose", "fitting_rust", "wire_damage", "foreign_object",
            "bird", "pin_missing", "spacer_damage"  # 500kV 特有
        ],
        "small_target_min_px": 20,
    },
    "capacitor": {
        "typical_models": ["CW1-500", "TBB35-12000"],
        "bank_capacity_mvar": [60, 120, 180],
        "detection_classes": [
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle", "fuse_blown"  # 500kV 特有
        ],
    },
    "meter": {
        "types": ["SF6压力表", "SF6密度继电器", "油温表", "油位计", "气体继电器"],
        "sf6_pressure_range_mpa": [0.45, 0.65],
        "oil_temp_range_celsius": [-20, 110],
        "detection_classes": ["meter", "pointer", "scale", "digital_display", "density_relay"],
    },
}

# 模型库配置
MODEL_LIBRARY_220KV = {
    "transformer": {
        "defect_detection": "models/220kV/transformer/transformer_defect_220kv.onnx",
        "oil_segmentation": "models/220kV/transformer/oil_seg_220kv.onnx",
        "silica_classifier": "models/220kV/transformer/silica_220kv.onnx",
        "thermal_anomaly": "models/220kV/transformer/thermal_220kv.onnx",
    },
    "switch": {
        "state_detection": "models/220kV/switch/switch_state_220kv.onnx",
        "indicator_ocr": "models/220kV/switch/indicator_ocr_220kv.onnx",
    },
    "busbar": {
        "defect_detection": "models/220kV/busbar/busbar_defect_220kv.onnx",
        "noise_filter": "models/220kV/busbar/noise_filter_220kv.onnx",
    },
    "capacitor": {
        "unit_detection": "models/220kV/capacitor/capacitor_220kv.onnx",
        "intrusion_detection": "models/220kV/capacitor/intrusion_220kv.onnx",
    },
    "meter": {
        "keypoint_detection": "models/220kV/meter/meter_keypoint_220kv.onnx",
        "ocr": "models/220kV/meter/meter_ocr_220kv.onnx",
    },
}

MODEL_LIBRARY_500KV = {
    "transformer": {
        "defect_detection": "models/500kV/transformer/transformer_defect_500kv.onnx",
        "oil_segmentation": "models/500kV/transformer/oil_seg_500kv.onnx",
        "silica_classifier": "models/500kV/transformer/silica_500kv.onnx",
        "thermal_anomaly": "models/500kV/transformer/thermal_500kv.onnx",
        "bushing_detection": "models/500kV/transformer/bushing_500kv.onnx",
    },
    "switch": {
        "state_detection": "models/500kV/switch/switch_state_500kv.onnx",
        "indicator_ocr": "models/500kV/switch/indicator_ocr_500kv.onnx",
        "gis_position": "models/500kV/switch/gis_position_500kv.onnx",
    },
    "busbar": {
        "defect_detection": "models/500kV/busbar/busbar_defect_500kv.onnx",
        "noise_filter": "models/500kV/busbar/noise_filter_500kv.onnx",
        "spacer_detection": "models/500kV/busbar/spacer_500kv.onnx",
    },
    "capacitor": {
        "unit_detection": "models/500kV/capacitor/capacitor_500kv.onnx",
        "intrusion_detection": "models/500kV/capacitor/intrusion_500kv.onnx",
    },
    "meter": {
        "keypoint_detection": "models/500kV/meter/meter_keypoint_500kv.onnx",
        "ocr": "models/500kV/meter/meter_ocr_500kv.onnx",
        "density_relay": "models/500kV/meter/density_relay_500kv.onnx",
    },
}


# =============================================================================
# 电压等级管理器（单例）
# =============================================================================
class VoltageManager:
    """电压等级管理器"""
    
    _instance = None
    _config_file = "configs/voltage_config.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._current_level: Optional[str] = None
        self._load_config()
        self._initialized = True
    
    def _load_config(self):
        """从配置文件加载当前电压等级"""
        config_path = Path(self._config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self._current_level = config.get("current_voltage_level")
                    logger.info(f"加载电压等级配置: {self._current_level}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
    
    def _save_config(self):
        """保存当前电压等级到配置文件"""
        config_path = Path(self._config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "current_voltage_level": self._current_level,
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存电压等级配置: {self._current_level}")
    
    def get_current_level(self) -> Optional[str]:
        """获取当前电压等级"""
        return self._current_level
    
    def set_voltage_level(self, level: str) -> bool:
        """设置电压等级"""
        if level not in ["220kV", "500kV"]:
            logger.error(f"不支持的电压等级: {level}")
            return False
        
        old_level = self._current_level
        self._current_level = level
        self._save_config()
        
        # 触发模型库切换
        self._switch_model_library(old_level, level)
        
        logger.info(f"电压等级已切换: {old_level} -> {level}")
        return True
    
    def _switch_model_library(self, old_level: Optional[str], new_level: str):
        """切换模型库"""
        logger.info(f"切换模型库: {old_level} -> {new_level}")
        
        # TODO: 实现实际的模型卸载/加载逻辑
        # 1. 卸载旧模型
        # 2. 加载新模型
        # 3. 更新插件配置
        
        # 这里可以触发事件通知各插件更新配置
        self._notify_plugins_config_changed(new_level)
    
    def _notify_plugins_config_changed(self, level: str):
        """通知插件配置已更改"""
        # TODO: 实现插件通知机制
        logger.info(f"通知所有插件: 电压等级已切换至 {level}")
    
    def get_equipment_config(self, equipment_type: str) -> Dict[str, Any]:
        """获取设备配置"""
        if self._current_level == "220kV":
            return CONFIG_220KV.get(equipment_type, {})
        elif self._current_level == "500kV":
            return CONFIG_500KV.get(equipment_type, {})
        return {}
    
    def get_all_models(self) -> Dict[str, Dict[str, str]]:
        """获取所有模型路径"""
        if self._current_level == "220kV":
            return MODEL_LIBRARY_220KV
        elif self._current_level == "500kV":
            return MODEL_LIBRARY_500KV
        return {}
    
    def get_model_path(self, plugin: str, model_name: str) -> Optional[str]:
        """获取指定模型路径"""
        models = self.get_all_models()
        return models.get(plugin, {}).get(model_name)


# 创建全局管理器实例
voltage_manager = VoltageManager()


# =============================================================================
# API 路由
# =============================================================================
def create_voltage_router() -> APIRouter:
    """创建电压等级管理路由"""
    
    router = APIRouter(prefix="/api/voltage", tags=["电压等级管理"])
    
    @router.get("/current", response_model=VoltageResponse)
    async def get_current_voltage():
        """获取当前电压等级"""
        level = voltage_manager.get_current_level()
        return VoltageResponse(
            success=True,
            voltage_level=level,
            message="获取成功" if level else "未设置电压等级"
        )
    
    @router.post("/set", response_model=VoltageResponse)
    async def set_voltage(request: VoltageSetRequest):
        """设置电压等级"""
        if request.level not in ["220kV", "500kV"]:
            raise HTTPException(status_code=400, detail="不支持的电压等级，请选择 220kV 或 500kV")
        
        success = voltage_manager.set_voltage_level(request.level)
        
        if success:
            return VoltageResponse(
                success=True,
                voltage_level=request.level,
                message=f"已成功切换到 {request.level}"
            )
        else:
            raise HTTPException(status_code=500, detail="切换电压等级失败")
    
    @router.get("/config/{equipment_type}", response_model=ConfigResponse)
    async def get_equipment_config(
        equipment_type: str,
        level: Optional[str] = Query(None, description="指定电压等级，不填则使用当前等级")
    ):
        """获取设备配置"""
        if equipment_type not in ["transformer", "switch", "busbar", "capacitor", "meter"]:
            raise HTTPException(status_code=400, detail="无效的设备类型")
        
        # 如果指定了电压等级，临时使用该等级的配置
        if level:
            if level == "220kV":
                config = CONFIG_220KV.get(equipment_type, {})
            elif level == "500kV":
                config = CONFIG_500KV.get(equipment_type, {})
            else:
                raise HTTPException(status_code=400, detail="不支持的电压等级")
        else:
            config = voltage_manager.get_equipment_config(equipment_type)
        
        return ConfigResponse(
            success=True,
            equipment_type=equipment_type,
            config=config
        )
    
    @router.get("/models", response_model=ModelsResponse)
    async def get_all_models():
        """获取所有模型路径"""
        models = voltage_manager.get_all_models()
        return ModelsResponse(
            success=True,
            models=models
        )
    
    @router.get("/model-status", response_model=ModelStatusResponse)
    async def get_model_status(path: str = Query(..., description="模型文件路径")):
        """检查模型文件状态"""
        model_path = Path(path)
        exists = model_path.exists()
        size_mb = model_path.stat().st_size / (1024 * 1024) if exists else 0
        
        return ModelStatusResponse(
            success=True,
            path=path,
            exists=exists,
            size_mb=round(size_mb, 2)
        )
    
    @router.get("/detection-classes/{equipment_type}")
    async def get_detection_classes(equipment_type: str):
        """获取检测类别列表"""
        config = voltage_manager.get_equipment_config(equipment_type)
        classes = config.get("detection_classes", [])
        
        return {
            "success": True,
            "equipment_type": equipment_type,
            "classes": classes,
            "count": len(classes)
        }
    
    @router.get("/thermal-thresholds")
    async def get_thermal_thresholds():
        """获取热成像阈值"""
        config = voltage_manager.get_equipment_config("transformer")
        thresholds = config.get("thermal_threshold_celsius", {})
        
        return {
            "success": True,
            "voltage_level": voltage_manager.get_current_level(),
            "thresholds": thresholds
        }
    
    @router.get("/angle-reference/{switch_type}")
    async def get_angle_reference(switch_type: str):
        """获取开关角度参考值"""
        config = voltage_manager.get_equipment_config("switch")
        angle_ref = config.get("angle_reference", {}).get(switch_type, {})
        
        return {
            "success": True,
            "switch_type": switch_type,
            "angle_reference": angle_ref
        }
    
    return router


# =============================================================================
# 便捷集成函数
# =============================================================================
def integrate_voltage_routes(app):
    """
    将电压等级路由集成到 FastAPI 应用
    
    使用方法:
        from platform_core.voltage_api import integrate_voltage_routes
        
        app = FastAPI()
        integrate_voltage_routes(app)
    """
    router = create_voltage_router()
    app.include_router(router)
    logger.info("电压等级管理 API 路由已集成")


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    # 简单测试
    manager = VoltageManager()
    
    print("当前电压等级:", manager.get_current_level())
    
    print("\n设置为 500kV...")
    manager.set_voltage_level("500kV")
    
    print("当前电压等级:", manager.get_current_level())
    
    print("\n主变配置:")
    config = manager.get_equipment_config("transformer")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    print("\n模型列表:")
    models = manager.get_all_models()
    for plugin, model_dict in models.items():
        print(f"  {plugin}:")
        for name, path in model_dict.items():
            print(f"    - {name}: {path}")
