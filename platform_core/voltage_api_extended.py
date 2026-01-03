#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变电站电压等级管理 API 路由
==========================================

提供完整的 RESTful API 接口用于：
1. 电压等级查询与切换
2. 设备配置获取
3. 模型状态检查
4. 插件功能管理

使用方法:
    from platform_core.voltage_api_extended import create_voltage_router, integrate_voltage_routes

    # FastAPI
    app = FastAPI()
    integrate_voltage_routes(app)

    # 或者单独获取路由
    router = create_voltage_router()
    app.include_router(router)

作者: 破夜绘明团队
日期: 2025
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .voltage_adapter_extended import (
    VoltageAdapterManager,
    VoltageCategory,
    get_all_voltage_categories,
    VOLTAGE_CONFIGS,
    MODEL_LIBRARIES,
    PLUGIN_CAPABILITIES
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic 模型定义
# =============================================================================
if FASTAPI_AVAILABLE:
    
    class VoltageSetRequest(BaseModel):
        """设置电压等级请求"""
        level: str = Field(..., description="电压等级，如 '500kV_AC', '220kV', '35kV' 等")
    
    
    class VoltageResponse(BaseModel):
        """电压等级响应"""
        success: bool
        voltage_level: Optional[str] = None
        category: Optional[str] = None
        message: str = ""
    
    
    class ConfigResponse(BaseModel):
        """设备配置响应"""
        success: bool
        equipment_type: str
        config: Dict[str, Any]
    
    
    class ModelStatusResponse(BaseModel):
        """模型状态响应"""
        success: bool
        voltage_level: str
        model_status: Dict[str, Dict[str, bool]]
        ready_count: int
        total_count: int
    
    
    class PluginListResponse(BaseModel):
        """插件列表响应"""
        success: bool
        voltage_level: str
        plugins: List[Dict[str, Any]]
    
    
    class VoltageCategoryResponse(BaseModel):
        """电压分类响应"""
        category: str
        description: str
        levels: List[str]
    
    
    class VoltageInfoResponse(BaseModel):
        """完整电压等级信息响应"""
        success: bool
        info: Dict[str, Any]


# =============================================================================
# 全局管理器实例
# =============================================================================
voltage_manager = VoltageAdapterManager()


# =============================================================================
# API 路由创建
# =============================================================================
def create_voltage_router() -> "APIRouter":
    """创建电压等级管理路由"""
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI 未安装，无法创建路由")
    
    router = APIRouter(prefix="/api/voltage", tags=["电压等级管理"])
    
    # -------------------------------------------------------------------------
    # 基础查询接口
    # -------------------------------------------------------------------------
    @router.get("/current", response_model=VoltageResponse)
    async def get_current_voltage():
        """获取当前电压等级"""
        level = voltage_manager.get_current_level()
        category = voltage_manager.get_current_category()
        return VoltageResponse(
            success=True,
            voltage_level=level,
            category=category,
            message="获取成功" if level else "未设置电压等级"
        )
    
    @router.post("/set", response_model=VoltageResponse)
    async def set_voltage(request: VoltageSetRequest):
        """设置电压等级"""
        # 验证电压等级
        if request.level not in VOLTAGE_CONFIGS:
            # 尝试标准化
            normalized = voltage_manager._normalize_voltage_level(request.level)
            if normalized not in VOLTAGE_CONFIGS:
                available = list(VOLTAGE_CONFIGS.keys())
                raise HTTPException(
                    status_code=400, 
                    detail=f"不支持的电压等级: {request.level}。可用等级: {available}"
                )
        
        success = voltage_manager.set_voltage_level(request.level)
        
        if success:
            return VoltageResponse(
                success=True,
                voltage_level=voltage_manager.get_current_level(),
                category=voltage_manager.get_current_category(),
                message=f"已成功切换到 {voltage_manager.get_current_level()}"
            )
        else:
            raise HTTPException(status_code=500, detail="切换电压等级失败")
    
    @router.get("/categories")
    async def get_voltage_categories():
        """获取所有电压分类"""
        categories = get_all_voltage_categories()
        return {
            "success": True,
            "categories": categories
        }
    
    @router.get("/available")
    async def get_available_levels():
        """获取所有可用的电压等级"""
        return {
            "success": True,
            "levels": voltage_manager.get_available_voltage_levels()
        }
    
    # -------------------------------------------------------------------------
    # 设备配置接口
    # -------------------------------------------------------------------------
    @router.get("/config/{equipment_type}", response_model=ConfigResponse)
    async def get_equipment_config(
        equipment_type: str,
        level: Optional[str] = Query(None, description="指定电压等级，不填则使用当前等级")
    ):
        """获取设备配置"""
        valid_types = ["transformer", "switch", "busbar", "capacitor", "meter", "dc_system", "gis"]
        if equipment_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"无效的设备类型。可用类型: {valid_types}"
            )
        
        # 如果指定了电压等级，临时使用该等级的配置
        if level:
            temp_manager = VoltageAdapterManager()
            if not temp_manager.set_voltage_level(level):
                raise HTTPException(status_code=400, detail=f"不支持的电压等级: {level}")
            config = temp_manager.get_equipment_config(equipment_type)
        else:
            config = voltage_manager.get_equipment_config(equipment_type)
        
        return ConfigResponse(
            success=True,
            equipment_type=equipment_type,
            config=config
        )
    
    @router.get("/config")
    async def get_all_config():
        """获取所有设备配置"""
        config = voltage_manager.get_all_equipment_config()
        return {
            "success": True,
            "voltage_level": voltage_manager.get_current_level(),
            "config": config
        }
    
    # -------------------------------------------------------------------------
    # 模型相关接口
    # -------------------------------------------------------------------------
    @router.get("/models")
    async def get_all_models():
        """获取所有模型路径"""
        return {
            "success": True,
            "voltage_level": voltage_manager.get_current_level(),
            "models": voltage_manager.get_all_model_paths()
        }
    
    @router.get("/model-status", response_model=ModelStatusResponse)
    async def get_model_status():
        """检查模型文件状态"""
        status = voltage_manager.check_model_status()
        
        # 统计就绪数量
        ready_count = 0
        total_count = 0
        for equipment_models in status.values():
            for is_ready in equipment_models.values():
                total_count += 1
                if is_ready:
                    ready_count += 1
        
        return ModelStatusResponse(
            success=True,
            voltage_level=voltage_manager.get_current_level() or "未设置",
            model_status=status,
            ready_count=ready_count,
            total_count=total_count
        )
    
    @router.get("/model/{equipment_type}/{model_name}")
    async def get_model_path(equipment_type: str, model_name: str):
        """获取指定模型路径"""
        path = voltage_manager.get_model_path(equipment_type, model_name)
        if path:
            exists = Path(path).exists()
            return {
                "success": True,
                "model_path": path,
                "exists": exists
            }
        else:
            raise HTTPException(status_code=404, detail="模型未找到")
    
    # -------------------------------------------------------------------------
    # 检测类别接口
    # -------------------------------------------------------------------------
    @router.get("/detection-classes/{equipment_type}")
    async def get_detection_classes(equipment_type: str):
        """获取检测类别"""
        classes = voltage_manager.get_detection_classes(equipment_type)
        return {
            "success": True,
            "equipment_type": equipment_type,
            "detection_classes": classes
        }
    
    @router.get("/detection-classes")
    async def get_all_detection_classes():
        """获取所有检测类别"""
        all_classes = {}
        for equipment_type in ["transformer", "switch", "busbar", "capacitor", "meter"]:
            all_classes[equipment_type] = voltage_manager.get_detection_classes(equipment_type)
        return {
            "success": True,
            "voltage_level": voltage_manager.get_current_level(),
            "detection_classes": all_classes
        }
    
    # -------------------------------------------------------------------------
    # 阈值和参数接口
    # -------------------------------------------------------------------------
    @router.get("/thermal-thresholds")
    async def get_thermal_thresholds():
        """获取热成像阈值"""
        thresholds = voltage_manager.get_thermal_thresholds()
        return {
            "success": True,
            "voltage_level": voltage_manager.get_current_level(),
            "thresholds": thresholds
        }
    
    @router.get("/angle-reference/{switch_type}")
    async def get_angle_reference(switch_type: str):
        """获取开关角度参考值"""
        valid_types = ["breaker", "isolator", "grounding"]
        if switch_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"无效的开关类型。可用类型: {valid_types}"
            )
        
        angle_ref = voltage_manager.get_angle_reference(switch_type)
        return {
            "success": True,
            "switch_type": switch_type,
            "angle_reference": angle_ref
        }
    
    # -------------------------------------------------------------------------
    # 插件功能接口
    # -------------------------------------------------------------------------
    @router.get("/plugins", response_model=PluginListResponse)
    async def get_supported_plugins():
        """获取当前电压等级支持的插件功能"""
        plugins = voltage_manager.get_supported_plugins()
        return PluginListResponse(
            success=True,
            voltage_level=voltage_manager.get_current_level() or "未设置",
            plugins=plugins
        )
    
    @router.get("/plugins/all")
    async def get_all_plugins():
        """获取所有可用的插件功能"""
        all_plugins = []
        for plugin_id, capability in PLUGIN_CAPABILITIES.items():
            all_plugins.append({
                "id": plugin_id,
                "name": capability.name,
                "description": capability.description,
                "supported_voltage_levels": capability.supported_voltage_levels,
                "detection_types": capability.detection_types,
                "requires_models": capability.requires_models
            })
        return {
            "success": True,
            "plugins": all_plugins
        }
    
    @router.get("/plugins/{plugin_id}")
    async def get_plugin_info(plugin_id: str):
        """获取指定插件的详细信息"""
        if plugin_id not in PLUGIN_CAPABILITIES:
            raise HTTPException(status_code=404, detail=f"插件未找到: {plugin_id}")
        
        capability = PLUGIN_CAPABILITIES[plugin_id]
        current_level = voltage_manager.get_current_level()
        
        # 检查是否支持当前电压等级
        supported = (
            capability.supported_voltage_levels == ["all"] or
            current_level in capability.supported_voltage_levels
        )
        
        return {
            "success": True,
            "plugin": {
                "id": plugin_id,
                "name": capability.name,
                "description": capability.description,
                "supported_voltage_levels": capability.supported_voltage_levels,
                "detection_types": capability.detection_types,
                "requires_models": capability.requires_models,
                "supported_at_current_level": supported
            }
        }
    
    # -------------------------------------------------------------------------
    # 特殊功能接口
    # -------------------------------------------------------------------------
    @router.get("/special-features")
    async def get_special_features():
        """获取当前电压等级的特殊功能特性"""
        features = voltage_manager.get_special_features()
        return {
            "success": True,
            "voltage_level": voltage_manager.get_current_level(),
            "special_features": features
        }
    
    # -------------------------------------------------------------------------
    # 综合信息接口
    # -------------------------------------------------------------------------
    @router.get("/info", response_model=VoltageInfoResponse)
    async def get_voltage_info():
        """获取当前电压等级的完整信息"""
        info = voltage_manager.get_voltage_level_info()
        return VoltageInfoResponse(
            success=True,
            info=info
        )
    
    @router.get("/info/{level}")
    async def get_specific_voltage_info(level: str):
        """获取指定电压等级的完整信息"""
        temp_manager = VoltageAdapterManager()
        if not temp_manager.set_voltage_level(level):
            raise HTTPException(status_code=400, detail=f"不支持的电压等级: {level}")
        
        info = temp_manager.get_voltage_level_info()
        return {
            "success": True,
            "info": info
        }
    
    # -------------------------------------------------------------------------
    # 比较接口
    # -------------------------------------------------------------------------
    @router.get("/compare")
    async def compare_voltage_levels(
        level1: str = Query(..., description="第一个电压等级"),
        level2: str = Query(..., description="第二个电压等级")
    ):
        """比较两个电压等级的配置差异"""
        manager1 = VoltageAdapterManager()
        manager2 = VoltageAdapterManager()
        
        if not manager1.set_voltage_level(level1):
            raise HTTPException(status_code=400, detail=f"不支持的电压等级: {level1}")
        if not manager2.set_voltage_level(level2):
            raise HTTPException(status_code=400, detail=f"不支持的电压等级: {level2}")
        
        comparison = {
            "level1": {
                "voltage_level": manager1.get_current_level(),
                "category": manager1.get_current_category(),
                "thermal_thresholds": manager1.get_thermal_thresholds(),
                "plugin_count": len(manager1.get_supported_plugins()),
                "special_features": manager1.get_special_features()
            },
            "level2": {
                "voltage_level": manager2.get_current_level(),
                "category": manager2.get_current_category(),
                "thermal_thresholds": manager2.get_thermal_thresholds(),
                "plugin_count": len(manager2.get_supported_plugins()),
                "special_features": manager2.get_special_features()
            },
            "differences": {
                "category_same": manager1.get_current_category() == manager2.get_current_category(),
                "thermal_threshold_diff": {
                    k: manager1.get_thermal_thresholds().get(k, 0) - manager2.get_thermal_thresholds().get(k, 0)
                    for k in ["normal", "warning", "alarm"]
                }
            }
        }
        
        return {
            "success": True,
            "comparison": comparison
        }
    
    return router


# =============================================================================
# 集成函数
# =============================================================================
def integrate_voltage_routes(app: "FastAPI"):
    """
    将电压等级管理路由集成到 FastAPI 应用
    
    Args:
        app: FastAPI 应用实例
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI 未安装，无法集成路由")
    
    router = create_voltage_router()
    app.include_router(router)
    logger.info("电压等级管理 API 路由已集成")


# =============================================================================
# Flask 支持 (可选)
# =============================================================================
def create_flask_blueprint():
    """
    创建 Flask Blueprint (如果使用 Flask)
    """
    try:
        from flask import Blueprint, jsonify, request
    except ImportError:
        raise ImportError("Flask 未安装")
    
    bp = Blueprint('voltage', __name__, url_prefix='/api/voltage')
    
    @bp.route('/current', methods=['GET'])
    def get_current():
        return jsonify({
            "success": True,
            "voltage_level": voltage_manager.get_current_level(),
            "category": voltage_manager.get_current_category()
        })
    
    @bp.route('/set', methods=['POST'])
    def set_voltage():
        data = request.get_json()
        level = data.get('level')
        success = voltage_manager.set_voltage_level(level)
        return jsonify({
            "success": success,
            "voltage_level": voltage_manager.get_current_level(),
            "category": voltage_manager.get_current_category()
        })
    
    @bp.route('/categories', methods=['GET'])
    def get_categories():
        return jsonify({
            "success": True,
            "categories": get_all_voltage_categories()
        })
    
    # ... 更多路由可根据需要添加
    
    return bp


# =============================================================================
# API 文档生成
# =============================================================================
API_DOCUMENTATION = """
# 变电站电压等级管理 API 文档

## 基础端点

### GET /api/voltage/current
获取当前电压等级

### POST /api/voltage/set
设置电压等级
- Body: {"level": "500kV_AC"}

### GET /api/voltage/categories
获取所有电压分类

### GET /api/voltage/available
获取所有可用的电压等级

## 设备配置端点

### GET /api/voltage/config/{equipment_type}
获取指定设备类型的配置
- equipment_type: transformer, switch, busbar, capacitor, meter, dc_system, gis

### GET /api/voltage/config
获取所有设备配置

## 模型端点

### GET /api/voltage/models
获取所有模型路径

### GET /api/voltage/model-status
检查模型文件状态

### GET /api/voltage/model/{equipment_type}/{model_name}
获取指定模型路径

## 检测类别端点

### GET /api/voltage/detection-classes/{equipment_type}
获取检测类别

### GET /api/voltage/detection-classes
获取所有检测类别

## 阈值和参数端点

### GET /api/voltage/thermal-thresholds
获取热成像阈值

### GET /api/voltage/angle-reference/{switch_type}
获取开关角度参考值

## 插件端点

### GET /api/voltage/plugins
获取当前电压等级支持的插件

### GET /api/voltage/plugins/all
获取所有可用插件

### GET /api/voltage/plugins/{plugin_id}
获取指定插件详情

## 综合信息端点

### GET /api/voltage/info
获取当前电压等级完整信息

### GET /api/voltage/info/{level}
获取指定电压等级完整信息

### GET /api/voltage/compare?level1=xxx&level2=xxx
比较两个电压等级
"""


if __name__ == "__main__":
    print(API_DOCUMENTATION)
