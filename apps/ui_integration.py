#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI服务器集成模块
==========================================

将高级插件集成到现有UI服务器

使用方法:
    在 ui_server.py 的 create_app() 函数中调用:

    from apps.ui_integration import integrate_all_advanced_features
    integrate_all_advanced_features(app, templates)

作者: 破夜绘明团队
日期: 2025
"""

import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)


def integrate_advanced_plugins_routes(app: "FastAPI"):
    """
    集成高级插件API路由
    
    Args:
        app: FastAPI应用实例
    """
    try:
        from apps.advanced_plugins_api import integrate_advanced_plugins_routes as _integrate
        _integrate(app)
        logger.info("✓ 高级插件API路由已集成")
        return True
    except ImportError as e:
        logger.warning(f"✗ 高级插件API导入失败: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ 高级插件API集成失败: {e}")
        return False


def register_advanced_pages(app: "FastAPI", templates: "Jinja2Templates"):
    """
    注册高级插件页面路由

    Args:
        app: FastAPI应用实例
        templates: Jinja2模板引擎
    """

    @app.get("/advanced-plugins", response_class=HTMLResponse)
    async def advanced_plugins_page(request: Request):
        """高级监测插件页面"""
        return templates.TemplateResponse(
            "pages/advanced_plugins.html",
            {
                "request": request,
                "active_tab": "advanced",
                "active_plugin": None,
                "version": "2.0.0",
            },
        )

    @app.get("/acoustic", response_class=HTMLResponse)
    async def acoustic_page(request: Request):
        """声学监测页面（重定向到高级插件页面）"""
        return templates.TemplateResponse(
            "pages/advanced_plugins.html",
            {
                "request": request,
                "active_tab": "advanced",
                "active_plugin": "acoustic",
                "version": "2.0.0",
            },
        )
    
    @app.get("/gas-detection", response_class=HTMLResponse)
    async def gas_detection_page(request: Request):
        """气体检测页面"""
        return templates.TemplateResponse(
            "pages/advanced_plugins.html",
            {
                "request": request,
                "active_tab": "advanced",
                "active_plugin": "gas",
                "version": "2.0.0",
            },
        )
    
    @app.get("/hyperspectral", response_class=HTMLResponse)
    async def hyperspectral_page(request: Request):
        """高光谱检测页面"""
        return templates.TemplateResponse(
            "pages/advanced_plugins.html",
            {
                "request": request,
                "active_tab": "advanced",
                "active_plugin": "hyperspectral",
                "version": "2.0.0",
            },
        )
    
    @app.get("/slam", response_class=HTMLResponse)
    async def slam_page(request: Request):
        """SLAM建图页面"""
        return templates.TemplateResponse(
            "pages/advanced_plugins.html",
            {
                "request": request,
                "active_tab": "advanced",
                "active_plugin": "slam",
                "version": "2.0.0",
            },
        )
    
    @app.get("/multimodal-fusion", response_class=HTMLResponse)
    async def multimodal_fusion_page(request: Request):
        """多模态融合页面"""
        return templates.TemplateResponse(
            "pages/advanced_plugins.html",
            {
                "request": request,
                "active_tab": "advanced",
                "active_plugin": "fusion",
                "version": "2.0.0",
            },
        )
    
    logger.info("✓ 高级插件页面路由已注册")


def integrate_all_advanced_features(app: "FastAPI", templates: "Jinja2Templates"):
    """
    集成所有高级功能
    
    Args:
        app: FastAPI应用实例
        templates: Jinja2模板引擎
    """
    # 1. 集成API路由
    integrate_advanced_plugins_routes(app)
    
    # 2. 注册页面路由
    register_advanced_pages(app, templates)
    
    logger.info("=" * 50)
    logger.info("高级功能集成完成")
    logger.info("=" * 50)


# =============================================================================
# 扩展的模块配置
# =============================================================================
EXTENDED_MODULES = {
    # 原有模块
    "transformer": {
        "name": "主变自主巡视",
        "icon": "box",
        "description": "主变压器本体及附属设施的外观缺陷识别、状态识别和热成像分析",
        "plugin_id": "transformer_inspection",
    },
    "switch": {
        "name": "开关间隔巡视",
        "icon": "toggle-on",
        "description": "断路器、隔离开关、接地开关的分合位状态识别和逻辑校验",
        "plugin_id": "switch_inspection",
    },
    "busbar": {
        "name": "母线自主巡视",
        "icon": "diagram-3",
        "description": "绝缘子串、金具、导线连接点的远距小目标检测",
        "plugin_id": "busbar_inspection",
    },
    "capacitor": {
        "name": "电容器巡视",
        "icon": "battery-charging",
        "description": "电容器组的结构完整性检测和区域入侵检测",
        "plugin_id": "capacitor_inspection",
    },
    "meter": {
        "name": "表计读数",
        "icon": "speedometer2",
        "description": "站内全类型模拟表计和数字表计的任意角度读数识别",
        "plugin_id": "meter_reading",
    },
    # 高级模块
    "acoustic": {
        "name": "声学监测",
        "icon": "soundwave",
        "description": "局部放电检测、超声波分析、声音异常识别",
        "plugin_id": "acoustic_monitoring",
        "advanced": True,
    },
    "gas": {
        "name": "气体检测",
        "icon": "cloud-haze2",
        "description": "SF6监测、油中溶解气体分析(DGA)、趋势预测",
        "plugin_id": "gas_detection",
        "advanced": True,
    },
    "hyperspectral": {
        "name": "高光谱检测",
        "icon": "rainbow",
        "description": "高光谱图像分析、缺陷检测、材料识别",
        "plugin_id": "hyperspectral_detection",
        "advanced": True,
    },
    "slam": {
        "name": "SLAM建图",
        "icon": "map",
        "description": "激光雷达建图、定位导航、回环检测",
        "plugin_id": "slam_mapping",
        "advanced": True,
    },
    "fusion": {
        "name": "多模态融合",
        "icon": "diagram-3-fill",
        "description": "多传感器数据融合、综合故障诊断、智能决策",
        "plugin_id": "multimodal_fusion",
        "advanced": True,
    },
}


def get_extended_modules():
    """获取扩展模块配置"""
    return EXTENDED_MODULES
