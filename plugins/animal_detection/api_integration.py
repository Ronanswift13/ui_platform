#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - API集成层
================================

将 AnimalDetectionPlugin 集成到现有 indoor_api.py 路由中
替换 IndoorDataGenerator.generate_animal_data() 的随机数据

用法:
    # 在 apps/indoor_api.py 中
    from plugins.animal_detection.api_integration import (
        create_animal_detection_router,
        integrate_animal_detection,
    )

    # 注册路由
    router.include_router(create_animal_detection_router())

    # 或替换现有数据生成器
    integrate_animal_detection(router)

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .plugin import AnimalDetectionPlugin

logger = logging.getLogger(__name__)


# =============================================================================
# 响应模型
# =============================================================================

class AnimalDetectionResponse(BaseModel):
    """动物检测API响应 (兼容前端 IndoorMonitorState.modules.animal)"""
    timestamp: int
    status: str              # online / alarm / warning
    detections: List[Dict]
    todayCount: int
    deterrentActive: bool
    # 扩展字段
    trace_id: Optional[str] = None
    risk_level: Optional[str] = None
    suggestion: Optional[str] = None
    simulation_mode: bool = False


class StatisticsResponse(BaseModel):
    """统计响应"""
    period_type: str
    period_label: str
    total_invasions: int
    class_distribution: Dict[str, int]
    deterrent_stats: Dict[str, Any]


# =============================================================================
# 插件单例管理
# =============================================================================

_plugin_instance: Optional[AnimalDetectionPlugin] = None


def get_plugin() -> AnimalDetectionPlugin:
    """获取或创建插件单例"""
    global _plugin_instance
    if _plugin_instance is None:
        _plugin_instance = AnimalDetectionPlugin()
        # 加载默认配置
        import json as _json
        from pathlib import Path
        config_path = Path(__file__).parent / "configs" / "default.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = _json.load(f)
        else:
            config = {"simulation": {"enabled": False}}
        _plugin_instance.init(config)
    return _plugin_instance


# =============================================================================
# 路由创建
# =============================================================================

def create_animal_detection_router(prefix: str = "/api/indoor/animal") -> APIRouter:
    """
    创建动物检测API路由

    端点:
    - GET  /api/indoor/animal          → 获取当前检测状态
    - GET  /api/indoor/animal/tracks   → 获取活跃跟踪
    - GET  /api/indoor/animal/stats    → 获取统计
    - GET  /api/indoor/animal/trend    → 获取趋势数据
    - GET  /api/indoor/animal/report   → 导出报告
    - GET  /api/indoor/animal/health   → 健康检查
    - GET  /api/indoor/animal/events   → 最近事件
    """
    animal_router = APIRouter(prefix=prefix, tags=["动物入侵检测"])

    @animal_router.get("")
    async def get_animal_data() -> Dict[str, Any]:
        """
        获取动物检测数据

        兼容前端 IndoorMonitorState 格式
        """
        plugin = get_plugin()

        last_event = plugin._last_event
        if last_event is None:
            return {
                "timestamp": int(time.time() * 1000),
                "status": "online",
                "detections": [],
                "todayCount": plugin._statistics.get_today_count() if plugin._statistics else 0,
                "deterrentActive": False,
                "simulation_mode": plugin._simulation_mode,
            }

        # 转换为前端兼容格式
        detections = []
        for i, det in enumerate(last_event.detections):
            d = {
                "id": i + 1,
                "type": det.animal_class,
                "confidence": round(det.confidence, 3),
                "bbox": det.bbox.to_dict() if det.bbox else {},
                "track_id": det.track_id,
                "stay_duration_s": round(det.stay_duration_s, 1),
                "is_thermal_validated": det.is_thermal_validated,
            }
            detections.append(d)

        status = "alarm" if last_event.total_detections > 0 else "online"

        return {
            "timestamp": int(last_event.timestamp * 1000),
            "status": status,
            "detections": detections,
            "todayCount": plugin._statistics.get_today_count() if plugin._statistics else 0,
            "deterrentActive": last_event.deterrent_active,
            "trace_id": last_event.trace_id,
            "risk_level": last_event.risk_level,
            "suggestion": last_event.suggestion,
            "simulation_mode": plugin._simulation_mode,
        }

    @animal_router.get("/tracks")
    async def get_active_tracks() -> Dict[str, Any]:
        """获取活跃跟踪轨迹"""
        plugin = get_plugin()
        return {
            "tracks": plugin.get_active_tracks(),
            "timestamp": int(time.time() * 1000),
        }

    @animal_router.get("/stats")
    async def get_statistics(
        period: str = Query("daily", enum=["daily", "weekly", "trend"])
    ) -> Dict[str, Any]:
        """获取统计数据"""
        plugin = get_plugin()
        return plugin.get_statistics(period)

    @animal_router.get("/trend")
    async def get_trend(
        days: int = Query(30, ge=1, le=365)
    ) -> Dict[str, Any]:
        """获取趋势数据"""
        plugin = get_plugin()
        if plugin._statistics:
            return plugin._statistics.get_trend_data(days)
        return {"daily_counts": [], "total": 0}

    @animal_router.get("/report")
    async def export_report(
        format: str = Query("json", enum=["json", "csv"]),
    ) -> Any:
        """导出报告"""
        plugin = get_plugin()
        data = plugin.export_report(format=format)
        if format == "csv":
            from fastapi.responses import Response
            return Response(
                content=data,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=animal_report.csv"},
            )
        return json.loads(data)

    @animal_router.get("/health")
    async def healthcheck() -> Dict[str, Any]:
        """健康检查"""
        plugin = get_plugin()
        return plugin.healthcheck()

    @animal_router.get("/events")
    async def get_recent_events(
        limit: int = Query(50, ge=1, le=500)
    ) -> Dict[str, Any]:
        """获取最近事件"""
        plugin = get_plugin()
        return {
            "events": plugin.get_recent_events(limit),
            "count": len(plugin.get_recent_events(limit)),
        }

    @animal_router.get("/config")
    async def get_ui_config() -> Dict[str, Any]:
        """获取UI配置"""
        plugin = get_plugin()
        return plugin.get_ui_config()

    return animal_router


# =============================================================================
# 集成到现有路由
# =============================================================================

def integrate_animal_detection(existing_router: APIRouter):
    """
    将动物检测集成到现有 indoor_api 路由中

    替换 _data_generator.generate_animal_data() 为真实插件输出
    """
    animal_router = create_animal_detection_router(prefix="")

    # 覆盖现有的 /animal 端点
    # 需要在 indoor_api.py 中调用此函数
    existing_router.include_router(animal_router)
    logger.info("动物检测插件已集成到室内API路由")
