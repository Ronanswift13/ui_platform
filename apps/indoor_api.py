"""
室内监测中心 API 路由
=====================

为室内监测中心提供RESTful API和WebSocket实时推送

输变电激光星芒破夜绘明监测平台 V3.5

API端点:
- GET  /api/indoor/fence       - 电子围栏数据
- GET  /api/indoor/slam         - SLAM建图数据
- GET  /api/indoor/environment - 环境监测数据
- WS   /ws/indoor              - 实时数据推送

版本: 2.0.0 - 重构版（移除模拟数据，接入真实插件）
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# 导入插件管理器
from platform_core.plugin_manager import PluginManager

logger = logging.getLogger(__name__)

# =============================================================================
# 数据模型
# =============================================================================

class PersonData(BaseModel):
    """人员数据"""
    id: int
    name: str
    x: float
    y: float
    cabinet: int
    status: str  # safe, warning, danger
    authorized: bool = True


class CabinetData(BaseModel):
    """机柜数据"""
    id: int
    name: str
    status: str  # idle, occupied, alarm
    person: Optional[str] = None


class FenceData(BaseModel):
    """电子围栏数据"""
    timestamp: int
    status: str
    persons: List[PersonData]
    cabinets: List[CabinetData]
    yellowLine: Dict[str, float]
    totalPersons: int
    alarmCount: int


class EnvironmentData(BaseModel):
    """环境监测数据"""
    timestamp: int
    status: str
    temperature: float
    humidity: float
    pressure: float
    gas_levels: Dict[str, Any]


# =============================================================================
# 插件管理器实例
# =============================================================================

_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """获取插件管理器单例"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


# =============================================================================
# 创建路由
# =============================================================================

router = APIRouter(prefix="/api/indoor", tags=["室内监测中心"])

# WebSocket连接管理
_websocket_connections: List[WebSocket] = []


@router.get("/fence")
async def get_fence_data():
    """获取电子围栏数据"""
    pm = get_plugin_manager()

    try:
        # 尝试获取室内围栏插件
        plugin = pm.get_plugin("indoor_fence")
        if plugin is None:
            plugin = pm.load_plugin("indoor_fence")

        if plugin and hasattr(plugin, 'get_current_state'):
            # 调用插件获取真实数据
            state = plugin.get_current_state()
            return {
                "timestamp": int(time.time() * 1000),
                "status": state.get("status", "online"),
                "persons": state.get("persons", []),
                "cabinets": state.get("cabinets", []),
                "yellowLine": state.get("yellowLine", {"y": 0.15}),
                "totalPersons": len(state.get("persons", [])),
                "alarmCount": state.get("alarmCount", 0)
            }
    except Exception as e:
        logger.error(f"获取电子围栏数据失败: {e}")

    # 返回空数据（插件未就绪）
    return {
        "timestamp": int(time.time() * 1000),
        "status": "offline",
        "persons": [],
        "cabinets": [],
        "yellowLine": {"y": 0.15},
        "totalPersons": 0,
        "alarmCount": 0
    }


@router.get("/slam")
async def get_slam_data():
    """获取SLAM建图数据"""
    pm = get_plugin_manager()

    try:
        plugin = pm.get_plugin("slam_mapping")
        if plugin is None:
            plugin = pm.load_plugin("slam_mapping")

        if plugin and hasattr(plugin, 'get_map_data'):
            map_data = plugin.get_map_data()
            return {
                "timestamp": int(time.time() * 1000),
                "status": "online",
                "map": map_data
            }
    except Exception as e:
        logger.error(f"获取SLAM数据失败: {e}")

    return {
        "timestamp": int(time.time() * 1000),
        "status": "offline",
        "map": None
    }


@router.get("/environment")
async def get_environment_data():
    """获取环境监测数据"""
    pm = get_plugin_manager()

    try:
        plugin = pm.get_plugin("gas_detection")
        if plugin is None:
            plugin = pm.load_plugin("gas_detection")

        if plugin and hasattr(plugin, 'get_readings'):
            readings = plugin.get_readings()
            return {
                "timestamp": int(time.time() * 1000),
                "status": readings.get("status", "good"),
                "temperature": readings.get("temperature", 25.0),
                "humidity": readings.get("humidity", 50.0),
                "pressure": readings.get("pressure", 1013.0),
                "gas_levels": readings.get("gas_levels", {})
            }
    except Exception as e:
        logger.error(f"获取环境数据失败: {e}")

    return {
        "timestamp": int(time.time() * 1000),
        "status": "offline",
        "temperature": 0.0,
        "humidity": 0.0,
        "pressure": 0.0,
        "gas_levels": {}
    }


@router.get("/all")
async def get_all_data():
    """获取所有模块数据"""
    return {
        "fence": await get_fence_data(),
        "slam": await get_slam_data(),
        "environment": await get_environment_data()
    }


# =============================================================================
# WebSocket 实时推送
# =============================================================================

async def broadcast_update(module: str, data: Dict[str, Any]):
    """广播更新到所有连接的客户端"""
    if not _websocket_connections:
        return

    message = json.dumps({
        "type": "update",
        "module": module,
        "data": data
    })

    disconnected = []
    for ws in _websocket_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        _websocket_connections.remove(ws)


async def indoor_data_pusher():
    """定时推送数据"""
    while True:
        try:
            # 获取所有模块数据
            all_data = await get_all_data()

            # 广播每个模块的数据
            for module, data in all_data.items():
                await broadcast_update(module, data)

        except Exception as e:
            logger.error(f"数据推送失败: {e}")

        await asyncio.sleep(1)


# WebSocket端点需要在主应用中注册
def create_websocket_route(app):
    """创建WebSocket路由"""

    @app.websocket("/ws/indoor")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        _websocket_connections.append(websocket)
        logger.info(f"WebSocket连接: {len(_websocket_connections)}个活跃连接")

        try:
            # 发送初始数据
            await websocket.send_json({
                "type": "connected",
                "message": "室内监测中心 WebSocket 已连接"
            })

            # 保持连接并接收消息
            while True:
                data = await websocket.receive_text()
                # 处理来自客户端的消息
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass

        except WebSocketDisconnect:
            _websocket_connections.remove(websocket)
            logger.info(f"WebSocket断开: 剩余{len(_websocket_connections)}个连接")


# =============================================================================
# 集成函数
# =============================================================================

def integrate_indoor_api(app):
    """
    将室内监测API集成到FastAPI应用

    使用方法:
        from apps.indoor_api import integrate_indoor_api
        integrate_indoor_api(app)
    """
    # 注册REST API路由
    app.include_router(router)

    # 注册WebSocket路由
    create_websocket_route(app)

    # 启动后台数据推送任务
    @app.on_event("startup")
    async def start_indoor_pusher():
        asyncio.create_task(indoor_data_pusher())

    logger.info("室内监测中心API已集成 (V2.0 - 真实插件版本)")


# =============================================================================
# 独立运行测试
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 添加项目根目录到路径
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    # 测试插件连接
    pm = get_plugin_manager()

    print("=== 室内监测插件状态 ===")
    for plugin_id in ["indoor_fence", "slam_mapping", "gas_detection"]:
        try:
            plugin = pm.get_plugin(plugin_id)
            if plugin:
                print(f"✓ {plugin_id}: {plugin.status.value}")
            else:
                print(f"✗ {plugin_id}: 未加载")
        except Exception as e:
            print(f"✗ {plugin_id}: 错误 - {e}")

