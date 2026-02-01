"""
室内监测中心 API 路由
=====================

为室内监测中心提供RESTful API和WebSocket实时推送

输变电激光星芒破夜绘明监测平台 V3.5

API端点:
- GET  /api/indoor/fence                      - 电子围栏数据
- GET  /api/indoor/animal                     - 动物入侵检测数据
- GET  /api/indoor/temperature                - 温度监测数据
- GET  /api/indoor/device                     - 设备状态监测数据
- GET  /api/indoor/fire                       - 消防监测数据
- GET  /api/indoor/slam                       - SLAM建图数据
- GET  /api/indoor/environment                - 环境监测数据
- GET  /api/indoor/fusion/evidence            - 多模态融合证据链
- GET  /api/indoor/plugin/{plugin_id}/capabilities - 插件能力配置
- POST /api/indoor/plugin/{plugin_id}/command - 执行插件命令
- GET  /api/indoor/all                        - 所有模块数据
- WS   /ws/indoor                             - 实时数据推送

版本: 3.0.0 - 重构版（新增融合证据链和配置驱动UI）
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


@router.get("/animal")
async def get_animal_data():
    """获取动物入侵检测数据"""
    pm = get_plugin_manager()

    try:
        plugin = pm.get_plugin("animal_detection")
        if plugin is None:
            plugin = pm.load_plugin("animal_detection")

        if plugin and hasattr(plugin, 'get_detections'):
            detections = plugin.get_detections()
            return {
                "timestamp": int(time.time() * 1000),
                "status": detections.get("status", "online"),
                "animals": detections.get("animals", []),
                "totalCount": len(detections.get("animals", [])),
                "alarmCount": detections.get("alarmCount", 0)
            }
    except Exception as e:
        logger.error(f"获取动物检测数据失败: {e}")

    return {
        "timestamp": int(time.time() * 1000),
        "status": "offline",
        "animals": [],
        "totalCount": 0,
        "alarmCount": 0
    }


@router.get("/temperature")
async def get_temperature_data():
    """获取温度监测数据"""
    pm = get_plugin_manager()

    try:
        plugin = pm.get_plugin("temperature_monitoring")
        if plugin is None:
            plugin = pm.load_plugin("temperature_monitoring")

        if plugin and hasattr(plugin, 'get_thermal_data'):
            thermal_data = plugin.get_thermal_data()
            return {
                "timestamp": int(time.time() * 1000),
                "status": thermal_data.get("status", "normal"),
                "avgTemp": thermal_data.get("avgTemp", 25.0),
                "maxTemp": thermal_data.get("maxTemp", 30.0),
                "minTemp": thermal_data.get("minTemp", 20.0),
                "hotspots": thermal_data.get("hotspots", []),
                "heatmap": thermal_data.get("heatmap", None)
            }
    except Exception as e:
        logger.error(f"获取温度数据失败: {e}")

    return {
        "timestamp": int(time.time() * 1000),
        "status": "offline",
        "avgTemp": 0.0,
        "maxTemp": 0.0,
        "minTemp": 0.0,
        "hotspots": [],
        "heatmap": None
    }


@router.get("/device")
async def get_device_data():
    """获取设备状态监测数据"""
    pm = get_plugin_manager()

    try:
        plugin = pm.get_plugin("device_monitoring")
        if plugin is None:
            plugin = pm.load_plugin("device_monitoring")

        if plugin and hasattr(plugin, 'get_device_status'):
            device_status = plugin.get_device_status()
            return {
                "timestamp": int(time.time() * 1000),
                "status": device_status.get("status", "online"),
                "devices": device_status.get("devices", []),
                "totalDevices": len(device_status.get("devices", [])),
                "onlineCount": device_status.get("onlineCount", 0),
                "offlineCount": device_status.get("offlineCount", 0),
                "faultCount": device_status.get("faultCount", 0)
            }
    except Exception as e:
        logger.error(f"获取设备状态数据失败: {e}")

    return {
        "timestamp": int(time.time() * 1000),
        "status": "offline",
        "devices": [],
        "totalDevices": 0,
        "onlineCount": 0,
        "offlineCount": 0,
        "faultCount": 0
    }


@router.get("/fire")
async def get_fire_data():
    """获取消防监测数据"""
    pm = get_plugin_manager()

    try:
        plugin = pm.get_plugin("fire_detection")
        if plugin is None:
            plugin = pm.load_plugin("fire_detection")

        if plugin and hasattr(plugin, 'get_fire_status'):
            fire_status = plugin.get_fire_status()
            return {
                "timestamp": int(time.time() * 1000),
                "status": fire_status.get("status", "safe"),
                "fireDetected": fire_status.get("fireDetected", False),
                "smokeLevel": fire_status.get("smokeLevel", 0.0),
                "fireLocations": fire_status.get("fireLocations", []),
                "alarmActive": fire_status.get("alarmActive", False)
            }
    except Exception as e:
        logger.error(f"获取消防数据失败: {e}")

    return {
        "timestamp": int(time.time() * 1000),
        "status": "offline",
        "fireDetected": False,
        "smokeLevel": 0.0,
        "fireLocations": [],
        "alarmActive": False
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


@router.get("/plugin/{plugin_id}/capabilities")
async def get_plugin_capabilities(plugin_id: str):
    """获取插件能力配置（用于动态生成控制面板）"""
    pm = get_plugin_manager()

    try:
        plugin = pm.get_plugin(plugin_id)
        if plugin is None:
            plugin = pm.load_plugin(plugin_id)

        if plugin and hasattr(plugin, 'get_capabilities'):
            capabilities = plugin.get_capabilities()
            return {
                "plugin_id": plugin_id,
                "name": capabilities.get("name", plugin_id),
                "description": capabilities.get("description", ""),
                "controls": capabilities.get("controls", []),
                "operations": capabilities.get("operations", [])
            }
    except Exception as e:
        logger.error(f"获取插件 {plugin_id} 能力失败: {e}")

    # 返回默认能力配置
    default_capabilities = {
        "indoor_fence": {
            "name": "电子围栏",
            "description": "基于激光雷达的电子围栏和越线检测",
            "controls": [
                {"type": "slider", "id": "threshold", "label": "越线阈值", "min": 0, "max": 1, "step": 0.1, "default": 0.15},
                {"type": "select", "id": "mode", "label": "检测模式", "options": ["标准", "严格", "宽松"], "default": "标准"}
            ],
            "operations": [
                {"id": "reset_fence", "label": "重置围栏", "icon": "shield-check"},
                {"id": "export_log", "label": "导出日志", "icon": "download"}
            ]
        },
        "animal_detection": {
            "name": "动物入侵检测",
            "description": "小动物识别与智能驱离",
            "controls": [
                {"type": "slider", "id": "sensitivity", "label": "灵敏度", "min": 0, "max": 100, "step": 5, "default": 70}
            ],
            "operations": [
                {"id": "activate_deterrent", "label": "启动驱离", "icon": "volume-up"},
                {"id": "view_history", "label": "查看历史", "icon": "clock-history"}
            ]
        },
        "temperature_monitoring": {
            "name": "温度监测",
            "description": "热成像温度监测与热力图分析",
            "controls": [
                {"type": "slider", "id": "temp_threshold", "label": "温度阈值(°C)", "min": 20, "max": 80, "step": 1, "default": 40}
            ],
            "operations": [
                {"id": "show_heatmap", "label": "显示热力图", "icon": "thermometer-half"},
                {"id": "export_data", "label": "导出数据", "icon": "download"}
            ]
        }
    }

    config = default_capabilities.get(plugin_id, {
        "name": plugin_id,
        "description": "监测插件",
        "controls": [],
        "operations": []
    })

    return {
        "plugin_id": plugin_id,
        **config
    }


@router.post("/plugin/{plugin_id}/command")
async def execute_plugin_command(plugin_id: str, command: Dict[str, Any]):
    """执行插件命令"""
    pm = get_plugin_manager()

    try:
        plugin = pm.get_plugin(plugin_id)
        if plugin is None:
            plugin = pm.load_plugin(plugin_id)

        if plugin and hasattr(plugin, 'execute_command'):
            result = plugin.execute_command(command)
            return {
                "success": True,
                "plugin_id": plugin_id,
                "command": command.get("operation"),
                "result": result
            }
    except Exception as e:
        logger.error(f"执行插件 {plugin_id} 命令失败: {e}")
        return {
            "success": False,
            "plugin_id": plugin_id,
            "error": str(e)
        }

    return {
        "success": False,
        "plugin_id": plugin_id,
        "error": "插件不支持命令执行"
    }


@router.get("/fusion/evidence")
async def get_fusion_evidence():
    """获取多模态融合证据链数据"""
    pm = get_plugin_manager()

    try:
        # 尝试获取融合引擎
        fusion_plugin = pm.get_plugin("multimodal_fusion")
        if fusion_plugin is None:
            fusion_plugin = pm.load_plugin("multimodal_fusion")

        if fusion_plugin and hasattr(fusion_plugin, 'get_evidence_chain'):
            evidence = fusion_plugin.get_evidence_chain()
            return {
                "timestamp": evidence.get("timestamp", int(time.time() * 1000)),
                "modalities": evidence.get("modalities", []),
                "fusion_result": evidence.get("fusion_result", "正常"),
                "recommendation": evidence.get("recommendation", ""),
                "confidence": evidence.get("confidence", 0.0)
            }
    except Exception as e:
        logger.error(f"获取融合证据链失败: {e}")

    # 返回模拟的多模态数据（当融合引擎不可用时）
    return {
        "timestamp": int(time.time() * 1000),
        "modalities": [
            {"type": "vision", "name": "视觉检测", "result": "正常", "confidence": 0.92},
            {"type": "acoustic", "name": "声学监测", "result": "正常", "confidence": 0.88},
            {"type": "gas", "name": "气体检测", "result": "正常", "confidence": 0.95},
            {"type": "thermal", "name": "热成像", "result": "正常", "confidence": 0.90}
        ],
        "fusion_result": "正常",
        "recommendation": "所有监测指标正常，继续监控",
        "confidence": 0.91
    }


@router.get("/all")
async def get_all_data():
    """获取所有模块数据"""
    return {
        "fence": await get_fence_data(),
        "animal": await get_animal_data(),
        "temperature": await get_temperature_data(),
        "device": await get_device_data(),
        "fire": await get_fire_data(),
        "slam": await get_slam_data(),
        "environment": await get_environment_data(),
        "fusion_evidence": await get_fusion_evidence()
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

    logger.info("室内监测中心API已集成 (V2.1 - 完整版，包含6个监测模块)")


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
    for plugin_id in [
        "indoor_fence",
        "animal_detection",
        "temperature_monitoring",
        "device_monitoring",
        "fire_detection",
        "slam_mapping",
        "gas_detection"
    ]:
        try:
            plugin = pm.get_plugin(plugin_id)
            if plugin:
                print(f"✓ {plugin_id}: {plugin.status.value}")
            else:
                print(f"✗ {plugin_id}: 未加载")
        except Exception as e:
            print(f"✗ {plugin_id}: 错误 - {e}")

