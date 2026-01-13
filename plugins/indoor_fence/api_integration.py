#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内电子围栏 API 集成层
==========================================

提供与统一监控平台的集成接口:
- RESTful API 端点
- WebSocket 实时推送
- 前端组件配置

基于 V2.0 平台架构设计
作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
import logging
import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import threading

# FastAPI 可选依赖
try:
    from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None

from .plugin import IndoorFencePlugin

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic 模型 (用于 API 请求/响应)
# =============================================================================

if FASTAPI_AVAILABLE:
    class AuthorizationRequest(BaseModel):
        """授权请求"""
        person_id: str
        cabinet_ids: List[int]

    class AllowListRequest(BaseModel):
        """授权列表请求"""
        cabinet_ids: List[int]

    class ConfigUpdateRequest(BaseModel):
        """配置更新请求"""
        key: str
        value: Any

    class ProcessRequest(BaseModel):
        """处理请求"""
        frame_data: Optional[str] = None  # Base64 编码的帧数据
        context: Optional[Dict] = None


# =============================================================================
# WebSocket 管理器
# =============================================================================

class WebSocketManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self._connections: List[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        """接受新连接"""
        await websocket.accept()
        with self._lock:
            self._connections.append(websocket)
        logger.info(f"[WebSocket] 新连接, 当前连接数: {len(self._connections)}")

    def disconnect(self, websocket: WebSocket):
        """断开连接"""
        with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)
        logger.info(f"[WebSocket] 连接断开, 当前连接数: {len(self._connections)}")

    async def broadcast(self, message: Dict):
        """广播消息"""
        disconnected = []
        with self._lock:
            connections = self._connections.copy()

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # 清理断开的连接
        for conn in disconnected:
            self.disconnect(conn)

    @property
    def connection_count(self) -> int:
        """获取连接数"""
        return len(self._connections)


# =============================================================================
# API 路由创建器
# =============================================================================

def create_indoor_fence_router(
    plugin: IndoorFencePlugin,
    prefix: str = "/api/indoor_fence"
) -> Optional[APIRouter]:
    """
    创建室内电子围栏 API 路由

    Args:
        plugin: 插件实例
        prefix: 路由前缀

    Returns:
        FastAPI 路由器
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI 不可用, API 路由未创建")
        return None

    router = APIRouter(prefix=prefix, tags=["室内电子围栏"])
    ws_manager = WebSocketManager()

    # ==================== 状态查询 ====================

    @router.get("/status")
    async def get_status():
        """获取插件状态"""
        health = plugin.healthcheck()
        return {
            "id": plugin.id,
            "version": plugin.version,
            "status": plugin.status.value if hasattr(plugin.status, 'value') else str(plugin.status),
            "healthy": health.healthy,
            "message": health.message,
            "details": health.details,
        }

    @router.get("/health")
    async def health_check():
        """健康检查"""
        health = plugin.healthcheck()
        return {
            "healthy": health.healthy,
            "message": health.message,
            "timestamp": datetime.now().isoformat(),
        }

    # ==================== 人员跟踪 ====================

    @router.get("/persons")
    async def get_tracked_persons():
        """获取跟踪人员列表"""
        return plugin.get_tracked_persons()

    @router.get("/persons/{person_id}")
    async def get_person_detail(person_id: str):
        """获取人员详情"""
        persons = plugin.get_tracked_persons()
        for p in persons:
            if p["person_id"] == person_id:
                return p
        raise HTTPException(status_code=404, detail="人员未找到")

    # ==================== 机柜管理 ====================

    @router.get("/cabinets")
    async def get_cabinet_status():
        """获取机柜状态"""
        return plugin.get_cabinet_status()

    @router.get("/cabinets/{cabinet_id}")
    async def get_cabinet_detail(cabinet_id: int):
        """获取机柜详情"""
        cabinets = plugin.get_cabinet_status()
        if cabinet_id in cabinets:
            return cabinets[cabinet_id]
        raise HTTPException(status_code=404, detail="机柜未找到")

    # ==================== 区域配置 ====================

    @router.get("/zones")
    async def get_zone_config():
        """获取区域配置"""
        return plugin.get_zone_config()

    # ==================== 授权管理 ====================

    @router.post("/authorization")
    async def set_authorization(request: AuthorizationRequest):
        """设置人员授权"""
        plugin.set_authorization(request.person_id, request.cabinet_ids)
        return {
            "status": "ok",
            "message": f"已设置授权: {request.person_id} -> 机柜{request.cabinet_ids}",
        }

    @router.delete("/authorization/{person_id}")
    async def clear_authorization(person_id: str):
        """清除人员授权"""
        plugin.clear_authorization(person_id)
        return {
            "status": "ok",
            "message": f"已清除授权: {person_id}",
        }

    @router.put("/allow_list")
    async def update_allow_list(request: AllowListRequest):
        """更新授权机柜列表"""
        plugin.update_allow_list(request.cabinet_ids)
        return {
            "status": "ok",
            "allow_list": request.cabinet_ids,
        }

    # ==================== 事件查询 ====================

    @router.get("/events")
    async def get_recent_events(limit: int = 100):
        """获取最近事件"""
        return plugin.get_recent_events(limit)

    # ==================== UI 配置 ====================

    @router.get("/ui/config")
    async def get_ui_config():
        """获取 UI 配置"""
        return plugin.get_ui_config()

    # ==================== WebSocket ====================

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket 实时推送"""
        await ws_manager.connect(websocket)

        try:
            while True:
                # 接收客户端消息
                data = await websocket.receive_json()

                # 处理消息
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "subscribe":
                    # 订阅特定事件
                    await websocket.send_json({
                        "type": "subscribed",
                        "topics": data.get("topics", []),
                    })

                elif msg_type == "get_status":
                    health = plugin.healthcheck()
                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "healthy": health.healthy,
                            "message": health.message,
                            "details": health.details,
                        },
                    })

        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket 错误: {e}")
            ws_manager.disconnect(websocket)

    # ==================== 统计信息 ====================

    @router.get("/stats")
    async def get_statistics():
        """获取统计信息"""
        health = plugin.healthcheck()
        details = health.details or {}
        return {
            "frame_count": details.get("frame_count", 0),
            "alert_count": details.get("alert_count", 0),
            "tracked_persons": details.get("tracked_persons", 0),
            "cabinet_count": details.get("cabinet_count", 0),
            "last_process_time_ms": details.get("last_process_time_ms", 0),
            "websocket_connections": ws_manager.connection_count,
        }

    return router


# =============================================================================
# 平台集成辅助函数
# =============================================================================

def integrate_with_platform(app, plugin: IndoorFencePlugin):
    """
    将室内电子围栏插件集成到平台

    Args:
        app: FastAPI 应用实例
        plugin: 插件实例
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI 不可用, 无法集成")
        return

    # 创建路由
    router = create_indoor_fence_router(plugin)

    if router:
        # 注册路由
        app.include_router(router)
        logger.info("室内电子围栏 API 已注册")


# =============================================================================
# 实时推送服务
# =============================================================================

class RealtimePushService:
    """
    实时推送服务

    周期性处理并推送结果到 WebSocket
    """

    def __init__(
        self,
        plugin: IndoorFencePlugin,
        ws_manager: WebSocketManager,
        interval_ms: int = 100
    ):
        self.plugin = plugin
        self.ws_manager = ws_manager
        self.interval = interval_ms / 1000.0

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """启动服务"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("实时推送服务已启动")

    def stop(self):
        """停止服务"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("实时推送服务已停止")

    def _run_loop(self):
        """运行循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._running:
            try:
                # 推送状态更新
                if self.ws_manager.connection_count > 0:
                    health = self.plugin.healthcheck()
                    loop.run_until_complete(
                        self.ws_manager.broadcast({
                            "type": "status_update",
                            "data": {
                                "healthy": health.healthy,
                                "message": health.message,
                                "details": health.details,
                                "timestamp": datetime.now().isoformat(),
                            },
                        })
                    )

                # 等待
                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"推送服务错误: {e}")
                time.sleep(1.0)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'create_indoor_fence_router',
    'integrate_with_platform',
    'WebSocketManager',
    'RealtimePushService',
]


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("室内电子围栏 API 集成层测试")
    print("=" * 50)

    print("\n[1] 检查 FastAPI 可用性...")
    if FASTAPI_AVAILABLE:
        print("FastAPI 可用")
    else:
        print("FastAPI 不可用, 请安装: pip install fastapi")

    print("\n测试完成!")
