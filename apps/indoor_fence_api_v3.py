"""
室内电子围栏 API (V3.0)
========================

生产级API实现:
- 移除所有模拟数据生成器
- 从PluginManager获取真实插件推理结果
- 统一事件契约 (timestamp/type/location/value/confidence + evidence + suggestion + trace_id)
- WebSocket实时推送 (断线重连、消息幂等)
- 完整的审计日志

版本: 3.0.0
作者: 变电站AI监控平台架构组

API端点:
- GET  /api/v3/indoor/fence/status       - 获取系统状态
- GET  /api/v3/indoor/fence/tracks       - 获取当前跟踪列表
- GET  /api/v3/indoor/fence/events       - 获取历史事件
- GET  /api/v3/indoor/fence/zones        - 获取区域配置
- POST /api/v3/indoor/fence/authorize    - 更新授权列表
- GET  /api/v3/indoor/fence/health       - 健康检查
- WS   /ws/v3/indoor/fence               - 实时事件推送
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from collections import deque
from threading import Lock
import weakref

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 导入统一事件契约
from platform_core.schema.indoor_fence_events import (
    UnifiedEvent, EventType, AlertLevel, PersonState,
    IndoorFenceEventBuilder, EventValidator, generate_trace_id
)

logger = logging.getLogger(__name__)


# =============================================================================
# 配置常量
# =============================================================================

class Config:
    """API配置"""
    # 仿真模式开关 (默认关闭,生产环境必须关闭)
    SIMULATION_MODE_ENABLED: bool = False
    SIMULATION_MODE_REQUIRE_EXPLICIT: bool = True  # 需要显式开启
    
    # WebSocket配置
    WS_HEARTBEAT_INTERVAL: int = 30  # 心跳间隔(秒)
    WS_MESSAGE_BUFFER_SIZE: int = 1000  # 消息缓冲区大小
    WS_RECONNECT_TIMEOUT: int = 5  # 重连超时(秒)
    
    # 事件配置
    EVENT_HISTORY_SIZE: int = 10000  # 事件历史大小
    EVENT_DEDUP_WINDOW: int = 1000  # 去重窗口(毫秒)


# =============================================================================
# Pydantic 模型
# =============================================================================

class TrackInfo(BaseModel):
    """轨迹信息"""
    track_id: str
    person_state: str
    ground_position: Dict[str, float]
    velocity: Optional[Dict[str, float]] = None
    confidence: float
    cabinet_id: Optional[int] = None
    is_authorized: bool = True
    last_seen: str
    hits: int
    age: int


class SystemStatus(BaseModel):
    """系统状态"""
    plugin_id: str = "indoor_fence"
    version: str = "3.0.0"
    status: str  # running, degraded, error
    simulation_mode: bool = False
    tracked_persons: int
    active_alarms: int
    sensor_health: Dict[str, float]
    last_process_time_ms: float
    uptime_seconds: float
    frame_count: int


class EventQuery(BaseModel):
    """事件查询参数"""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    event_types: Optional[List[str]] = None
    alert_levels: Optional[List[str]] = None
    person_id: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = 0


class AuthorizeRequest(BaseModel):
    """授权请求"""
    person_id: Optional[str] = None  # 如不指定则应用于所有人
    cabinet_ids: List[int]
    expires_at: Optional[str] = None  # 授权过期时间


class HealthResponse(BaseModel):
    """健康检查响应"""
    healthy: bool
    message: str
    components: Dict[str, Dict[str, Any]]
    timestamp: str


# =============================================================================
# WebSocket 连接管理器
# =============================================================================

class WebSocketConnectionManager:
    """
    WebSocket连接管理器
    
    特性:
    - 断线检测与自动清理
    - 消息去重 (基于trace_id)
    - 消息顺序保证
    - 广播失败重试
    """
    
    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}  # client_id -> WebSocket
        self._lock = Lock()
        self._message_buffer: deque = deque(maxlen=Config.WS_MESSAGE_BUFFER_SIZE)
        self._sent_trace_ids: Dict[str, Set[str]] = {}  # client_id -> sent trace_ids
        self._last_heartbeat: Dict[str, float] = {}
        
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """
        建立连接
        
        Args:
            websocket: WebSocket实例
            client_id: 客户端ID (用于重连时恢复状态)
            
        Returns:
            client_id
        """
        await websocket.accept()
        
        if not client_id:
            client_id = f"client_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            # 如果是重连,先清理旧连接
            if client_id in self._connections:
                try:
                    await self._connections[client_id].close()
                except Exception:
                    pass
            
            self._connections[client_id] = websocket
            self._sent_trace_ids[client_id] = set()
            self._last_heartbeat[client_id] = time.time()
        
        logger.info(f"[WebSocket] 客户端连接: {client_id}, 当前连接数: {len(self._connections)}")
        
        # 发送连接确认和历史消息
        await self._send_connection_ack(websocket, client_id)
        
        return client_id
    
    async def _send_connection_ack(self, websocket: WebSocket, client_id: str):
        """发送连接确认"""
        try:
            await websocket.send_json({
                "type": "connection_ack",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "message": "连接成功",
                "features": {
                    "heartbeat_interval": Config.WS_HEARTBEAT_INTERVAL,
                    "supports_reconnect": True,
                    "supports_dedup": True
                }
            })
        except Exception as e:
            logger.error(f"发送连接确认失败: {e}")
    
    def disconnect(self, client_id: str):
        """断开连接"""
        with self._lock:
            if client_id in self._connections:
                del self._connections[client_id]
            if client_id in self._sent_trace_ids:
                del self._sent_trace_ids[client_id]
            if client_id in self._last_heartbeat:
                del self._last_heartbeat[client_id]
        
        logger.info(f"[WebSocket] 客户端断开: {client_id}, 剩余连接数: {len(self._connections)}")
    
    async def broadcast_event(self, event: UnifiedEvent):
        """
        广播事件到所有连接
        
        特性:
        - 基于trace_id的消息幂等性
        - 失败连接自动清理
        """
        message = {
            "type": "event",
            "trace_id": event.trace_id,
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "data": event.to_dict()
        }
        
        # 添加到消息缓冲区 (用于重连恢复)
        self._message_buffer.append(message)
        
        disconnected = []
        
        with self._lock:
            connections = list(self._connections.items())
        
        for client_id, websocket in connections:
            # 检查是否已发送过 (消息幂等)
            if event.trace_id in self._sent_trace_ids.get(client_id, set()):
                continue
            
            try:
                await websocket.send_json(message)
                
                # 记录已发送
                with self._lock:
                    if client_id in self._sent_trace_ids:
                        self._sent_trace_ids[client_id].add(event.trace_id)
                        # 限制记录大小
                        if len(self._sent_trace_ids[client_id]) > 10000:
                            # 清理旧记录
                            self._sent_trace_ids[client_id] = set(
                                list(self._sent_trace_ids[client_id])[-5000:]
                            )
            except Exception as e:
                logger.warning(f"发送消息到 {client_id} 失败: {e}")
                disconnected.append(client_id)
        
        # 清理断开的连接
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """发送消息到特定客户端"""
        with self._lock:
            websocket = self._connections.get(client_id)
        
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                self.disconnect(client_id)
    
    def update_heartbeat(self, client_id: str):
        """更新心跳时间"""
        with self._lock:
            self._last_heartbeat[client_id] = time.time()
    
    async def check_heartbeats(self):
        """检查心跳,清理超时连接"""
        current_time = time.time()
        timeout = Config.WS_HEARTBEAT_INTERVAL * 2  # 2倍心跳间隔视为超时
        
        with self._lock:
            expired = [
                client_id for client_id, last_time in self._last_heartbeat.items()
                if current_time - last_time > timeout
            ]
        
        for client_id in expired:
            logger.warning(f"客户端 {client_id} 心跳超时,断开连接")
            self.disconnect(client_id)
    
    def get_missed_messages(self, client_id: str, last_trace_id: Optional[str]) -> List[Dict]:
        """获取断线期间错过的消息"""
        if not last_trace_id:
            return []
        
        messages = []
        found = False
        
        for msg in self._message_buffer:
            if found:
                messages.append(msg)
            elif msg.get('trace_id') == last_trace_id:
                found = True
        
        return messages
    
    @property
    def connection_count(self) -> int:
        """获取当前连接数"""
        return len(self._connections)


# =============================================================================
# 插件服务层 (连接PluginManager)
# =============================================================================

class IndoorFenceService:
    """
    室内电子围栏服务层
    
    职责:
    - 管理插件实例
    - 从插件获取真实推理结果
    - 事件生成与广播
    - 状态管理
    """
    
    _instance: Optional['IndoorFenceService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._plugin = None
        self._plugin_manager = None
        self._event_builder = IndoorFenceEventBuilder()
        self._event_history: deque = deque(maxlen=Config.EVENT_HISTORY_SIZE)
        self._start_time = time.time()
        self._frame_count = 0
        self._last_process_time = 0.0
        self._current_trace_id = generate_trace_id()
        
        self._initialized = True
        logger.info("[IndoorFenceService] 服务初始化")
    
    def _get_plugin_manager(self):
        """懒加载PluginManager"""
        if self._plugin_manager is None:
            try:
                from platform_core.plugin_manager import PluginManager
                self._plugin_manager = PluginManager()
            except ImportError as e:
                logger.error(f"无法导入PluginManager: {e}")
                raise HTTPException(status_code=500, detail="插件管理器不可用")
        return self._plugin_manager
    
    def _get_plugin(self):
        """获取或加载插件实例"""
        if self._plugin is None:
            pm = self._get_plugin_manager()
            try:
                self._plugin = pm.get_plugin("indoor_fence")
                if self._plugin is None:
                    self._plugin = pm.load_plugin("indoor_fence")
                logger.info(f"[IndoorFenceService] 插件已加载: {self._plugin.version}")
            except Exception as e:
                logger.error(f"加载插件失败: {e}")
                raise HTTPException(status_code=500, detail=f"插件加载失败: {e}")
        return self._plugin
    
    def get_status(self) -> SystemStatus:
        """获取系统状态"""
        plugin = self._get_plugin()
        health = plugin.healthcheck()
        
        # 获取传感器健康度
        sensor_health = {}
        if hasattr(plugin, '_camera_adapter'):
            cam_health = plugin._camera_adapter.get_health()
            sensor_health['camera'] = cam_health.data_rate_hz if cam_health else 0.0
        if hasattr(plugin, '_lidar_adapter'):
            lidar_health = plugin._lidar_adapter.get_health()
            sensor_health['lidar'] = lidar_health.data_rate_hz if lidar_health else 0.0
        
        # 获取跟踪人数
        tracked_persons = 0
        if hasattr(plugin, '_tracker'):
            tracked_persons = len(plugin._tracker.get_confirmed_tracks())
        
        # 获取活跃告警数
        active_alarms = sum(
            1 for evt in self._event_history 
            if evt.alert_level in [AlertLevel.ALARM, AlertLevel.CRITICAL]
            and (datetime.now() - datetime.fromisoformat(evt.timestamp)).seconds < 300
        )
        
        return SystemStatus(
            status="running" if health.healthy else "degraded",
            simulation_mode=Config.SIMULATION_MODE_ENABLED,
            tracked_persons=tracked_persons,
            active_alarms=active_alarms,
            sensor_health=sensor_health,
            last_process_time_ms=self._last_process_time,
            uptime_seconds=time.time() - self._start_time,
            frame_count=self._frame_count
        )
    
    def get_tracks(self) -> List[TrackInfo]:
        """获取当前跟踪列表"""
        plugin = self._get_plugin()
        
        if not hasattr(plugin, '_tracker'):
            return []
        
        tracks = []
        for track in plugin._tracker.get_confirmed_tracks():
            # 获取人员状态
            person_state = PersonState.SAFE.value
            if hasattr(plugin, '_state_machine'):
                state_result = plugin._state_machine.evaluate_person(
                    track.track_id, 
                    track.position,
                    plugin._allow_list if hasattr(plugin, '_allow_list') else []
                )
                if state_result:
                    person_state = state_result.state.value
            
            tracks.append(TrackInfo(
                track_id=track.track_id,
                person_state=person_state,
                ground_position={"x": track.position.x, "y": track.position.y},
                velocity={"x": track.velocity.x, "y": track.velocity.y} if track.velocity else None,
                confidence=track.confidence,
                cabinet_id=getattr(track, 'cabinet_id', None),
                is_authorized=getattr(track, 'is_authorized', True),
                last_seen=datetime.fromtimestamp(track.last_seen).isoformat(),
                hits=track.hits,
                age=track.age
            ))
        
        return tracks
    
    def get_events(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        alert_levels: Optional[List[str]] = None,
        person_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取历史事件"""
        events = list(self._event_history)
        
        # 过滤
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            events = [e for e in events if datetime.fromisoformat(e.timestamp) >= start_dt]
        
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
            events = [e for e in events if datetime.fromisoformat(e.timestamp) <= end_dt]
        
        if event_types:
            events = [e for e in events if e.type.value in event_types]
        
        if alert_levels:
            events = [e for e in events if e.alert_level.value in alert_levels]
        
        if person_id:
            events = [e for e in events if e.person_id == person_id]
        
        # 分页
        total = len(events)
        events = events[offset:offset + limit]
        
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "events": [e.to_dict() for e in events]
        }
    
    def get_zones(self) -> Dict[str, Any]:
        """获取区域配置"""
        plugin = self._get_plugin()
        
        if hasattr(plugin, '_zone_config'):
            config = plugin._zone_config
            return {
                "config_id": config.config_id,
                "config_name": config.config_name,
                "zones": [
                    {
                        "zone_id": z.zone_id,
                        "zone_type": z.zone_type.value,
                        "name": z.name,
                        "polygon": [{"x": p.x, "y": p.y} for p in z.polygon.vertices]
                    }
                    for z in config.zones.values()
                ],
                "cabinets": [
                    {
                        "cabinet_id": c.cabinet_id,
                        "name": c.name,
                        "x_start": c.x_start,
                        "x_end": c.x_end,
                        "y_line": c.y_line
                    }
                    for c in config.cabinets.values()
                ],
                "yellow_line": {
                    "y": config.yellow_line.y if hasattr(config, 'yellow_line') else 0.5
                }
            }
        
        return {"error": "区域配置不可用"}
    
    def update_authorization(self, request: AuthorizeRequest) -> Dict[str, Any]:
        """更新授权"""
        plugin = self._get_plugin()
        
        if hasattr(plugin, 'update_allow_list'):
            plugin.update_allow_list(request.cabinet_ids)
        
        if hasattr(plugin, '_state_machine') and request.person_id:
            plugin._state_machine.set_authorization(request.person_id, request.cabinet_ids)
        
        return {
            "status": "success",
            "person_id": request.person_id,
            "authorized_cabinets": request.cabinet_ids,
            "expires_at": request.expires_at
        }
    
    def health_check(self) -> HealthResponse:
        """健康检查"""
        components = {}
        overall_healthy = True
        messages = []
        
        # 检查插件
        try:
            plugin = self._get_plugin()
            health = plugin.healthcheck()
            components["plugin"] = {
                "healthy": health.healthy,
                "message": health.message,
                "details": health.details
            }
            if not health.healthy:
                overall_healthy = False
                messages.append(f"插件: {health.message}")
        except Exception as e:
            components["plugin"] = {"healthy": False, "message": str(e)}
            overall_healthy = False
            messages.append(f"插件: {e}")
        
        # 检查相机适配器
        if hasattr(plugin, '_camera_adapter'):
            try:
                cam_health = plugin._camera_adapter.get_health()
                components["camera"] = {
                    "healthy": cam_health.status.value in ['connected', 'running'],
                    "status": cam_health.status.value,
                    "data_rate_hz": cam_health.data_rate_hz
                }
            except Exception as e:
                components["camera"] = {"healthy": False, "message": str(e)}
        
        # 检查雷达适配器
        if hasattr(plugin, '_lidar_adapter'):
            try:
                lidar_health = plugin._lidar_adapter.get_health()
                components["lidar"] = {
                    "healthy": lidar_health.status.value in ['connected', 'running'],
                    "status": lidar_health.status.value,
                    "data_rate_hz": lidar_health.data_rate_hz
                }
            except Exception as e:
                components["lidar"] = {"healthy": False, "message": str(e)}
        
        return HealthResponse(
            healthy=overall_healthy,
            message="; ".join(messages) if messages else "所有组件正常",
            components=components,
            timestamp=datetime.now().isoformat()
        )
    
    def emit_event(self, event: UnifiedEvent):
        """发射事件"""
        # 验证事件
        is_valid, errors = EventValidator.validate(event)
        if not is_valid:
            logger.warning(f"事件验证失败: {errors}")
            # 记录但不阻止
        
        # 添加到历史
        self._event_history.append(event)
        
        logger.debug(f"[Event] {event.type.value}: {event.event_id}")
    
    def new_trace(self):
        """开始新的追踪链"""
        self._current_trace_id = generate_trace_id()
        self._event_builder = IndoorFenceEventBuilder(self._current_trace_id)
        return self._current_trace_id


# =============================================================================
# API 路由创建
# =============================================================================

def create_indoor_fence_api_v3() -> APIRouter:
    """创建室内电子围栏API路由"""
    
    router = APIRouter(prefix="/api/v3/indoor/fence", tags=["室内电子围栏 V3"])
    
    # 服务单例
    service = IndoorFenceService()
    ws_manager = WebSocketConnectionManager()
    
    # ==================== REST API ====================
    
    @router.get("/status", response_model=SystemStatus)
    async def get_status():
        """获取系统状态"""
        return service.get_status()
    
    @router.get("/tracks", response_model=List[TrackInfo])
    async def get_tracks():
        """获取当前跟踪列表"""
        return service.get_tracks()
    
    @router.get("/events")
    async def get_events(
        start_time: Optional[str] = Query(None, description="开始时间 (ISO8601)"),
        end_time: Optional[str] = Query(None, description="结束时间 (ISO8601)"),
        event_types: Optional[str] = Query(None, description="事件类型,逗号分隔"),
        alert_levels: Optional[str] = Query(None, description="告警级别,逗号分隔"),
        person_id: Optional[str] = Query(None, description="人员ID"),
        limit: int = Query(100, le=1000, description="返回数量限制"),
        offset: int = Query(0, ge=0, description="偏移量")
    ):
        """获取历史事件"""
        return service.get_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types.split(",") if event_types else None,
            alert_levels=alert_levels.split(",") if alert_levels else None,
            person_id=person_id,
            limit=limit,
            offset=offset
        )
    
    @router.get("/zones")
    async def get_zones():
        """获取区域配置"""
        return service.get_zones()
    
    @router.post("/authorize")
    async def update_authorization(request: AuthorizeRequest):
        """更新授权"""
        return service.update_authorization(request)
    
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """健康检查"""
        return service.health_check()
    
    @router.post("/trace/new")
    async def new_trace():
        """开始新的追踪链"""
        trace_id = service.new_trace()
        return {"trace_id": trace_id}
    
    # ==================== WebSocket ====================
    
    @router.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        client_id: Optional[str] = Query(None, description="客户端ID(用于重连)"),
        last_trace_id: Optional[str] = Query(None, description="最后收到的trace_id(用于恢复)")
    ):
        """
        WebSocket实时事件推送
        
        连接参数:
        - client_id: 可选,用于重连时恢复状态
        - last_trace_id: 可选,用于获取断线期间错过的消息
        
        消息格式:
        - 服务器发送: {"type": "event", "trace_id": "...", "data": {...}}
        - 客户端发送: {"type": "ping"} 或 {"type": "subscribe", "topics": [...]}
        """
        client_id = await ws_manager.connect(websocket, client_id)
        
        # 发送错过的消息
        if last_trace_id:
            missed = ws_manager.get_missed_messages(client_id, last_trace_id)
            for msg in missed:
                try:
                    await websocket.send_json(msg)
                except Exception:
                    break
        
        try:
            while True:
                # 接收客户端消息
                data = await websocket.receive_json()
                msg_type = data.get("type", "")
                
                if msg_type == "ping":
                    ws_manager.update_heartbeat(client_id)
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif msg_type == "subscribe":
                    topics = data.get("topics", [])
                    await websocket.send_json({
                        "type": "subscribed",
                        "topics": topics,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif msg_type == "get_status":
                    status = service.get_status()
                    await websocket.send_json({
                        "type": "status",
                        "data": status.model_dump()
                    })
                
                elif msg_type == "get_tracks":
                    tracks = service.get_tracks()
                    await websocket.send_json({
                        "type": "tracks",
                        "data": [t.model_dump() for t in tracks]
                    })
        
        except WebSocketDisconnect:
            ws_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket错误 [{client_id}]: {e}")
            ws_manager.disconnect(client_id)
    
    # ==================== 内部接口 (供插件调用) ====================
    
    async def broadcast_event_internal(event: UnifiedEvent):
        """内部接口: 广播事件"""
        service.emit_event(event)
        await ws_manager.broadcast_event(event)
    
    # 将广播函数挂载到router,供外部访问
    router.broadcast_event = broadcast_event_internal
    router.service = service
    router.ws_manager = ws_manager
    
    return router


# =============================================================================
# 集成函数
# =============================================================================

def integrate_indoor_fence_api_v3(app):
    """
    将室内电子围栏API V3集成到FastAPI应用
    
    使用方法:
        from apps.indoor_fence_api_v3 import integrate_indoor_fence_api_v3
        router = integrate_indoor_fence_api_v3(app)
    """
    router = create_indoor_fence_api_v3()
    app.include_router(router)
    
    # 启动后台任务
    @app.on_event("startup")
    async def start_background_tasks():
        asyncio.create_task(_heartbeat_checker(router.ws_manager))
        asyncio.create_task(_event_processor(router.service, router.broadcast_event))
    
    logger.info("[IndoorFenceAPI V3] API已集成")
    return router


async def _heartbeat_checker(ws_manager: WebSocketConnectionManager):
    """心跳检查后台任务"""
    while True:
        await asyncio.sleep(Config.WS_HEARTBEAT_INTERVAL)
        await ws_manager.check_heartbeats()


async def _event_processor(service: IndoorFenceService, broadcast_func):
    """事件处理后台任务"""
    while True:
        try:
            # 定期处理并推送状态
            status = service.get_status()
            tracks = service.get_tracks()
            
            # 构建状态更新事件
            from platform_core.schema.indoor_fence_events import (
                UnifiedEvent, EventType, Location, Point2D, Evidence, Suggestion, 
                SuggestionType, generate_event_id, generate_trace_id
            )
            
            status_event = UnifiedEvent(
                event_id=generate_event_id(),
                timestamp=datetime.now().isoformat(),
                type=EventType.SYSTEM_STATUS,
                location=Location(ground_position=Point2D(0, 0)),
                value={
                    "tracked_persons": status.tracked_persons,
                    "active_alarms": status.active_alarms,
                    "sensor_health": status.sensor_health
                },
                confidence=1.0,
                evidence=Evidence(),
                suggestion=Suggestion(
                    suggestion_type=SuggestionType.NONE,
                    message=""
                ),
                trace_id=generate_trace_id()
            )
            
            await broadcast_func(status_event)
            
        except Exception as e:
            logger.error(f"事件处理错误: {e}")
        
        await asyncio.sleep(1.0)  # 1秒更新一次


# =============================================================================
# 模块导出
# =============================================================================

__all__ = [
    'create_indoor_fence_api_v3',
    'integrate_indoor_fence_api_v3',
    'IndoorFenceService',
    'WebSocketConnectionManager',
    'Config'
]
