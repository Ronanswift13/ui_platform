# -*- coding: utf-8 -*-
"""
全景监测仪表盘API
==================

实现方案中的全景可视化与运维平台集成功能：
1. 统一数据接口
2. 实时状态监控
3. 历史趋势分析
4. 告警联动
5. 3D地图集成

作者: AI巡检系统
版本: 1.0.0
"""

from __future__ import annotations
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# =============================================================================
# 数据模型
# =============================================================================

class DeviceStatus(str, Enum):
    """设备状态"""
    NORMAL = "normal"
    ATTENTION = "attention"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class AlarmLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    ATTENTION = "attention"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_name: str
    device_type: str
    location: Dict[str, float]  # {"x": 0, "y": 0, "z": 0}
    status: DeviceStatus = DeviceStatus.NORMAL
    health_index: float = 100.0
    last_inspection: Optional[float] = None
    modalities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringData:
    """监测数据"""
    device_id: str
    modality: str
    timestamp: float
    values: Dict[str, float]
    status: str = "normal"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlarmRecord:
    """告警记录"""
    alarm_id: str
    device_id: str
    device_name: str
    alarm_type: str
    level: AlarmLevel
    message: str
    timestamp: float
    source_modality: str
    acknowledged: bool = False
    resolved: bool = False
    handler: Optional[str] = None
    resolve_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendData:
    """趋势数据"""
    device_id: str
    metric: str
    timestamps: List[float]
    values: List[float]
    predictions: Optional[List[float]] = None
    thresholds: Optional[Dict[str, float]] = None


# =============================================================================
# 中央状态缓存
# =============================================================================

class CentralStateCache:
    """
    中央状态缓存
    
    存储和管理所有设备的实时状态
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # 设备信息
        self.devices: Dict[str, DeviceInfo] = {}
        
        # 实时监测数据 (device_id -> modality -> data_list)
        self.monitoring_data: Dict[str, Dict[str, List[MonitoringData]]] = defaultdict(lambda: defaultdict(list))
        
        # 告警记录
        self.alarms: Dict[str, AlarmRecord] = {}
        self.active_alarms: List[str] = []
        
        # 健康指数历史
        self.health_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        
        # 融合结果缓存
        self.fusion_results: Dict[str, Dict] = {}
        
        # 锁
        self._lock = threading.RLock()
        
        # WebSocket连接
        self.websocket_clients: List[WebSocket] = []
        
    def register_device(self, device: DeviceInfo):
        """注册设备"""
        with self._lock:
            self.devices[device.device_id] = device
            logger.info(f"注册设备: {device.device_id} ({device.device_name})")
    
    def update_device_status(self, device_id: str, status: DeviceStatus, 
                            health_index: Optional[float] = None):
        """更新设备状态"""
        with self._lock:
            if device_id not in self.devices:
                return
            
            self.devices[device_id].status = status
            if health_index is not None:
                self.devices[device_id].health_index = health_index
                self.health_history[device_id].append((time.time(), health_index))
                
                # 限制历史长度
                if len(self.health_history[device_id]) > self.max_history:
                    self.health_history[device_id].pop(0)
    
    def add_monitoring_data(self, data: MonitoringData):
        """添加监测数据"""
        with self._lock:
            self.monitoring_data[data.device_id][data.modality].append(data)
            
            # 限制历史长度
            if len(self.monitoring_data[data.device_id][data.modality]) > self.max_history:
                self.monitoring_data[data.device_id][data.modality].pop(0)
            
            # 更新设备状态
            if data.device_id in self.devices:
                if data.status != 'normal':
                    status_map = {
                        'attention': DeviceStatus.ATTENTION,
                        'warning': DeviceStatus.WARNING,
                        'alarm': DeviceStatus.ALARM,
                        'critical': DeviceStatus.CRITICAL
                    }
                    new_status = status_map.get(data.status, DeviceStatus.NORMAL)
                    
                    # 取最严重的状态
                    current = self.devices[data.device_id].status
                    priority = [DeviceStatus.NORMAL, DeviceStatus.ATTENTION, 
                               DeviceStatus.WARNING, DeviceStatus.ALARM, DeviceStatus.CRITICAL]
                    if priority.index(new_status) > priority.index(current):
                        self.devices[data.device_id].status = new_status
    
    def add_alarm(self, alarm: AlarmRecord):
        """添加告警"""
        with self._lock:
            self.alarms[alarm.alarm_id] = alarm
            if not alarm.resolved:
                self.active_alarms.append(alarm.alarm_id)
            
            # 广播告警
            asyncio.create_task(self._broadcast_alarm(alarm))
    
    async def _broadcast_alarm(self, alarm: AlarmRecord):
        """广播告警到所有WebSocket客户端"""
        message = {
            'type': 'alarm',
            'data': asdict(alarm)
        }
        message['data']['level'] = alarm.level.value
        
        for client in self.websocket_clients[:]:
            try:
                await client.send_json(message)
            except Exception as e:
                logger.warning(f"WebSocket发送失败: {e}")
                self.websocket_clients.remove(client)
    
    def acknowledge_alarm(self, alarm_id: str, handler: str) -> bool:
        """确认告警"""
        with self._lock:
            if alarm_id in self.alarms:
                self.alarms[alarm_id].acknowledged = True
                self.alarms[alarm_id].handler = handler
                return True
            return False
    
    def resolve_alarm(self, alarm_id: str) -> bool:
        """解决告警"""
        with self._lock:
            if alarm_id in self.alarms:
                self.alarms[alarm_id].resolved = True
                self.alarms[alarm_id].resolve_time = time.time()
                if alarm_id in self.active_alarms:
                    self.active_alarms.remove(alarm_id)
                return True
            return False
    
    def update_fusion_result(self, device_id: str, result: Dict):
        """更新融合结果"""
        with self._lock:
            self.fusion_results[device_id] = {
                **result,
                'timestamp': time.time()
            }
    
    def get_device_overview(self) -> Dict[str, Any]:
        """获取设备概览"""
        with self._lock:
            status_count = defaultdict(int)
            for device in self.devices.values():
                status_count[device.status.value] += 1
            
            return {
                'total_devices': len(self.devices),
                'by_status': dict(status_count),
                'active_alarms': len(self.active_alarms),
                'average_health': sum(d.health_index for d in self.devices.values()) / max(len(self.devices), 1)
            }
    
    def get_device_details(self, device_id: str) -> Optional[Dict]:
        """获取设备详情"""
        with self._lock:
            if device_id not in self.devices:
                return None
            
            device = self.devices[device_id]
            
            # 收集各模态最新数据
            latest_data = {}
            for modality, data_list in self.monitoring_data[device_id].items():
                if data_list:
                    latest = data_list[-1]
                    latest_data[modality] = asdict(latest)
            
            # 收集相关告警
            device_alarms = [
                asdict(a) for a in self.alarms.values() 
                if a.device_id == device_id and not a.resolved
            ]
            for alarm in device_alarms:
                alarm['level'] = alarm['level'].value if isinstance(alarm['level'], AlarmLevel) else alarm['level']
            
            return {
                'device': {
                    **asdict(device),
                    'status': device.status.value
                },
                'latest_data': latest_data,
                'active_alarms': device_alarms,
                'fusion_result': self.fusion_results.get(device_id),
                'health_history': self.health_history.get(device_id, [])[-100:]
            }
    
    def get_trend_data(self, device_id: str, modality: str, 
                      metric: str, hours: int = 24) -> TrendData:
        """获取趋势数据"""
        with self._lock:
            timestamps = []
            values = []
            
            cutoff = time.time() - hours * 3600
            
            if device_id in self.monitoring_data and modality in self.monitoring_data[device_id]:
                for data in self.monitoring_data[device_id][modality]:
                    if data.timestamp >= cutoff:
                        timestamps.append(data.timestamp)
                        if metric in data.values:
                            values.append(data.values[metric])
            
            return TrendData(
                device_id=device_id,
                metric=metric,
                timestamps=timestamps,
                values=values
            )


# =============================================================================
# 全景监测服务
# =============================================================================

class PanoramicMonitoringService:
    """
    全景监测服务
    
    提供统一的监测数据接口
    """
    
    def __init__(self, cache: CentralStateCache):
        self.cache = cache
        
        # 插件引用
        self.plugins: Dict[str, Any] = {}
        
        # 告警处理器
        self.alarm_handlers: List[Callable[[AlarmRecord], None]] = []
        
    def register_plugin(self, plugin_id: str, plugin: Any):
        """注册插件"""
        self.plugins[plugin_id] = plugin
        logger.info(f"注册插件到全景监测: {plugin_id}")
    
    def register_alarm_handler(self, handler: Callable[[AlarmRecord], None]):
        """注册告警处理器"""
        self.alarm_handlers.append(handler)
    
    def process_plugin_result(self, plugin_id: str, device_id: str, result: Dict):
        """处理插件结果"""
        # 确定模态
        modality = self._get_modality_from_plugin(plugin_id)
        
        # 创建监测数据
        monitoring_data = MonitoringData(
            device_id=device_id,
            modality=modality,
            timestamp=result.get('timestamp', time.time()),
            values=result.get('values', {}),
            status=result.get('status', 'normal'),
            confidence=result.get('confidence', 1.0),
            metadata=result.get('metadata', {})
        )
        
        self.cache.add_monitoring_data(monitoring_data)
        
        # 处理告警
        alarms = result.get('alarms', [])
        for alarm_data in alarms:
            alarm = self._create_alarm(device_id, modality, alarm_data)
            self.cache.add_alarm(alarm)
            
            # 调用告警处理器
            for handler in self.alarm_handlers:
                try:
                    handler(alarm)
                except Exception as e:
                    logger.error(f"告警处理器错误: {e}")
        
        # 更新设备状态
        self._update_device_from_result(device_id, result)
    
    def process_fusion_result(self, device_id: str, result: Dict):
        """处理融合结果"""
        self.cache.update_fusion_result(device_id, result)
        
        # 更新设备状态
        overall_status = result.get('overall_status', 'normal')
        health_index = result.get('confidence', 1.0) * 100
        
        status_map = {
            'normal': DeviceStatus.NORMAL,
            'attention': DeviceStatus.ATTENTION,
            'warning': DeviceStatus.WARNING,
            'alarm': DeviceStatus.ALARM,
            'critical': DeviceStatus.CRITICAL
        }
        
        self.cache.update_device_status(
            device_id,
            status_map.get(overall_status, DeviceStatus.NORMAL),
            health_index
        )
        
        # 处理融合告警
        for alarm_data in result.get('alarms', []):
            alarm = self._create_alarm(device_id, 'fusion', alarm_data)
            self.cache.add_alarm(alarm)
    
    def _get_modality_from_plugin(self, plugin_id: str) -> str:
        """从插件ID获取模态"""
        modality_map = {
            'acoustic_monitoring': 'acoustic',
            'gas_detection': 'gas',
            'hyperspectral_detection': 'hyperspectral',
            'transformer_inspection': 'visual',
            'thermal_monitoring': 'thermal',
            'slam_mapping': 'lidar'
        }
        return modality_map.get(plugin_id, 'unknown')
    
    def _create_alarm(self, device_id: str, modality: str, alarm_data: Dict) -> AlarmRecord:
        """创建告警记录"""
        device = self.cache.devices.get(device_id)
        device_name = device.device_name if device else device_id
        
        level_str = alarm_data.get('level', 'warning')
        try:
            level = AlarmLevel(level_str)
        except ValueError:
            level = AlarmLevel.WARNING
        
        return AlarmRecord(
            alarm_id=f"alarm_{int(time.time() * 1000)}",
            device_id=device_id,
            device_name=device_name,
            alarm_type=alarm_data.get('type', 'unknown'),
            level=level,
            message=alarm_data.get('message', ''),
            timestamp=alarm_data.get('timestamp', time.time()),
            source_modality=modality,
            metadata=alarm_data.get('metadata', {})
        )
    
    def _update_device_from_result(self, device_id: str, result: Dict):
        """从结果更新设备状态"""
        status = result.get('status', 'normal')
        
        if status == 'normal':
            return
        
        status_map = {
            'attention': DeviceStatus.ATTENTION,
            'warning': DeviceStatus.WARNING,
            'alarm': DeviceStatus.ALARM,
            'critical': DeviceStatus.CRITICAL
        }
        
        new_status = status_map.get(status, DeviceStatus.NORMAL)
        self.cache.update_device_status(device_id, new_status)


# =============================================================================
# 告警联动服务
# =============================================================================

class AlarmLinkageService:
    """
    告警联动服务
    
    处理告警的通知和联动
    """
    
    def __init__(self):
        # 通知目标
        self.websocket_endpoint: Optional[str] = None
        self.http_endpoint: Optional[str] = None
        self.email_recipients: List[str] = []
        self.sms_recipients: List[str] = []
        
        # 告警规则
        self.escalation_rules: List[Dict] = []
        
        # 静默规则
        self.silence_rules: List[Dict] = []
        
        # 告警计数 (用于升级)
        self.alarm_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def configure(self, config: Dict):
        """配置告警联动"""
        self.websocket_endpoint = config.get('websocket_endpoint')
        self.http_endpoint = config.get('http_endpoint')
        self.email_recipients = config.get('email_recipients', [])
        self.sms_recipients = config.get('sms_recipients', [])
        self.escalation_rules = config.get('escalation_rules', [])
        self.silence_rules = config.get('silence_rules', [])
    
    async def handle_alarm(self, alarm: AlarmRecord):
        """处理告警"""
        # 检查静默规则
        if self._is_silenced(alarm):
            logger.debug(f"告警被静默: {alarm.alarm_id}")
            return
        
        # 计数
        self.alarm_counts[alarm.device_id][alarm.alarm_type] += 1
        
        # 检查升级
        alarm = self._check_escalation(alarm)
        
        # 发送通知
        await self._send_notifications(alarm)
    
    def _is_silenced(self, alarm: AlarmRecord) -> bool:
        """检查是否被静默"""
        for rule in self.silence_rules:
            if self._match_rule(alarm, rule):
                return True
        return False
    
    def _check_escalation(self, alarm: AlarmRecord) -> AlarmRecord:
        """检查并应用升级规则"""
        count = self.alarm_counts[alarm.device_id][alarm.alarm_type]
        
        for rule in self.escalation_rules:
            if count >= rule.get('threshold', 3):
                target_level = AlarmLevel(rule.get('escalate_to', 'critical'))
                if self._level_priority(target_level) > self._level_priority(alarm.level):
                    alarm.level = target_level
                    alarm.message = f"[升级] {alarm.message}"
                break
        
        return alarm
    
    def _level_priority(self, level: AlarmLevel) -> int:
        """获取告警级别优先级"""
        priorities = {
            AlarmLevel.INFO: 0,
            AlarmLevel.ATTENTION: 1,
            AlarmLevel.WARNING: 2,
            AlarmLevel.ALARM: 3,
            AlarmLevel.CRITICAL: 4
        }
        return priorities.get(level, 0)
    
    def _match_rule(self, alarm: AlarmRecord, rule: Dict) -> bool:
        """匹配规则"""
        if 'device_id' in rule and alarm.device_id != rule['device_id']:
            return False
        if 'alarm_type' in rule and alarm.alarm_type != rule['alarm_type']:
            return False
        if 'level' in rule:
            if isinstance(rule['level'], list):
                if alarm.level.value not in rule['level']:
                    return False
            elif alarm.level.value != rule['level']:
                return False
        return True
    
    async def _send_notifications(self, alarm: AlarmRecord):
        """发送通知"""
        tasks = []
        
        # WebSocket通知 (由cache处理)
        
        # HTTP通知
        if self.http_endpoint:
            tasks.append(self._send_http_notification(alarm))
        
        # 邮件通知 (仅高级别)
        if alarm.level in [AlarmLevel.ALARM, AlarmLevel.CRITICAL] and self.email_recipients:
            tasks.append(self._send_email_notification(alarm))
        
        # 短信通知 (仅critical)
        if alarm.level == AlarmLevel.CRITICAL and self.sms_recipients:
            tasks.append(self._send_sms_notification(alarm))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_http_notification(self, alarm: AlarmRecord):
        """发送HTTP通知"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                data = asdict(alarm)
                data['level'] = alarm.level.value
                await session.post(self.http_endpoint, json=data)
        except Exception as e:
            logger.error(f"HTTP通知发送失败: {e}")
    
    async def _send_email_notification(self, alarm: AlarmRecord):
        """发送邮件通知"""
        # 实际实现需要配置邮件服务器
        logger.info(f"邮件通知: {alarm.message} -> {self.email_recipients}")
    
    async def _send_sms_notification(self, alarm: AlarmRecord):
        """发送短信通知"""
        # 实际实现需要配置短信服务
        logger.info(f"短信通知: {alarm.message} -> {self.sms_recipients}")


# =============================================================================
# FastAPI路由
# =============================================================================

if FASTAPI_AVAILABLE:
    
    class DeviceOverviewResponse(BaseModel):
        """设备概览响应"""
        total_devices: int
        by_status: Dict[str, int]
        active_alarms: int
        average_health: float
    
    class DeviceDetailResponse(BaseModel):
        """设备详情响应"""
        device: Dict[str, Any]
        latest_data: Dict[str, Dict]
        active_alarms: List[Dict]
        fusion_result: Optional[Dict]
        health_history: List[List[float]]
    
    class AlarmListResponse(BaseModel):
        """告警列表响应"""
        total: int
        active: int
        alarms: List[Dict]
    
    class TrendDataResponse(BaseModel):
        """趋势数据响应"""
        device_id: str
        metric: str
        timestamps: List[float]
        values: List[float]
        predictions: Optional[List[float]]
        thresholds: Optional[Dict[str, float]]


def create_panoramic_router(cache: CentralStateCache, 
                           service: PanoramicMonitoringService) -> APIRouter:
    """创建全景监测API路由"""
    
    router = APIRouter(prefix="/api/panoramic", tags=["全景监测"])
    
    @router.get("/overview", response_model=DeviceOverviewResponse, summary="获取设备概览")
    async def get_overview():
        """获取所有设备的状态概览"""
        return cache.get_device_overview()
    
    @router.get("/devices", summary="获取设备列表")
    async def list_devices(
        status: Optional[str] = Query(None, description="按状态筛选"),
        device_type: Optional[str] = Query(None, description="按类型筛选")
    ):
        """获取设备列表"""
        devices = []
        for device in cache.devices.values():
            if status and device.status.value != status:
                continue
            if device_type and device.device_type != device_type:
                continue
            
            devices.append({
                **asdict(device),
                'status': device.status.value
            })
        
        return {
            'total': len(devices),
            'devices': devices
        }
    
    @router.get("/devices/{device_id}", summary="获取设备详情")
    async def get_device_detail(device_id: str):
        """获取设备详细信息"""
        detail = cache.get_device_details(device_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="设备不存在")
        return detail
    
    @router.get("/devices/{device_id}/trend", summary="获取趋势数据")
    async def get_device_trend(
        device_id: str,
        modality: str = Query(..., description="模态类型"),
        metric: str = Query(..., description="指标名称"),
        hours: int = Query(24, description="时间范围(小时)")
    ):
        """获取设备的趋势数据"""
        trend = cache.get_trend_data(device_id, modality, metric, hours)
        return {
            'device_id': trend.device_id,
            'metric': trend.metric,
            'timestamps': trend.timestamps,
            'values': trend.values,
            'predictions': trend.predictions,
            'thresholds': trend.thresholds
        }
    
    @router.get("/alarms", summary="获取告警列表")
    async def list_alarms(
        active_only: bool = Query(True, description="仅活动告警"),
        device_id: Optional[str] = Query(None, description="按设备筛选"),
        level: Optional[str] = Query(None, description="按级别筛选"),
        limit: int = Query(100, description="返回数量限制")
    ):
        """获取告警列表"""
        alarms = []
        
        for alarm in cache.alarms.values():
            if active_only and alarm.resolved:
                continue
            if device_id and alarm.device_id != device_id:
                continue
            if level and alarm.level.value != level:
                continue
            
            alarm_dict = asdict(alarm)
            alarm_dict['level'] = alarm.level.value
            alarms.append(alarm_dict)
        
        # 按时间排序
        alarms.sort(key=lambda x: x['timestamp'], reverse=True)
        alarms = alarms[:limit]
        
        return {
            'total': len(cache.alarms),
            'active': len(cache.active_alarms),
            'alarms': alarms
        }
    
    @router.post("/alarms/{alarm_id}/acknowledge", summary="确认告警")
    async def acknowledge_alarm(alarm_id: str, handler: str = Query(..., description="处理人")):
        """确认告警"""
        success = cache.acknowledge_alarm(alarm_id, handler)
        if not success:
            raise HTTPException(status_code=404, detail="告警不存在")
        return {'success': True, 'message': '告警已确认'}
    
    @router.post("/alarms/{alarm_id}/resolve", summary="解决告警")
    async def resolve_alarm(alarm_id: str):
        """解决告警"""
        success = cache.resolve_alarm(alarm_id)
        if not success:
            raise HTTPException(status_code=404, detail="告警不存在")
        return {'success': True, 'message': '告警已解决'}
    
    @router.get("/fusion/{device_id}", summary="获取融合结果")
    async def get_fusion_result(device_id: str):
        """获取设备的多模态融合结果"""
        result = cache.fusion_results.get(device_id)
        if result is None:
            raise HTTPException(status_code=404, detail="无融合结果")
        return result
    
    @router.get("/statistics", summary="获取统计信息")
    async def get_statistics():
        """获取系统统计信息"""
        overview = cache.get_device_overview()
        
        # 告警统计
        alarm_by_level = defaultdict(int)
        alarm_by_type = defaultdict(int)
        for alarm in cache.alarms.values():
            if not alarm.resolved:
                alarm_by_level[alarm.level.value] += 1
                alarm_by_type[alarm.alarm_type] += 1
        
        # 模态数据统计
        modality_counts = defaultdict(int)
        for device_data in cache.monitoring_data.values():
            for modality in device_data:
                modality_counts[modality] += len(device_data[modality])
        
        return {
            'overview': overview,
            'alarm_by_level': dict(alarm_by_level),
            'alarm_by_type': dict(alarm_by_type),
            'monitoring_data_by_modality': dict(modality_counts),
            'timestamp': time.time()
        }
    
    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket实时数据推送"""
        await websocket.accept()
        cache.websocket_clients.append(websocket)
        
        try:
            while True:
                # 接收客户端消息
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # 处理订阅请求
                if message.get('type') == 'subscribe':
                    await websocket.send_json({
                        'type': 'subscribed',
                        'channels': message.get('channels', [])
                    })
                
                # 心跳
                elif message.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong'})
                
        except WebSocketDisconnect:
            cache.websocket_clients.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
            if websocket in cache.websocket_clients:
                cache.websocket_clients.remove(websocket)
    
    return router


def integrate_panoramic_routes(app, cache: CentralStateCache = None, 
                               service: PanoramicMonitoringService = None):
    """集成全景监测路由到FastAPI应用"""
    if cache is None:
        cache = CentralStateCache()
    if service is None:
        service = PanoramicMonitoringService(cache)
    
    router = create_panoramic_router(cache, service)
    app.include_router(router)
    
    logger.info("全景监测API路由已集成")
    return cache, service


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'DeviceStatus',
    'AlarmLevel',
    'DeviceInfo',
    'MonitoringData',
    'AlarmRecord',
    'TrendData',
    'CentralStateCache',
    'PanoramicMonitoringService',
    'AlarmLinkageService',
    'create_panoramic_router',
    'integrate_panoramic_routes'
]
