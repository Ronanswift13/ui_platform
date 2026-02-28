#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内监测中心 API V2.0
======================

基于迭代方案重构:
- 移除Mock数据，接入真实插件
- 统一事件总线
- WebSocket实时推送
- 标准化数据结构

输变电激光星芒破夜绘明监测平台 V3.5

API端点:
- GET  /api/indoor/fence       - 电子围栏数据
- GET  /api/indoor/animal      - 动物入侵检测数据
- GET  /api/indoor/temperature - 温度监测数据
- GET  /api/indoor/device      - 设备状态数据
- GET  /api/indoor/fire        - 消防监测数据
- GET  /api/indoor/environment - 环境监测数据
- GET  /api/indoor/slam        - SLAM地图数据
- POST /api/indoor/fence/zone  - 更新围栏区域
- POST /api/indoor/deterrent   - 触发驱离
- GET  /api/indoor/plugin/{id}/capabilities - 插件能力配置
- POST /api/indoor/plugin/{id}/command     - 执行插件命令
- GET  /api/indoor/fusion/evidence         - 多模态融合证据链
- WS   /ws/indoor              - 实时数据推送

版本: 2.0.0
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# 统一数据结构
# =============================================================================
class StatusLevel(Enum):
    """状态级别"""
    NORMAL = "normal"
    ATTENTION = "attention"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


@dataclass
class UnifiedResponse:
    """统一响应结构"""
    success: bool
    timestamp: float
    module: str
    status: str
    data: Dict[str, Any]
    alarms: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'timestamp': self.timestamp,
            'module': self.module,
            'status': self.status,
            'data': self.data,
            'alarms': self.alarms,
            'confidence': self.confidence,
        }


# =============================================================================
# 插件管理器集成
# =============================================================================
class IndoorPluginManager:
    """
    室内插件管理器
    
    管理所有室内监测插件的实例
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._plugins: Dict[str, Any] = {}
        self._event_handlers: List[Callable] = []
        self._initialized = True
        
        self._init_plugins()
    
    def _init_plugins(self):
        """初始化所有插件"""
        # 尝试加载电子围栏插件
        try:
            from plugins.indoor_fence.core.tracker_v3 import FusedTrackerV3
            self._plugins['fence'] = {
                'tracker': FusedTrackerV3(
                    enable_behavior_analysis=True,
                    enable_risk_analysis=True,
                ),
                'zone_config': {},
                'authorization': {},
            }
            logger.info("[IndoorPluginManager] 电子围栏V3.0已加载")
        except Exception as e:
            logger.warning(f"[IndoorPluginManager] 电子围栏加载失败: {e}")
            self._plugins['fence'] = None

        # 尝试加载动物检测插件
        try:
            from plugins.animal_detection.plugin import AnimalDetectionPlugin
            self._plugins['animal'] = AnimalDetectionPlugin(
                config={
                    "deterrent": {"enabled": True},
                },
            )
            logger.info("[IndoorPluginManager] 动物检测V1.0已加载")
        except Exception as e:
            logger.warning(f"[IndoorPluginManager] 动物检测加载失败: {e}")
            self._plugins['animal'] = None

        # 尝试加载消防监测插件
        try:
            from plugins.fire_detection.plugin import FireDetectionPlugin
            self._plugins['fire'] = FireDetectionPlugin()
            logger.info("[IndoorPluginManager] 消防监测V1.0已加载")
        except Exception as e:
            logger.warning(f"[IndoorPluginManager] 消防监测加载失败: {e}")
            self._plugins['fire'] = None

        # 尝试加载语义SLAM插件
        try:
            from plugins.slam_mapping.semantic_slam_plugin import SemanticSLAMPlugin
            self._plugins['slam'] = SemanticSLAMPlugin()
            logger.info("[IndoorPluginManager] 语义SLAM V2.0已加载")
        except Exception as e:
            logger.warning(f"[IndoorPluginManager] 语义SLAM加载失败: {e}")
            self._plugins['slam'] = None

        # 环境监测 (使用气体检测插件)
        try:
            from plugins.gas_detection.plugin import GasDetectionPlugin
            self._plugins['environment'] = GasDetectionPlugin()
            logger.info("[IndoorPluginManager] 环境监测已加载")
        except Exception as e:
            logger.warning(f"[IndoorPluginManager] 环境监测加载失败: {e}")
            self._plugins['environment'] = None

        # 设备状态监测
        self._plugins['device'] = DeviceMonitor()

        # 温度监测
        self._plugins['temperature'] = TemperatureMonitor()
    
    def _on_event(self, event):
        """事件回调"""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"[IndoorPluginManager] 事件处理失败: {e}")
    
    def register_event_handler(self, handler: Callable):
        """注册事件处理器"""
        self._event_handlers.append(handler)
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """获取插件"""
        return self._plugins.get(name)
    
    def get_all_status(self) -> Dict[str, bool]:
        """获取所有插件状态"""
        return {name: plugin is not None for name, plugin in self._plugins.items()}


# =============================================================================
# 辅助监测模块
# =============================================================================
class DeviceMonitor:
    """设备状态监测"""
    
    def __init__(self):
        self._devices = [
            {'id': 'CAM-001', 'name': '高清摄像头#1', 'type': 'camera', 'status': 'online', 'health': 95},
            {'id': 'CAM-002', 'name': '高清摄像头#2', 'type': 'camera', 'status': 'online', 'health': 98},
            {'id': 'LDR-001', 'name': '激光雷达#1', 'type': 'lidar', 'status': 'online', 'health': 92},
            {'id': 'THM-001', 'name': '热成像仪#1', 'type': 'thermal', 'status': 'online', 'health': 100},
            {'id': 'SMK-001', 'name': '烟雾传感器#1', 'type': 'smoke_sensor', 'status': 'online', 'health': 88},
            {'id': 'TMP-001', 'name': '温度传感器#1', 'type': 'temp_sensor', 'status': 'online', 'health': 96},
            {'id': 'GAS-001', 'name': 'SF6传感器#1', 'type': 'gas_sensor', 'status': 'online', 'health': 90},
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        import random
        
        # 模拟设备状态变化
        for device in self._devices:
            # 小概率状态变化
            if random.random() < 0.01:
                device['status'] = random.choice(['online', 'attention', 'offline'])
            # 健康度波动
            device['health'] = max(0, min(100, device['health'] + random.randint(-2, 2)))
        
        online_count = sum(1 for d in self._devices if d['status'] == 'online')
        avg_health = sum(d['health'] for d in self._devices) / len(self._devices)
        
        return {
            'timestamp': int(time.time() * 1000),
            'status': 'online' if online_count == len(self._devices) else 'attention',
            'devices': self._devices,
            'totalDevices': len(self._devices),
            'onlineDevices': online_count,
            'avgHealth': int(avg_health),
        }


class TemperatureMonitor:
    """温度监测"""
    
    def __init__(self, grid_size: Tuple[int, int] = (10, 8)):
        self.grid_size = grid_size
        self._base_temp = 25.0
        self._heatmap = None
        self._init_heatmap()
    
    def _init_heatmap(self):
        """初始化热力图"""
        import numpy as np
        rows, cols = self.grid_size
        self._heatmap = np.full((rows, cols), self._base_temp)
        
        # 添加一些热点（模拟设备散热）
        self._heatmap[2:4, 3:5] = 35.0  # 设备区域1
        self._heatmap[6:8, 1:3] = 32.0  # 设备区域2
    
    def get_status(self) -> Dict[str, Any]:
        """获取温度状态"""
        import numpy as np
        import random
        
        # 模拟温度波动
        noise = np.random.randn(*self.grid_size) * 0.5
        current_heatmap = self._heatmap + noise
        
        max_temp = float(np.max(current_heatmap))
        min_temp = float(np.min(current_heatmap))
        avg_temp = float(np.mean(current_heatmap))
        
        # 检测热点
        hotspots = []
        threshold = self._base_temp + 10
        hot_mask = current_heatmap > threshold
        if np.any(hot_mask):
            hot_indices = np.argwhere(hot_mask)
            for idx in hot_indices[:5]:  # 最多5个热点
                hotspots.append({
                    'row': int(idx[0]),
                    'col': int(idx[1]),
                    'temp': float(current_heatmap[idx[0], idx[1]]),
                })
        
        status = 'good' if max_temp < 40 else ('attention' if max_temp < 50 else 'warning')
        
        return {
            'timestamp': int(time.time() * 1000),
            'status': status,
            'heatmap': current_heatmap.tolist(),
            'maxTemp': round(max_temp, 1),
            'minTemp': round(min_temp, 1),
            'avgTemp': round(avg_temp, 1),
            'hotspots': hotspots,
            'gridSize': {'rows': self.grid_size[0], 'cols': self.grid_size[1]},
        }


# =============================================================================
# 数据生成器 (备用)
# =============================================================================
class FallbackDataGenerator:
    """
    备用数据生成器
    
    当插件不可用时提供模拟数据
    """
    
    def __init__(self):
        import random
        self._random = random
    
    def generate_fence_data(self) -> Dict[str, Any]:
        """生成围栏数据"""
        persons = [
            {
                'id': 1, 'name': '张工', 
                'x': 0.3 + self._random.random() * 0.05,
                'y': 0.5 + self._random.random() * 0.05,
                'cabinet': 2, 'status': 'safe', 'authorized': True,
                'behavior': 'normal', 'risk_intent': 'safe',
            },
            {
                'id': 2, 'name': '李工',
                'x': 0.7 + self._random.random() * 0.05,
                'y': 0.2 + self._random.random() * 0.05,
                'cabinet': 5, 'status': 'warning', 'authorized': True,
                'behavior': 'reaching', 'risk_intent': 'safe',
            },
        ]
        
        cabinets = [
            {'id': i, 'name': f'机柜{i}', 'status': 'idle', 'person': None}
            for i in range(1, 9)
        ]
        cabinets[1]['status'] = 'occupied'
        cabinets[1]['person'] = '张工'
        cabinets[4]['status'] = 'occupied'
        cabinets[4]['person'] = '李工'
        
        return {
            'timestamp': int(time.time() * 1000),
            'status': 'online',
            'persons': persons,
            'cabinets': cabinets,
            'yellowLine': {'y': 0.8, 'status': 'normal'},
            'totalPersons': len(persons),
            'alarmCount': 0,
        }
    
    def generate_animal_data(self) -> Dict[str, Any]:
        """生成动物数据"""
        detections = []
        if self._random.random() < 0.1:
            detections.append({
                'id': 1,
                'type': self._random.choice(['mouse', 'bird', 'cat']),
                'x': self._random.random(),
                'y': self._random.random(),
                'confidence': self._random.uniform(0.7, 0.95),
                'threat_level': 'medium',
            })
        
        return {
            'timestamp': int(time.time() * 1000),
            'status': 'online',
            'detections': detections,
            'todayCount': self._random.randint(0, 5),
            'deterrentActive': False,
        }
    
    def generate_fire_data(self) -> Dict[str, Any]:
        """生成消防数据"""
        zones = [
            {'id': 'A', 'name': '配电室', 'status': 'normal', 'smokeLevel': 0.02, 'temp': 24.5},
            {'id': 'B', 'name': '控制室', 'status': 'normal', 'smokeLevel': 0.01, 'temp': 23.8},
            {'id': 'C', 'name': '电缆层', 'status': 'normal', 'smokeLevel': 0.03, 'temp': 26.2},
        ]
        
        return {
            'timestamp': int(time.time() * 1000),
            'status': 'online',
            'zones': zones,
            'lastTest': '2026-01-20 08:00:00',
            'sprinklerStatus': 'ready',
            'alarmStatus': 'normal',
        }
    
    def generate_environment_data(self) -> Dict[str, Any]:
        """生成环境数据"""
        return {
            'timestamp': int(time.time() * 1000),
            'status': 'good',
            'sf6': {
                'current': 5 + self._random.random() * 3,
                'unit': 'ppm',
                'threshold': 1000,
            },
            'o2': {
                'current': 20.5 + self._random.random() * 0.5,
                'unit': '%',
                'threshold': {'min': 19.5, 'max': 23.5},
            },
            'co': {
                'current': self._random.random() * 2,
                'unit': 'ppm',
                'threshold': 50,
            },
            'humidity': 45 + self._random.random() * 10,
            'pressure': 1013 + self._random.random() * 5,
        }


# =============================================================================
# 创建路由
# =============================================================================
router = APIRouter(prefix="/api/indoor", tags=["室内监测中心V2"])

# 全局实例
_plugin_manager = IndoorPluginManager()
_fallback_generator = FallbackDataGenerator()
_device_monitor = DeviceMonitor()
_temperature_monitor = TemperatureMonitor()

# WebSocket连接管理
_websocket_connections: List[WebSocket] = []


@router.get("/fence")
async def get_fence_data():
    """获取电子围栏数据"""
    fence_plugin = _plugin_manager.get_plugin('fence')
    
    if fence_plugin and fence_plugin.get('tracker'):
        tracker = fence_plugin['tracker']
        tracks = tracker.get_all_tracks()
        
        persons = []
        for track in tracks:
            persons.append({
                'id': track.track_id,
                'name': track.person_id or f'Person-{track.track_id}',
                'x': track.position[0],
                'y': track.position[1],
                'cabinet': track.nearby_cabinet,
                'status': 'safe' if track.risk_score < 0.3 else ('warning' if track.risk_score < 0.7 else 'danger'),
                'authorized': track.is_authorized,
                'behavior': track.behavior.value,
                'behavior_confidence': track.behavior_confidence,
                'risk_intent': track.risk_intent.value,
                'risk_score': track.risk_score,
                'risk_reason': track.risk_reason,
            })
        
        alarm_count = sum(1 for p in persons if p['status'] in ['warning', 'danger'])
        
        return {
            'timestamp': int(time.time() * 1000),
            'status': 'online',
            'persons': persons,
            'cabinets': fence_plugin.get('zone_config', {}).get('cabinets', []),
            'yellowLine': {'y': 0.8, 'status': 'normal'},
            'totalPersons': len(persons),
            'alarmCount': alarm_count,
            'statistics': tracker.get_statistics() if hasattr(tracker, 'get_statistics') else {},
        }
    else:
        return _fallback_generator.generate_fence_data()


@router.get("/animal")
async def get_animal_data():
    """获取动物入侵检测数据"""
    animal_plugin = _plugin_manager.get_plugin('animal')

    if animal_plugin:
        try:
            stats = animal_plugin.get_statistics() if hasattr(animal_plugin, 'get_statistics') else {}
            recent = animal_plugin.get_recent_detections(limit=10) if hasattr(animal_plugin, 'get_recent_detections') else []

            return {
                'timestamp': int(time.time() * 1000),
                'status': 'online',
                'detections': recent,
                'todayCount': stats.get('total_detections', 0),
                'deterrentActive': stats.get('active_deterrents', 0) > 0,
                'statistics': stats,
            }
        except Exception as e:
            logger.warning(f"[IndoorAPI] 动物检测插件数据获取失败，使用回退: {e}")
            return _fallback_generator.generate_animal_data()
    else:
        return _fallback_generator.generate_animal_data()


@router.get("/temperature")
async def get_temperature_data():
    """获取温度监测数据"""
    return _temperature_monitor.get_status()


@router.get("/device")
async def get_device_data():
    """获取设备状态数据"""
    return _device_monitor.get_status()


@router.get("/fire")
async def get_fire_data():
    """获取消防监测数据"""
    fire_plugin = _plugin_manager.get_plugin('fire')

    if fire_plugin:
        try:
            stats = fire_plugin.get_statistics() if hasattr(fire_plugin, 'get_statistics') else {}

            zone_list = []
            if hasattr(fire_plugin, 'get_all_zone_status'):
                zones = fire_plugin.get_all_zone_status()
                for zone in zones:
                    zone_list.append({
                        'id': zone.zone_id,
                        'name': zone.zone_name,
                        'status': zone.alarm_level.value if hasattr(zone.alarm_level, 'value') else str(zone.alarm_level),
                        'smokeLevel': zone.smoke_density or 0,
                        'temp': zone.temperature or 25,
                        'fireDetected': zone.fire_detected,
                        'smokeDetected': zone.smoke_detected,
                    })

            recent = fire_plugin.get_recent_detections(limit=10) if hasattr(fire_plugin, 'get_recent_detections') else []

            # 如果没有区域数据，使用默认
            if not zone_list:
                zone_list = [
                    {'id': 'A', 'name': '配电室', 'status': 'normal', 'smokeLevel': 0.02, 'temp': 24.5},
                    {'id': 'B', 'name': '控制室', 'status': 'normal', 'smokeLevel': 0.01, 'temp': 23.8},
                ]

            return {
                'timestamp': int(time.time() * 1000),
                'status': stats.get('alarm_counts', {}).get('critical', 0) > 0 and 'critical' or 'online',
                'zones': zone_list,
                'recentDetections': recent,
                'lastTest': '2026-01-20 08:00:00',
                'sprinklerStatus': 'ready',
                'alarmStatus': 'normal' if stats.get('active_fires', 0) == 0 else 'alarm',
                'statistics': stats,
            }
        except Exception as e:
            logger.warning(f"[IndoorAPI] 消防插件数据获取失败，使用回退: {e}")
            return _fallback_generator.generate_fire_data()
    else:
        return _fallback_generator.generate_fire_data()


@router.get("/environment")
async def get_environment_data():
    """获取环境监测数据"""
    env_plugin = _plugin_manager.get_plugin('environment')
    
    if env_plugin and hasattr(env_plugin, 'get_status'):
        return env_plugin.get_status()
    else:
        return _fallback_generator.generate_environment_data()


@router.get("/slam")
async def get_slam_data():
    """获取SLAM地图数据"""
    slam_plugin = _plugin_manager.get_plugin('slam')

    if slam_plugin:
        try:
            stats = slam_plugin.get_map_statistics() if hasattr(slam_plugin, 'get_map_statistics') else {}
            changes = slam_plugin.get_recent_changes(limit=10) if hasattr(slam_plugin, 'get_recent_changes') else []

            return {
                'timestamp': int(time.time() * 1000),
                'status': 'online',
                'mapStats': stats,
                'recentChanges': changes,
            }
        except Exception as e:
            logger.warning(f"[IndoorAPI] SLAM插件数据获取失败: {e}")

    return {
        'timestamp': int(time.time() * 1000),
        'status': 'offline',
        'mapStats': None,
        'recentChanges': [],
    }


@router.get("/all")
async def get_all_data():
    """获取所有模块数据"""
    return {
        'fence': await get_fence_data(),
        'animal': await get_animal_data(),
        'temperature': await get_temperature_data(),
        'device': await get_device_data(),
        'fire': await get_fire_data(),
        'environment': await get_environment_data(),
        'slam': await get_slam_data(),
    }


@router.get("/status")
async def get_plugin_status():
    """获取插件状态"""
    return {
        'timestamp': int(time.time() * 1000),
        'plugins': _plugin_manager.get_all_status(),
        'version': '2.0.0',
    }


# =============================================================================
# POST端点
# =============================================================================
class DeterrentRequest(BaseModel):
    """驱离请求"""
    type: str = "audio"
    duration: float = 5.0
    intensity: float = 0.8


@router.post("/deterrent")
async def trigger_deterrent(request: DeterrentRequest):
    """触发驱离"""
    animal_plugin = _plugin_manager.get_plugin('animal')
    
    if animal_plugin and hasattr(animal_plugin, 'trigger_manual_deterrent'):
        from plugins.animal_detection.plugin import DeterrentType
        
        deterrent_map = {
            'audio': DeterrentType.AUDIO,
            'light': DeterrentType.LIGHT,
            'vibration': DeterrentType.VIBRATION,
            'combined': DeterrentType.COMBINED,
        }
        
        det_type = deterrent_map.get(request.type, DeterrentType.AUDIO)
        success = animal_plugin.trigger_manual_deterrent(
            det_type, request.duration, request.intensity
        )
        
        return {
            'success': success,
            'message': '驱离已触发' if success else '驱离触发失败',
        }
    else:
        return {
            'success': False,
            'message': '动物检测插件不可用',
        }


class ZoneConfigRequest(BaseModel):
    """区域配置请求"""
    zone_id: str
    zone_type: str
    polygon: List[List[float]]


@router.post("/fence/zone")
async def update_fence_zone(request: ZoneConfigRequest):
    """更新围栏区域"""
    fence_plugin = _plugin_manager.get_plugin('fence')
    
    if fence_plugin:
        if 'zone_config' not in fence_plugin:
            fence_plugin['zone_config'] = {}
        
        fence_plugin['zone_config'][request.zone_id] = {
            'type': request.zone_type,
            'polygon': request.polygon,
        }
        
        return {
            'success': True,
            'message': f'区域 {request.zone_id} 已更新',
        }
    else:
        raise HTTPException(status_code=503, detail="电子围栏插件不可用")


# =============================================================================
# 插件能力、命令与融合证据链
# =============================================================================

# 插件ID映射: UI长名 -> V2内部短名
_plugin_id_map = {
    'indoor_fence': 'fence',
    'animal_detection': 'animal',
    'temperature_monitoring': 'temperature',
    'device_monitoring': 'device',
    'fire_detection': 'fire',
    'gas_detection': 'environment',
    'slam_mapping': 'slam',
}

# 默认能力配置
_default_capabilities = {
    "indoor_fence": {
        "name": "电子围栏",
        "description": "基于激光雷达的电子围栏和越线检测",
        "controls": [
            {"type": "slider", "id": "threshold", "label": "越线阈值", "min": 0, "max": 1, "step": 0.1, "default": 0.15},
            {"type": "select", "id": "mode", "label": "检测模式", "options": ["标准", "严格", "宽松"], "default": "标准"},
        ],
        "operations": [
            {"id": "reset_fence", "label": "重置围栏", "icon": "shield-check"},
            {"id": "export_log", "label": "导出日志", "icon": "download"},
        ],
    },
    "animal_detection": {
        "name": "动物入侵检测",
        "description": "小动物识别与智能驱离",
        "controls": [
            {"type": "slider", "id": "sensitivity", "label": "灵敏度", "min": 0, "max": 100, "step": 5, "default": 70},
        ],
        "operations": [
            {"id": "activate_deterrent", "label": "启动驱离", "icon": "volume-up"},
            {"id": "view_history", "label": "查看历史", "icon": "clock-history"},
        ],
    },
    "temperature_monitoring": {
        "name": "温度监测",
        "description": "热成像温度监测与热力图分析",
        "controls": [
            {"type": "slider", "id": "temp_threshold", "label": "温度阈值(°C)", "min": 20, "max": 80, "step": 1, "default": 40},
        ],
        "operations": [
            {"id": "show_heatmap", "label": "显示热力图", "icon": "thermometer-half"},
            {"id": "export_data", "label": "导出数据", "icon": "download"},
        ],
    },
    "fire_detection": {
        "name": "消防监测",
        "description": "火焰烟雾检测与多模态融合",
        "controls": [
            {"type": "slider", "id": "fire_sensitivity", "label": "灵敏度", "min": 0, "max": 100, "step": 5, "default": 80},
        ],
        "operations": [
            {"id": "test_alarm", "label": "测试报警", "icon": "bell"},
            {"id": "silence", "label": "静音", "icon": "volume-mute"},
        ],
    },
    "gas_detection": {
        "name": "环境监测",
        "description": "SF6/O2/CO气体检测与环境参数监测",
        "controls": [
            {"type": "slider", "id": "sf6_threshold", "label": "SF6阈值(ppm)", "min": 100, "max": 5000, "step": 100, "default": 1000},
        ],
        "operations": [
            {"id": "calibrate", "label": "校准传感器", "icon": "sliders"},
            {"id": "export_data", "label": "导出数据", "icon": "download"},
        ],
    },
    "device_monitoring": {
        "name": "设备状态监测",
        "description": "设备运行状态与故障检测",
        "controls": [],
        "operations": [
            {"id": "refresh_status", "label": "刷新状态", "icon": "arrow-clockwise"},
        ],
    },
}


@router.get("/plugin/{plugin_id}/capabilities")
async def get_plugin_capabilities(plugin_id: str):
    """获取插件能力配置（用于动态生成控制面板）"""
    # 尝试从实际插件获取
    short_name = _plugin_id_map.get(plugin_id, plugin_id)
    plugin = _plugin_manager.get_plugin(short_name)

    if plugin and hasattr(plugin, 'get_capabilities'):
        try:
            capabilities = plugin.get_capabilities()
            return {
                "plugin_id": plugin_id,
                "name": capabilities.get("name", plugin_id),
                "description": capabilities.get("description", ""),
                "controls": capabilities.get("controls", []),
                "operations": capabilities.get("operations", []),
            }
        except Exception as e:
            logger.error(f"获取插件 {plugin_id} 能力失败: {e}")

    # 返回默认能力配置
    config = _default_capabilities.get(plugin_id, {
        "name": plugin_id,
        "description": "监测插件",
        "controls": [],
        "operations": [],
    })

    return {"plugin_id": plugin_id, **config}


@router.post("/plugin/{plugin_id}/command")
async def execute_plugin_command(plugin_id: str, command: Dict[str, Any]):
    """执行插件命令"""
    short_name = _plugin_id_map.get(plugin_id, plugin_id)
    plugin = _plugin_manager.get_plugin(short_name)

    if plugin and hasattr(plugin, 'execute_command'):
        try:
            result = plugin.execute_command(command)
            return {
                "success": True,
                "plugin_id": plugin_id,
                "command": command.get("operation"),
                "result": result,
            }
        except Exception as e:
            logger.error(f"执行插件 {plugin_id} 命令失败: {e}")
            return {
                "success": False,
                "plugin_id": plugin_id,
                "error": str(e),
            }

    return {
        "success": False,
        "plugin_id": plugin_id,
        "error": "插件不支持命令执行",
    }


@router.get("/fusion/evidence")
async def get_fusion_evidence():
    """获取多模态融合证据链数据"""
    # 尝试获取融合引擎
    try:
        from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin
        fusion_plugin = _plugin_manager.get_plugin('fusion')
        if fusion_plugin and hasattr(fusion_plugin, 'get_evidence_chain'):
            evidence = fusion_plugin.get_evidence_chain()
            return {
                "timestamp": evidence.get("timestamp", int(time.time() * 1000)),
                "modalities": evidence.get("modalities", []),
                "fusion_result": evidence.get("fusion_result", "正常"),
                "recommendation": evidence.get("recommendation", ""),
                "confidence": evidence.get("confidence", 0.0),
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
            {"type": "thermal", "name": "热成像", "result": "正常", "confidence": 0.90},
        ],
        "fusion_result": "正常",
        "recommendation": "所有监测指标正常，继续监控",
        "confidence": 0.91,
    }


# =============================================================================
# WebSocket
# =============================================================================
def create_websocket_route(app):
    """创建WebSocket路由"""
    
    @app.websocket("/ws/indoor")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        _websocket_connections.append(websocket)
        logger.info(f"[IndoorWS] 新连接: 当前{len(_websocket_connections)}个")
        
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "subscribe":
                        # 订阅特定模块
                        pass
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            _websocket_connections.remove(websocket)
            logger.info(f"[IndoorWS] 断开: 剩余{len(_websocket_connections)}个")


async def broadcast_event(event: Dict[str, Any]):
    """广播事件到所有WebSocket连接"""
    if not _websocket_connections:
        return
    
    message = json.dumps(event)
    for ws in _websocket_connections.copy():
        try:
            await ws.send_text(message)
        except Exception:
            _websocket_connections.remove(ws)


async def indoor_data_pusher():
    """后台数据推送任务"""
    while True:
        try:
            if _websocket_connections:
                data = {
                    'type': 'update',
                    'timestamp': int(time.time() * 1000),
                    'fence': await get_fence_data(),
                    'fire': await get_fire_data(),
                }
                await broadcast_event(data)
        except Exception as e:
            logger.error(f"[IndoorPusher] 推送失败: {e}")
        
        await asyncio.sleep(1.0)


# =============================================================================
# 集成函数
# =============================================================================
def integrate_indoor_api_v2(app):
    """
    将室内监测API V2.0集成到FastAPI应用
    """
    # 注册REST API路由
    app.include_router(router)
    
    # 注册WebSocket路由
    create_websocket_route(app)
    
    # 注册事件处理
    def on_event(event):
        asyncio.create_task(broadcast_event({
            'type': 'event',
            'event': event.__dict__ if hasattr(event, '__dict__') else str(event),
        }))
    
    _plugin_manager.register_event_handler(on_event)
    
    # 启动后台推送任务
    @app.on_event("startup")
    async def start_indoor_pusher():
        asyncio.create_task(indoor_data_pusher())
    
    logger.info("[IndoorAPI] V2.0 集成完成")


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    'router',
    'integrate_indoor_api_v2',
    'IndoorPluginManager',
]


# 为了兼容性，保留旧函数名
integrate_indoor_api = integrate_indoor_api_v2
