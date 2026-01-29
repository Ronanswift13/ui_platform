#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内监测中心 3D可视化 API
==========================

为3D数字孪生提供数据接口:
- 点云数据
- 轨迹数据
- 热力图数据
- 告警管理

输变电激光星芒破夜绘明监测平台 V3.5

版本: 1.0.0
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import math
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# 数据模型
# =============================================================================
class AlarmLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


class AlarmStatus(str, Enum):
    """告警状态"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISPERSING = "dispersing"
    REPORTED = "reported"
    RESOLVED = "resolved"


class AlarmType(str, Enum):
    """告警类型"""
    FENCE_VIOLATION = "fence_violation"
    ANIMAL_INTRUSION = "animal_intrusion"
    FIRE_DETECTED = "fire_detected"
    SMOKE_DETECTED = "smoke_detected"
    TEMPERATURE_HIGH = "temperature_high"
    GAS_ABNORMAL = "gas_abnormal"
    DEVICE_OFFLINE = "device_offline"


@dataclass
class Point3D:
    """3D点"""
    x: float
    y: float
    z: float
    intensity: float = 1.0
    semantic: int = 0  # 语义标签

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    x: float
    y: float
    z: float = 0.0
    timestamp: float = 0.0
    speed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    """轨迹"""
    id: str
    object_id: str
    object_type: str  # person, animal, fire
    points: List[TrajectoryPoint] = field(default_factory=list)
    color: str = "#3b82f6"
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'object_id': self.object_id,
            'type': self.object_type,
            'points': [p.to_dict() for p in self.points],
            'color': self.color,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'speed': self._calculate_avg_speed(),
        }

    def _calculate_avg_speed(self) -> float:
        if len(self.points) < 2:
            return 0.0
        speeds = [p.speed for p in self.points if p.speed > 0]
        return sum(speeds) / len(speeds) if speeds else 0.0


@dataclass
class Alarm:
    """告警"""
    id: str
    type: AlarmType
    level: AlarmLevel
    status: AlarmStatus
    message: str
    x: float
    y: float
    z: float = 0.0
    timestamp: float = 0.0
    source_id: str = ""
    source_type: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    confirmed_at: Optional[float] = None
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'level': self.level.value,
            'status': self.status.value,
            'message': self.message,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'timestamp': self.timestamp,
            'source_id': self.source_id,
            'source_type': self.source_type,
            'details': self.details,
            'confirmed_at': self.confirmed_at,
            'resolved_at': self.resolved_at,
        }


# =============================================================================
# 数据存储
# =============================================================================
class Indoor3DDataStore:
    """3D数据存储"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._point_cloud: List[Point3D] = []
        self._trajectories: Dict[str, Trajectory] = {}
        self._alarms: Dict[str, Alarm] = {}
        self._heatmap_data: Dict[str, Any] = {}
        self._zones: List[Dict[str, Any]] = []

        self._initialized = True
        self._init_demo_data()

    def _init_demo_data(self):
        """初始化演示数据"""
        # 生成演示点云
        self._generate_demo_point_cloud()

        # 生成演示区域
        self._generate_demo_zones()

        # 生成演示告警
        self._generate_demo_alarms()

        logger.info("[Indoor3DDataStore] 演示数据初始化完成")

    def _generate_demo_point_cloud(self):
        """生成演示点云 - 模拟室内环境"""
        points = []

        # 地面点
        for x in range(-10, 11):
            for z in range(-10, 11):
                points.append(Point3D(
                    x=x * 0.5,
                    y=0,
                    z=z * 0.5,
                    intensity=0.3,
                    semantic=0  # GROUND
                ))

        # 墙壁点
        for i in range(-10, 11):
            for h in range(0, 6):
                # 四面墙
                points.append(Point3D(x=-5, y=h * 0.5, z=i * 0.5, intensity=0.5, semantic=1))
                points.append(Point3D(x=5, y=h * 0.5, z=i * 0.5, intensity=0.5, semantic=1))
                points.append(Point3D(x=i * 0.5, y=h * 0.5, z=-5, intensity=0.5, semantic=1))
                points.append(Point3D(x=i * 0.5, y=h * 0.5, z=5, intensity=0.5, semantic=1))

        # 机柜点 (模拟8个机柜)
        cabinet_positions = [
            (-3, -3), (-1, -3), (1, -3), (3, -3),
            (-3, 3), (-1, 3), (1, 3), (3, 3),
        ]
        for cx, cz in cabinet_positions:
            for dx in range(-1, 2):
                for dz in range(-1, 2):
                    for h in range(0, 8):
                        points.append(Point3D(
                            x=cx + dx * 0.2,
                            y=h * 0.25,
                            z=cz + dz * 0.2,
                            intensity=0.7,
                            semantic=2  # CABINET
                        ))

        self._point_cloud = points

    def _generate_demo_zones(self):
        """生成演示区域"""
        self._zones = [
            {
                'id': 'zone-danger-1',
                'name': '高压危险区',
                'type': 'danger',
                'polygon': [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]],
            },
            {
                'id': 'zone-warning-1',
                'name': '黄线警戒区',
                'type': 'warning',
                'polygon': [[0.1, 0.7], [0.9, 0.7], [0.9, 0.8], [0.1, 0.8]],
            },
            {
                'id': 'zone-safe-1',
                'name': '安全作业区',
                'type': 'safe',
                'polygon': [[0.6, 0.2], [0.8, 0.2], [0.8, 0.5], [0.6, 0.5]],
            },
        ]

    def _generate_demo_alarms(self):
        """生成演示告警"""
        now = time.time()

        demo_alarms = [
            Alarm(
                id='alarm-001',
                type=AlarmType.FENCE_VIOLATION,
                level=AlarmLevel.ALARM,
                status=AlarmStatus.PENDING,
                message='李工越过黄线警戒区',
                x=0.7,
                y=0.75,
                timestamp=now - 120,
                source_id='person-2',
                source_type='person',
                details={'person_name': '李工', 'zone': '黄线警戒区'},
            ),
            Alarm(
                id='alarm-002',
                type=AlarmType.TEMPERATURE_HIGH,
                level=AlarmLevel.WARNING,
                status=AlarmStatus.PENDING,
                message='GIS室温度异常升高',
                x=0.3,
                y=0.4,
                timestamp=now - 300,
                source_id='temp-sensor-1',
                source_type='sensor',
                details={'temperature': 48.5, 'threshold': 45},
            ),
            Alarm(
                id='alarm-003',
                type=AlarmType.ANIMAL_INTRUSION,
                level=AlarmLevel.WARNING,
                status=AlarmStatus.CONFIRMED,
                message='检测到老鼠入侵',
                x=0.5,
                y=0.3,
                timestamp=now - 600,
                source_id='animal-1',
                source_type='animal',
                details={'animal_type': 'mouse', 'confidence': 0.85},
                confirmed_at=now - 500,
            ),
        ]

        for alarm in demo_alarms:
            self._alarms[alarm.id] = alarm

    # =========================================================================
    # 点云操作
    # =========================================================================
    def get_point_cloud(self,
                        semantic_filter: Optional[List[int]] = None,
                        bounds: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """获取点云数据"""
        points = self._point_cloud

        if semantic_filter:
            points = [p for p in points if p.semantic in semantic_filter]

        if bounds:
            points = [p for p in points if
                      bounds.get('min_x', -float('inf')) <= p.x <= bounds.get('max_x', float('inf')) and
                      bounds.get('min_y', -float('inf')) <= p.y <= bounds.get('max_y', float('inf')) and
                      bounds.get('min_z', -float('inf')) <= p.z <= bounds.get('max_z', float('inf'))]

        return [p.to_dict() for p in points]

    def update_point_cloud(self, points: List[Dict[str, Any]]):
        """更新点云数据"""
        self._point_cloud = [Point3D(**p) for p in points]

    # =========================================================================
    # 轨迹操作
    # =========================================================================
    def get_trajectories(self,
                         object_type: Optional[str] = None,
                         time_range: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """获取轨迹数据"""
        trajectories = list(self._trajectories.values())

        if object_type:
            trajectories = [t for t in trajectories if t.object_type == object_type]

        if time_range:
            start, end = time_range
            trajectories = [t for t in trajectories if
                           t.start_time >= start and t.end_time <= end]

        return [t.to_dict() for t in trajectories]

    def add_trajectory_point(self, object_id: str, object_type: str, point: Dict[str, Any]):
        """添加轨迹点"""
        trajectory_id = f"traj-{object_id}"

        if trajectory_id not in self._trajectories:
            self._trajectories[trajectory_id] = Trajectory(
                id=trajectory_id,
                object_id=object_id,
                object_type=object_type,
                start_time=point.get('timestamp', time.time()),
            )

        trajectory = self._trajectories[trajectory_id]
        trajectory.points.append(TrajectoryPoint(**point))
        trajectory.end_time = point.get('timestamp', time.time())

        # 限制轨迹点数量
        if len(trajectory.points) > 100:
            trajectory.points = trajectory.points[-100:]
            trajectory.start_time = trajectory.points[0].timestamp

    def clear_trajectories(self, object_id: Optional[str] = None):
        """清除轨迹"""
        if object_id:
            trajectory_id = f"traj-{object_id}"
            if trajectory_id in self._trajectories:
                del self._trajectories[trajectory_id]
        else:
            self._trajectories.clear()

    # =========================================================================
    # 热力图操作
    # =========================================================================
    def get_heatmap(self, heatmap_type: str = 'temperature') -> Dict[str, Any]:
        """获取热力图数据"""
        if heatmap_type not in self._heatmap_data:
            self._heatmap_data[heatmap_type] = self._generate_heatmap(heatmap_type)

        return self._heatmap_data[heatmap_type]

    def _generate_heatmap(self, heatmap_type: str) -> Dict[str, Any]:
        """生成热力图数据"""
        rows, cols = 10, 10

        if heatmap_type == 'temperature':
            # 温度热力图
            data = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    base_temp = 25
                    # 添加热点
                    if 3 <= i <= 5 and 3 <= j <= 5:
                        base_temp = 35 + random.random() * 10
                    elif 7 <= i <= 9 and 1 <= j <= 3:
                        base_temp = 30 + random.random() * 5
                    else:
                        base_temp = 22 + random.random() * 5
                    row.append(round(base_temp, 1))
                data.append(row)

            return {
                'type': 'temperature',
                'data': data,
                'min': 22,
                'max': 50,
                'unit': '°C',
                'timestamp': int(time.time() * 1000),
            }

        elif heatmap_type == 'gas':
            # 气体浓度热力图
            data = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    concentration = random.random() * 10
                    if 2 <= i <= 4 and 6 <= j <= 8:
                        concentration = 15 + random.random() * 10
                    row.append(round(concentration, 2))
                data.append(row)

            return {
                'type': 'gas',
                'data': data,
                'min': 0,
                'max': 30,
                'unit': 'ppm',
                'timestamp': int(time.time() * 1000),
            }

        elif heatmap_type == 'density':
            # 人员密度热力图
            data = [[0] * cols for _ in range(rows)]
            # 模拟人员位置
            person_positions = [(3, 5), (7, 2), (5, 8)]
            for px, py in person_positions:
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = px + di, py + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            distance = math.sqrt(di**2 + dj**2)
                            data[ni][nj] += max(0, 1 - distance / 3)

            return {
                'type': 'density',
                'data': data,
                'min': 0,
                'max': 3,
                'unit': '人',
                'timestamp': int(time.time() * 1000),
            }

        return {'type': heatmap_type, 'data': [], 'timestamp': int(time.time() * 1000)}

    def update_heatmap(self, heatmap_type: str, data: List[List[float]]):
        """更新热力图数据"""
        if heatmap_type in self._heatmap_data:
            self._heatmap_data[heatmap_type]['data'] = data
            self._heatmap_data[heatmap_type]['timestamp'] = int(time.time() * 1000)

    # =========================================================================
    # 区域操作
    # =========================================================================
    def get_zones(self) -> List[Dict[str, Any]]:
        """获取区域数据"""
        return self._zones

    def add_zone(self, zone: Dict[str, Any]):
        """添加区域"""
        self._zones.append(zone)

    def update_zone(self, zone_id: str, zone_data: Dict[str, Any]):
        """更新区域"""
        for i, zone in enumerate(self._zones):
            if zone['id'] == zone_id:
                self._zones[i] = {**zone, **zone_data}
                return True
        return False

    def delete_zone(self, zone_id: str):
        """删除区域"""
        self._zones = [z for z in self._zones if z['id'] != zone_id]

    # =========================================================================
    # 告警操作
    # =========================================================================
    def get_alarms(self,
                   status: Optional[AlarmStatus] = None,
                   level: Optional[AlarmLevel] = None,
                   alarm_type: Optional[AlarmType] = None,
                   limit: int = 50) -> List[Dict[str, Any]]:
        """获取告警列表"""
        alarms = list(self._alarms.values())

        if status:
            alarms = [a for a in alarms if a.status == status]
        if level:
            alarms = [a for a in alarms if a.level == level]
        if alarm_type:
            alarms = [a for a in alarms if a.type == alarm_type]

        # 按时间排序
        alarms.sort(key=lambda a: a.timestamp, reverse=True)

        return [a.to_dict() for a in alarms[:limit]]

    def get_alarm(self, alarm_id: str) -> Optional[Dict[str, Any]]:
        """获取单个告警"""
        alarm = self._alarms.get(alarm_id)
        return alarm.to_dict() if alarm else None

    def add_alarm(self, alarm_data: Dict[str, Any]) -> str:
        """添加告警"""
        alarm_id = alarm_data.get('id', f"alarm-{int(time.time() * 1000)}")

        alarm = Alarm(
            id=alarm_id,
            type=AlarmType(alarm_data.get('type', 'fence_violation')),
            level=AlarmLevel(alarm_data.get('level', 'warning')),
            status=AlarmStatus.PENDING,
            message=alarm_data.get('message', ''),
            x=alarm_data.get('x', 0.5),
            y=alarm_data.get('y', 0.5),
            z=alarm_data.get('z', 0),
            timestamp=alarm_data.get('timestamp', time.time()),
            source_id=alarm_data.get('source_id', ''),
            source_type=alarm_data.get('source_type', ''),
            details=alarm_data.get('details', {}),
        )

        self._alarms[alarm_id] = alarm
        return alarm_id

    def update_alarm_status(self, alarm_id: str, status: AlarmStatus) -> bool:
        """更新告警状态"""
        if alarm_id not in self._alarms:
            return False

        alarm = self._alarms[alarm_id]
        alarm.status = status

        if status == AlarmStatus.CONFIRMED:
            alarm.confirmed_at = time.time()
        elif status == AlarmStatus.RESOLVED:
            alarm.resolved_at = time.time()

        return True

    def delete_alarm(self, alarm_id: str) -> bool:
        """删除告警"""
        if alarm_id in self._alarms:
            del self._alarms[alarm_id]
            return True
        return False

    def get_alarm_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        alarms = list(self._alarms.values())

        by_status = {}
        by_level = {}
        by_type = {}

        for alarm in alarms:
            by_status[alarm.status.value] = by_status.get(alarm.status.value, 0) + 1
            by_level[alarm.level.value] = by_level.get(alarm.level.value, 0) + 1
            by_type[alarm.type.value] = by_type.get(alarm.type.value, 0) + 1

        return {
            'total': len(alarms),
            'by_status': by_status,
            'by_level': by_level,
            'by_type': by_type,
            'pending_count': by_status.get('pending', 0),
            'critical_count': by_level.get('critical', 0),
        }


# =============================================================================
# API路由
# =============================================================================
router = APIRouter(prefix="/api/indoor/3d", tags=["室内3D可视化"])

# 全局数据存储
_data_store = Indoor3DDataStore()


# =========================================================================
# 点云API
# =========================================================================
@router.get("/pointcloud")
async def get_point_cloud(
    semantic: Optional[str] = Query(None, description="语义标签过滤，逗号分隔"),
    min_x: Optional[float] = None,
    max_x: Optional[float] = None,
    min_y: Optional[float] = None,
    max_y: Optional[float] = None,
    min_z: Optional[float] = None,
    max_z: Optional[float] = None,
):
    """获取点云数据"""
    semantic_filter = None
    if semantic:
        semantic_filter = [int(s) for s in semantic.split(',')]

    bounds = None
    if any([min_x, max_x, min_y, max_y, min_z, max_z]):
        bounds = {
            'min_x': min_x or -float('inf'),
            'max_x': max_x or float('inf'),
            'min_y': min_y or -float('inf'),
            'max_y': max_y or float('inf'),
            'min_z': min_z or -float('inf'),
            'max_z': max_z or float('inf'),
        }

    points = _data_store.get_point_cloud(semantic_filter, bounds)

    return {
        'timestamp': int(time.time() * 1000),
        'count': len(points),
        'points': points,
        'semantic_labels': {
            0: 'GROUND',
            1: 'WALL',
            2: 'CABINET',
            3: 'DOOR',
            4: 'EQUIPMENT',
            5: 'CABLE',
            6: 'PERSON',
            7: 'OBSTACLE',
        },
    }


# =========================================================================
# 轨迹API
# =========================================================================
@router.get("/trajectories")
async def get_trajectories(
    object_type: Optional[str] = Query(None, description="对象类型: person, animal, fire"),
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
):
    """获取轨迹数据"""
    time_range = None
    if start_time and end_time:
        time_range = (start_time, end_time)

    trajectories = _data_store.get_trajectories(object_type, time_range)

    return {
        'timestamp': int(time.time() * 1000),
        'count': len(trajectories),
        'trajectories': trajectories,
    }


class TrajectoryPointRequest(BaseModel):
    """轨迹点请求"""
    object_id: str
    object_type: str
    x: float
    y: float
    z: float = 0.0
    speed: float = 0.0


@router.post("/trajectories/point")
async def add_trajectory_point(request: TrajectoryPointRequest):
    """添加轨迹点"""
    point = {
        'x': request.x,
        'y': request.y,
        'z': request.z,
        'speed': request.speed,
        'timestamp': time.time(),
    }

    _data_store.add_trajectory_point(request.object_id, request.object_type, point)

    return {'success': True, 'message': '轨迹点已添加'}


@router.delete("/trajectories/{object_id}")
async def clear_trajectory(object_id: str):
    """清除指定对象的轨迹"""
    _data_store.clear_trajectories(object_id)
    return {'success': True, 'message': f'轨迹 {object_id} 已清除'}


# =========================================================================
# 热力图API
# =========================================================================
@router.get("/heatmap/{heatmap_type}")
async def get_heatmap(heatmap_type: str):
    """获取热力图数据

    Args:
        heatmap_type: 热力图类型 (temperature, gas, density)
    """
    if heatmap_type not in ['temperature', 'gas', 'density']:
        raise HTTPException(status_code=400, detail=f"不支持的热力图类型: {heatmap_type}")

    heatmap = _data_store.get_heatmap(heatmap_type)

    return {
        'timestamp': int(time.time() * 1000),
        'heatmap': heatmap,
    }


# =========================================================================
# 区域API
# =========================================================================
@router.get("/zones")
async def get_zones():
    """获取区域数据"""
    zones = _data_store.get_zones()

    return {
        'timestamp': int(time.time() * 1000),
        'count': len(zones),
        'zones': zones,
    }


class ZoneRequest(BaseModel):
    """区域请求"""
    id: str
    name: str
    type: str  # danger, warning, safe
    polygon: List[List[float]]


@router.post("/zones")
async def add_zone(request: ZoneRequest):
    """添加区域"""
    zone = {
        'id': request.id,
        'name': request.name,
        'type': request.type,
        'polygon': request.polygon,
    }

    _data_store.add_zone(zone)

    return {'success': True, 'message': f'区域 {request.id} 已添加'}


@router.put("/zones/{zone_id}")
async def update_zone(zone_id: str, request: ZoneRequest):
    """更新区域"""
    zone_data = {
        'name': request.name,
        'type': request.type,
        'polygon': request.polygon,
    }

    success = _data_store.update_zone(zone_id, zone_data)

    if success:
        return {'success': True, 'message': f'区域 {zone_id} 已更新'}
    else:
        raise HTTPException(status_code=404, detail=f"区域 {zone_id} 不存在")


@router.delete("/zones/{zone_id}")
async def delete_zone(zone_id: str):
    """删除区域"""
    _data_store.delete_zone(zone_id)
    return {'success': True, 'message': f'区域 {zone_id} 已删除'}


# =========================================================================
# 告警API
# =========================================================================
@router.get("/alarms")
async def get_alarms(
    status: Optional[str] = Query(None, description="状态过滤"),
    level: Optional[str] = Query(None, description="级别过滤"),
    alarm_type: Optional[str] = Query(None, description="类型过滤"),
    limit: int = Query(50, description="返回数量限制"),
):
    """获取告警列表"""
    status_enum = AlarmStatus(status) if status else None
    level_enum = AlarmLevel(level) if level else None
    type_enum = AlarmType(alarm_type) if alarm_type else None

    alarms = _data_store.get_alarms(status_enum, level_enum, type_enum, limit)
    statistics = _data_store.get_alarm_statistics()

    return {
        'timestamp': int(time.time() * 1000),
        'count': len(alarms),
        'alarms': alarms,
        'statistics': statistics,
    }


@router.get("/alarms/{alarm_id}")
async def get_alarm(alarm_id: str):
    """获取单个告警详情"""
    alarm = _data_store.get_alarm(alarm_id)

    if alarm:
        return {
            'timestamp': int(time.time() * 1000),
            'alarm': alarm,
        }
    else:
        raise HTTPException(status_code=404, detail=f"告警 {alarm_id} 不存在")


class AlarmRequest(BaseModel):
    """告警请求"""
    type: str
    level: str
    message: str
    x: float
    y: float
    z: float = 0.0
    source_id: str = ""
    source_type: str = ""
    details: Dict[str, Any] = {}


@router.post("/alarms")
async def add_alarm(request: AlarmRequest):
    """添加告警"""
    alarm_data = {
        'type': request.type,
        'level': request.level,
        'message': request.message,
        'x': request.x,
        'y': request.y,
        'z': request.z,
        'source_id': request.source_id,
        'source_type': request.source_type,
        'details': request.details,
    }

    alarm_id = _data_store.add_alarm(alarm_data)

    return {
        'success': True,
        'alarm_id': alarm_id,
        'message': '告警已创建',
    }


class AlarmStatusRequest(BaseModel):
    """告警状态更新请求"""
    status: str


@router.put("/alarms/{alarm_id}/status")
async def update_alarm_status(alarm_id: str, request: AlarmStatusRequest):
    """更新告警状态"""
    try:
        status = AlarmStatus(request.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"无效的状态: {request.status}")

    success = _data_store.update_alarm_status(alarm_id, status)

    if success:
        return {'success': True, 'message': f'告警 {alarm_id} 状态已更新为 {request.status}'}
    else:
        raise HTTPException(status_code=404, detail=f"告警 {alarm_id} 不存在")


@router.post("/alarms/{alarm_id}/confirm")
async def confirm_alarm(alarm_id: str):
    """确认告警"""
    success = _data_store.update_alarm_status(alarm_id, AlarmStatus.CONFIRMED)

    if success:
        return {'success': True, 'message': f'告警 {alarm_id} 已确认'}
    else:
        raise HTTPException(status_code=404, detail=f"告警 {alarm_id} 不存在")


@router.post("/alarms/{alarm_id}/disperse")
async def disperse_alarm(alarm_id: str):
    """驱离处理"""
    success = _data_store.update_alarm_status(alarm_id, AlarmStatus.DISPERSING)

    if success:
        # 触发驱离设备
        return {'success': True, 'message': f'告警 {alarm_id} 驱离处理中'}
    else:
        raise HTTPException(status_code=404, detail=f"告警 {alarm_id} 不存在")


@router.post("/alarms/{alarm_id}/report")
async def report_alarm(alarm_id: str):
    """上报告警"""
    success = _data_store.update_alarm_status(alarm_id, AlarmStatus.REPORTED)

    if success:
        return {'success': True, 'message': f'告警 {alarm_id} 已上报'}
    else:
        raise HTTPException(status_code=404, detail=f"告警 {alarm_id} 不存在")


@router.post("/alarms/{alarm_id}/resolve")
async def resolve_alarm(alarm_id: str):
    """解决告警"""
    success = _data_store.update_alarm_status(alarm_id, AlarmStatus.RESOLVED)

    if success:
        return {'success': True, 'message': f'告警 {alarm_id} 已解决'}
    else:
        raise HTTPException(status_code=404, detail=f"告警 {alarm_id} 不存在")


@router.delete("/alarms/{alarm_id}")
async def delete_alarm(alarm_id: str):
    """删除告警"""
    success = _data_store.delete_alarm(alarm_id)

    if success:
        return {'success': True, 'message': f'告警 {alarm_id} 已删除'}
    else:
        raise HTTPException(status_code=404, detail=f"告警 {alarm_id} 不存在")


# =========================================================================
# 综合数据API
# =========================================================================
@router.get("/scene")
async def get_scene_data():
    """获取完整场景数据 (用于初始化3D场景)"""
    return {
        'timestamp': int(time.time() * 1000),
        'pointcloud': {
            'count': len(_data_store._point_cloud),
            'points': _data_store.get_point_cloud(),
        },
        'zones': _data_store.get_zones(),
        'trajectories': _data_store.get_trajectories(),
        'heatmaps': {
            'temperature': _data_store.get_heatmap('temperature'),
            'gas': _data_store.get_heatmap('gas'),
        },
        'alarms': _data_store.get_alarms(limit=20),
        'statistics': _data_store.get_alarm_statistics(),
    }


# =============================================================================
# 集成函数
# =============================================================================
def integrate_indoor_3d_api(app):
    """
    将室内3D可视化API集成到FastAPI应用
    """
    app.include_router(router)
    logger.info("[Indoor3DAPI] 3D可视化API已集成")


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    'router',
    'integrate_indoor_3d_api',
    'Indoor3DDataStore',
    'Alarm',
    'AlarmLevel',
    'AlarmStatus',
    'AlarmType',
]
