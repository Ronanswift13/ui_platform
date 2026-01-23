"""
室内监测中心 API 路由
=====================

为室内监测中心提供RESTful API和WebSocket实时推送

输变电激光星芒破夜绘明监测平台 V3.5

API端点:
- GET  /api/indoor/fence       - 电子围栏数据
- GET  /api/indoor/animal      - 动物入侵检测数据
- GET  /api/indoor/temperature - 温度监测数据
- GET  /api/indoor/device      - 设备状态数据
- GET  /api/indoor/fire        - 消防监测数据
- GET  /api/indoor/environment - 环境监测数据
- WS   /ws/indoor              - 实时数据推送

版本: 1.0.0
"""

from __future__ import annotations
import asyncio
import json
import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

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


class AnimalDetection(BaseModel):
    """动物检测"""
    id: int
    type: str  # mouse, bird, snake, etc.
    confidence: float
    bbox: Dict[str, float]


class AnimalData(BaseModel):
    """动物入侵数据"""
    timestamp: int
    status: str
    detections: List[AnimalDetection]
    todayCount: int
    deterrentActive: bool


class TemperatureData(BaseModel):
    """温度监测数据"""
    timestamp: int
    status: str
    heatmap: List[List[float]]
    maxTemp: float
    minTemp: float
    avgTemp: float
    hotspots: List[Dict[str, Any]]


class DeviceInfo(BaseModel):
    """设备信息"""
    id: str
    name: str
    type: str
    status: str
    health: int


class DeviceData(BaseModel):
    """设备状态数据"""
    timestamp: int
    status: str
    devices: List[DeviceInfo]
    totalDevices: int
    onlineDevices: int
    avgHealth: int


class FireZone(BaseModel):
    """消防区域"""
    id: str
    name: str
    status: str
    smokeLevel: float
    temp: float


class FireData(BaseModel):
    """消防监测数据"""
    timestamp: int
    status: str
    zones: List[FireZone]
    lastTest: str
    sprinklerStatus: str
    alarmStatus: str


class GasData(BaseModel):
    """气体数据"""
    current: float
    unit: str
    threshold: Any
    history: List[float]


class EnvironmentData(BaseModel):
    """环境监测数据"""
    timestamp: int
    status: str
    sf6: GasData
    o2: GasData
    co: GasData
    humidity: float
    pressure: float


# =============================================================================
# 数据生成器（模拟数据）
# =============================================================================

class IndoorDataGenerator:
    """室内监测数据生成器"""
    
    def __init__(self):
        self._fence_persons = [
            {"id": 1, "name": "张工", "x": 0.3, "y": 0.5, "cabinet": 2, "status": "safe", "authorized": True},
            {"id": 2, "name": "李工", "x": 0.7, "y": 0.2, "cabinet": 5, "status": "danger", "authorized": True},
        ]
        self._animal_detection_chance = 0.1
        self._history_length = 30
        
        # 初始化历史数据
        self._sf6_history = [5 + random.random() * 3 for _ in range(self._history_length)]
        self._o2_history = [20.5 + random.random() * 0.5 for _ in range(self._history_length)]
        self._co_history = [random.random() * 2 for _ in range(self._history_length)]
    
    def generate_fence_data(self) -> Dict[str, Any]:
        """生成电子围栏数据"""
        timestamp = int(time.time() * 1000)
        
        # 随机更新人员位置
        for person in self._fence_persons:
            person["x"] = max(0.1, min(0.9, person["x"] + (random.random() - 0.5) * 0.02))
            person["y"] = max(0.1, min(0.9, person["y"] + (random.random() - 0.5) * 0.02))
            
            # 检查越线
            if person["y"] < 0.15:
                person["status"] = "danger"
            elif person["y"] < 0.25:
                person["status"] = "warning"
            else:
                person["status"] = "safe"
        
        # 生成机柜状态
        cabinets = []
        for i in range(1, 7):
            person = next((p for p in self._fence_persons if p["cabinet"] == i), None)
            status = "idle"
            if person:
                status = "alarm" if person["status"] == "danger" else "occupied"
            
            cabinets.append({
                "id": i,
                "name": f"机柜-{str(i).zfill(2)}",
                "status": status,
                "person": person["name"] if person else None
            })
        
        alarm_count = sum(1 for p in self._fence_persons if p["status"] == "danger")
        
        return {
            "timestamp": timestamp,
            "status": "alarm" if alarm_count > 0 else "online",
            "persons": self._fence_persons,
            "cabinets": cabinets,
            "yellowLine": {"y": 0.15},
            "totalPersons": len(self._fence_persons),
            "alarmCount": alarm_count
        }
    
    def generate_animal_data(self) -> Dict[str, Any]:
        """生成动物入侵数据"""
        timestamp = int(time.time() * 1000)
        
        detections = []
        if random.random() < self._animal_detection_chance:
            detections.append({
                "id": 1,
                "type": random.choice(["mouse", "bird", "cat"]),
                "confidence": 0.75 + random.random() * 0.2,
                "bbox": {
                    "x": 0.3 + random.random() * 0.4,
                    "y": 0.3 + random.random() * 0.4,
                    "width": 0.1,
                    "height": 0.08
                }
            })
        
        return {
            "timestamp": timestamp,
            "status": "alarm" if detections else "online",
            "detections": detections,
            "todayCount": random.randint(0, 5),
            "deterrentActive": len(detections) > 0
        }
    
    def generate_temperature_data(self) -> Dict[str, Any]:
        """生成温度监测数据"""
        timestamp = int(time.time() * 1000)
        
        # 生成热力图 (8x6)
        heatmap = []
        for y in range(6):
            row = []
            for x in range(8):
                temp = 25 + random.random() * 5
                # 创建热点
                if (x == 3 and y == 2) or (x == 6 and y == 4):
                    temp += 15 + random.random() * 10
                row.append(round(temp, 1))
            heatmap.append(row)
        
        flat = [t for row in heatmap for t in row]
        max_temp = max(flat)
        min_temp = min(flat)
        avg_temp = sum(flat) / len(flat)
        
        hotspots = []
        if max_temp > 40:
            hotspots.append({"x": 3, "y": 2, "temp": max_temp})
        
        return {
            "timestamp": timestamp,
            "status": "warning" if max_temp > 45 else "attention",
            "heatmap": heatmap,
            "maxTemp": round(max_temp, 1),
            "minTemp": round(min_temp, 1),
            "avgTemp": round(avg_temp, 1),
            "hotspots": hotspots
        }
    
    def generate_device_data(self) -> Dict[str, Any]:
        """生成设备状态数据"""
        timestamp = int(time.time() * 1000)
        
        devices = [
            {"id": "cam-01", "name": "摄像机-01", "type": "camera", "status": "online", "health": 98},
            {"id": "cam-02", "name": "摄像机-02", "type": "camera", "status": "online", "health": 95},
            {"id": "lidar-01", "name": "激光雷达", "type": "lidar", "status": "online", "health": 100},
            {"id": "sensor-01", "name": "温湿度传感器", "type": "sensor", "status": "online", "health": 92},
            {"id": "relay-01", "name": "继电器模块", "type": "relay", "status": "online", "health": 100},
            {"id": "speaker-01", "name": "声光报警器", "type": "alarm", "status": "standby", "health": 100},
        ]
        
        # 随机波动健康度
        for device in devices:
            device["health"] = max(80, min(100, device["health"] + random.randint(-2, 2)))
        
        online_count = sum(1 for d in devices if d["status"] in ["online", "standby"])
        avg_health = sum(d["health"] for d in devices) // len(devices)
        
        return {
            "timestamp": timestamp,
            "status": "online" if online_count == len(devices) else "warning",
            "devices": devices,
            "totalDevices": len(devices),
            "onlineDevices": online_count,
            "avgHealth": avg_health
        }
    
    def generate_fire_data(self) -> Dict[str, Any]:
        """生成消防监测数据"""
        timestamp = int(time.time() * 1000)
        
        zones = [
            {"id": "zone-a", "name": "A区-主控室", "status": "normal", "smokeLevel": 0.02 + random.random() * 0.01, "temp": 24 + random.random() * 2},
            {"id": "zone-b", "name": "B区-GIS室", "status": "normal", "smokeLevel": 0.01 + random.random() * 0.01, "temp": 23 + random.random() * 2},
            {"id": "zone-c", "name": "C区-继保室", "status": "normal", "smokeLevel": 0.015 + random.random() * 0.01, "temp": 25 + random.random() * 2},
            {"id": "zone-d", "name": "D区-蓄电池室", "status": "normal", "smokeLevel": 0.03 + random.random() * 0.02, "temp": 26 + random.random() * 2},
        ]
        
        return {
            "timestamp": timestamp,
            "status": "online",
            "zones": zones,
            "lastTest": "2026-01-20 09:00:00",
            "sprinklerStatus": "ready",
            "alarmStatus": "armed"
        }
    
    def generate_environment_data(self) -> Dict[str, Any]:
        """生成环境监测数据"""
        timestamp = int(time.time() * 1000)
        
        # 更新历史数据
        self._sf6_history.pop(0)
        self._sf6_history.append(5 + random.random() * 3)
        
        self._o2_history.pop(0)
        self._o2_history.append(20.5 + random.random() * 0.5)
        
        self._co_history.pop(0)
        self._co_history.append(random.random() * 2)
        
        return {
            "timestamp": timestamp,
            "status": "good",
            "sf6": {
                "current": self._sf6_history[-1],
                "unit": "ppm",
                "threshold": 1000,
                "history": self._sf6_history.copy()
            },
            "o2": {
                "current": self._o2_history[-1],
                "unit": "%",
                "threshold": {"min": 19.5, "max": 23.5},
                "history": self._o2_history.copy()
            },
            "co": {
                "current": self._co_history[-1],
                "unit": "ppm",
                "threshold": 50,
                "history": self._co_history.copy()
            },
            "humidity": 45 + random.random() * 10,
            "pressure": 1013 + random.random() * 5
        }


# =============================================================================
# 创建路由
# =============================================================================

router = APIRouter(prefix="/api/indoor", tags=["室内监测中心"])

# 数据生成器实例
_data_generator = IndoorDataGenerator()

# WebSocket连接管理
_websocket_connections: List[WebSocket] = []


@router.get("/fence")
async def get_fence_data():
    """获取电子围栏数据"""
    return _data_generator.generate_fence_data()


@router.get("/animal")
async def get_animal_data():
    """获取动物入侵检测数据"""
    return _data_generator.generate_animal_data()


@router.get("/temperature")
async def get_temperature_data():
    """获取温度监测数据"""
    return _data_generator.generate_temperature_data()


@router.get("/device")
async def get_device_data():
    """获取设备状态数据"""
    return _data_generator.generate_device_data()


@router.get("/fire")
async def get_fire_data():
    """获取消防监测数据"""
    return _data_generator.generate_fire_data()


@router.get("/environment")
async def get_environment_data():
    """获取环境监测数据"""
    return _data_generator.generate_environment_data()


@router.get("/all")
async def get_all_data():
    """获取所有模块数据"""
    return {
        "fence": _data_generator.generate_fence_data(),
        "animal": _data_generator.generate_animal_data(),
        "temperature": _data_generator.generate_temperature_data(),
        "device": _data_generator.generate_device_data(),
        "fire": _data_generator.generate_fire_data(),
        "environment": _data_generator.generate_environment_data()
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
    modules = ["fence", "animal", "temperature", "device", "fire", "environment"]
    generators = {
        "fence": _data_generator.generate_fence_data,
        "animal": _data_generator.generate_animal_data,
        "temperature": _data_generator.generate_temperature_data,
        "device": _data_generator.generate_device_data,
        "fire": _data_generator.generate_fire_data,
        "environment": _data_generator.generate_environment_data,
    }
    
    while True:
        for module in modules:
            data = generators[module]()
            await broadcast_update(module, data)
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
    
    logger.info("室内监测中心API已集成")


# =============================================================================
# 独立运行测试
# =============================================================================

if __name__ == "__main__":
    # 测试数据生成
    generator = IndoorDataGenerator()
    
    print("=== 电子围栏数据 ===")
    print(json.dumps(generator.generate_fence_data(), indent=2, ensure_ascii=False))
    
    print("\n=== 动物入侵数据 ===")
    print(json.dumps(generator.generate_animal_data(), indent=2, ensure_ascii=False))
    
    print("\n=== 温度监测数据 ===")
    print(json.dumps(generator.generate_temperature_data(), indent=2, ensure_ascii=False))
    
    print("\n=== 设备状态数据 ===")
    print(json.dumps(generator.generate_device_data(), indent=2, ensure_ascii=False))
    
    print("\n=== 消防监测数据 ===")
    print(json.dumps(generator.generate_fire_data(), indent=2, ensure_ascii=False))
    
    print("\n=== 环境监测数据 ===")
    print(json.dumps(generator.generate_environment_data(), indent=2, ensure_ascii=False))
