#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室外监测中心 API 路由
=====================

为室外监测中心提供RESTful API和WebSocket实时推送

输变电激光星芒破夜绘明监测平台 V3.5

版本: 1.0.0
"""

from __future__ import annotations
import asyncio
import json
import logging
import random
import time
import math
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from plugins.switch_inspection.switch_consistency import (
        ConsistencyChecker, SensorAdapter, VideoJudgment, SensorReading,
        ConsistencyResult, ConsistencyStatus,
    )
    CONSISTENCY_AVAILABLE = True
except ImportError:
    CONSISTENCY_AVAILABLE = False


class AlarmLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DetectionStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    ERROR = "error"


@dataclass
class Detection:
    id: str
    label: str
    confidence: float
    bbox: Optional[Dict[str, float]] = None
    status: str = "normal"
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alarm:
    id: str
    type: str
    level: AlarmLevel
    message: str
    timestamp: str
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleResult:
    module_id: str
    module_name: str
    status: DetectionStatus
    timestamp: int
    detections: List[Detection]
    alarms: List[Alarm]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "detections": [asdict(d) for d in self.detections],
            "alarms": [{"id": a.id, "type": a.type, "level": a.level.value, 
                       "message": a.message, "timestamp": a.timestamp} for a in self.alarms],
            "metrics": self.metrics,
            "metadata": self.metadata
        }


class OutdoorDataGenerator:
    """室外监测数据生成器"""
    
    def __init__(self):
        self._det_counter = 0
        self._alm_counter = 0
    
    def _det_id(self) -> str:
        self._det_counter += 1
        return f"det_{self._det_counter:06d}"
    
    def _alm_id(self) -> str:
        self._alm_counter += 1
        return f"alm_{self._alm_counter:06d}"
    
    def _ts(self) -> int:
        return int(time.time() * 1000)
    
    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_busbar_data(self) -> Dict:
        """母线巡视数据"""
        dets, alms = [], []
        
        # 绝缘子
        dets.append(Detection(self._det_id(), "绝缘子状态", 0.92, 
                              {"x": 0.2, "y": 0.3, "width": 0.1, "height": 0.15},
                              "normal", "表面清洁无裂纹"))
        # 金具
        dets.append(Detection(self._det_id(), "金具连接", 0.89,
                              {"x": 0.35, "y": 0.25, "width": 0.08, "height": 0.08},
                              "normal", "连接紧固"))
        # 导线
        dets.append(Detection(self._det_id(), "导线连接点", 0.94,
                              None, "normal", "接触良好",
                              {"temperature": 38 + random.random() * 5}))
        
        # 裂纹检测
        if random.random() > 0.9:
            dets.append(Detection(self._det_id(), "微小裂纹", 0.75, 
                                  {"x": 0.45, "y": 0.4, "width": 0.05, "height": 0.03},
                                  "alarm", "检测到微小裂纹",
                                  {"crack_length_mm": 2.5, "growth_rate": 0.15}))
            alms.append(Alarm(self._alm_id(), "crack_detected", AlarmLevel.WARNING,
                             "绝缘子表面检测到微小裂纹", self._now()))
        
        status = DetectionStatus.ALARM if alms else DetectionStatus.NORMAL
        return ModuleResult("busbar", "母线巡视", status, self._ts(), dets, alms,
                           {"defects": len([d for d in dets if d.status != "normal"]),
                            "processing_time_ms": 45}).to_dict()

    def generate_switch_data(self) -> Dict:
        """开关巡视数据"""
        dets, alms = [], []
        
        states = ["合", "分"]
        dets.append(Detection(self._det_id(), "断路器状态", 0.96, None, "normal",
                              f"状态: {random.choice(states)}位"))
        dets.append(Detection(self._det_id(), "隔离开关状态", 0.94, None, "normal",
                              f"状态: {random.choice(states)}位"))
        dets.append(Detection(self._det_id(), "接地开关状态", 0.95, None, "normal", "状态: 分位"))
        
        # 动作轨迹分析
        if random.random() > 0.85:
            dtw = random.random() * 0.3
            stuck = dtw > 0.2
            dets.append(Detection(self._det_id(), "动作轨迹分析", 0.88, None,
                                  "alarm" if stuck else "normal",
                                  "检测到轻微卡涩" if stuck else "轨迹正常",
                                  {"dtw_distance": dtw}))
            if stuck:
                alms.append(Alarm(self._alm_id(), "mechanical_stuck", AlarmLevel.WARNING,
                                 "开关动作轨迹异常", self._now()))
        
        status = DetectionStatus.ALARM if alms else DetectionStatus.NORMAL
        return ModuleResult("switch", "开关巡视", status, self._ts(), dets, alms,
                           {"processing_time_ms": 38}).to_dict()

    def generate_transformer_data(self) -> Dict:
        """变压器巡视数据"""
        dets, alms = [], []
        
        # 油位
        oil = 0.7 + random.random() * 0.25
        dets.append(Detection(self._det_id(), "油位", 0.96, None,
                              "normal" if oil > 0.6 else "warning", f"油位: {oil*100:.1f}%"))
        
        # 硅胶
        colors = ["蓝色", "浅蓝", "粉色"]
        color = random.choices(colors, weights=[0.7, 0.2, 0.1])[0]
        silica_st = "normal" if color == "蓝色" else "warning" if color == "浅蓝" else "alarm"
        dets.append(Detection(self._det_id(), "硅胶状态", 0.91,
                              {"x": 0.6, "y": 0.2, "width": 0.1, "height": 0.1},
                              silica_st, f"颜色: {color}"))
        if silica_st == "alarm":
            alms.append(Alarm(self._alm_id(), "silica_degraded", AlarmLevel.WARNING,
                             "呼吸器硅胶变色需更换", self._now()))
        
        # 温度
        temp = 45 + random.random() * 30
        temp_st = "normal" if temp < 65 else "warning" if temp < 80 else "alarm"
        dets.append(Detection(self._det_id(), "温度监测", 0.94, None, temp_st,
                              f"最高温度: {temp:.1f}°C", {"max_temp": temp}))
        if temp_st == "alarm":
            alms.append(Alarm(self._alm_id(), "overheating", AlarmLevel.ERROR,
                             f"温度过高: {temp:.1f}°C", self._now()))
        
        # 声纹
        audio = random.choices(["正常哼鸣", "轻微异响"], weights=[0.9, 0.1])[0]
        dets.append(Detection(self._det_id(), "声纹分析", 0.85, None,
                              "normal" if audio == "正常哼鸣" else "warning", f"声音: {audio}"))
        
        status = DetectionStatus.ALARM if any(a.level in [AlarmLevel.ERROR] for a in alms) else \
                 DetectionStatus.WARNING if alms else DetectionStatus.NORMAL
        return ModuleResult("transformer", "变压器巡视", status, self._ts(), dets, alms,
                           {"temperature_max": temp, "processing_time_ms": 65}).to_dict()

    def generate_capacitor_data(self) -> Dict:
        """电容器巡视数据"""
        dets, alms = [], []
        
        # 倾斜检测
        tilt = abs(random.gauss(0, 2))
        tilt_st = "normal" if tilt < 2 else "warning" if tilt < 5 else "alarm"
        dets.append(Detection(self._det_id(), "倾斜检测", 0.93, None, tilt_st,
                              f"倾斜角度: {tilt:.2f}°",
                              {"tilt_angle_deg": tilt, "method": "LSD+VP"}))
        if tilt_st == "alarm":
            alms.append(Alarm(self._alm_id(), "tilt_error", AlarmLevel.ERROR,
                             f"电容器严重倾斜: {tilt:.1f}°", self._now()))
        elif tilt_st == "warning":
            alms.append(Alarm(self._alm_id(), "tilt_warning", AlarmLevel.WARNING,
                             f"电容器轻微倾斜: {tilt:.1f}°", self._now()))
        
        # 结构
        dets.append(Detection(self._det_id(), "结构完整性", 0.95, None, "normal", "结构完整"))
        
        # 区域
        intrusion = random.random() > 0.98
        dets.append(Detection(self._det_id(), "区域监控", 0.97, None,
                              "alarm" if intrusion else "normal",
                              "检测到入侵" if intrusion else "区域安全"))
        if intrusion:
            alms.append(Alarm(self._alm_id(), "intrusion", AlarmLevel.CRITICAL,
                             "电容器区域检测到入侵", self._now()))
        
        status = DetectionStatus.ALARM if any(d.status == "alarm" for d in dets) else \
                 DetectionStatus.WARNING if any(d.status == "warning" for d in dets) else DetectionStatus.NORMAL
        return ModuleResult("capacitor", "电容器巡视", status, self._ts(), dets, alms,
                           {"tilt_angle": tilt, "processing_time_ms": 42}).to_dict()

    def generate_meter_data(self) -> Dict:
        """表计读数数据"""
        dets, alms = [], []
        
        # 油温表
        oil_temp = 35 + random.random() * 25
        dets.append(Detection(self._det_id(), "油温表", 0.94,
                              {"x": 0.1, "y": 0.2, "width": 0.12, "height": 0.12},
                              "normal" if oil_temp < 55 else "warning",
                              f"读数: {oil_temp:.1f}°C",
                              {"value": oil_temp, "unit": "°C", "type": "analog_pointer"}))
        
        # 油位计
        oil_level = 0.6 + random.random() * 0.35
        dets.append(Detection(self._det_id(), "油位计", 0.91,
                              {"x": 0.25, "y": 0.15, "width": 0.08, "height": 0.2},
                              "normal" if oil_level > 0.5 else "warning",
                              f"读数: {oil_level*100:.0f}%",
                              {"value": oil_level * 100, "unit": "%"}))
        
        # SF6压力表
        sf6 = 0.4 + random.random() * 0.2
        sf6_st = "normal" if 0.45 < sf6 < 0.55 else "warning"
        dets.append(Detection(self._det_id(), "SF6压力表", 0.92,
                              {"x": 0.4, "y": 0.2, "width": 0.12, "height": 0.12},
                              sf6_st, f"读数: {sf6:.2f} MPa",
                              {"value": sf6, "unit": "MPa"}))
        if sf6_st == "warning":
            alms.append(Alarm(self._alm_id(), "pressure_abnormal", AlarmLevel.WARNING,
                             f"SF6压力异常: {sf6:.2f} MPa", self._now()))
        
        # 数字表
        led = random.randint(100, 999)
        dets.append(Detection(self._det_id(), "数字电流表", 0.97,
                              {"x": 0.55, "y": 0.25, "width": 0.15, "height": 0.08},
                              "normal", f"读数: {led} A",
                              {"value": led, "unit": "A", "type": "digital_led"}))
        
        status = DetectionStatus.WARNING if alms else DetectionStatus.NORMAL
        return ModuleResult("meter", "表计读数", status, self._ts(), dets, alms,
                           {"meters_read": len(dets), "processing_time_ms": 55}).to_dict()

    def generate_bird_data(self) -> Dict:
        """鸟类监控数据"""
        dets, alms = [], []
        
        count = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
        species_list = ["麻雀", "喜鹊", "乌鸦", "鸽子"]
        behaviors = ["飞行", "停留", "盘旋"]
        
        for _ in range(count):
            sp = random.choice(species_list)
            bh = random.choice(behaviors)
            dist = 5 + random.random() * 20
            risk = 0.3 * {"麻雀": 0.2, "喜鹊": 0.5, "乌鸦": 0.6, "鸽子": 0.3}[sp] + \
                   0.4 * {"飞行": 0.2, "停留": 0.5, "盘旋": 0.4}[bh] + \
                   0.3 * max(0, 1 - dist / 25)
            
            st = "normal" if risk < 0.3 else "warning" if risk < 0.7 else "alarm"
            dets.append(Detection(self._det_id(), f"鸟类-{sp}", 0.82,
                                  {"x": random.random() * 0.7, "y": random.random() * 0.5,
                                   "width": 0.08, "height": 0.08},
                                  st, f"{sp} {bh} 距离{dist:.1f}m",
                                  {"species": sp, "behavior": bh, "distance_m": dist, "risk_score": risk}))
            if st == "alarm":
                alms.append(Alarm(self._alm_id(), "bird_risk", AlarmLevel.WARNING,
                                 f"高风险鸟类: {sp} {bh}", self._now()))
        
        status = DetectionStatus.ALARM if alms else DetectionStatus.WARNING if count > 0 else DetectionStatus.NORMAL
        return ModuleResult("bird", "鸟类监控", status, self._ts(), dets, alms,
                           {"bird_count": count, "processing_time_ms": 35}).to_dict()

    def generate_acoustic_data(self) -> Dict:
        """声学监测数据"""
        dets, alms = [], []
        
        # 局放
        pd = random.random() * 0.5
        pd_st = "normal" if pd < 0.2 else "warning" if pd < 0.4 else "alarm"
        dets.append(Detection(self._det_id(), "局部放电", 0.87, None, pd_st,
                              f"放电强度: {pd*100:.1f}%",
                              {"pd_level": pd, "frequency_hz": 45000, "model": "AST"}))
        if pd_st == "alarm":
            alms.append(Alarm(self._alm_id(), "partial_discharge", AlarmLevel.ERROR,
                             "检测到明显局部放电", self._now()))
        
        # 机械噪声
        mech = random.random() * 0.3
        dets.append(Detection(self._det_id(), "机械噪声", 0.84, None,
                              "normal" if mech < 0.15 else "warning",
                              f"异响程度: {mech*100:.1f}%"))
        
        health = 1 - (pd * 0.6 + mech * 0.4)
        status = DetectionStatus.ALARM if pd_st == "alarm" else \
                 DetectionStatus.WARNING if any(d.status == "warning" for d in dets) else DetectionStatus.NORMAL
        return ModuleResult("acoustic", "声学监测", status, self._ts(), dets, alms,
                           {"pd_level": pd, "health_index": health, "processing_time_ms": 28}).to_dict()

    def generate_gas_data(self) -> Dict:
        """气体检测数据"""
        dets, alms = [], []
        
        sf6 = 5 + random.random() * 15
        h2 = random.random() * 50
        co = random.random() * 20
        
        sf6_st = "normal" if sf6 < 15 else "warning"
        dets.append(Detection(self._det_id(), "SF6浓度", 0.95, None, sf6_st,
                              f"浓度: {sf6:.1f} ppm", {"value": sf6, "unit": "ppm"}))
        dets.append(Detection(self._det_id(), "H2浓度", 0.93, None,
                              "normal" if h2 < 30 else "warning",
                              f"浓度: {h2:.1f} ppm", {"value": h2, "unit": "ppm"}))
        dets.append(Detection(self._det_id(), "CO浓度", 0.94, None, "normal",
                              f"浓度: {co:.1f} ppm", {"value": co, "unit": "ppm"}))
        
        # 泄漏
        leak = random.random() > 0.95
        if leak:
            dets.append(Detection(self._det_id(), "泄漏定位", 0.78,
                                  {"x": 0.4, "y": 0.5, "width": 0.2, "height": 0.15},
                                  "alarm", "检测到气体羽流",
                                  {"optical_detection": True}))
            alms.append(Alarm(self._alm_id(), "gas_leak", AlarmLevel.ERROR,
                             "检测到气体泄漏", self._now()))
        
        health = max(0, 1 - (sf6 / 100 + h2 / 100 + co / 50) / 3)
        status = DetectionStatus.ALARM if leak else \
                 DetectionStatus.WARNING if any(d.status == "warning" for d in dets) else DetectionStatus.NORMAL
        return ModuleResult("gas", "气体检测", status, self._ts(), dets, alms,
                           {"sf6_ppm": sf6, "h2_ppm": h2, "health_index": health, "processing_time_ms": 32}).to_dict()

    def generate_hyperspectral_data(self) -> Dict:
        """高光谱检测数据"""
        dets, alms = [], []
        
        materials = random.sample(["金属", "油迹", "水", "锈蚀", "绝缘材料"], k=random.randint(2, 3))
        for mat in materials:
            st = "alarm" if mat in ["油迹", "锈蚀"] else "normal"
            dets.append(Detection(self._det_id(), f"材料-{mat}", 0.83,
                                  {"x": random.random() * 0.6, "y": random.random() * 0.6,
                                   "width": 0.15, "height": 0.15},
                                  st, f"识别到{mat}区域", {"material": mat, "sam_angle": 0.1}))
            if mat == "油迹":
                alms.append(Alarm(self._alm_id(), "oil_detected", AlarmLevel.WARNING,
                                 "高光谱检测到油迹", self._now()))
        
        if "锈蚀" in materials:
            depth = random.random() * 2
            grade = "轻度" if depth < 0.5 else "中度" if depth < 1 else "重度"
            dets.append(Detection(self._det_id(), "腐蚀分析", 0.79, None,
                                  "warning" if grade != "轻度" else "normal",
                                  f"腐蚀深度: {depth:.2f}mm ({grade})",
                                  {"depth_mm": depth, "grade": grade}))
        
        status = DetectionStatus.ALARM if any(d.status == "alarm" for d in dets) else \
                 DetectionStatus.WARNING if any(d.status == "warning" for d in dets) else DetectionStatus.NORMAL
        return ModuleResult("hyperspectral", "高光谱检测", status, self._ts(), dets, alms,
                           {"materials_detected": len(materials), "processing_time_ms": 85}).to_dict()

    def generate_slam_data(self) -> Dict:
        """SLAM建图数据"""
        dets, alms = [], []
        
        # 定位状态
        localization_quality = 0.8 + random.random() * 0.18
        dets.append(Detection(self._det_id(), "定位质量", localization_quality, None,
                              "normal" if localization_quality > 0.85 else "warning",
                              f"定位精度: {localization_quality*100:.1f}%",
                              {"icp_fitness": localization_quality, "pose_uncertainty": 0.05}))
        
        # 点云统计
        point_count = 50000 + random.randint(-10000, 20000)
        dets.append(Detection(self._det_id(), "点云数据", 0.95, None, "normal",
                              f"点云数量: {point_count}",
                              {"point_count": point_count, "voxel_size_m": 0.1}))
        
        # 语义标注
        semantic_objects = random.randint(5, 15)
        dets.append(Detection(self._det_id(), "语义识别", 0.88, None, "normal",
                              f"识别目标: {semantic_objects}个",
                              {"object_count": semantic_objects, "categories": ["设备", "围栏", "道路"]}))
        
        # 路径规划
        path_length = 50 + random.random() * 100
        dets.append(Detection(self._det_id(), "路径规划", 0.92, None, "normal",
                              f"规划路径: {path_length:.1f}m",
                              {"path_length_m": path_length, "algorithm": "A*"}))
        
        status = DetectionStatus.WARNING if localization_quality < 0.85 else DetectionStatus.NORMAL
        return ModuleResult("slam", "SLAM建图", status, self._ts(), dets, alms,
                           {"point_count": point_count, "localization_quality": localization_quality,
                            "processing_time_ms": 120}).to_dict()

    def generate_fusion_data(self) -> Dict:
        """多模态融合数据"""
        dets, alms = [], []
        
        # 综合健康评估
        health = 0.7 + random.random() * 0.28
        health_grade = "优良" if health > 0.9 else "良好" if health > 0.7 else "注意"
        dets.append(Detection(self._det_id(), "综合健康", health, None,
                              "normal" if health > 0.7 else "warning",
                              f"健康指数: {health*100:.1f}% ({health_grade})",
                              {"health_index": health, "grade": health_grade}))
        
        # 故障诊断
        fault_types = ["正常", "正常", "接触不良", "局部过热", "绝缘老化"]
        fault = random.choice(fault_types)
        fault_conf = 0.75 + random.random() * 0.2
        dets.append(Detection(self._det_id(), "故障诊断", fault_conf, None,
                              "alarm" if fault != "正常" else "normal",
                              f"诊断结果: {fault}",
                              {"fault_type": fault, "bayesian_inference": True}))
        if fault != "正常":
            alms.append(Alarm(self._alm_id(), "fault_detected", AlarmLevel.WARNING,
                             f"融合诊断检测到: {fault}", self._now()))
        
        # 传感器状态
        sensors = {"thermal": 0.95, "acoustic": 0.88, "gas": 0.92, "visual": 0.97}
        sensor_health = sum(sensors.values()) / len(sensors)
        dets.append(Detection(self._det_id(), "传感器状态", sensor_health, None, "normal",
                              f"传感器健康度: {sensor_health*100:.1f}%",
                              {"sensor_details": sensors}))
        
        # 趋势预测
        trend = random.choice(["稳定", "轻微上升", "轻微下降"])
        dets.append(Detection(self._det_id(), "趋势分析", 0.82, None, "normal",
                              f"风险趋势: {trend}", {"trend": trend, "forecast_hours": 24}))
        
        status = DetectionStatus.WARNING if fault != "正常" or health < 0.7 else DetectionStatus.NORMAL
        return ModuleResult("fusion", "多模态融合", status, self._ts(), dets, alms,
                           {"health_index": health, "sensor_count": len(sensors),
                            "processing_time_ms": 150}).to_dict()


# ==================== API路由创建 ====================

_data_generator = OutdoorDataGenerator()
_websocket_connections: List = []

# 一致性校核引擎单例
_sensor_adapter: Optional["SensorAdapter"] = None
_consistency_checker: Optional["ConsistencyChecker"] = None

def _get_consistency_checker() -> Optional["ConsistencyChecker"]:
    global _sensor_adapter, _consistency_checker
    if not CONSISTENCY_AVAILABLE:
        return None
    if _consistency_checker is None:
        _sensor_adapter = SensorAdapter()
        _consistency_checker = ConsistencyChecker(sensor_adapter=_sensor_adapter)
    return _consistency_checker


def create_outdoor_router() -> "APIRouter":
    """创建室外监测API路由"""
    if not FASTAPI_AVAILABLE:
        return None

    router = APIRouter(prefix="/api/outdoor", tags=["室外监测中心"])
    
    @router.get("/busbar")
    async def get_busbar_data():
        return _data_generator.generate_busbar_data()
    
    @router.get("/switch")
    async def get_switch_data():
        data = _data_generator.generate_switch_data()
        # 附加一致性校核结果(如可用)
        checker = _get_consistency_checker()
        if checker and data.get("detections"):
            consistency_results = []
            for det in data["detections"]:
                label = det.get("label", "")
                if "断路器" in label:
                    dev_type, dev_id = "breaker", "CB-001"
                elif "隔离开关" in label:
                    dev_type, dev_id = "isolator", "DS-001"
                elif "接地开关" in label:
                    dev_type, dev_id = "grounding", "ES-001"
                else:
                    continue
                state = "closed" if "合" in det.get("details", "") else "open"
                vj = VideoJudgment(
                    device_id=dev_id, device_type=dev_type, state=state,
                    confidence=det.get("confidence", 0.0), clarity_score=0.85,
                    evidence_sources=["dl", "color"],
                    timestamp=datetime.now().isoformat(),
                )
                result = checker.check_consistency(
                    device_id=dev_id, device_type=dev_type,
                    station_id="station-demo", bay_id="bay-01",
                    video_judgment=vj,
                )
                consistency_results.append(result.to_dict())
            data["consistency_results"] = consistency_results
        return data

    @router.post("/switch/consistency")
    async def run_switch_consistency(
        device_id: str = Query(...),
        device_type: str = Query("breaker"),
        video_state: str = Query(""),
        video_confidence: float = Query(0.0),
        video_clarity: float = Query(0.85),
        sensor_state: str = Query(""),
        sensor_quality: float = Query(1.0),
        sensor_protocol: str = Query("manual"),
        station_id: str = Query(""),
        bay_id: str = Query(""),
    ):
        """手动触发一致性校核"""
        checker = _get_consistency_checker()
        if not checker:
            raise HTTPException(status_code=503, detail="一致性校核模块不可用")

        vj = None
        if video_state:
            vj = VideoJudgment(
                device_id=device_id, device_type=device_type,
                state=video_state, confidence=video_confidence,
                clarity_score=video_clarity, evidence_sources=["api"],
                timestamp=datetime.now().isoformat(),
            )

        sr = None
        if sensor_state:
            sr = SensorReading(
                device_id=device_id, state=sensor_state,
                source_protocol=sensor_protocol,
                timestamp=datetime.now().isoformat(),
                quality=sensor_quality,
            )

        result = checker.check_consistency(
            device_id=device_id, device_type=device_type,
            station_id=station_id, bay_id=bay_id,
            video_judgment=vj, sensor_reading=sr,
        )
        return {"success": True, "data": result.to_dict()}

    @router.post("/switch/sensor")
    async def set_sensor_reading(
        device_id: str = Query(...),
        state: str = Query(...),
        protocol: str = Query("manual"),
        quality: float = Query(1.0),
    ):
        """设置传感/遥信读数(供协议适配器回调)"""
        checker = _get_consistency_checker()
        if not checker:
            raise HTTPException(status_code=503, detail="一致性校核模块不可用")

        reading = checker.sensor_adapter.set_from_protocol(
            device_id=device_id, state=state,
            source_protocol=protocol, quality=quality,
        )
        return {"success": True, "data": reading.to_dict()}

    @router.get("/transformer")
    async def get_transformer_data():
        return _data_generator.generate_transformer_data()
    
    @router.get("/capacitor")
    async def get_capacitor_data():
        return _data_generator.generate_capacitor_data()
    
    @router.get("/meter")
    async def get_meter_data():
        return _data_generator.generate_meter_data()
    
    @router.get("/bird")
    async def get_bird_data():
        return _data_generator.generate_bird_data()
    
    @router.get("/acoustic")
    async def get_acoustic_data():
        return _data_generator.generate_acoustic_data()
    
    @router.get("/gas")
    async def get_gas_data():
        return _data_generator.generate_gas_data()
    
    @router.get("/hyperspectral")
    async def get_hyperspectral_data():
        return _data_generator.generate_hyperspectral_data()
    
    @router.get("/slam")
    async def get_slam_data():
        return _data_generator.generate_slam_data()
    
    @router.get("/fusion")
    async def get_fusion_data():
        return _data_generator.generate_fusion_data()
    
    @router.get("/all")
    async def get_all_data():
        return {
            "busbar": _data_generator.generate_busbar_data(),
            "switch": _data_generator.generate_switch_data(),
            "transformer": _data_generator.generate_transformer_data(),
            "capacitor": _data_generator.generate_capacitor_data(),
            "meter": _data_generator.generate_meter_data(),
            "bird": _data_generator.generate_bird_data(),
            "acoustic": _data_generator.generate_acoustic_data(),
            "gas": _data_generator.generate_gas_data(),
            "hyperspectral": _data_generator.generate_hyperspectral_data(),
            "slam": _data_generator.generate_slam_data(),
            "fusion": _data_generator.generate_fusion_data(),
        }
    
    @router.post("/detect")
    async def run_detection(module: str = Query(...), camera_id: str = Query("cam1")):
        generators = {
            "busbar": _data_generator.generate_busbar_data,
            "switch": _data_generator.generate_switch_data,
            "transformer": _data_generator.generate_transformer_data,
            "capacitor": _data_generator.generate_capacitor_data,
            "meter": _data_generator.generate_meter_data,
            "bird": _data_generator.generate_bird_data,
            "acoustic": _data_generator.generate_acoustic_data,
            "gas": _data_generator.generate_gas_data,
            "hyperspectral": _data_generator.generate_hyperspectral_data,
            "slam": _data_generator.generate_slam_data,
            "fusion": _data_generator.generate_fusion_data,
        }
        if module not in generators:
            raise HTTPException(status_code=400, detail=f"未知模块: {module}")
        return generators[module]()
    
    return router


async def outdoor_data_pusher():
    """WebSocket数据推送任务"""
    while True:
        if _websocket_connections:
            data = {
                "type": "outdoor_update",
                "timestamp": int(time.time() * 1000),
                "data": _data_generator.generate_fusion_data()
            }
            disconnected = []
            for ws in _websocket_connections:
                try:
                    await ws.send_json(data)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                _websocket_connections.remove(ws)
        await asyncio.sleep(2)


def create_websocket_route(app):
    """创建WebSocket路由"""
    if not FASTAPI_AVAILABLE:
        return
    
    @app.websocket("/ws/outdoor")
    async def websocket_outdoor(websocket: WebSocket):
        await websocket.accept()
        _websocket_connections.append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data) if data else {}
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            if websocket in _websocket_connections:
                _websocket_connections.remove(websocket)


def integrate_outdoor_api(app):
    """集成室外监测API到FastAPI应用"""
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI不可用")
        return
    
    router = create_outdoor_router()
    if router:
        app.include_router(router)
    create_websocket_route(app)
    
    @app.on_event("startup")
    async def start_outdoor_pusher():
        asyncio.create_task(outdoor_data_pusher())
    
    logger.info("室外监测中心API已集成")


if __name__ == "__main__":
    gen = OutdoorDataGenerator()
    print("=== 母线巡视 ===")
    print(json.dumps(gen.generate_busbar_data(), indent=2, ensure_ascii=False))
    print("\n=== 变压器巡视 ===")
    print(json.dumps(gen.generate_transformer_data(), indent=2, ensure_ascii=False))
    print("\n=== 多模态融合 ===")
    print(json.dumps(gen.generate_fusion_data(), indent=2, ensure_ascii=False))
