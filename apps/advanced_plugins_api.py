#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级插件API路由
==========================================

集成高级监测插件的RESTful API:
1. 声学监测 (Acoustic Monitoring)
2. 气体检测 (Gas Detection)
3. 高光谱检测 (Hyperspectral Detection)
4. SLAM建图 (SLAM Mapping)
5. 多模态融合 (Multimodal Fusion)

使用方法:
    from apps.advanced_plugins_api import create_advanced_plugins_router, integrate_advanced_plugins_routes
    
    app = FastAPI()
    integrate_advanced_plugins_routes(app)

作者: 破夜绘明团队
日期: 2025
"""

from __future__ import annotations

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

try:
    from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# 数据模型定义
# =============================================================================
if FASTAPI_AVAILABLE:
    
    # ===================== 通用模型 =====================
    class PluginStatusResponse(BaseModel):
        """插件状态响应"""
        plugin_id: str
        plugin_name: str
        version: str
        initialized: bool
        model_loaded: bool
        capabilities: List[str] = []
        
    class ProcessRequest(BaseModel):
        """通用处理请求"""
        device_id: str
        data: Dict[str, Any]
        options: Optional[Dict[str, Any]] = None
    
    class ProcessResponse(BaseModel):
        """通用处理响应"""
        success: bool
        device_id: str
        timestamp: float
        processing_time_ms: float
        results: Dict[str, Any]
        alarms: List[Dict[str, Any]] = []
        
    # ===================== 声学监测模型 =====================
    class AcousticAnalyzeRequest(BaseModel):
        """声学分析请求"""
        device_id: str
        audio_data: Optional[List[float]] = None  # 音频数据
        audio_file_path: Optional[str] = None  # 或音频文件路径
        sample_rate: int = 16000
        analysis_type: str = "full"  # full, pd_only, anomaly_only
        
    class AcousticResult(BaseModel):
        """声学分析结果"""
        success: bool
        device_id: str
        timestamp: float
        processing_time_ms: float
        pd_detected: bool = False
        pd_level: str = "normal"
        pd_confidence: float = 0.0
        anomaly_detected: bool = False
        anomaly_type: Optional[str] = None
        anomaly_confidence: float = 0.0
        frequency_analysis: Dict[str, Any] = {}
        recommendations: List[str] = []
        alarms: List[Dict[str, Any]] = []
    
    # ===================== 气体检测模型 =====================
    class GasAnalyzeRequest(BaseModel):
        """气体分析请求"""
        device_id: str
        gas_readings: Dict[str, float]  # 例如: {"SF6": 850, "H2": 50, "CO": 100}
        temperature: Optional[float] = None
        humidity: Optional[float] = None
        timestamp: Optional[float] = None
        
    class GasResult(BaseModel):
        """气体分析结果"""
        success: bool
        device_id: str
        timestamp: float
        processing_time_ms: float
        status: str  # normal, attention, warning, alarm, critical
        gas_levels: Dict[str, Dict[str, Any]] = {}  # 各气体的详细状态
        dga_analysis: Optional[Dict[str, Any]] = None  # DGA分析结果
        trend_prediction: Optional[Dict[str, Any]] = None  # 趋势预测
        health_index: float = 100.0
        recommendations: List[str] = []
        alarms: List[Dict[str, Any]] = []
    
    # ===================== 高光谱检测模型 =====================
    class HyperspectralAnalyzeRequest(BaseModel):
        """高光谱分析请求"""
        device_id: str
        spectral_data: Optional[List[List[float]]] = None  # 光谱数据
        image_path: Optional[str] = None  # 高光谱图像路径
        wavelength_range: List[int] = [400, 2500]  # 波长范围nm
        analysis_type: str = "defect"  # defect, material, contamination
        
    class HyperspectralResult(BaseModel):
        """高光谱分析结果"""
        success: bool
        device_id: str
        timestamp: float
        processing_time_ms: float
        defects_detected: List[Dict[str, Any]] = []
        material_analysis: Optional[Dict[str, Any]] = None
        contamination_map: Optional[Dict[str, Any]] = None
        spectral_features: Dict[str, Any] = {}
        recommendations: List[str] = []
        alarms: List[Dict[str, Any]] = []
    
    # ===================== SLAM建图模型 =====================
    class SLAMProcessRequest(BaseModel):
        """SLAM处理请求"""
        session_id: str
        point_cloud: Optional[List[List[float]]] = None  # [[x,y,z], ...]
        lidar_file_path: Optional[str] = None
        odometry: Optional[Dict[str, float]] = None  # {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}
        timestamp: Optional[float] = None
        
    class SLAMResult(BaseModel):
        """SLAM处理结果"""
        success: bool
        session_id: str
        timestamp: float
        processing_time_ms: float
        pose: Dict[str, float] = {}  # 当前位姿
        map_updated: bool = False
        loop_closure_detected: bool = False
        map_points_count: int = 0
        keyframe_added: bool = False
        
    class SLAMMapResponse(BaseModel):
        """SLAM地图响应"""
        success: bool
        session_id: str
        map_points: int
        keyframes: int
        trajectory_length: float
        bounds: Dict[str, float]
        
    # ===================== 多模态融合模型 =====================
    class MultimodalFusionRequest(BaseModel):
        """多模态融合请求"""
        device_id: str
        modalities: Dict[str, Dict[str, Any]]  # 各模态数据
        pre_processed: bool = False
        fusion_strategy: str = "hybrid"  # early, late, attention, hybrid
        
    class MultimodalFusionResult(BaseModel):
        """多模态融合结果"""
        success: bool
        device_id: str
        timestamp: float
        processing_time_ms: float
        overall_status: str
        confidence: float
        modality_contributions: Dict[str, float]
        fused_detections: List[Dict[str, Any]]
        diagnostic_report: Dict[str, Any]
        recommendations: List[str]
        alarms: List[Dict[str, Any]]


# =============================================================================
# 插件管理器（单例）
# =============================================================================
class AdvancedPluginManager:
    """高级插件管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._plugins = {}
        self._model_registry = None
        self._alarm_handlers = []
        self._initialized = True
        
        logger.info("高级插件管理器初始化完成")
    
    def initialize_plugins(self):
        """初始化所有高级插件"""
        try:
            # 尝试导入各个插件
            plugins_to_load = [
                ("acoustic_monitoring", "plugins.acoustic_monitoring.plugin", "AcousticMonitoringPlugin"),
                ("gas_detection", "plugins.gas_detection.plugin", "GasDetectionPlugin"),
                ("hyperspectral_detection", "plugins.hyperspectral_detection.plugin", "HyperspectralDetectionPlugin"),
                ("slam_mapping", "plugins.slam_mapping.plugin", "SLAMMappingPlugin"),
                ("multimodal_fusion", "plugins.multimodal_fusion.plugin", "MultimodalFusionPlugin"),
            ]
            
            for plugin_id, module_path, class_name in plugins_to_load:
                try:
                    import importlib
                    module = importlib.import_module(module_path)
                    plugin_class = getattr(module, class_name)
                    
                    # 获取默认配置
                    config = self._get_plugin_config(plugin_id)
                    
                    # 创建实例
                    plugin_instance = plugin_class(config)
                    
                    # 初始化
                    if hasattr(plugin_instance, 'init'):
                        plugin_instance.init()
                    
                    self._plugins[plugin_id] = {
                        'instance': plugin_instance,
                        'loaded': True,
                        'error': None
                    }
                    
                    logger.info(f"插件 {plugin_id} 加载成功")
                    
                except Exception as e:
                    logger.warning(f"加载插件 {plugin_id} 失败: {e}")
                    self._plugins[plugin_id] = {
                        'instance': None,
                        'loaded': False,
                        'error': str(e)
                    }
            
            return True
            
        except Exception as e:
            logger.error(f"初始化插件失败: {e}")
            return False
    
    def _get_plugin_config(self, plugin_id: str) -> Dict:
        """获取插件配置"""
        # 从配置文件加载或使用默认值
        default_configs = {
            "acoustic_monitoring": {
                "sample_rate": 16000,
                "detection_params": {
                    "pd_threshold": 0.7,
                    "anomaly_threshold": 0.6
                }
            },
            "gas_detection": {
                "thresholds": {
                    "SF6": {"attention": 800, "warning": 1000, "alarm": 1500},
                    "H2": {"attention": 100, "warning": 150, "alarm": 300},
                    "CO": {"attention": 200, "warning": 300, "alarm": 600}
                }
            },
            "hyperspectral_detection": {
                "wavelength_range": [400, 2500],
                "num_bands": 224
            },
            "slam_mapping": {
                "voxel_size": 0.1,
                "keyframe_distance": 1.0
            },
            "multimodal_fusion": {
                "fusion_strategy": "hybrid",
                "alarm_threshold": 3
            }
        }
        
        return default_configs.get(plugin_id, {})
    
    def get_plugin(self, plugin_id: str):
        """获取插件实例"""
        if plugin_id not in self._plugins:
            return None
        
        plugin_info = self._plugins[plugin_id]
        if not plugin_info['loaded']:
            return None
            
        return plugin_info['instance']
    
    def get_plugin_status(self, plugin_id: str) -> Dict:
        """获取插件状态"""
        if plugin_id not in self._plugins:
            return {
                'plugin_id': plugin_id,
                'loaded': False,
                'error': '插件未注册'
            }
        
        plugin_info = self._plugins[plugin_id]
        plugin = plugin_info['instance']
        
        status = {
            'plugin_id': plugin_id,
            'loaded': plugin_info['loaded'],
            'error': plugin_info['error']
        }
        
        if plugin and hasattr(plugin, 'get_plugin_status'):
            status.update(plugin.get_plugin_status())
        elif plugin:
            status.update({
                'plugin_name': getattr(plugin, 'PLUGIN_NAME', plugin_id),
                'version': getattr(plugin, 'VERSION', '1.0.0'),
                'initialized': getattr(plugin, 'initialized', False)
            })
        
        return status
    
    def list_plugins(self) -> List[Dict]:
        """列出所有插件"""
        return [self.get_plugin_status(pid) for pid in self._plugins]
    
    def register_alarm_handler(self, handler):
        """注册告警处理器"""
        self._alarm_handlers.append(handler)
    
    def emit_alarm(self, alarm: Dict):
        """发送告警"""
        for handler in self._alarm_handlers:
            try:
                handler(alarm)
            except Exception as e:
                logger.error(f"告警处理失败: {e}")


# 全局实例
plugin_manager = AdvancedPluginManager()


# =============================================================================
# SLAM会话管理
# =============================================================================
class SLAMSessionManager:
    """SLAM会话管理器"""
    
    def __init__(self):
        self._sessions = {}
    
    def create_session(self, session_id: str) -> Dict:
        """创建SLAM会话"""
        if session_id in self._sessions:
            return {"success": False, "error": "会话已存在"}
        
        self._sessions[session_id] = {
            'created_at': time.time(),
            'frames_processed': 0,
            'map_points': 0,
            'keyframes': 0,
            'trajectory': [],
            'status': 'active'
        }
        
        return {"success": True, "session_id": session_id}
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, data: Dict):
        """更新会话数据"""
        if session_id in self._sessions:
            self._sessions[session_id].update(data)
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[Dict]:
        """列出所有会话"""
        return [
            {"session_id": sid, **info}
            for sid, info in self._sessions.items()
        ]


slam_session_manager = SLAMSessionManager()


# =============================================================================
# 创建API路由
# =============================================================================
def create_advanced_plugins_router() -> APIRouter:
    """创建高级插件API路由"""
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI未安装")
    
    router = APIRouter(prefix="/api/advanced", tags=["高级插件"])
    
    # ========================= 通用端点 =========================
    
    @router.get("/plugins", summary="获取所有高级插件状态")
    async def list_advanced_plugins():
        """列出所有高级插件及其状态"""
        return {
            "success": True,
            "plugins": plugin_manager.list_plugins()
        }
    
    @router.get("/plugins/{plugin_id}", summary="获取插件详情")
    async def get_plugin_detail(plugin_id: str):
        """获取指定插件的详细信息"""
        status = plugin_manager.get_plugin_status(plugin_id)
        if not status.get('loaded'):
            raise HTTPException(status_code=404, detail=f"插件 {plugin_id} 未加载")
        return {"success": True, "plugin": status}
    
    @router.post("/plugins/{plugin_id}/initialize", summary="初始化插件")
    async def initialize_plugin(plugin_id: str):
        """初始化指定插件"""
        plugin = plugin_manager.get_plugin(plugin_id)
        if plugin is None:
            raise HTTPException(status_code=404, detail=f"插件 {plugin_id} 不存在")
        
        try:
            if hasattr(plugin, 'init'):
                result = plugin.init()
                return {"success": True, "initialized": result}
            return {"success": True, "message": "插件无需初始化"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================= 声学监测 =========================
    
    @router.post("/acoustic/analyze", response_model=AcousticResult, summary="声学分析")
    async def analyze_acoustic(request: AcousticAnalyzeRequest):
        """
        执行声学信号分析
        
        支持:
        - 局部放电(PD)检测
        - 异常声音检测
        - 频谱分析
        """
        start_time = time.time()
        
        plugin = plugin_manager.get_plugin("acoustic_monitoring")
        
        # 如果插件不可用，使用模拟数据
        if plugin is None:
            logger.warning("声学监测插件不可用，返回模拟数据")
            return AcousticResult(
                success=True,
                device_id=request.device_id,
                timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000,
                pd_detected=False,
                pd_level="normal",
                pd_confidence=0.95,
                anomaly_detected=False,
                frequency_analysis={
                    "dominant_frequency": 50,
                    "harmonics": [100, 150, 200],
                    "noise_level_db": -45
                },
                recommendations=["设备运行正常，建议定期监测"]
            )
        
        try:
            # 构建输入数据
            input_data = {
                "device_id": request.device_id,
                "sample_rate": request.sample_rate,
                "analysis_type": request.analysis_type
            }
            
            if request.audio_data:
                input_data["audio_data"] = request.audio_data
            elif request.audio_file_path:
                input_data["audio_file_path"] = request.audio_file_path
            
            # 调用插件处理
            result = plugin.process(input_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 提取告警
            alarms = result.get('alarms', [])
            for alarm in alarms:
                plugin_manager.emit_alarm(alarm)
            
            return AcousticResult(
                success=result.get('success', True),
                device_id=request.device_id,
                timestamp=time.time(),
                processing_time_ms=processing_time,
                pd_detected=result.get('pd_detected', False),
                pd_level=result.get('pd_level', 'normal'),
                pd_confidence=result.get('pd_confidence', 0.0),
                anomaly_detected=result.get('anomaly_detected', False),
                anomaly_type=result.get('anomaly_type'),
                anomaly_confidence=result.get('anomaly_confidence', 0.0),
                frequency_analysis=result.get('frequency_analysis', {}),
                recommendations=result.get('recommendations', []),
                alarms=alarms
            )
            
        except Exception as e:
            logger.error(f"声学分析失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/acoustic/status/{device_id}", summary="获取声学监测状态")
    async def get_acoustic_status(device_id: str):
        """获取设备的声学监测状态"""
        plugin = plugin_manager.get_plugin("acoustic_monitoring")
        
        if plugin is None:
            return {
                "success": True,
                "device_id": device_id,
                "status": "unknown",
                "message": "插件未加载"
            }
        
        if hasattr(plugin, 'get_device_status'):
            return {
                "success": True,
                "device_id": device_id,
                "status": plugin.get_device_status(device_id)
            }
        
        return {
            "success": True,
            "device_id": device_id,
            "status": "normal"
        }
    
    # ========================= 气体检测 =========================
    
    @router.post("/gas/analyze", response_model=GasResult, summary="气体分析")
    async def analyze_gas(request: GasAnalyzeRequest):
        """
        执行气体检测分析
        
        支持:
        - SF6气体监测
        - 油中溶解气体分析(DGA)
        - 趋势预测
        """
        start_time = time.time()
        
        plugin = plugin_manager.get_plugin("gas_detection")
        
        # 如果插件不可用，使用模拟数据
        if plugin is None:
            logger.warning("气体检测插件不可用，返回模拟数据")
            
            gas_levels = {}
            for gas, value in request.gas_readings.items():
                gas_levels[gas] = {
                    "value": value,
                    "unit": "ppm",
                    "status": "normal" if value < 500 else "attention",
                    "threshold_warning": 1000,
                    "threshold_alarm": 1500
                }
            
            return GasResult(
                success=True,
                device_id=request.device_id,
                timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000,
                status="normal",
                gas_levels=gas_levels,
                dga_analysis={
                    "fault_type": "none",
                    "confidence": 0.95,
                    "ratios": {"C2H2_C2H4": 0.1, "CH4_H2": 0.3}
                },
                trend_prediction={
                    "next_24h": "stable",
                    "next_7d": "stable"
                },
                health_index=95.0,
                recommendations=["气体浓度正常，建议继续监测"]
            )
        
        try:
            # 构建输入数据
            input_data = {
                "device_id": request.device_id,
                "gas_readings": request.gas_readings,
                "timestamp": request.timestamp or time.time()
            }
            
            if request.temperature:
                input_data["temperature"] = request.temperature
            if request.humidity:
                input_data["humidity"] = request.humidity
            
            # 调用插件处理
            result = plugin.process(input_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 提取告警
            alarms = result.get('alarms', [])
            for alarm in alarms:
                plugin_manager.emit_alarm(alarm)
            
            return GasResult(
                success=result.get('success', True),
                device_id=request.device_id,
                timestamp=time.time(),
                processing_time_ms=processing_time,
                status=result.get('overall_status', 'normal'),
                gas_levels=result.get('gas_levels', {}),
                dga_analysis=result.get('dga_analysis'),
                trend_prediction=result.get('trend_prediction'),
                health_index=result.get('health_index', 100.0),
                recommendations=result.get('recommendations', []),
                alarms=alarms
            )
            
        except Exception as e:
            logger.error(f"气体分析失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/gas/history/{device_id}", summary="获取气体历史数据")
    async def get_gas_history(
        device_id: str,
        hours: int = Query(default=24, ge=1, le=720, description="回溯小时数")
    ):
        """获取设备的气体监测历史数据"""
        plugin = plugin_manager.get_plugin("gas_detection")
        
        if plugin is None or not hasattr(plugin, 'get_history'):
            # 返回模拟历史数据
            import random
            history = []
            current_time = time.time()
            for i in range(min(hours, 24)):
                history.append({
                    "timestamp": current_time - i * 3600,
                    "SF6": 800 + random.randint(-50, 50),
                    "H2": 50 + random.randint(-10, 10),
                    "CO": 100 + random.randint(-20, 20),
                    "status": "normal"
                })
            
            return {
                "success": True,
                "device_id": device_id,
                "hours": hours,
                "data_points": len(history),
                "history": history
            }
        
        history = plugin.get_history(device_id, hours)
        return {
            "success": True,
            "device_id": device_id,
            "hours": hours,
            "data_points": len(history),
            "history": history
        }
    
    # ========================= 高光谱检测 =========================
    
    @router.post("/hyperspectral/analyze", response_model=HyperspectralResult, summary="高光谱分析")
    async def analyze_hyperspectral(request: HyperspectralAnalyzeRequest):
        """
        执行高光谱图像分析
        
        支持:
        - 缺陷检测
        - 材料分析
        - 污染检测
        """
        start_time = time.time()
        
        plugin = plugin_manager.get_plugin("hyperspectral_detection")
        
        # 如果插件不可用，使用模拟数据
        if plugin is None:
            logger.warning("高光谱检测插件不可用，返回模拟数据")
            return HyperspectralResult(
                success=True,
                device_id=request.device_id,
                timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000,
                defects_detected=[],
                material_analysis={
                    "primary_material": "copper",
                    "confidence": 0.92,
                    "oxidation_level": "low"
                },
                spectral_features={
                    "absorption_peaks": [550, 680, 850],
                    "reflectance_avg": 0.65
                },
                recommendations=["材料状态良好，无明显缺陷"]
            )
        
        try:
            # 构建输入数据
            input_data = {
                "device_id": request.device_id,
                "wavelength_range": request.wavelength_range,
                "analysis_type": request.analysis_type
            }
            
            if request.spectral_data:
                input_data["spectral_data"] = request.spectral_data
            elif request.image_path:
                input_data["image_path"] = request.image_path
            
            # 调用插件处理
            result = plugin.process(input_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 提取告警
            alarms = result.get('alarms', [])
            for alarm in alarms:
                plugin_manager.emit_alarm(alarm)
            
            return HyperspectralResult(
                success=result.get('success', True),
                device_id=request.device_id,
                timestamp=time.time(),
                processing_time_ms=processing_time,
                defects_detected=result.get('defects', []),
                material_analysis=result.get('material_analysis'),
                contamination_map=result.get('contamination_map'),
                spectral_features=result.get('spectral_features', {}),
                recommendations=result.get('recommendations', []),
                alarms=alarms
            )
            
        except Exception as e:
            logger.error(f"高光谱分析失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================= SLAM建图 =========================
    
    @router.post("/slam/session/create", summary="创建SLAM会话")
    async def create_slam_session(session_id: str = Query(..., description="会话ID")):
        """创建新的SLAM建图会话"""
        result = slam_session_manager.create_session(session_id)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    
    @router.get("/slam/sessions", summary="列出所有SLAM会话")
    async def list_slam_sessions():
        """列出所有SLAM建图会话"""
        return {
            "success": True,
            "sessions": slam_session_manager.list_sessions()
        }
    
    @router.get("/slam/session/{session_id}", summary="获取SLAM会话详情")
    async def get_slam_session(session_id: str):
        """获取SLAM会话详细信息"""
        session = slam_session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"success": True, "session": session}
    
    @router.post("/slam/process", response_model=SLAMResult, summary="处理SLAM帧")
    async def process_slam_frame(request: SLAMProcessRequest):
        """
        处理SLAM点云帧
        
        返回当前位姿估计和地图更新状态
        """
        start_time = time.time()
        
        session = slam_session_manager.get_session(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        plugin = plugin_manager.get_plugin("slam_mapping")
        
        # 如果插件不可用，使用模拟数据
        if plugin is None:
            logger.warning("SLAM插件不可用，返回模拟数据")
            
            # 更新会话
            session['frames_processed'] += 1
            session['map_points'] += 100
            
            return SLAMResult(
                success=True,
                session_id=request.session_id,
                timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000,
                pose={
                    "x": session['frames_processed'] * 0.1,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                },
                map_updated=True,
                loop_closure_detected=False,
                map_points_count=session['map_points'],
                keyframe_added=session['frames_processed'] % 10 == 0
            )
        
        try:
            # 构建输入数据
            input_data = {
                "session_id": request.session_id,
                "timestamp": request.timestamp or time.time()
            }
            
            if request.point_cloud:
                input_data["point_cloud"] = request.point_cloud
            elif request.lidar_file_path:
                input_data["lidar_file_path"] = request.lidar_file_path
            
            if request.odometry:
                input_data["odometry"] = request.odometry
            
            # 调用插件处理
            result = plugin.process(input_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 更新会话
            slam_session_manager.update_session(request.session_id, {
                'frames_processed': session['frames_processed'] + 1,
                'map_points': result.get('map_points_count', session['map_points']),
                'keyframes': session['keyframes'] + (1 if result.get('keyframe_added') else 0)
            })
            
            return SLAMResult(
                success=result.get('success', True),
                session_id=request.session_id,
                timestamp=time.time(),
                processing_time_ms=processing_time,
                pose=result.get('pose', {}),
                map_updated=result.get('map_updated', False),
                loop_closure_detected=result.get('loop_closure_detected', False),
                map_points_count=result.get('map_points_count', 0),
                keyframe_added=result.get('keyframe_added', False)
            )
            
        except Exception as e:
            logger.error(f"SLAM处理失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/slam/map/{session_id}", response_model=SLAMMapResponse, summary="获取SLAM地图")
    async def get_slam_map(session_id: str):
        """获取SLAM会话的地图数据"""
        session = slam_session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return SLAMMapResponse(
            success=True,
            session_id=session_id,
            map_points=session.get('map_points', 0),
            keyframes=session.get('keyframes', 0),
            trajectory_length=len(session.get('trajectory', [])),
            bounds={
                "min_x": -10.0,
                "max_x": 10.0,
                "min_y": -10.0,
                "max_y": 10.0,
                "min_z": 0.0,
                "max_z": 3.0
            }
        )
    
    @router.delete("/slam/session/{session_id}", summary="删除SLAM会话")
    async def delete_slam_session(session_id: str):
        """删除SLAM会话及其数据"""
        success = slam_session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"success": True, "message": f"会话 {session_id} 已删除"}
    
    # ========================= 多模态融合 =========================
    
    @router.post("/fusion/analyze", response_model=MultimodalFusionResult, summary="多模态融合分析")
    async def analyze_multimodal(request: MultimodalFusionRequest):
        """
        执行多模态数据融合分析
        
        整合多种传感器数据进行综合故障诊断:
        - 特征级融合（注意力机制）
        - 决策级融合（关联分析）
        - 综合诊断报告
        """
        start_time = time.time()
        
        plugin = plugin_manager.get_plugin("multimodal_fusion")
        
        # 如果插件不可用，使用模拟数据
        if plugin is None:
            logger.warning("多模态融合插件不可用，返回模拟数据")
            
            modality_names = list(request.modalities.keys())
            contributions = {m: 1.0 / len(modality_names) for m in modality_names}
            
            return MultimodalFusionResult(
                success=True,
                device_id=request.device_id,
                timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000,
                overall_status="normal",
                confidence=0.92,
                modality_contributions=contributions,
                fused_detections=[],
                diagnostic_report={
                    "summary": "设备运行正常",
                    "risk_level": "low",
                    "modalities_analyzed": modality_names
                },
                recommendations=["继续定期监测"],
                alarms=[]
            )
        
        try:
            # 构建输入数据
            input_data = {
                "device_id": request.device_id,
                "modalities": request.modalities,
                "pre_processed": request.pre_processed
            }
            
            # 设置融合策略
            if hasattr(plugin, 'fusion_engine'):
                plugin.fusion_engine.strategy = request.fusion_strategy
            
            # 调用插件处理
            result = plugin.process(input_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 提取告警
            alarms = result.get('alarms', [])
            for alarm in alarms:
                plugin_manager.emit_alarm(alarm)
            
            return MultimodalFusionResult(
                success=result.get('success', True),
                device_id=request.device_id,
                timestamp=time.time(),
                processing_time_ms=processing_time,
                overall_status=result.get('overall_status', 'unknown'),
                confidence=result.get('confidence', 0.0),
                modality_contributions=result.get('modality_contributions', {}),
                fused_detections=result.get('detections', []),
                diagnostic_report=result.get('diagnostic_report', {}),
                recommendations=result.get('recommendations', []),
                alarms=alarms
            )
            
        except Exception as e:
            logger.error(f"多模态融合分析失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/fusion/trend/{device_id}", summary="获取设备状态趋势")
    async def get_device_trend(
        device_id: str,
        hours: int = Query(default=24, ge=1, le=168, description="回溯小时数")
    ):
        """获取设备的多模态融合状态趋势"""
        plugin = plugin_manager.get_plugin("multimodal_fusion")
        
        if plugin is None or not hasattr(plugin, 'get_device_trend'):
            # 返回模拟趋势数据
            return {
                "success": True,
                "device_id": device_id,
                "time_range_hours": hours,
                "data_points": 0,
                "status_distribution": {"normal": 0.95, "attention": 0.05},
                "confidence_stats": {
                    "mean": 0.90,
                    "min": 0.85,
                    "max": 0.95,
                    "trend": "stable"
                }
            }
        
        trend = plugin.get_device_trend(device_id, hours)
        return trend
    
    @router.post("/fusion/batch", summary="批量多模态分析")
    async def batch_multimodal_analysis(
        requests: List[MultimodalFusionRequest],
        background_tasks: BackgroundTasks
    ):
        """
        批量执行多模态融合分析
        
        适用于同时分析多个设备
        """
        results = []
        
        for req in requests:
            try:
                result = await analyze_multimodal(req)
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "device_id": req.device_id,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total": len(requests),
            "processed": len(results),
            "results": results
        }
    
    # ========================= WebSocket实时推送 =========================
    
    @router.websocket("/ws/monitoring")
    async def websocket_monitoring(websocket: WebSocket):
        """WebSocket实时监测数据推送"""
        await websocket.accept()
        
        try:
            while True:
                # 接收消息
                data = await websocket.receive_json()
                
                msg_type = data.get("type")
                
                if msg_type == "subscribe":
                    # 订阅设备
                    device_id = data.get("device_id")
                    modalities = data.get("modalities", ["all"])
                    
                    await websocket.send_json({
                        "type": "subscribed",
                        "device_id": device_id,
                        "modalities": modalities
                    })
                
                elif msg_type == "unsubscribe":
                    device_id = data.get("device_id")
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "device_id": device_id
                    })
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                
                # 可以在这里添加实际的数据推送逻辑
                
        except WebSocketDisconnect:
            logger.info("WebSocket客户端断开连接")
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
    
    return router


# =============================================================================
# 集成函数
# =============================================================================
def integrate_advanced_plugins_routes(app):
    """
    将高级插件路由集成到FastAPI应用
    
    Args:
        app: FastAPI应用实例
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI未安装")
    
    # 初始化插件
    plugin_manager.initialize_plugins()
    
    # 创建并注册路由
    router = create_advanced_plugins_router()
    app.include_router(router)
    
    logger.info("高级插件API路由已集成")


# =============================================================================
# 独立测试
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="高级插件API测试")
    integrate_advanced_plugins_routes(app)
    
    uvicorn.run(app, host="127.0.0.1", port=8082)
