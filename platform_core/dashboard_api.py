"""
统一仪表盘API
输变电激光星芒破夜绘明监测平台

基于UPDATE.md优化方案实现:
- 统一的UI组件格式规范
- 所有功能模块状态汇总
- 告警与异常综合展示
- 系统健康度监控
- 深度学习模型状态

版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# 数据模型定义
# =============================================================================

class ModuleStatus(str, Enum):
    """模块状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    WARNING = "warning"
    ERROR = "error"
    LOADING = "loading"


class AlarmLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ModuleHealth(BaseModel):
    """模块健康状态"""
    module_id: str
    module_name: str
    status: ModuleStatus
    health_score: float = 100.0
    last_heartbeat: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = {}


class SystemOverview(BaseModel):
    """系统概览"""
    total_modules: int
    online_modules: int
    warning_modules: int
    error_modules: int
    overall_health: float
    uptime_hours: float
    last_update: str


class AlarmSummary(BaseModel):
    """告警摘要"""
    total_alarms: int
    critical_count: int
    error_count: int
    warning_count: int
    info_count: int
    unresolved_count: int
    recent_alarms: List[Dict[str, Any]] = []


class DLModelStatus(BaseModel):
    """深度学习模型状态"""
    model_id: str
    model_name: str
    model_type: str
    loaded: bool
    device: str = "cpu"
    inference_count: int = 0
    avg_latency_ms: float = 0.0
    last_inference: Optional[str] = None
    memory_mb: float = 0.0


class PluginStatus(BaseModel):
    """插件状态"""
    plugin_id: str
    plugin_name: str
    version: str
    status: str
    dl_model_enabled: bool = False
    dl_model_id: Optional[str] = None
    capabilities: List[str] = []
    inference_count: int = 0
    error_count: int = 0
    last_inference: Optional[str] = None


class DashboardData(BaseModel):
    """仪表盘数据"""
    timestamp: str
    system_overview: SystemOverview
    alarm_summary: AlarmSummary
    module_health: List[ModuleHealth]
    dl_models: List[DLModelStatus]
    plugins: List[PluginStatus]
    metrics: Dict[str, Any] = {}


# =============================================================================
# 仪表盘服务
# =============================================================================

class DashboardService:
    """
    仪表盘服务

    统一汇总所有模块状态、告警和指标
    """

    def __init__(self):
        self._start_time = time.time()
        self._plugin_manager = None
        self._model_registry = None
        self._alarm_manager = None

        # 缓存
        self._cache_ttl = 5.0  # 秒
        self._last_cache_update = 0.0
        self._cached_data: Optional[DashboardData] = None

    def set_plugin_manager(self, pm):
        """设置插件管理器"""
        self._plugin_manager = pm

    def set_model_registry(self, mr):
        """设置模型注册器"""
        self._model_registry = mr

    def set_alarm_manager(self, am):
        """设置告警管理器"""
        self._alarm_manager = am

    def get_dashboard_data(self, force_refresh: bool = False) -> DashboardData:
        """
        获取仪表盘数据

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            仪表盘数据
        """
        now = time.time()

        # 检查缓存
        if not force_refresh and self._cached_data is not None:
            if now - self._last_cache_update < self._cache_ttl:
                return self._cached_data

        # 收集数据
        module_health = self._collect_module_health()
        dl_models = self._collect_dl_models()
        plugins = self._collect_plugins()
        alarm_summary = self._collect_alarm_summary()

        # 计算系统概览
        system_overview = self._calculate_system_overview(module_health)

        # 收集指标
        metrics = self._collect_metrics()

        data = DashboardData(
            timestamp=datetime.now().isoformat(),
            system_overview=system_overview,
            alarm_summary=alarm_summary,
            module_health=module_health,
            dl_models=dl_models,
            plugins=plugins,
            metrics=metrics,
        )

        # 更新缓存
        self._cached_data = data
        self._last_cache_update = now

        return data

    def _collect_module_health(self) -> List[ModuleHealth]:
        """收集模块健康状态"""
        modules = []

        # 定义所有功能模块
        module_definitions = [
            ("transformer_inspection", "主变自主巡视", "A组"),
            ("switch_inspection", "开关间隔巡视", "B组"),
            ("busbar_inspection", "母线及设备巡视", "C组"),
            ("capacitor_inspection", "电容器/电抗器", "D组"),
            ("meter_reading", "表计读数", "E组"),
            ("gas_detection", "气体泄漏检测", "F组"),
            ("indoor_fence", "室内电子围栏", "G组"),
            ("acoustic_monitoring", "声学监测", "H组"),
            ("bird_monitoring", "鸟害监测", "I组"),
            ("hyperspectral", "高光谱检测", "扩展"),
            ("multimodal_fusion", "多模态融合", "扩展"),
            ("slam_mapping", "SLAM建图", "扩展"),
        ]

        for module_id, module_name, group in module_definitions:
            status, health_score, error_msg, metrics = self._get_module_status(module_id)

            modules.append(ModuleHealth(
                module_id=module_id,
                module_name=f"{module_name} ({group})",
                status=status,
                health_score=health_score,
                last_heartbeat=datetime.now().isoformat(),
                error_message=error_msg,
                metrics=metrics,
            ))

        return modules

    def _get_module_status(self, module_id: str) -> tuple:
        """获取单个模块状态"""
        status = ModuleStatus.ONLINE
        health_score = 100.0
        error_msg = None
        metrics = {}

        # 尝试从插件管理器获取状态
        if self._plugin_manager is not None:
            try:
                plugin = self._plugin_manager.get_plugin(module_id)
                if plugin is not None:
                    health = plugin.healthcheck()
                    if hasattr(health, 'healthy'):
                        if health.healthy:
                            status = ModuleStatus.ONLINE
                            health_score = 100.0
                        else:
                            status = ModuleStatus.ERROR
                            health_score = 0.0
                            error_msg = health.message if hasattr(health, 'message') else "未知错误"

                        if hasattr(health, 'details'):
                            metrics = health.details
                else:
                    status = ModuleStatus.OFFLINE
                    health_score = 0.0
            except Exception as e:
                status = ModuleStatus.ERROR
                health_score = 0.0
                error_msg = str(e)
        else:
            # 模拟数据
            import random
            if random.random() > 0.1:
                status = ModuleStatus.ONLINE
                health_score = random.uniform(85, 100)
            else:
                status = ModuleStatus.WARNING
                health_score = random.uniform(60, 85)

        return status, health_score, error_msg, metrics

    def _collect_dl_models(self) -> List[DLModelStatus]:
        """收集深度学习模型状态"""
        models = []

        # 定义深度学习模型
        model_definitions = [
            ("yolov8_vit_transformer", "YOLOv8-ViT主变检测", "detection"),
            ("yolov8_vit_switch", "YOLOv8-ViT开关检测", "detection"),
            ("yolov8_vit_busbar", "YOLOv8-ViT母线检测", "detection"),
            ("yolov8_vit_capacitor", "YOLOv8-ViT电容检测", "detection"),
            ("gl_translstm_gas", "GL-TransLSTM气体预测", "timeseries"),
            ("acoustic_cnn_lstm", "声学CNN-LSTM检测", "audio"),
            ("deep_sort_tracker", "DeepSORT多目标跟踪", "tracking"),
            ("keypoint_hrnet", "HRNet关键点检测", "keypoint"),
        ]

        for model_id, model_name, model_type in model_definitions:
            loaded, device, inf_count, avg_latency, last_inf, mem = self._get_model_status(model_id)

            models.append(DLModelStatus(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                loaded=loaded,
                device=device,
                inference_count=inf_count,
                avg_latency_ms=avg_latency,
                last_inference=last_inf,
                memory_mb=mem,
            ))

        return models

    def _get_model_status(self, model_id: str) -> tuple:
        """获取单个模型状态"""
        loaded = False
        device = "cpu"
        inference_count = 0
        avg_latency = 0.0
        last_inference = None
        memory_mb = 0.0

        if self._model_registry is not None:
            try:
                info = self._model_registry.get_model_info(model_id)
                if info is not None:
                    loaded = info.get("loaded", False)
                    device = info.get("device", "cpu")
                    inference_count = info.get("inference_count", 0)
                    avg_latency = info.get("avg_latency_ms", 0.0)
                    last_inference = info.get("last_inference")
                    memory_mb = info.get("memory_mb", 0.0)
            except Exception:
                pass
        else:
            # 模拟数据
            import random
            loaded = random.random() > 0.3
            if loaded:
                device = random.choice(["cpu", "cuda:0"])
                inference_count = random.randint(0, 1000)
                avg_latency = random.uniform(20, 100)
                memory_mb = random.uniform(50, 500)

        return loaded, device, inference_count, avg_latency, last_inference, memory_mb

    def _collect_plugins(self) -> List[PluginStatus]:
        """收集插件状态"""
        plugins = []

        plugin_definitions = [
            ("transformer_inspection", "主变自主巡视", "1.0.0", ["defect_detection", "state_recognition", "thermal_analysis"]),
            ("switch_inspection", "开关间隔巡视", "1.0.0", ["state_recognition", "logic_validation", "clarity_evaluation"]),
            ("busbar_inspection", "母线及设备巡视", "1.0.0", ["defect_detection", "pin_counting"]),
            ("capacitor_inspection", "电容器电抗器", "1.0.0", ["defect_detection", "tilt_detection"]),
            ("meter_reading", "表计读数", "1.0.0", ["dial_reading", "digital_reading", "led_reading"]),
            ("gas_detection", "气体泄漏检测", "3.0.0", ["concentration_monitoring", "trend_prediction", "leak_detection"]),
            ("indoor_fence", "室内电子围栏", "2.0.0", ["person_tracking", "zone_monitoring", "violation_detection"]),
            ("acoustic_monitoring", "声学监测", "1.0.0", ["pd_detection", "frequency_analysis"]),
            ("bird_monitoring", "鸟害监测", "1.0.0", ["bird_detection", "tracking"]),
        ]

        for plugin_id, plugin_name, version, capabilities in plugin_definitions:
            status, dl_enabled, dl_model_id, inf_count, err_count, last_inf = self._get_plugin_status(plugin_id)

            plugins.append(PluginStatus(
                plugin_id=plugin_id,
                plugin_name=plugin_name,
                version=version,
                status=status,
                dl_model_enabled=dl_enabled,
                dl_model_id=dl_model_id,
                capabilities=capabilities,
                inference_count=inf_count,
                error_count=err_count,
                last_inference=last_inf,
            ))

        return plugins

    def _get_plugin_status(self, plugin_id: str) -> tuple:
        """获取单个插件状态"""
        status = "ready"
        dl_enabled = True
        dl_model_id = None
        inference_count = 0
        error_count = 0
        last_inference = None

        # 模型ID映射
        model_map = {
            "transformer_inspection": "yolov8_vit_transformer",
            "switch_inspection": "yolov8_vit_switch",
            "busbar_inspection": "yolov8_vit_busbar",
            "capacitor_inspection": "yolov8_vit_capacitor",
            "gas_detection": "gl_translstm_gas",
            "acoustic_monitoring": "acoustic_cnn_lstm",
            "indoor_fence": "deep_sort_tracker",
            "meter_reading": "keypoint_hrnet",
        }

        dl_model_id = model_map.get(plugin_id)

        if self._plugin_manager is not None:
            try:
                plugin = self._plugin_manager.get_plugin(plugin_id)
                if plugin is not None:
                    status = plugin.status.value if hasattr(plugin, 'status') else "ready"

                    if hasattr(plugin, '_inference_count'):
                        inference_count = plugin._inference_count
                    if hasattr(plugin, '_error_count'):
                        error_count = plugin._error_count
                    if hasattr(plugin, '_last_inference_time'):
                        last_inference = plugin._last_inference_time.isoformat() if plugin._last_inference_time else None
                else:
                    status = "unloaded"
            except Exception:
                status = "error"
        else:
            # 模拟数据
            import random
            status = random.choice(["ready", "running", "ready", "ready"])
            inference_count = random.randint(0, 500)
            error_count = random.randint(0, 10)

        return status, dl_enabled, dl_model_id, inference_count, error_count, last_inference

    def _collect_alarm_summary(self) -> AlarmSummary:
        """收集告警摘要"""
        total = 0
        critical = 0
        error = 0
        warning = 0
        info = 0
        unresolved = 0
        recent = []

        if self._alarm_manager is not None:
            try:
                summary = self._alarm_manager.get_summary()
                total = summary.get("total", 0)
                critical = summary.get("critical", 0)
                error = summary.get("error", 0)
                warning = summary.get("warning", 0)
                info = summary.get("info", 0)
                unresolved = summary.get("unresolved", 0)
                recent = self._alarm_manager.get_recent(limit=10)
            except Exception:
                pass
        else:
            # 模拟数据
            import random
            total = random.randint(0, 50)
            critical = random.randint(0, 2)
            error = random.randint(0, 5)
            warning = random.randint(0, 15)
            info = total - critical - error - warning
            unresolved = random.randint(0, min(10, total))

            recent = [
                {
                    "id": f"alarm_{i}",
                    "level": random.choice(["warning", "error", "info"]),
                    "module": random.choice(["transformer_inspection", "switch_inspection", "gas_detection"]),
                    "message": f"模拟告警 {i}",
                    "timestamp": datetime.now().isoformat(),
                }
                for i in range(min(5, total))
            ]

        return AlarmSummary(
            total_alarms=total,
            critical_count=critical,
            error_count=error,
            warning_count=warning,
            info_count=info,
            unresolved_count=unresolved,
            recent_alarms=recent,
        )

    def _calculate_system_overview(self, module_health: List[ModuleHealth]) -> SystemOverview:
        """计算系统概览"""
        total = len(module_health)
        online = sum(1 for m in module_health if m.status == ModuleStatus.ONLINE)
        warning = sum(1 for m in module_health if m.status == ModuleStatus.WARNING)
        error = sum(1 for m in module_health if m.status in [ModuleStatus.ERROR, ModuleStatus.OFFLINE])

        # 计算整体健康度
        if total > 0:
            overall_health = sum(m.health_score for m in module_health) / total
        else:
            overall_health = 100.0

        # 计算运行时间
        uptime_hours = (time.time() - self._start_time) / 3600

        return SystemOverview(
            total_modules=total,
            online_modules=online,
            warning_modules=warning,
            error_modules=error,
            overall_health=round(overall_health, 2),
            uptime_hours=round(uptime_hours, 2),
            last_update=datetime.now().isoformat(),
        )

    def _collect_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        import random

        # 这里应该从实际监控系统获取
        return {
            "cpu_usage": random.uniform(20, 60),
            "memory_usage": random.uniform(40, 70),
            "gpu_usage": random.uniform(30, 80),
            "disk_usage": random.uniform(30, 50),
            "network_rx_mbps": random.uniform(10, 100),
            "network_tx_mbps": random.uniform(5, 50),
            "fps_avg": random.uniform(25, 30),
            "latency_avg_ms": random.uniform(30, 60),
            "inference_per_second": random.uniform(5, 20),
        }


# =============================================================================
# 全局实例
# =============================================================================

_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service() -> DashboardService:
    """获取仪表盘服务实例"""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service


# =============================================================================
# API路由
# =============================================================================

def create_dashboard_router() -> APIRouter:
    """创建仪表盘API路由"""

    router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

    @router.get("/", response_model=DashboardData)
    async def get_dashboard():
        """获取仪表盘数据"""
        service = get_dashboard_service()
        return service.get_dashboard_data()

    @router.get("/overview", response_model=SystemOverview)
    async def get_overview():
        """获取系统概览"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()
        return data.system_overview

    @router.get("/alarms", response_model=AlarmSummary)
    async def get_alarms():
        """获取告警摘要"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()
        return data.alarm_summary

    @router.get("/modules", response_model=List[ModuleHealth])
    async def get_modules():
        """获取模块健康状态"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()
        return data.module_health

    @router.get("/modules/{module_id}", response_model=ModuleHealth)
    async def get_module(module_id: str):
        """获取单个模块状态"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()

        for module in data.module_health:
            if module.module_id == module_id:
                return module

        raise HTTPException(status_code=404, detail=f"模块 {module_id} 不存在")

    @router.get("/dl-models", response_model=List[DLModelStatus])
    async def get_dl_models():
        """获取深度学习模型状态"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()
        return data.dl_models

    @router.get("/dl-models/{model_id}", response_model=DLModelStatus)
    async def get_dl_model(model_id: str):
        """获取单个模型状态"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()

        for model in data.dl_models:
            if model.model_id == model_id:
                return model

        raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")

    @router.get("/plugins", response_model=List[PluginStatus])
    async def get_plugins():
        """获取插件状态"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()
        return data.plugins

    @router.get("/plugins/{plugin_id}", response_model=PluginStatus)
    async def get_plugin(plugin_id: str):
        """获取单个插件状态"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()

        for plugin in data.plugins:
            if plugin.plugin_id == plugin_id:
                return plugin

        raise HTTPException(status_code=404, detail=f"插件 {plugin_id} 不存在")

    @router.get("/metrics")
    async def get_metrics():
        """获取系统指标"""
        service = get_dashboard_service()
        data = service.get_dashboard_data()
        return data.metrics

    @router.post("/refresh")
    async def refresh_dashboard():
        """强制刷新仪表盘数据"""
        service = get_dashboard_service()
        data = service.get_dashboard_data(force_refresh=True)
        return {"success": True, "timestamp": data.timestamp}

    @router.websocket("/ws")
    async def websocket_dashboard(websocket: WebSocket):
        """WebSocket实时仪表盘更新"""
        await websocket.accept()
        service = get_dashboard_service()

        try:
            import asyncio

            while True:
                # 每5秒推送一次数据
                data = service.get_dashboard_data(force_refresh=True)
                await websocket.send_json(data.dict())
                await asyncio.sleep(5)

        except WebSocketDisconnect:
            logger.info("[Dashboard WS] 客户端断开连接")
        except Exception as e:
            logger.error(f"[Dashboard WS] 错误: {e}")

    return router


def integrate_dashboard_routes(app):
    """将仪表盘路由集成到主应用"""
    router = create_dashboard_router()
    app.include_router(router)
    logger.info("[Dashboard] API路由已集成")
