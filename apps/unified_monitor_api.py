#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一监测API
==========================================

提供统一的检测接口，聚合所有插件的推理结果

端点:
    - GET  /api/ui/monitor     - 获取聚合监测数据
    - GET  /api/ui/regions     - 获取识别区域列表
    - POST /api/ui/train       - 启动模型训练
    - GET  /api/ui/models      - 查询模型库状态

作者: 破夜绘明团队
版本: 2.0.0
日期: 2025
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# 数据模型
# =============================================================================

class ModuleType(str, Enum):
    """模块类型"""
    TRANSFORMER = "transformer"
    SWITCH = "switch"
    BUSBAR = "busbar"
    CAPACITOR = "capacitor"
    METER = "meter"
    ACOUSTIC = "acoustic"
    GAS = "gas"
    HYPERSPECTRAL = "hyperspectral"
    SLAM = "slam"
    FUSION = "fusion"


class DetectionLevel(str, Enum):
    """检测等级"""
    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"


class DetectionItem(BaseModel):
    """检测项"""
    id: int
    label: str
    confidence: float = Field(ge=0, le=1)
    x: float = Field(ge=0, le=1, description="归一化X坐标")
    y: float = Field(ge=0, le=1, description="归一化Y坐标")
    width: float = Field(ge=0, le=1, description="归一化宽度")
    height: float = Field(ge=0, le=1, description="归一化高度")
    level: DetectionLevel = DetectionLevel.NORMAL
    module: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegionItem(BaseModel):
    """识别区域"""
    id: int
    name: str
    type: str
    status: str = "normal"
    confidence: float = 0.9
    x: float = 0
    y: float = 0
    width: float = 0.1
    height: float = 0.1


class AlarmItem(BaseModel):
    """告警项"""
    id: int
    type: str
    level: str = "warning"
    message: str
    time: str
    module: str
    device_id: str = ""
    acknowledged: bool = False


class ModuleStatus(BaseModel):
    """模块状态"""
    status: str = "normal"
    loaded: bool = False
    error: Optional[str] = None
    detected_items: List[DetectionItem] = Field(default_factory=list)
    last_update: float = 0


class MonitorResponse(BaseModel):
    """监测响应"""
    success: bool = True
    timestamp: float
    camera_id: str
    module: str
    voltage: str

    # 各模块数据
    transformer: Optional[ModuleStatus] = None
    switch: Optional[ModuleStatus] = None
    busbar: Optional[ModuleStatus] = None
    capacitor: Optional[ModuleStatus] = None
    meter: Optional[ModuleStatus] = None
    acoustic: Optional[Dict[str, Any]] = None
    gas: Optional[Dict[str, Any]] = None
    hyperspectral: Optional[Dict[str, Any]] = None
    slam: Optional[Dict[str, Any]] = None
    fusion: Optional[Dict[str, Any]] = None

    # 聚合检测结果
    detections: List[DetectionItem] = Field(default_factory=list)

    # 告警
    alarms: List[AlarmItem] = Field(default_factory=list)

    # 性能指标
    metrics: Dict[str, float] = Field(default_factory=dict)


class RegionsResponse(BaseModel):
    """区域列表响应"""
    success: bool = True
    module: str
    camera_id: str
    regions: List[RegionItem]
    total: int


class TrainRequest(BaseModel):
    """训练请求"""
    plugin_id: str
    voltage_level: str = "110kv"
    dataset_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    """训练响应"""
    success: bool
    task_id: str
    plugin_id: str
    status: str
    message: str


class ModelInfo(BaseModel):
    """模型信息"""
    model_id: str
    plugin_id: str
    version: str
    status: str
    trained_at: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    """模型列表响应"""
    success: bool = True
    models: List[ModelInfo]
    total: int


# =============================================================================
# 统一监测服务
# =============================================================================

class UnifiedMonitorService:
    """统一监测服务"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._plugin_manager = None
        self._model_registry = None
        self._cache: Dict[str, Any] = {}
        self._initialized = True

        logger.info("统一监测服务初始化完成")

    def _get_plugin_manager(self):
        """懒加载插件管理器"""
        if self._plugin_manager is None:
            try:
                from apps.advanced_plugins_api import plugin_manager
                self._plugin_manager = plugin_manager
            except ImportError:
                logger.warning("高级插件管理器不可用")
        return self._plugin_manager

    def get_monitor_data(
        self,
        camera_id: str,
        module: str,
        voltage: str,
        site_id: Optional[str] = None
    ) -> MonitorResponse:
        """
        获取聚合监测数据

        调用各插件的 process() 方法并聚合结果
        """
        result = MonitorResponse(
            timestamp=time.time(),
            camera_id=camera_id,
            module=module,
            voltage=voltage
        )

        plugin_manager = self._get_plugin_manager()
        all_detections = []
        all_alarms = []

        # 获取各模块数据
        modules_to_query = self._get_modules_for_voltage(voltage, module)

        for mod_id in modules_to_query:
            try:
                mod_data = self._get_module_data(mod_id, camera_id, plugin_manager)

                # 设置模块数据
                if mod_id == "transformer":
                    result.transformer = mod_data
                elif mod_id == "switch":
                    result.switch = mod_data
                elif mod_id == "busbar":
                    result.busbar = mod_data
                elif mod_id == "capacitor":
                    result.capacitor = mod_data
                elif mod_id == "meter":
                    result.meter = mod_data
                elif mod_id == "acoustic":
                    result.acoustic = self._convert_to_dict(mod_data)
                elif mod_id == "gas":
                    result.gas = self._convert_to_dict(mod_data)
                elif mod_id == "hyperspectral":
                    result.hyperspectral = self._convert_to_dict(mod_data)
                elif mod_id == "slam":
                    result.slam = self._convert_to_dict(mod_data)
                elif mod_id == "fusion":
                    result.fusion = self._convert_to_dict(mod_data)

                # 收集检测结果
                if mod_data and hasattr(mod_data, 'detected_items'):
                    all_detections.extend(mod_data.detected_items)

            except Exception as e:
                logger.warning(f"获取模块 {mod_id} 数据失败: {e}")

        # 聚合检测结果
        result.detections = all_detections

        # 获取告警
        result.alarms = self._get_alarms(modules_to_query)

        # 性能指标
        result.metrics = self._get_metrics()

        return result

    def _get_modules_for_voltage(self, voltage: str, current_module: str) -> List[str]:
        """根据电压等级获取需要查询的模块"""
        # 基础模块始终查询
        basic_modules = ["transformer", "switch", "busbar", "capacitor", "meter"]

        # 高级模块按需查询
        advanced_modules = ["acoustic", "gas", "hyperspectral", "slam", "fusion"]

        # 如果当前模块是高级模块，优先查询
        if current_module in advanced_modules:
            return [current_module] + basic_modules

        return basic_modules + advanced_modules

    def _get_module_data(
        self,
        module_id: str,
        camera_id: str,
        plugin_manager
    ) -> Optional[ModuleStatus]:
        """获取单个模块的数据"""

        # 模块到插件ID的映射
        plugin_mapping = {
            "transformer": "transformer_inspection",
            "switch": "switch_inspection",
            "busbar": "busbar_inspection",
            "capacitor": "capacitor_inspection",
            "meter": "meter_reading",
            "acoustic": "acoustic_monitoring",
            "gas": "gas_detection",
            "hyperspectral": "hyperspectral_detection",
            "slam": "slam_mapping",
            "fusion": "multimodal_fusion"
        }

        plugin_id = plugin_mapping.get(module_id)
        if not plugin_id:
            return None

        # 尝试获取插件
        plugin = None
        if plugin_manager:
            plugin = plugin_manager.get_plugin(plugin_id)

        if plugin and hasattr(plugin, 'process'):
            try:
                # 调用插件处理
                result = plugin.process({
                    "camera_id": camera_id,
                    "timestamp": time.time()
                })

                return ModuleStatus(
                    status=result.get("status", "normal"),
                    loaded=True,
                    detected_items=result.get("detections", []),
                    last_update=time.time()
                )
            except Exception as e:
                logger.warning(f"插件 {plugin_id} 处理失败: {e}")
                return ModuleStatus(
                    status="error",
                    loaded=True,
                    error=str(e),
                    last_update=time.time()
                )

        # 返回模拟数据
        return self._get_mock_module_data(module_id, camera_id)

    def _get_mock_module_data(self, module_id: str, camera_id: str) -> ModuleStatus:
        """获取模拟模块数据"""
        import random

        # 模拟检测结果
        mock_detections = {
            "transformer": [
                DetectionItem(id=1, label="储油柜", confidence=0.95, x=0.1, y=0.1, width=0.15, height=0.2, module=module_id),
                DetectionItem(id=2, label="冷却系统", confidence=0.92, x=0.3, y=0.4, width=0.2, height=0.3, module=module_id),
                DetectionItem(id=3, label="套管", confidence=0.88, x=0.6, y=0.1, width=0.1, height=0.4, module=module_id),
            ],
            "switch": [
                DetectionItem(id=1, label="断路器", confidence=0.93, x=0.2, y=0.3, width=0.15, height=0.25, module=module_id),
                DetectionItem(id=2, label="隔离开关", confidence=0.91, x=0.5, y=0.2, width=0.12, height=0.2, module=module_id),
            ],
            "meter": [
                DetectionItem(id=1, label="油温表", confidence=0.89, x=0.15, y=0.6, width=0.08, height=0.1, module=module_id),
                DetectionItem(id=2, label="油位计", confidence=0.87, x=0.7, y=0.5, width=0.06, height=0.12, module=module_id),
            ]
        }

        detections = mock_detections.get(module_id, [])

        return ModuleStatus(
            status="normal",
            loaded=False,  # 标记为模拟数据
            detected_items=detections,
            last_update=time.time()
        )

    def _convert_to_dict(self, data) -> Optional[Dict[str, Any]]:
        """将数据转换为字典"""
        if data is None:
            return None
        if isinstance(data, dict):
            return data
        if hasattr(data, 'dict'):
            return data.dict()
        if hasattr(data, '__dict__'):
            return data.__dict__
        return {"data": data}

    def _get_alarms(self, modules: List[str]) -> List[AlarmItem]:
        """获取告警列表"""
        # TODO: 从告警中心获取实际告警
        return []

    def _get_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        import random
        return {
            "fps": random.uniform(24, 30),
            "latency": random.uniform(30, 60),
            "gpu_usage": random.uniform(40, 80),
            "memory_usage": random.uniform(50, 70)
        }

    def get_regions(
        self,
        module: str,
        camera_id: str,
        voltage: str
    ) -> RegionsResponse:
        """获取识别区域列表"""

        # 根据模块和电压等级获取配置的区域
        regions = self._load_regions_config(module, camera_id, voltage)

        return RegionsResponse(
            module=module,
            camera_id=camera_id,
            regions=regions,
            total=len(regions)
        )

    def _load_regions_config(
        self,
        module: str,
        camera_id: str,
        voltage: str
    ) -> List[RegionItem]:
        """加载区域配置"""

        # 默认区域配置
        default_regions = {
            "transformer": [
                RegionItem(id=1, name="储油柜", type="设备外观查看", status="normal", confidence=0.95),
                RegionItem(id=2, name="冷却系统", type="设备外观查看", status="normal", confidence=0.92),
                RegionItem(id=3, name="取气阀", type="设备外观查看", status="warning", confidence=0.88),
                RegionItem(id=4, name="地层及接地引", type="设备外观查看", status="normal", confidence=0.91),
                RegionItem(id=5, name="本体油位窗A侧", type="表计读数", status="normal", confidence=0.89),
                RegionItem(id=6, name="本体顶部绝缘件", type="设备外观查看", status="normal", confidence=0.93),
                RegionItem(id=7, name="风扇/端子箱/5", type="设备外观查看", status="normal", confidence=0.90),
                RegionItem(id=8, name="套压输入附着物", type="表计读数", status="normal", confidence=0.87),
            ],
            "switch": [
                RegionItem(id=1, name="断路器本体", type="状态识别", status="normal", confidence=0.94),
                RegionItem(id=2, name="隔离开关A", type="状态识别", status="normal", confidence=0.91),
                RegionItem(id=3, name="隔离开关B", type="状态识别", status="normal", confidence=0.92),
                RegionItem(id=4, name="接地开关", type="状态识别", status="normal", confidence=0.89),
                RegionItem(id=5, name="机构箱", type="设备外观查看", status="normal", confidence=0.87),
            ],
            "busbar": [
                RegionItem(id=1, name="绝缘子串1", type="缺陷检测", status="normal", confidence=0.93),
                RegionItem(id=2, name="绝缘子串2", type="缺陷检测", status="normal", confidence=0.91),
                RegionItem(id=3, name="金具连接点", type="设备外观查看", status="normal", confidence=0.88),
                RegionItem(id=4, name="导线接头", type="设备外观查看", status="warning", confidence=0.85),
            ],
            "capacitor": [
                RegionItem(id=1, name="电容器组1", type="设备外观查看", status="normal", confidence=0.92),
                RegionItem(id=2, name="电容器组2", type="设备外观查看", status="normal", confidence=0.90),
                RegionItem(id=3, name="放电线圈", type="设备外观查看", status="normal", confidence=0.88),
                RegionItem(id=4, name="围栏区域", type="入侵检测", status="normal", confidence=0.95),
            ],
            "meter": [
                RegionItem(id=1, name="油温表", type="表计读数", status="normal", confidence=0.91),
                RegionItem(id=2, name="油位计", type="表计读数", status="normal", confidence=0.89),
                RegionItem(id=3, name="压力表", type="表计读数", status="normal", confidence=0.87),
                RegionItem(id=4, name="SF6压力表", type="表计读数", status="normal", confidence=0.90),
            ],
        }

        return default_regions.get(module, [])


# =============================================================================
# 全局服务实例
# =============================================================================
monitor_service = UnifiedMonitorService()


# =============================================================================
# 创建路由
# =============================================================================

def create_unified_monitor_router() -> APIRouter:
    """创建统一监测路由"""

    router = APIRouter(prefix="/api/ui", tags=["unified-monitor"])

    @router.get("/monitor", response_model=MonitorResponse, summary="获取聚合监测数据")
    async def get_monitor_data(
        camera_id: str = Query(default="cam1", description="摄像头ID"),
        module: str = Query(default="transformer", description="当前模块"),
        voltage: str = Query(default="110kv", description="电压等级"),
        site_id: Optional[str] = Query(default=None, description="站点ID")
    ):
        """
        获取聚合监测数据

        该接口调用各插件的 process() 方法，并将结果聚合后返回。
        返回格式统一，包含各模块状态、检测结果和告警信息。
        """
        try:
            return monitor_service.get_monitor_data(
                camera_id=camera_id,
                module=module,
                voltage=voltage,
                site_id=site_id
            )
        except Exception as e:
            logger.error(f"获取监测数据失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/regions", response_model=RegionsResponse, summary="获取识别区域列表")
    async def get_regions(
        module: str = Query(default="transformer", description="模块ID"),
        camera: str = Query(default="cam1", description="摄像头ID"),
        voltage: str = Query(default="110kv", description="电压等级")
    ):
        """获取指定模块和摄像头的识别区域列表"""
        try:
            return monitor_service.get_regions(
                module=module,
                camera_id=camera,
                voltage=voltage
            )
        except Exception as e:
            logger.error(f"获取区域列表失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/train", response_model=TrainResponse, summary="启动模型训练")
    async def start_training(
        request: TrainRequest,
        background_tasks: BackgroundTasks
    ):
        """
        启动模型训练任务

        在后台异步执行训练，返回任务ID供状态查询
        """
        import uuid

        task_id = str(uuid.uuid4())[:8]

        # TODO: 实际启动训练任务
        # background_tasks.add_task(run_training, task_id, request)

        return TrainResponse(
            success=True,
            task_id=task_id,
            plugin_id=request.plugin_id,
            status="queued",
            message=f"训练任务已提交，任务ID: {task_id}"
        )

    @router.get("/train/{task_id}", summary="查询训练状态")
    async def get_training_status(task_id: str):
        """查询训练任务状态"""
        # TODO: 从训练管理器获取状态
        return {
            "success": True,
            "task_id": task_id,
            "status": "running",
            "progress": 45.5,
            "current_epoch": 23,
            "total_epochs": 50,
            "metrics": {
                "loss": 0.0234,
                "accuracy": 0.956,
                "mAP": 0.892
            }
        }

    @router.get("/models", response_model=ModelsResponse, summary="查询模型库状态")
    async def list_models(
        plugin_id: Optional[str] = Query(default=None, description="插件ID过滤")
    ):
        """获取模型库中的所有模型"""

        # TODO: 从模型注册器获取实际数据
        models = [
            ModelInfo(
                model_id="transformer_defect_yolov8",
                plugin_id="transformer_inspection",
                version="1.2.0",
                status="loaded",
                trained_at="2025-01-15",
                metrics={"mAP": 0.912, "fps": 45.2}
            ),
            ModelInfo(
                model_id="switch_state_yolov8",
                plugin_id="switch_inspection",
                version="1.1.0",
                status="loaded",
                trained_at="2025-01-10",
                metrics={"accuracy": 0.978, "fps": 52.1}
            ),
            ModelInfo(
                model_id="meter_reading_hrnet",
                plugin_id="meter_reading",
                version="2.0.0",
                status="loaded",
                trained_at="2025-01-20",
                metrics={"accuracy": 0.965, "fps": 38.5}
            ),
            ModelInfo(
                model_id="acoustic_anomaly_cnn",
                plugin_id="acoustic_monitoring",
                version="1.0.0",
                status="available",
                metrics={"f1_score": 0.891}
            ),
            ModelInfo(
                model_id="gas_prediction_lstm",
                plugin_id="gas_detection",
                version="1.0.0",
                status="available",
                metrics={"mae": 0.023}
            ),
        ]

        # 过滤
        if plugin_id:
            models = [m for m in models if m.plugin_id == plugin_id]

        return ModelsResponse(
            models=models,
            total=len(models)
        )

    @router.post("/models/{model_id}/load", summary="加载模型")
    async def load_model(model_id: str):
        """加载指定模型到推理引擎"""
        # TODO: 实际加载模型
        return {
            "success": True,
            "model_id": model_id,
            "message": f"模型 {model_id} 已加载"
        }

    @router.delete("/models/{model_id}/unload", summary="卸载模型")
    async def unload_model(model_id: str):
        """从推理引擎卸载模型"""
        # TODO: 实际卸载模型
        return {
            "success": True,
            "model_id": model_id,
            "message": f"模型 {model_id} 已卸载"
        }

    return router


# =============================================================================
# 集成函数
# =============================================================================

def integrate_unified_monitor_routes(app):
    """
    将统一监测路由集成到FastAPI应用

    Args:
        app: FastAPI应用实例
    """
    router = create_unified_monitor_router()
    app.include_router(router)

    logger.info("✓ 统一监测API路由已集成 (/api/ui/*)")


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'MonitorResponse',
    'RegionsResponse',
    'TrainRequest',
    'TrainResponse',
    'ModelsResponse',
    'UnifiedMonitorService',
    'monitor_service',
    'create_unified_monitor_router',
    'integrate_unified_monitor_routes'
]
