# -*- coding: utf-8 -*-
"""
多模态融合系统集成模块
将增强版融合引擎、数据采集管理器、训练流水线和全景监测API集成到现有平台
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """集成配置"""
    data_dir: str = "./collected_data"
    models_dir: str = "./models"
    enable_auto_training: bool = True
    enable_panoramic_monitoring: bool = True
    enable_alarm_linkage: bool = True
    fusion_strategy: str = "hybrid"
    min_samples_for_retrain: int = 100
    health_check_interval: int = 60
    websocket_broadcast_interval: float = 1.0


class MultimodalSystemIntegration:
    """
    多模态系统集成器
    负责协调各模块之间的交互
    """
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self._initialized = False
        self._running = False
        
        # 核心组件引用
        self.fusion_engine = None
        self.data_manager = None
        self.training_pipeline = None
        self.monitoring_service = None
        self.alarm_service = None
        self.state_cache = None
        
        # 插件管理
        self.plugins: Dict[str, Any] = {}
        self.plugin_processors: Dict[str, Callable] = {}
        
        # 事件处理器
        self.event_handlers: Dict[str, List[Callable]] = {
            'detection': [],
            'fusion_complete': [],
            'alarm': [],
            'health_update': [],
            'training_complete': [],
        }
        
        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # 状态
        self._last_health_check = None
        self._system_status = "uninitialized"
        
    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            logger.info("开始初始化多模态系统集成...")
            
            # 1. 初始化增强版融合引擎
            self._init_fusion_engine()
            
            # 2. 初始化数据采集管理器
            self._init_data_manager()
            
            # 3. 初始化训练流水线
            self._init_training_pipeline()
            
            # 4. 初始化全景监测服务
            if self.config.enable_panoramic_monitoring:
                self._init_monitoring_service()
            
            # 5. 初始化告警联动服务
            if self.config.enable_alarm_linkage:
                self._init_alarm_service()
            
            self._initialized = True
            self._system_status = "initialized"
            logger.info("多模态系统集成初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"多模态系统集成初始化失败: {e}")
            self._system_status = "error"
            return False
    
    def _init_fusion_engine(self):
        """初始化融合引擎"""
        try:
            from plugins.multimodal_fusion.fusion_engine_enhanced import (
                EnhancedFusionEngine,
                FusionConfig
            )
            
            fusion_config = FusionConfig(
                default_strategy=self.config.fusion_strategy,
                enable_bayesian=True,
                enable_attention=True,
                attention_hidden_dim=256,
                num_attention_heads=8,
            )
            
            self.fusion_engine = EnhancedFusionEngine(fusion_config)
            logger.info("融合引擎初始化成功")
            
        except ImportError as e:
            logger.warning(f"无法导入增强版融合引擎，使用基础版本: {e}")
            self._init_basic_fusion_engine()
    
    def _init_basic_fusion_engine(self):
        """初始化基础融合引擎（回退方案）"""
        try:
            from plugins.multimodal_fusion.fusion_engine import FusionEngine
            self.fusion_engine = FusionEngine({})
            logger.info("基础融合引擎初始化成功")
        except ImportError:
            logger.error("无法初始化任何融合引擎")
            raise
    
    def _init_data_manager(self):
        """初始化数据采集管理器"""
        try:
            from platform_core.data_collection_manager import (
                DataCollectionManager,
                AcousticCollector,
                GasCollector,
                AcousticPreprocessor,
                assess_acoustic_quality
            )
            from platform_core.data_collection_manager import ModalityType
            
            self.data_manager = DataCollectionManager(
                data_dir=self.config.data_dir
            )
            
            # 注册采集器
            self.data_manager.register_collector(
                ModalityType.ACOUSTIC, 
                AcousticCollector()
            )
            self.data_manager.register_collector(
                ModalityType.GAS, 
                GasCollector()
            )
            
            # 注册预处理器
            self.data_manager.register_preprocessor(
                ModalityType.ACOUSTIC,
                AcousticPreprocessor()
            )
            
            # 注册质量评估器
            self.data_manager.register_quality_assessor(
                ModalityType.ACOUSTIC,
                assess_acoustic_quality
            )
            
            logger.info("数据采集管理器初始化成功")
            
        except ImportError as e:
            logger.warning(f"无法导入数据采集管理器: {e}")
            self.data_manager = None
    
    def _init_training_pipeline(self):
        """初始化训练流水线"""
        try:
            from platform_core.training_pipeline import (
                TrainingPipeline,
                AutoTrainingScheduler
            )
            
            self.training_pipeline = TrainingPipeline(
                models_dir=self.config.models_dir
            )
            
            if self.config.enable_auto_training and self.data_manager:
                self.auto_scheduler = AutoTrainingScheduler(
                    pipeline=self.training_pipeline,
                    data_manager=self.data_manager,
                    min_new_samples=self.config.min_samples_for_retrain
                )
                logger.info("自动训练调度器初始化成功")
            else:
                self.auto_scheduler = None
            
            logger.info("训练流水线初始化成功")
            
        except ImportError as e:
            logger.warning(f"无法导入训练流水线: {e}")
            self.training_pipeline = None
    
    def _init_monitoring_service(self):
        """初始化全景监测服务"""
        try:
            from apps.panoramic_monitoring_api import (
                CentralStateCache,
                PanoramicMonitoringService
            )
            
            self.state_cache = CentralStateCache()
            self.monitoring_service = PanoramicMonitoringService(self.state_cache)
            
            logger.info("全景监测服务初始化成功")
            
        except ImportError as e:
            logger.warning(f"无法导入全景监测服务: {e}")
            self.monitoring_service = None
    
    def _init_alarm_service(self):
        """初始化告警联动服务"""
        try:
            from apps.panoramic_monitoring_api import AlarmLinkageService
            
            self.alarm_service = AlarmLinkageService()
            
            # 如果有监测服务，注册告警处理器
            if self.monitoring_service:
                self.monitoring_service.register_alarm_handler(
                    self.alarm_service.handle_alarm
                )
            
            logger.info("告警联动服务初始化成功")
            
        except ImportError as e:
            logger.warning(f"无法导入告警联动服务: {e}")
            self.alarm_service = None
    
    def register_plugin(self, plugin_id: str, plugin: Any):
        """注册插件"""
        self.plugins[plugin_id] = plugin
        
        # 如果有监测服务，也注册到监测服务
        if self.monitoring_service:
            self.monitoring_service.register_plugin(plugin_id, plugin)
        
        logger.info(f"插件已注册: {plugin_id}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            logger.debug(f"事件处理器已注册: {event_type}")
    
    def _emit_event(self, event_type: str, data: Any):
        """触发事件"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"事件处理器执行失败 [{event_type}]: {e}")
    
    async def process_multimodal_data(
        self,
        device_id: str,
        modality_data: Dict[str, Dict],
        save_to_collection: bool = True
    ) -> Dict:
        """
        处理多模态数据
        
        Args:
            device_id: 设备ID
            modality_data: 各模态数据 {modality: {features, detections, ...}}
            save_to_collection: 是否保存到数据采集
            
        Returns:
            处理结果
        """
        if not self._initialized:
            return {"success": False, "error": "系统未初始化"}
        
        result = {
            "success": True,
            "device_id": device_id,
            "timestamp": datetime.now().isoformat(),
            "modality_results": {},
            "fusion_result": None,
            "alarms": []
        }
        
        try:
            # 1. 处理各模态数据
            for modality, data in modality_data.items():
                modality_result = await self._process_single_modality(
                    device_id, modality, data
                )
                result["modality_results"][modality] = modality_result
                
                # 触发检测事件
                self._emit_event('detection', {
                    "device_id": device_id,
                    "modality": modality,
                    "result": modality_result
                })
                
                # 保存到数据采集
                if save_to_collection and self.data_manager:
                    self._save_to_collection(device_id, modality, data)
            
            # 2. 执行多模态融合
            if self.fusion_engine and len(modality_data) > 1:
                fusion_result = await self._perform_fusion(
                    device_id, modality_data, result["modality_results"]
                )
                result["fusion_result"] = fusion_result
                
                # 触发融合完成事件
                self._emit_event('fusion_complete', {
                    "device_id": device_id,
                    "result": fusion_result
                })
                
                # 处理融合结果中的告警
                if fusion_result and fusion_result.get("alarms"):
                    result["alarms"].extend(fusion_result["alarms"])
            
            # 3. 更新监测状态
            if self.monitoring_service:
                await self._update_monitoring_state(device_id, result)
            
            # 4. 处理告警
            for alarm in result["alarms"]:
                self._emit_event('alarm', alarm)
            
        except Exception as e:
            logger.error(f"多模态数据处理失败: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    async def _process_single_modality(
        self, 
        device_id: str, 
        modality: str, 
        data: Dict
    ) -> Dict:
        """处理单个模态数据"""
        # 查找对应的插件处理器
        plugin_id = self._get_plugin_for_modality(modality)
        
        if plugin_id and plugin_id in self.plugins:
            plugin = self.plugins[plugin_id]
            try:
                # 调用插件的推理方法
                if hasattr(plugin, 'process') and callable(plugin.process):
                    return await self._run_in_executor(
                        plugin.process, data
                    )
                elif hasattr(plugin, 'infer') and callable(plugin.infer):
                    return await self._run_in_executor(
                        plugin.infer, data.get('image'), data.get('rois', [])
                    )
            except Exception as e:
                logger.error(f"插件处理失败 [{plugin_id}]: {e}")
        
        # 返回原始数据
        return {"raw_data": data, "processed": False}
    
    def _get_plugin_for_modality(self, modality: str) -> Optional[str]:
        """获取模态对应的插件ID"""
        modality_plugin_map = {
            "acoustic": "acoustic_monitoring",
            "gas": "gas_detection",
            "hyperspectral": "hyperspectral_detection",
            "thermal": "thermal_imaging",
            "visual": "visual_inspection",
            "lidar": "slam_mapping",
        }
        return modality_plugin_map.get(modality)
    
    async def _perform_fusion(
        self,
        device_id: str,
        modality_data: Dict,
        modality_results: Dict
    ) -> Dict:
        """执行多模态融合"""
        try:
            from plugins.multimodal_fusion.fusion_engine_enhanced import ModalityData
            
            # 构建融合输入
            fusion_inputs = {}
            for modality, data in modality_data.items():
                fusion_inputs[modality] = ModalityData(
                    modality_type=modality,
                    features=data.get('features', []),
                    confidence=modality_results.get(modality, {}).get('confidence', 0.5),
                    detections=modality_results.get(modality, {}).get('detections', []),
                    metadata=data.get('metadata', {}),
                    timestamp=data.get('timestamp', 0)
                )
            
            # 执行融合
            result = await self._run_in_executor(
                self.fusion_engine.fuse, fusion_inputs
            )
            
            return {
                "success": result.success if hasattr(result, 'success') else True,
                "overall_status": result.overall_status if hasattr(result, 'overall_status') else "unknown",
                "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
                "fused_detections": result.fused_detections if hasattr(result, 'fused_detections') else [],
                "modality_contributions": result.modality_contributions if hasattr(result, 'modality_contributions') else {},
                "diagnostic_report": result.diagnostic_report if hasattr(result, 'diagnostic_report') else {},
                "recommendations": result.recommendations if hasattr(result, 'recommendations') else [],
                "alarms": self._extract_alarms(result) if hasattr(result, 'fused_detections') else []
            }
            
        except Exception as e:
            logger.error(f"融合执行失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_alarms(self, fusion_result) -> List[Dict]:
        """从融合结果中提取告警"""
        alarms = []
        
        # 根据整体状态生成告警
        status = getattr(fusion_result, 'overall_status', 'normal')
        if status in ['warning', 'alarm', 'critical']:
            alarm = {
                "level": status.upper(),
                "message": f"多模态融合检测到{status}状态",
                "timestamp": datetime.now().isoformat(),
                "source": "multimodal_fusion",
                "details": {
                    "confidence": getattr(fusion_result, 'confidence', 0),
                    "recommendations": getattr(fusion_result, 'recommendations', [])
                }
            }
            alarms.append(alarm)
        
        return alarms
    
    def _save_to_collection(self, device_id: str, modality: str, data: Dict):
        """保存数据到采集管理器"""
        if not self.data_manager:
            return
        
        try:
            from platform_core.data_collection_manager import ModalityType
            
            modality_type_map = {
                "acoustic": ModalityType.ACOUSTIC,
                "gas": ModalityType.GAS,
                "hyperspectral": ModalityType.HYPERSPECTRAL,
                "thermal": ModalityType.THERMAL,
                "visual": ModalityType.VISUAL,
                "lidar": ModalityType.LIDAR,
            }
            
            modality_type = modality_type_map.get(modality)
            if modality_type:
                self.data_manager.collect_sample(
                    task_id="auto_collection",
                    modality=modality_type,
                    device_id=device_id,
                    raw_data=data
                )
        except Exception as e:
            logger.error(f"保存数据到采集管理器失败: {e}")
    
    async def _update_monitoring_state(self, device_id: str, result: Dict):
        """更新监测状态"""
        if not self.monitoring_service or not self.state_cache:
            return
        
        try:
            from apps.panoramic_monitoring_api import (
                MonitoringData,
                DeviceStatus
            )
            
            # 更新各模态数据
            for modality, modality_result in result.get("modality_results", {}).items():
                monitoring_data = MonitoringData(
                    device_id=device_id,
                    modality=modality,
                    timestamp=datetime.now().timestamp(),
                    values=modality_result.get("values", {}),
                    status=modality_result.get("status", "normal"),
                    confidence=modality_result.get("confidence", 0.5),
                    detections=modality_result.get("detections", []),
                    metadata=modality_result.get("metadata", {})
                )
                self.state_cache.add_monitoring_data(device_id, modality, monitoring_data)
            
            # 更新融合结果
            if result.get("fusion_result"):
                fusion = result["fusion_result"]
                self.state_cache.fusion_results[device_id] = fusion
                
                # 根据融合结果更新设备状态
                status_map = {
                    "normal": DeviceStatus.NORMAL,
                    "attention": DeviceStatus.ATTENTION,
                    "warning": DeviceStatus.WARNING,
                    "alarm": DeviceStatus.ALARM,
                    "critical": DeviceStatus.CRITICAL,
                }
                new_status = status_map.get(
                    fusion.get("overall_status", "normal"),
                    DeviceStatus.NORMAL
                )
                self.state_cache.update_device_status(device_id, new_status)
            
            # 处理告警
            for alarm in result.get("alarms", []):
                await self._process_alarm(device_id, alarm)
                
        except Exception as e:
            logger.error(f"更新监测状态失败: {e}")
    
    async def _process_alarm(self, device_id: str, alarm: Dict):
        """处理告警"""
        if not self.state_cache:
            return
        
        try:
            from apps.panoramic_monitoring_api import AlarmInfo, AlarmLevel
            
            level_map = {
                "INFO": AlarmLevel.INFO,
                "ATTENTION": AlarmLevel.ATTENTION,
                "WARNING": AlarmLevel.WARNING,
                "ALARM": AlarmLevel.ALARM,
                "CRITICAL": AlarmLevel.CRITICAL,
            }
            
            alarm_info = AlarmInfo(
                alarm_id=f"ALM_{device_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                device_id=device_id,
                level=level_map.get(alarm.get("level", "WARNING"), AlarmLevel.WARNING),
                message=alarm.get("message", ""),
                source=alarm.get("source", "unknown"),
                timestamp=datetime.now().timestamp(),
                details=alarm.get("details", {})
            )
            
            self.state_cache.add_alarm(alarm_info)
            
        except Exception as e:
            logger.error(f"处理告警失败: {e}")
    
    async def _run_in_executor(self, func: Callable, *args) -> Any:
        """在线程池中执行同步函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func, *args)
    
    async def check_and_trigger_training(self) -> Dict:
        """检查并触发自动训练"""
        if not self.auto_scheduler:
            return {"triggered": False, "reason": "自动训练调度器未启用"}
        
        try:
            results = self.auto_scheduler.check_and_retrain()
            
            for result in results:
                self._emit_event('training_complete', result)
            
            return {
                "triggered": len(results) > 0,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"自动训练检查失败: {e}")
            return {"triggered": False, "error": str(e)}
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        status = {
            "status": self._system_status,
            "initialized": self._initialized,
            "components": {
                "fusion_engine": self.fusion_engine is not None,
                "data_manager": self.data_manager is not None,
                "training_pipeline": self.training_pipeline is not None,
                "monitoring_service": self.monitoring_service is not None,
                "alarm_service": self.alarm_service is not None,
            },
            "plugins": list(self.plugins.keys()),
            "config": {
                "fusion_strategy": self.config.fusion_strategy,
                "auto_training_enabled": self.config.enable_auto_training,
                "panoramic_monitoring_enabled": self.config.enable_panoramic_monitoring,
            }
        }
        
        # 添加各组件详细状态
        if self.fusion_engine and hasattr(self.fusion_engine, 'get_status'):
            status["fusion_engine_status"] = self.fusion_engine.get_status()
        
        if self.data_manager and hasattr(self.data_manager, 'get_statistics'):
            status["data_collection_stats"] = self.data_manager.get_statistics()
        
        if self.training_pipeline and hasattr(self.training_pipeline, 'get_training_statistics'):
            status["training_stats"] = self.training_pipeline.get_training_statistics()
        
        return status
    
    async def health_check(self) -> Dict:
        """执行健康检查"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "overall": "healthy",
            "components": {}
        }
        
        # 检查融合引擎
        if self.fusion_engine:
            try:
                health["components"]["fusion_engine"] = {
                    "status": "healthy",
                    "strategy": self.config.fusion_strategy
                }
            except Exception as e:
                health["components"]["fusion_engine"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["overall"] = "degraded"
        
        # 检查数据管理器
        if self.data_manager:
            try:
                stats = self.data_manager.get_statistics() if hasattr(self.data_manager, 'get_statistics') else {}
                health["components"]["data_manager"] = {
                    "status": "healthy",
                    "sample_count": stats.get("total_samples", 0)
                }
            except Exception as e:
                health["components"]["data_manager"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["overall"] = "degraded"
        
        # 检查监测服务
        if self.monitoring_service:
            try:
                health["components"]["monitoring_service"] = {
                    "status": "healthy",
                    "device_count": len(self.state_cache.devices) if self.state_cache else 0
                }
            except Exception as e:
                health["components"]["monitoring_service"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["overall"] = "degraded"
        
        # 检查插件
        health["plugins"] = {}
        for plugin_id, plugin in self.plugins.items():
            try:
                if hasattr(plugin, 'healthcheck'):
                    plugin_health = plugin.healthcheck()
                    health["plugins"][plugin_id] = {
                        "status": "healthy" if plugin_health else "unhealthy"
                    }
                else:
                    health["plugins"][plugin_id] = {"status": "unknown"}
            except Exception as e:
                health["plugins"][plugin_id] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        self._last_health_check = health
        self._emit_event('health_update', health)
        
        return health
    
    def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭多模态系统集成...")
        
        self._running = False
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        # 清理各组件
        for plugin_id, plugin in self.plugins.items():
            try:
                if hasattr(plugin, 'cleanup'):
                    plugin.cleanup()
            except Exception as e:
                logger.error(f"插件清理失败 [{plugin_id}]: {e}")
        
        self._system_status = "shutdown"
        logger.info("多模态系统集成已关闭")


# =============================================================================
# FastAPI集成函数
# =============================================================================

def integrate_multimodal_system(app, config: IntegrationConfig = None):
    """
    将多模态系统集成到FastAPI应用
    
    Args:
        app: FastAPI应用实例
        config: 集成配置
    """
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
    
    # 创建集成实例
    integration = MultimodalSystemIntegration(config)
    
    # 初始化
    if not integration.initialize():
        logger.error("多模态系统初始化失败")
    
    # 创建路由
    router = APIRouter(prefix="/api/multimodal", tags=["multimodal"])
    
    class MultimodalRequest(BaseModel):
        device_id: str
        modality_data: Dict[str, Dict]
        save_to_collection: bool = True
    
    @router.post("/process")
    async def process_multimodal(request: MultimodalRequest):
        """处理多模态数据"""
        result = await integration.process_multimodal_data(
            request.device_id,
            request.modality_data,
            request.save_to_collection
        )
        return result
    
    @router.get("/status")
    async def get_system_status():
        """获取系统状态"""
        return integration.get_system_status()
    
    @router.get("/health")
    async def health_check():
        """健康检查"""
        return await integration.health_check()
    
    @router.post("/training/check")
    async def check_training():
        """检查并触发自动训练"""
        return await integration.check_and_trigger_training()
    
    # 注册路由
    app.include_router(router)
    
    # 注册启动和关闭事件
    @app.on_event("startup")
    async def startup():
        logger.info("多模态系统API启动")
    
    @app.on_event("shutdown")
    async def shutdown():
        integration.shutdown()
    
    # 返回集成实例供外部使用
    return integration


# =============================================================================
# 插件集成辅助函数
# =============================================================================

def integrate_with_existing_plugins(integration: MultimodalSystemIntegration, plugin_manager):
    """
    与现有插件管理器集成
    
    Args:
        integration: 多模态系统集成实例
        plugin_manager: 现有的插件管理器实例
    """
    # 获取已加载的插件
    if hasattr(plugin_manager, '_plugins'):
        for plugin_id, plugin in plugin_manager._plugins.items():
            integration.register_plugin(plugin_id, plugin)
            logger.info(f"已集成现有插件: {plugin_id}")
    
    # 如果有模型注册器，设置到融合引擎
    if hasattr(plugin_manager, '_model_registry') and integration.fusion_engine:
        if hasattr(integration.fusion_engine, 'set_model_registry'):
            integration.fusion_engine.set_model_registry(plugin_manager._model_registry)
            logger.info("已设置模型注册器到融合引擎")


def setup_alarm_notifications(integration: MultimodalSystemIntegration, config: Dict):
    """
    设置告警通知
    
    Args:
        integration: 多模态系统集成实例
        config: 通知配置
            {
                'http_endpoint': 'http://...',
                'email_recipients': ['...'],
                'sms_recipients': ['...'],
                'escalation_rules': [...]
            }
    """
    if integration.alarm_service:
        integration.alarm_service.configure(config)
        logger.info("告警通知配置完成")


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # 创建配置
        config = IntegrationConfig(
            data_dir="./test_data",
            models_dir="./test_models",
            enable_auto_training=True,
            enable_panoramic_monitoring=True,
            fusion_strategy="hybrid"
        )
        
        # 创建集成实例
        integration = MultimodalSystemIntegration(config)
        
        # 初始化
        if integration.initialize():
            print("系统初始化成功")
            
            # 获取状态
            status = integration.get_system_status()
            print(f"系统状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
            
            # 模拟处理多模态数据
            result = await integration.process_multimodal_data(
                device_id="T001",
                modality_data={
                    "acoustic": {
                        "features": [0.1, 0.2, 0.3],
                        "timestamp": 1234567890
                    },
                    "gas": {
                        "features": {"SF6": 99.5, "H2": 0.01},
                        "timestamp": 1234567890
                    }
                }
            )
            print(f"处理结果: {json.dumps(result, indent=2, ensure_ascii=False, default=str)}")
            
            # 健康检查
            health = await integration.health_check()
            print(f"健康状态: {json.dumps(health, indent=2, ensure_ascii=False)}")
        else:
            print("系统初始化失败")
        
        # 关闭
        integration.shutdown()
    
    asyncio.run(main())
