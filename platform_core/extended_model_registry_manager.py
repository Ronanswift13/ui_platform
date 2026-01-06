"""
扩展模型注册中心管理器
输变电站全自动AI巡检方案 - 多模态模型统一管理

功能:
1. 统一管理所有类型模型(图像、点云、音频、时序、多模态)
2. 提供异步推理队列支持
3. 模型热更新和版本管理
4. 健康监控和性能统计

版本: 2.0.0
"""

from __future__ import annotations
import os
import yaml
import time
import logging
import threading
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

from platform_core.extended_inference_engine import (
    ExtendedModelConfig,
    ExtendedInferenceResult,
    ExtendedBaseInferenceEngine,
    InferenceEngineFactory,
    ExtendedModelType,
    PointCloudInferenceEngine,
    AudioInferenceEngine,
    TimeSeriesInferenceEngine,
    MultimodalFusionEngine
)

logger = logging.getLogger(__name__)


# =============================================================================
# 模型包装器
# =============================================================================
@dataclass
class ExtendedModelWrapper:
    """扩展模型包装器"""
    config: ExtendedModelConfig
    engine: ExtendedBaseInferenceEngine
    load_time: float = 0.0
    inference_count: int = 0
    total_inference_time: float = 0.0
    last_inference_time: float = 0.0
    last_error: Optional[str] = None
    version: str = "1.0.0"
    
    def infer(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
        """执行推理并更新统计"""
        start_time = time.perf_counter()
        result = self.engine.infer(inputs)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        self.inference_count += 1
        self.total_inference_time += elapsed
        self.last_inference_time = elapsed
        
        if not result.success:
            self.last_error = result.error_message
        
        return result
    
    @property
    def avg_inference_time(self) -> float:
        """平均推理时间(ms)"""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count


# =============================================================================
# 异步推理队列
# =============================================================================
@dataclass
class InferenceTask:
    """推理任务"""
    task_id: str
    model_id: str
    inputs: Dict[str, Any]
    priority: int = 0
    callback: Optional[Callable[[ExtendedInferenceResult], None]] = None
    future: Optional[Future] = None
    created_time: float = field(default_factory=time.time)


class AsyncInferenceQueue:
    """异步推理队列"""
    
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_tasks: Dict[str, InferenceTask] = {}
        self._lock = threading.Lock()
        self._task_counter = 0
    
    def submit(self, model_id: str, inputs: Dict[str, Any],
               infer_func: Callable, priority: int = 0,
               callback: Optional[Callable] = None) -> str:
        """提交推理任务"""
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}_{model_id}"
        
        task = InferenceTask(
            task_id=task_id,
            model_id=model_id,
            inputs=inputs,
            priority=priority,
            callback=callback
        )
        
        # 提交到线程池
        future = self._executor.submit(self._execute_task, task, infer_func)
        task.future = future
        
        with self._lock:
            self._pending_tasks[task_id] = task
        
        return task_id
    
    def _execute_task(self, task: InferenceTask, 
                      infer_func: Callable) -> ExtendedInferenceResult:
        """执行推理任务"""
        try:
            result = infer_func(task.model_id, task.inputs)
            
            if task.callback:
                task.callback(result)
            
            return result
        finally:
            with self._lock:
                if task.task_id in self._pending_tasks:
                    del self._pending_tasks[task.task_id]
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[ExtendedInferenceResult]:
        """获取任务结果"""
        with self._lock:
            task = self._pending_tasks.get(task_id)
        
        if task is None or task.future is None:
            return None
        
        try:
            return task.future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"获取任务结果失败: {e}")
            return None
    
    def cancel(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            task = self._pending_tasks.get(task_id)
        
        if task and task.future:
            return task.future.cancel()
        return False
    
    def shutdown(self, wait: bool = True):
        """关闭队列"""
        self._executor.shutdown(wait=wait)


# =============================================================================
# 扩展模型注册中心
# =============================================================================
class ExtendedModelRegistry:
    """
    扩展模型注册中心
    
    支持多模态模型的统一管理
    """
    
    def __init__(self, backend: str = "onnx_cpu", device_id: int = 0):
        self._configs: Dict[str, ExtendedModelConfig] = {}
        self._models: Dict[str, ExtendedModelWrapper] = {}
        self._lock = threading.RLock()
        self._backend = backend
        self._device_id = device_id
        self._async_queue = AsyncInferenceQueue(max_workers=4)
    
    def register(self, model_id: str, config: ExtendedModelConfig) -> bool:
        """注册模型配置"""
        with self._lock:
            if model_id in self._configs:
                logger.warning(f"模型已注册,将更新配置: {model_id}")
            self._configs[model_id] = config
            logger.info(f"模型注册成功: {model_id} ({config.model_type})")
            return True
    
    def register_from_dict(self, model_id: str, config_dict: Dict) -> bool:
        """从字典注册模型"""
        config = ExtendedModelConfig.from_dict(model_id, config_dict)
        return self.register(model_id, config)
    
    def unregister(self, model_id: str) -> bool:
        """取消注册模型"""
        with self._lock:
            if model_id in self._models:
                self._models[model_id].engine.unload()
                del self._models[model_id]
            if model_id in self._configs:
                del self._configs[model_id]
                return True
            return False
    
    def load(self, model_id: str) -> bool:
        """加载模型"""
        with self._lock:
            if model_id in self._models:
                return True
            
            if model_id not in self._configs:
                logger.error(f"模型未注册: {model_id}")
                return False
            
            config = self._configs[model_id]
            
            # 创建推理引擎
            engine = InferenceEngineFactory.create(config)
            
            if not engine.load():
                logger.error(f"模型加载失败: {model_id}")
                return False
            
            # 创建包装器
            wrapper = ExtendedModelWrapper(
                config=config,
                engine=engine,
                load_time=time.time()
            )
            
            self._models[model_id] = wrapper
            logger.info(f"模型加载成功: {model_id}")
            return True
    
    def unload(self, model_id: str) -> bool:
        """卸载模型"""
        with self._lock:
            if model_id in self._models:
                self._models[model_id].engine.unload()
                del self._models[model_id]
                logger.info(f"模型已卸载: {model_id}")
                return True
            return False
    
    def is_loaded(self, model_id: str) -> bool:
        """检查模型是否已加载"""
        return model_id in self._models
    
    def is_registered(self, model_id: str) -> bool:
        """检查模型是否已注册"""
        return model_id in self._configs
    
    def list_models(self) -> List[str]:
        """列出所有注册的模型"""
        return list(self._configs.keys())
    
    def list_loaded_models(self) -> List[str]:
        """列出所有已加载的模型"""
        return list(self._models.keys())
    
    def get_config(self, model_id: str) -> Optional[ExtendedModelConfig]:
        """获取模型配置"""
        return self._configs.get(model_id)
    
    def infer(self, model_id: str, inputs: Dict[str, Any],
              auto_load: bool = True) -> ExtendedInferenceResult:
        """
        同步推理
        
        Args:
            model_id: 模型ID
            inputs: 输入数据字典
            auto_load: 是否自动加载模型
        
        Returns:
            推理结果
        """
        # 自动加载
        if not self.is_loaded(model_id):
            if auto_load:
                if not self.load(model_id):
                    return ExtendedInferenceResult(
                        success=False,
                        model_id=model_id,
                        inference_time_ms=0,
                        error_message=f"模型加载失败: {model_id}"
                    )
            else:
                return ExtendedInferenceResult(
                    success=False,
                    model_id=model_id,
                    inference_time_ms=0,
                    error_message=f"模型未加载: {model_id}"
                )
        
        return self._models[model_id].infer(inputs)
    
    def infer_async(self, model_id: str, inputs: Dict[str, Any],
                    callback: Optional[Callable] = None,
                    priority: int = 0) -> str:
        """
        异步推理
        
        Args:
            model_id: 模型ID
            inputs: 输入数据
            callback: 完成回调函数
            priority: 优先级
        
        Returns:
            任务ID
        """
        return self._async_queue.submit(
            model_id=model_id,
            inputs=inputs,
            infer_func=self.infer,
            priority=priority,
            callback=callback
        )
    
    def get_async_result(self, task_id: str,
                         timeout: Optional[float] = None) -> Optional[ExtendedInferenceResult]:
        """获取异步推理结果"""
        return self._async_queue.get_result(task_id, timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "registered_count": len(self._configs),
            "loaded_count": len(self._models),
            "backend": self._backend,
            "device_id": self._device_id,
            "models": {}
        }
        
        for model_id, wrapper in self._models.items():
            stats["models"][model_id] = {
                "model_type": wrapper.config.model_type,
                "inference_count": wrapper.inference_count,
                "avg_inference_time_ms": wrapper.avg_inference_time,
                "last_inference_time_ms": wrapper.last_inference_time,
                "load_time": wrapper.load_time,
                "last_error": wrapper.last_error,
                "version": wrapper.version
            }
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        status = {
            "healthy": True,
            "total_models": len(self._configs),
            "loaded_models": len(self._models),
            "model_status": {}
        }
        
        for model_id, config in self._configs.items():
            model_status = {
                "registered": True,
                "loaded": model_id in self._models,
                "model_path_exists": os.path.exists(config.model_path),
                "last_error": None
            }
            
            if model_id in self._models:
                model_status["last_error"] = self._models[model_id].last_error
                if self._models[model_id].last_error:
                    status["healthy"] = False
            
            if not model_status["model_path_exists"]:
                status["healthy"] = False
            
            status["model_status"][model_id] = model_status
        
        return status
    
    def shutdown(self):
        """关闭注册中心"""
        # 关闭异步队列
        self._async_queue.shutdown()
        
        # 卸载所有模型
        for model_id in list(self._models.keys()):
            self.unload(model_id)
        
        logger.info("模型注册中心已关闭")


# =============================================================================
# 扩展模型注册中心管理器
# =============================================================================
class ExtendedModelRegistryManager:
    """
    扩展模型注册中心管理器
    
    负责:
    1. 加载配置文件
    2. 初始化模型注册中心
    3. 提供全局访问接口
    """
    
    _instance: Optional['ExtendedModelRegistryManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return
        
        self._config_path = config_path or "configs/extended_models_config.yaml"
        self._registry: Optional[ExtendedModelRegistry] = None
        self._raw_config: Dict = {}
        self._initialized = False
    
    def initialize(self, backend: str = "onnx_cpu", device_id: int = 0) -> bool:
        """初始化管理器"""
        try:
            # 加载配置
            if not self._load_config():
                return False
            
            # 创建注册中心
            self._registry = ExtendedModelRegistry(backend=backend, device_id=device_id)
            
            # 注册所有模型
            self._register_all_models()
            
            self._initialized = True
            logger.info("扩展模型注册中心管理器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def _load_config(self) -> bool:
        """加载配置文件"""
        config_path = Path(self._config_path)
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在,将创建默认配置: {config_path}")
            self._create_default_config(config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}
            return True
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False
    
    def _create_default_config(self, config_path: Path) -> None:
        """创建默认配置文件"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            "version": "2.0.0",
            "global_settings": {
                "default_backend": "onnx_cpu",
                "device_id": 0,
                "enable_fp16": False,
                "max_batch_size": 1
            },
            "models": {
                "pointcloud_inspection": {},
                "audio_inspection": {},
                "timeseries_inspection": {},
                "multimodal_inspection": {}
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    def _register_all_models(self) -> None:
        """注册配置中的所有模型"""
        models_config = self._raw_config.get("models", {})

        if self._registry is None:
            logger.warning("模型注册中心未初始化，跳过模型注册")
            return

        for category, models in models_config.items():
            if not isinstance(models, dict):
                continue

            for model_id, config in models.items():
                if not isinstance(config, dict):
                    continue

                full_model_id = f"{category}_{model_id}" if category else model_id
                self._registry.register_from_dict(full_model_id, config)
    
    def get_registry(self) -> Optional[ExtendedModelRegistry]:
        """获取模型注册中心"""
        return self._registry
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def check_models(self) -> Dict[str, Any]:
        """检查所有模型状态"""
        if self._registry is None:
            return {}
        
        return self._registry.get_health_status()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self._registry is None:
            return {}
        
        stats = self._registry.get_stats()
        stats["config_path"] = self._config_path
        stats["config_version"] = self._raw_config.get("version", "unknown")
        return stats
    
    def reload_config(self) -> bool:
        """重新加载配置"""
        if not self._load_config():
            return False
        
        if self._registry:
            # 注册新模型(不卸载已加载的)
            self._register_all_models()
        
        return True
    
    def shutdown(self) -> None:
        """关闭管理器"""
        if self._registry:
            self._registry.shutdown()
        self._initialized = False


# =============================================================================
# 便捷函数
# =============================================================================
_manager_instance: Optional[ExtendedModelRegistryManager] = None


def get_extended_model_registry_manager(
    config_path: str = "configs/extended_models_config.yaml"
) -> ExtendedModelRegistryManager:
    """获取全局模型注册中心管理器"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ExtendedModelRegistryManager(config_path)
    return _manager_instance


def initialize_extended_models(
    config_path: str = "configs/extended_models_config.yaml",
    backend: str = "onnx_cpu",
    device_id: int = 0
) -> bool:
    """初始化扩展模型系统"""
    manager = get_extended_model_registry_manager(config_path)
    return manager.initialize(backend=backend, device_id=device_id)


def get_extended_registry() -> Optional[ExtendedModelRegistry]:
    """获取扩展模型注册中心"""
    manager = get_extended_model_registry_manager()
    if manager.is_initialized():
        return manager.get_registry()
    return None


def extended_infer(model_id: str, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
    """执行扩展推理"""
    registry = get_extended_registry()
    if registry is None:
        return ExtendedInferenceResult(
            success=False,
            model_id=model_id,
            inference_time_ms=0,
            error_message="模型注册中心未初始化"
        )
    return registry.infer(model_id, inputs)
