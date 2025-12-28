"""
模型注册中心管理器
输变电站全自动AI巡检方案 - 深度学习推理统一管理

功能:
1. 统一加载和管理所有ONNX/TensorRT模型
2. 提供模型缓存和懒加载机制
3. 支持模型热更新
4. 提供统一的推理接口
5. 模型健康检查和监控

使用方式:
    from platform_core.model_registry_manager import ModelRegistryManager
    
    # 初始化
    manager = ModelRegistryManager("configs/models_config.yaml")
    manager.initialize()
    
    # 获取模型进行推理
    result = manager.infer("transformer_defect_yolov8n", image)
    
    # 或者获取registry传递给检测器
    registry = manager.get_registry()
    detector = TransformerDetectorEnhanced(config, model_registry=registry)
"""

from __future__ import annotations
import os
import yaml
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)


class InferenceBackend(Enum):
    """推理后端类型"""
    ONNX_CPU = "onnx_cpu"
    ONNX_CUDA = "onnx_cuda"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class ModelType(Enum):
    """模型类型"""
    YOLOV8 = "yolov8"
    RTDETR = "rtdetr"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    KEYPOINT = "keypoint"
    OCR = "ocr"
    ANOMALY_DETECTION = "anomaly_detection"
    REGRESSION = "regression"


@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    model_path: str
    model_type: str
    input_size: Tuple[int, int]
    input_format: str = "NCHW"
    output_format: str = "default"
    classes: Optional[List[str]] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    description: str = ""
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, model_id: str, config: Dict) -> 'ModelConfig':
        """从字典创建配置"""
        return cls(
            model_id=model_id,
            model_path=config.get("model_path", ""),
            model_type=config.get("model_type", ""),
            input_size=tuple(config.get("input_size", [640, 640])),
            input_format=config.get("input_format", "NCHW"),
            output_format=config.get("output_format", "default"),
            classes=config.get("classes"),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            nms_threshold=config.get("nms_threshold", 0.45),
            description=config.get("description", ""),
            extra_config={k: v for k, v in config.items() 
                         if k not in ["model_id", "model_path", "model_type", 
                                      "input_size", "input_format", "output_format",
                                      "classes", "confidence_threshold", "nms_threshold",
                                      "description"]}
        )


@dataclass
class InferenceResult:
    """推理结果"""
    success: bool
    model_id: str
    outputs: Dict[str, np.ndarray]
    inference_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseInferenceEngine(ABC):
    """推理引擎基类"""
    
    @abstractmethod
    def load(self, model_path: str, config: ModelConfig) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行推理"""
        pass
    
    @abstractmethod
    def unload(self):
        """卸载模型"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass


class ONNXInferenceEngine(BaseInferenceEngine):
    """ONNX推理引擎"""
    
    def __init__(self, backend: InferenceBackend = InferenceBackend.ONNX_CPU,
                 device_id: int = 0, enable_fp16: bool = False):
        self._backend = backend
        self._device_id = device_id
        self._enable_fp16 = enable_fp16
        self._session = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._config: Optional[ModelConfig] = None
    
    def load(self, model_path: str, config: ModelConfig) -> bool:
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
            
            # 检查模型文件
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 配置执行提供者
            providers = self._get_providers()
            
            # 会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 创建会话
            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # 获取输入输出名称
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            self._config = config
            
            logger.info(f"模型加载成功: {config.model_id} ({model_path})")
            logger.info(f"  输入: {self._input_names}")
            logger.info(f"  输出: {self._output_names}")
            logger.info(f"  后端: {self._backend.value}")
            
            return True
            
        except ImportError:
            logger.error("未安装onnxruntime，请运行: pip install onnxruntime-gpu")
            return False
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def _get_providers(self) -> List[str]:
        """获取执行提供者列表"""
        if self._backend == InferenceBackend.ONNX_CUDA:
            return [
                ('CUDAExecutionProvider', {
                    'device_id': self._device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                'CPUExecutionProvider'
            ]
        elif self._backend == InferenceBackend.TENSORRT:
            return [
                ('TensorrtExecutionProvider', {
                    'device_id': self._device_id,
                    'trt_fp16_enable': self._enable_fp16,
                }),
                ('CUDAExecutionProvider', {'device_id': self._device_id}),
                'CPUExecutionProvider'
            ]
        else:
            return ['CPUExecutionProvider']
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行推理"""
        if self._session is None:
            raise RuntimeError("模型未加载")
        
        # 如果只提供了一个输入且输入名不匹配，自动映射
        if len(inputs) == 1 and len(self._input_names) == 1:
            input_key = list(inputs.keys())[0]
            if input_key != self._input_names[0]:
                inputs = {self._input_names[0]: inputs[input_key]}
        
        # 执行推理
        outputs = self._session.run(self._output_names, inputs)
        
        # 构建输出字典
        return {name: output for name, output in zip(self._output_names, outputs)}
    
    def unload(self):
        """卸载模型"""
        if self._session is not None:
            del self._session
            self._session = None
            logger.info(f"模型已卸载: {self._config.model_id if self._config else 'unknown'}")
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._session is not None


class ModelWrapper:
    """模型包装器 - 包含引擎和配置"""
    
    def __init__(self, config: ModelConfig, engine: BaseInferenceEngine):
        self.config = config
        self.engine = engine
        self.load_time: Optional[float] = None
        self.inference_count: int = 0
        self.total_inference_time: float = 0.0
        self._lock = threading.Lock()
    
    def infer(self, image: np.ndarray) -> InferenceResult:
        """执行推理"""
        start_time = time.time()
        
        try:
            # 预处理图像
            input_tensor = self._preprocess(image)
            
            # 执行推理
            with self._lock:
                outputs = self.engine.infer({"images": input_tensor})
            
            inference_time = (time.time() - start_time) * 1000
            
            # 更新统计
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return InferenceResult(
                success=True,
                model_id=self.config.model_id,
                outputs=outputs,
                inference_time_ms=inference_time,
                metadata={
                    "input_shape": input_tensor.shape,
                    "output_shapes": {k: v.shape for k, v in outputs.items()}
                }
            )
            
        except Exception as e:
            return InferenceResult(
                success=False,
                model_id=self.config.model_id,
                outputs={},
                inference_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        import cv2
        
        h, w = self.config.input_size
        
        # 调整尺寸
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))
        
        # BGR转RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # HWC转CHW
        if self.config.input_format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        
        # 添加batch维度
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @property
    def avg_inference_time(self) -> float:
        """平均推理时间(ms)"""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count


class ModelRegistry:
    """
    模型注册中心
    
    提供统一的模型管理接口，支持:
    - 模型注册和查询
    - 懒加载和缓存
    - 推理调用
    """
    
    def __init__(self, backend: InferenceBackend = InferenceBackend.ONNX_CUDA,
                 device_id: int = 0, enable_fp16: bool = False,
                 base_path: str = ""):
        self._backend = backend
        self._device_id = device_id
        self._enable_fp16 = enable_fp16
        self._base_path = base_path
        self._configs: Dict[str, ModelConfig] = {}
        self._models: Dict[str, ModelWrapper] = {}
        self._lock = threading.Lock()
    
    def register(self, config: ModelConfig) -> bool:
        """注册模型配置"""
        with self._lock:
            self._configs[config.model_id] = config
            logger.info(f"模型已注册: {config.model_id}")
            return True
    
    def register_from_dict(self, model_id: str, config_dict: Dict) -> bool:
        """从字典注册模型"""
        config = ModelConfig.from_dict(model_id, config_dict)
        return self.register(config)
    
    def get_config(self, model_id: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self._configs.get(model_id)
    
    def list_models(self) -> List[str]:
        """列出所有注册的模型"""
        return list(self._configs.keys())
    
    def is_loaded(self, model_id: str) -> bool:
        """检查模型是否已加载"""
        return model_id in self._models and self._models[model_id].engine.is_loaded()
    
    def load(self, model_id: str) -> bool:
        """加载模型"""
        if model_id not in self._configs:
            logger.error(f"模型未注册: {model_id}")
            return False
        
        if self.is_loaded(model_id):
            logger.info(f"模型已加载: {model_id}")
            return True
        
        config = self._configs[model_id]
        
        # 解析模型路径
        model_path = config.model_path
        if self._base_path and not os.path.isabs(model_path):
            model_path = os.path.join(self._base_path, model_path)
        
        # 检查模型文件
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return False
        
        # 创建推理引擎
        engine = ONNXInferenceEngine(
            backend=self._backend,
            device_id=self._device_id,
            enable_fp16=self._enable_fp16
        )
        
        # 加载模型
        if not engine.load(model_path, config):
            return False
        
        # 创建包装器
        wrapper = ModelWrapper(config, engine)
        wrapper.load_time = time.time()
        
        with self._lock:
            self._models[model_id] = wrapper
        
        return True
    
    def unload(self, model_id: str) -> bool:
        """卸载模型"""
        with self._lock:
            if model_id in self._models:
                self._models[model_id].engine.unload()
                del self._models[model_id]
                return True
        return False
    
    def infer(self, model_id: str, image: np.ndarray, 
              auto_load: bool = True) -> InferenceResult:
        """
        执行推理
        
        Args:
            model_id: 模型ID
            image: 输入图像 (BGR, HWC格式)
            auto_load: 是否自动加载未加载的模型
            
        Returns:
            InferenceResult: 推理结果
        """
        # 自动加载
        if not self.is_loaded(model_id):
            if auto_load:
                if not self.load(model_id):
                    return InferenceResult(
                        success=False,
                        model_id=model_id,
                        outputs={},
                        inference_time_ms=0,
                        error_message=f"模型加载失败: {model_id}"
                    )
            else:
                return InferenceResult(
                    success=False,
                    model_id=model_id,
                    outputs={},
                    inference_time_ms=0,
                    error_message=f"模型未加载: {model_id}"
                )
        
        # 执行推理
        return self._models[model_id].infer(image)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "registered_count": len(self._configs),
            "loaded_count": len(self._models),
            "models": {}
        }
        
        for model_id, wrapper in self._models.items():
            stats["models"][model_id] = {
                "inference_count": wrapper.inference_count,
                "avg_inference_time_ms": wrapper.avg_inference_time,
                "load_time": wrapper.load_time
            }
        
        return stats


class ModelRegistryManager:
    """
    模型注册中心管理器
    
    负责:
    1. 加载配置文件
    2. 初始化模型注册中心
    3. 提供全局访问接口
    """
    
    _instance: Optional['ModelRegistryManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return
        
        self._config_path = config_path or "configs/models_config.yaml"
        self._config: Dict[str, Any] = {}
        self._registry: Optional[ModelRegistry] = None
        self._base_path = ""
    
    def initialize(self, preload_models: Optional[List[str]] = None) -> bool:
        """
        初始化模型注册中心
        
        Args:
            preload_models: 需要预加载的模型ID列表
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 加载配置
            if not self._load_config():
                return False
            
            # 创建注册中心
            global_config = self._config.get("global", {})
            self._base_path = global_config.get("models_base_path", "models")
            
            backend_str = global_config.get("default_backend", "onnx_cpu")
            backend = InferenceBackend(backend_str)
            
            self._registry = ModelRegistry(
                backend=backend,
                device_id=global_config.get("cuda_device_id", 0),
                enable_fp16=global_config.get("enable_fp16", False),
                base_path=self._base_path
            )
            
            # 注册所有模型
            self._register_all_models()
            
            # 预加载模型
            if preload_models:
                for model_id in preload_models:
                    self._registry.load(model_id)
            
            # 预热
            warmup_iterations = global_config.get("warmup_iterations", 0)
            if warmup_iterations > 0 and preload_models:
                self._warmup(preload_models, warmup_iterations)
            
            self._initialized = True
            logger.info("模型注册中心初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"模型注册中心初始化失败: {e}")
            return False
    
    def _load_config(self) -> bool:
        """加载配置文件"""
        try:
            if not os.path.exists(self._config_path):
                logger.error(f"配置文件不存在: {self._config_path}")
                return False
            
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            
            logger.info(f"配置文件加载成功: {self._config_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return False
    
    def _register_all_models(self):
        """注册所有模型"""
        # 遍历各插件配置
        plugin_sections = [
            "transformer_inspection",
            "switch_inspection", 
            "busbar_inspection",
            "capacitor_inspection",
            "meter_reading",
            "common"
        ]
        
        for section in plugin_sections:
            if section not in self._config:
                continue
            
            section_config = self._config[section]
            for model_name, model_config in section_config.items():
                if isinstance(model_config, dict) and "model_path" in model_config:
                    model_id = model_config.get("model_id", f"{section}_{model_name}")
                    self._registry.register_from_dict(model_id, model_config)
    
    def _warmup(self, model_ids: List[str], iterations: int):
        """模型预热"""
        logger.info(f"开始模型预热 ({iterations}次)...")
        
        # 创建虚拟输入
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for model_id in model_ids:
            if self._registry.is_loaded(model_id):
                for i in range(iterations):
                    self._registry.infer(model_id, dummy_image)
                logger.info(f"  {model_id}: 预热完成")
    
    def get_registry(self) -> Optional[ModelRegistry]:
        """获取模型注册中心实例"""
        return self._registry
    
    def infer(self, model_id: str, image: np.ndarray) -> InferenceResult:
        """执行推理"""
        if self._registry is None:
            return InferenceResult(
                success=False,
                model_id=model_id,
                outputs={},
                inference_time_ms=0,
                error_message="模型注册中心未初始化"
            )
        return self._registry.infer(model_id, image)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self._registry is None:
            return {"error": "模型注册中心未初始化"}
        return self._registry.get_stats()
    
    def check_models(self) -> Dict[str, Dict[str, Any]]:
        """
        检查模型文件状态
        
        Returns:
            Dict: 各模型的文件状态
        """
        result = {}
        
        if self._registry is None:
            return {"error": "模型注册中心未初始化"}
        
        for model_id in self._registry.list_models():
            config = self._registry.get_config(model_id)
            if config:
                model_path = config.model_path
                if not os.path.isabs(model_path):
                    model_path = os.path.join(self._base_path, model_path)
                
                exists = os.path.exists(model_path)
                size = os.path.getsize(model_path) if exists else 0
                
                result[model_id] = {
                    "path": model_path,
                    "exists": exists,
                    "size_mb": round(size / 1024 / 1024, 2) if exists else 0,
                    "loaded": self._registry.is_loaded(model_id),
                    "type": config.model_type,
                    "description": config.description
                }
        
        return result


# 全局访问函数
_manager: Optional[ModelRegistryManager] = None


def get_model_registry_manager(config_path: Optional[str] = None) -> ModelRegistryManager:
    """获取模型注册中心管理器单例"""
    global _manager
    if _manager is None:
        _manager = ModelRegistryManager(config_path)
    return _manager


def get_model_registry() -> Optional[ModelRegistry]:
    """获取模型注册中心"""
    manager = get_model_registry_manager()
    return manager.get_registry()


def initialize_models(config_path: str = "configs/models_config.yaml",
                     preload: Optional[List[str]] = None) -> bool:
    """
    初始化模型系统
    
    Args:
        config_path: 配置文件路径
        preload: 需要预加载的模型列表
        
    Returns:
        bool: 是否成功
    """
    manager = get_model_registry_manager(config_path)
    return manager.initialize(preload_models=preload)


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 初始化
    manager = get_model_registry_manager("configs/models_config.yaml")
    
    if manager.initialize():
        # 检查模型状态
        print("\n模型状态检查:")
        status = manager.check_models()
        for model_id, info in status.items():
            status_icon = "✓" if info["exists"] else "✗"
            print(f"  [{status_icon}] {model_id}: {info['path']}")
            if info["exists"]:
                print(f"      大小: {info['size_mb']} MB")
    else:
        print("初始化失败")
