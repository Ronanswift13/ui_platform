"""
深度学习推理引擎
输变电激光监测平台 - 全自动AI巡检增强

提供统一的模型推理接口，支持:
- ONNX Runtime推理
- TensorRT加速推理
- 模型动态加载/卸载
- 批量推理优化
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None


class InferenceBackend(Enum):
    """推理后端类型"""
    ONNX_CPU = "onnx_cpu"
    ONNX_CUDA = "onnx_cuda"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class ModelType(Enum):
    """模型类型"""
    DETECTION = "detection"          # 目标检测
    CLASSIFICATION = "classification" # 分类
    SEGMENTATION = "segmentation"    # 语义分割
    KEYPOINT = "keypoint"            # 关键点检测
    OCR = "ocr"                      # 文字识别


@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    model_path: str
    model_type: ModelType
    input_size: Tuple[int, int] = (640, 640)
    input_names: List[str] = field(default_factory=lambda: ["images"])
    output_names: List[str] = field(default_factory=lambda: ["output0"])
    class_names: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    backend: InferenceBackend = InferenceBackend.ONNX_CPU
    device_id: int = 0
    fp16: bool = False
    dynamic_batch: bool = False
    max_batch_size: int = 1


@dataclass
class InferenceResult:
    """推理结果"""
    model_id: str
    detections: List[Dict[str, Any]] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    raw_outputs: Optional[Dict[str, np.ndarray]] = None


class BaseInferenceEngine(ABC):
    """推理引擎基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._loaded = False
        self._lock = threading.Lock()
    
    @abstractmethod
    def load(self) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def unload(self) -> bool:
        """卸载模型"""
        pass
    
    @abstractmethod
    def infer(self, image: np.ndarray) -> InferenceResult:
        """执行推理"""
        pass
    
    @abstractmethod
    def infer_batch(self, images: List[np.ndarray]) -> List[InferenceResult]:
        """批量推理"""
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理"""
        h, w = self.config.input_size
        
        # Resize
        if image.shape[:2] != (h, w):
            try:
                import cv2
                image = cv2.resize(image, (w, h))
            except ImportError:
                pass
        
        # BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]
        
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


class ONNXInferenceEngine(BaseInferenceEngine):
    """ONNX Runtime推理引擎"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._session: Optional[Any] = None
    
    def load(self) -> bool:
        if ort is None:
            raise ImportError("onnxruntime未安装，请执行: pip install onnxruntime")
        
        with self._lock:
            if self._loaded:
                return True
            
            try:
                # 配置providers
                providers = []
                if self.config.backend == InferenceBackend.ONNX_CUDA:
                    providers.append(('CUDAExecutionProvider', {
                        'device_id': self.config.device_id,
                    }))
                providers.append('CPUExecutionProvider')
                
                # 配置session options
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # 创建session
                self._session = ort.InferenceSession(
                    self.config.model_path,
                    sess_options=sess_options,
                    providers=providers
                )
                
                self._loaded = True
                print(f"[ONNXEngine] 模型加载成功: {self.config.model_id}")
                return True
                
            except Exception as e:
                print(f"[ONNXEngine] 模型加载失败: {e}")
                return False
    
    def unload(self) -> bool:
        with self._lock:
            if self._session is not None:
                del self._session
                self._session = None
            self._loaded = False
            return True
    
    def infer(self, image: np.ndarray) -> InferenceResult:
        if not self._loaded:
            raise RuntimeError("模型未加载")
        
        start_time = time.perf_counter()
        
        # 预处理
        preprocess_start = time.perf_counter()
        input_tensor = self.preprocess(image)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # 推理
        infer_start = time.perf_counter()
        if self._session is None:
            raise RuntimeError("推理会话未初始化")
        outputs = self._session.run(
            self.config.output_names,
            {self.config.input_names[0]: input_tensor}
        )
        inference_time = (time.perf_counter() - infer_start) * 1000
        
        # 后处理
        postprocess_start = time.perf_counter()
        detections = self._postprocess(outputs, (image.shape[0], image.shape[1]))
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000
        
        return InferenceResult(
            model_id=self.config.model_id,
            detections=detections,
            inference_time_ms=inference_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time,
            raw_outputs={name: out for name, out in zip(self.config.output_names, outputs)}
        )
    
    def infer_batch(self, images: List[np.ndarray]) -> List[InferenceResult]:
        return [self.infer(img) for img in images]
    
    def _postprocess(self, outputs: List[np.ndarray], orig_shape: Tuple[int, int]) -> List[Dict]:
        """后处理 - YOLO格式"""
        if self.config.model_type == ModelType.DETECTION:
            return self._postprocess_detection(outputs, orig_shape)
        elif self.config.model_type == ModelType.KEYPOINT:
            return self._postprocess_keypoint(outputs, orig_shape)
        elif self.config.model_type == ModelType.SEGMENTATION:
            return self._postprocess_segmentation(outputs, orig_shape)
        else:
            return []
    
    def _postprocess_detection(self, outputs: List[np.ndarray], orig_shape: Tuple[int, int]) -> List[Dict]:
        """YOLO检测后处理"""
        detections = []
        output = outputs[0]
        
        # YOLOv8输出格式: [batch, 84, 8400] -> 转置为 [batch, 8400, 84]
        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)
        
        if len(output.shape) == 3:
            output = output[0]  # 取第一个batch
        
        # 解析检测结果
        orig_h, orig_w = orig_shape
        input_h, input_w = self.config.input_size
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        for row in output:
            if len(row) < 5:
                continue
            
            # YOLO格式: cx, cy, w, h, conf, class_scores...
            cx, cy, w, h = row[:4]
            scores = row[4:]
            
            if len(scores) == 1:
                conf = scores[0]
                class_id = 0
            else:
                class_id = int(np.argmax(scores))
                conf = scores[class_id]
            
            if conf < self.config.confidence_threshold:
                continue
            
            # 转换为xyxy格式
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y
            
            class_name = self.config.class_names[class_id] if class_id < len(self.config.class_names) else f"class_{class_id}"
            
            detections.append({
                "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "confidence": float(conf),
                "class_id": class_id,
                "class_name": class_name,
            })
        
        # NMS
        detections = self._nms(detections)
        
        return detections
    
    def _postprocess_keypoint(self, outputs: List[np.ndarray], orig_shape: Tuple[int, int]) -> List[Dict]:
        """关键点检测后处理"""
        return []
    
    def _postprocess_segmentation(self, outputs: List[np.ndarray], orig_shape: Tuple[int, int]) -> List[Dict]:
        """语义分割后处理"""
        return []
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """非极大值抑制"""
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self._iou(best["bbox"], d["bbox"]) < self.config.nms_threshold
            ]
        
        return keep
    
    def _iou(self, box1: Dict, box2: Dict) -> float:
        """计算IoU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


class ModelRegistry:
    """模型注册表 - 单例模式"""
    
    _instance: Optional["ModelRegistry"] = None
    
    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._engines: Dict[str, BaseInferenceEngine] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._lock = threading.Lock()
        self._initialized = True
    
    def register(self, config: ModelConfig) -> bool:
        """注册模型配置"""
        with self._lock:
            self._configs[config.model_id] = config
            return True
    
    def load(self, model_id: str) -> bool:
        """加载模型"""
        with self._lock:
            if model_id not in self._configs:
                raise ValueError(f"模型未注册: {model_id}")
            
            if model_id in self._engines:
                return True
            
            config = self._configs[model_id]
            
            # 根据后端创建引擎
            if config.backend in [InferenceBackend.ONNX_CPU, InferenceBackend.ONNX_CUDA]:
                engine = ONNXInferenceEngine(config)
            else:
                raise ValueError(f"不支持的后端: {config.backend}")
            
            if engine.load():
                self._engines[model_id] = engine
                return True
            return False
    
    def unload(self, model_id: str) -> bool:
        """卸载模型"""
        with self._lock:
            if model_id in self._engines:
                self._engines[model_id].unload()
                del self._engines[model_id]
            return True
    
    def get_engine(self, model_id: str) -> Optional[BaseInferenceEngine]:
        """获取推理引擎"""
        return self._engines.get(model_id)
    
    def infer(self, model_id: str, image: np.ndarray) -> InferenceResult:
        """执行推理"""
        engine = self.get_engine(model_id)
        if engine is None:
            # 尝试自动加载
            if not self.load(model_id):
                raise RuntimeError(f"模型加载失败: {model_id}")
            engine = self.get_engine(model_id)

        if engine is None:
            raise RuntimeError(f"无法获取推理引擎: {model_id}")
        return engine.infer(image)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        return [
            {
                "model_id": model_id,
                "model_type": config.model_type.value,
                "backend": config.backend.value,
                "loaded": model_id in self._engines,
            }
            for model_id, config in self._configs.items()
        ]


# 便捷函数
_registry_instance: Optional[ModelRegistry] = None

def get_model_registry() -> ModelRegistry:
    """获取模型注册表实例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


def register_model(config: ModelConfig) -> bool:
    """注册模型"""
    return get_model_registry().register(config)


def infer(model_id: str, image: np.ndarray) -> InferenceResult:
    """执行推理"""
    return get_model_registry().infer(model_id, image)
