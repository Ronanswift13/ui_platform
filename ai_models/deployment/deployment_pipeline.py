#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统集成模块
ONNX/TensorRT转换、推理优化、边缘部署

功能:
1. PyTorch到ONNX转换
2. ONNX优化与量化
3. TensorRT引擎构建
4. 边缘设备部署 (Jetson/RK3588)
5. 推理性能基准测试

作者: AI巡检系统
版本: 1.0.0
"""

import os
import sys
import time
import json
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 配置类
# =============================================================================
@dataclass
class DeploymentConfig:
    """部署配置"""
    # 模型配置
    model_name: str = "model"
    input_shapes: Dict[str, List[int]] = field(default_factory=lambda: {"input": [1, 3, 640, 640]})
    output_names: List[str] = field(default_factory=lambda: ["output"])
    
    # 优化配置
    fp16: bool = True
    int8: bool = False
    dynamic_batch: bool = True
    max_batch_size: int = 8
    
    # TensorRT配置
    workspace_size: int = 1 << 30  # 1GB
    max_workspace_size: int = 1 << 32  # 4GB
    
    # 边缘设备配置
    target_device: str = "gpu"  # gpu, jetson, rk3588, cpu
    
    # 路径配置
    output_dir: str = "deployed_models"


# =============================================================================
# ONNX工具
# =============================================================================
class ONNXConverter:
    """PyTorch模型到ONNX转换器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
    
    def convert_pytorch_to_onnx(self,
                                 model,
                                 dummy_inputs: Dict[str, Any],
                                 output_path: str,
                                 opset_version: int = 13,
                                 dynamic_axes: Optional[Dict] = None) -> bool:
        """
        PyTorch模型转ONNX
        
        Args:
            model: PyTorch模型
            dummy_inputs: 虚拟输入字典
            output_path: 输出路径
            opset_version: ONNX opset版本
            dynamic_axes: 动态轴配置
        """
        try:
            import torch
            
            model.eval()
            
            # 准备输入
            if isinstance(dummy_inputs, dict):
                if len(dummy_inputs) == 1:
                    inputs = list(dummy_inputs.values())[0]
                    input_names = list(dummy_inputs.keys())
                else:
                    inputs = tuple(dummy_inputs.values())
                    input_names = list(dummy_inputs.keys())
            else:
                inputs = dummy_inputs
                input_names = ["input"]
            
            # 默认动态轴
            if dynamic_axes is None and self.config.dynamic_batch:
                dynamic_axes = {}
                for name in input_names:
                    dynamic_axes[name] = {0: "batch_size"}
                for name in self.config.output_names:
                    dynamic_axes[name] = {0: "batch_size"}
            
            # 导出
            torch.onnx.export(
                model,
                inputs,
                output_path,
                input_names=input_names,
                output_names=self.config.output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True
            )
            
            logger.info(f"ONNX模型导出成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            return False
    
    def optimize_onnx(self, 
                      input_path: str, 
                      output_path: str,
                      optimization_level: str = "all") -> bool:
        """
        优化ONNX模型
        
        Args:
            input_path: 输入ONNX路径
            output_path: 输出ONNX路径
            optimization_level: 优化级别 (basic, extended, all)
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            # 加载模型
            model = onnx.load(input_path)
            
            # 基本优化
            from onnx import optimizer as onnx_optimizer
            passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "eliminate_unused_initializer",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_bn_into_conv",
                "fuse_add_bias_into_conv",
            ]
            
            optimized_model = onnx_optimizer.optimize(model, passes)
            onnx.save(optimized_model, output_path)
            
            logger.info(f"ONNX优化完成: {output_path}")
            return True
            
        except ImportError:
            logger.warning("onnx优化器未安装,尝试使用onnxruntime优化")
            return self._optimize_with_ort(input_path, output_path)
        except Exception as e:
            logger.error(f"ONNX优化失败: {e}")
            return False
    
    def _optimize_with_ort(self, input_path: str, output_path: str) -> bool:
        """使用ONNXRuntime优化"""
        try:
            import onnxruntime as ort
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.optimized_model_filepath = output_path
            
            # 创建session会触发优化
            ort.InferenceSession(input_path, sess_options)
            
            logger.info(f"ONNXRuntime优化完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNXRuntime优化失败: {e}")
            return False
    
    def quantize_onnx(self,
                      input_path: str,
                      output_path: str,
                      quantization_type: str = "dynamic",
                      calibration_data: Optional[List[Dict]] = None) -> bool:
        """
        ONNX模型量化
        
        Args:
            input_path: 输入ONNX路径
            output_path: 输出ONNX路径
            quantization_type: 量化类型 (dynamic, static)
            calibration_data: 校准数据 (静态量化需要)
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, quantize_static
            from onnxruntime.quantization import QuantType
            
            if quantization_type == "dynamic":
                quantize_dynamic(
                    input_path,
                    output_path,
                    weight_type=QuantType.QUInt8
                )
            elif quantization_type == "static":
                if calibration_data is None:
                    logger.error("静态量化需要校准数据")
                    return False
                
                # 创建校准数据读取器
                from onnxruntime.quantization import CalibrationDataReader
                
                class DataReader(CalibrationDataReader):
                    def __init__(self, data):
                        self.data = data
                        self.idx = 0
                    
                    def get_next(self):
                        if self.idx >= len(self.data):
                            return None
                        result = self.data[self.idx]
                        self.idx += 1
                        return result
                
                reader = DataReader(calibration_data)
                quantize_static(
                    input_path,
                    output_path,
                    reader,
                    quant_format=QuantType.QInt8
                )
            
            logger.info(f"ONNX量化完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX量化失败: {e}")
            return False
    
    def verify_onnx(self, onnx_path: str) -> bool:
        """验证ONNX模型"""
        try:
            import onnx
            
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            logger.info(f"ONNX模型验证通过: {onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX验证失败: {e}")
            return False


# =============================================================================
# TensorRT工具
# =============================================================================
class TensorRTConverter:
    """TensorRT引擎构建器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.trt_available = self._check_tensorrt()
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _check_tensorrt(self) -> bool:
        """检查TensorRT是否可用"""
        try:
            import tensorrt as trt
            logger.info(f"TensorRT版本: {trt.__version__}")
            return True
        except ImportError:
            logger.warning("TensorRT未安装")
            return False
    
    def build_engine(self,
                     onnx_path: str,
                     engine_path: str,
                     fp16: bool = True,
                     int8: bool = False,
                     calibrator = None) -> bool:
        """
        构建TensorRT引擎
        
        Args:
            onnx_path: ONNX模型路径
            engine_path: 引擎输出路径
            fp16: 启用FP16
            int8: 启用INT8
            calibrator: INT8校准器
        """
        if not self.trt_available:
            logger.error("TensorRT不可用")
            return False
        
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # 创建builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # 解析ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX解析错误: {parser.get_error(error)}")
                    return False
            
            # 配置builder
            config = builder.create_builder_config()
            config.max_workspace_size = self.config.workspace_size
            
            # FP16
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用FP16")
            
            # INT8
            if int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                if calibrator:
                    config.int8_calibrator = calibrator
                logger.info("启用INT8")
            
            # 动态批次
            if self.config.dynamic_batch:
                profile = builder.create_optimization_profile()
                for i in range(network.num_inputs):
                    input_tensor = network.get_input(i)
                    shape = input_tensor.shape
                    
                    min_shape = [1] + list(shape[1:])
                    opt_shape = [self.config.max_batch_size // 2] + list(shape[1:])
                    max_shape = [self.config.max_batch_size] + list(shape[1:])
                    
                    profile.set_shape(
                        input_tensor.name,
                        min_shape, opt_shape, max_shape
                    )
                
                config.add_optimization_profile(profile)
            
            # 构建引擎
            logger.info("开始构建TensorRT引擎...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("TensorRT引擎构建失败")
                return False
            
            # 保存引擎
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT引擎构建成功: {engine_path}")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT引擎构建失败: {e}")
            return False
    
    def create_int8_calibrator(self,
                                calibration_data: List[np.ndarray],
                                cache_file: str = "calibration.cache"):
        """创建INT8校准器"""
        if not self.trt_available:
            return None
        
        try:
            import tensorrt as trt
            
            class Calibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, data, cache_file):
                    trt.IInt8EntropyCalibrator2.__init__(self)
                    self.data = data
                    self.cache_file = cache_file
                    self.current_index = 0
                    
                    # 分配GPU内存
                    import pycuda.driver as cuda
                    import pycuda.autoinit
                    
                    self.device_input = cuda.mem_alloc(
                        data[0].nbytes
                    )
                
                def get_batch_size(self):
                    return 1
                
                def get_batch(self, names):
                    if self.current_index >= len(self.data):
                        return None
                    
                    import pycuda.driver as cuda
                    
                    batch = self.data[self.current_index]
                    cuda.memcpy_htod(self.device_input, batch)
                    self.current_index += 1
                    
                    return [int(self.device_input)]
                
                def read_calibration_cache(self):
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, 'rb') as f:
                            return f.read()
                    return None
                
                def write_calibration_cache(self, cache):
                    with open(self.cache_file, 'wb') as f:
                        f.write(cache)
            
            return Calibrator(calibration_data, cache_file)
            
        except Exception as e:
            logger.error(f"创建INT8校准器失败: {e}")
            return None


# =============================================================================
# 推理引擎包装器
# =============================================================================
class InferenceEngine(ABC):
    """推理引擎抽象基类"""
    
    @abstractmethod
    def load(self, model_path: str) -> bool:
        pass
    
    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def get_latency(self) -> float:
        pass


class ONNXRuntimeEngine(InferenceEngine):
    """ONNXRuntime推理引擎"""
    
    def __init__(self, providers: Optional[List[str]] = None):
        self.session = None
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.input_names = []
        self.output_names = []
        self._latency = 0.0
    
    def load(self, model_path: str) -> bool:
        try:
            import onnxruntime as ort
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                model_path, 
                sess_options,
                providers=self.providers
            )
            
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            logger.info(f"ONNXRuntime引擎加载成功: {model_path}")
            logger.info(f"使用Provider: {self.session.get_providers()}")
            return True
            
        except Exception as e:
            logger.error(f"ONNXRuntime加载失败: {e}")
            return False
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.session is None:
            raise RuntimeError("模型未加载")
        
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, inputs)
        self._latency = (time.perf_counter() - start) * 1000
        
        return {name: out for name, out in zip(self.output_names, outputs)}
    
    def get_latency(self) -> float:
        return self._latency


class TensorRTEngine(InferenceEngine):
    """TensorRT推理引擎"""
    
    def __init__(self):
        self.engine = None
        self.context = None
        self.bindings = []
        self.input_names = []
        self.output_names = []
        self._latency = 0.0
        
        self.trt_available = self._check_tensorrt()
    
    def _check_tensorrt(self) -> bool:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            return True
        except ImportError:
            return False
    
    def load(self, model_path: str) -> bool:
        if not self.trt_available:
            logger.error("TensorRT不可用")
            return False
        
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # 加载引擎
            with open(model_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # 获取绑定信息
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                
                if self.engine.binding_is_input(i):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
            
            logger.info(f"TensorRT引擎加载成功: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT加载失败: {e}")
            return False
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.trt_available or self.engine is None:
            raise RuntimeError("TensorRT引擎未加载")
        
        import tensorrt as trt
        import pycuda.driver as cuda
        
        start = time.perf_counter()
        
        # 分配内存
        bindings = []
        outputs = {}
        
        for name in self.input_names:
            data = inputs[name]
            device_mem = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(device_mem, data)
            bindings.append(int(device_mem))
        
        for i, name in enumerate(self.output_names):
            binding_idx = self.engine.get_binding_index(name)
            shape = self.engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            output = np.empty(shape, dtype=dtype)
            device_mem = cuda.mem_alloc(output.nbytes)
            bindings.append(int(device_mem))
            outputs[name] = (output, device_mem)
        
        # 执行推理
        self.context.execute_v2(bindings)
        
        # 复制输出
        result = {}
        for name, (output, device_mem) in outputs.items():
            cuda.memcpy_dtoh(output, device_mem)
            result[name] = output
        
        self._latency = (time.perf_counter() - start) * 1000
        
        return result
    
    def get_latency(self) -> float:
        return self._latency


# =============================================================================
# 边缘部署工具
# =============================================================================
class EdgeDeployer:
    """边缘设备部署器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.device_info = self._detect_device()
    
    def _detect_device(self) -> Dict[str, Any]:
        """检测设备信息"""
        info = {
            "platform": sys.platform,
            "cuda_available": False,
            "tensorrt_available": False,
            "device_type": "cpu"
        }
        
        # 检查CUDA
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["device_type"] = "gpu"
        except ImportError:
            pass
        
        # 检查TensorRT
        try:
            import tensorrt as trt
            info["tensorrt_available"] = True
            info["tensorrt_version"] = trt.__version__
        except ImportError:
            pass
        
        # 检测Jetson
        if os.path.exists("/etc/nv_tegra_release"):
            info["device_type"] = "jetson"
            with open("/etc/nv_tegra_release") as f:
                info["jetson_version"] = f.read().strip()
        
        # 检测RK3588
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model") as f:
                model = f.read()
                if "rk3588" in model.lower():
                    info["device_type"] = "rk3588"
        
        logger.info(f"检测到设备: {info['device_type']}")
        return info
    
    def prepare_for_device(self, 
                           onnx_path: str,
                           output_dir: str) -> Dict[str, str]:
        """
        为目标设备准备模型
        
        Returns:
            dict: 各格式模型路径
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {"onnx": onnx_path}
        
        device_type = self.config.target_device
        
        if device_type in ["gpu", "jetson"]:
            # 构建TensorRT引擎
            trt_converter = TensorRTConverter(self.config)
            engine_path = os.path.join(output_dir, f"{self.config.model_name}.engine")
            
            if trt_converter.build_engine(
                onnx_path, engine_path,
                fp16=self.config.fp16,
                int8=self.config.int8
            ):
                results["tensorrt"] = engine_path
        
        elif device_type == "rk3588":
            # RK3588使用RKNN
            rknn_path = self._convert_to_rknn(onnx_path, output_dir)
            if rknn_path:
                results["rknn"] = rknn_path
        
        # 始终保留优化后的ONNX
        onnx_converter = ONNXConverter(self.config)
        optimized_path = os.path.join(output_dir, f"{self.config.model_name}_optimized.onnx")
        onnx_converter.optimize_onnx(onnx_path, optimized_path)
        results["onnx_optimized"] = optimized_path
        
        # 量化版本
        if self.config.int8:
            quantized_path = os.path.join(output_dir, f"{self.config.model_name}_int8.onnx")
            onnx_converter.quantize_onnx(onnx_path, quantized_path)
            results["onnx_int8"] = quantized_path
        
        return results
    
    def _convert_to_rknn(self, onnx_path: str, output_dir: str) -> Optional[str]:
        """转换到RKNN格式 (RK3588)"""
        try:
            from rknn.api import RKNN
            
            rknn = RKNN()
            
            # 配置
            rknn.config(
                target_platform='rk3588',
                optimization_level=3
            )
            
            # 加载ONNX
            ret = rknn.load_onnx(model=onnx_path)
            if ret != 0:
                logger.error("RKNN加载ONNX失败")
                return None
            
            # 构建
            ret = rknn.build(do_quantization=self.config.int8)
            if ret != 0:
                logger.error("RKNN构建失败")
                return None
            
            # 导出
            rknn_path = os.path.join(output_dir, f"{self.config.model_name}.rknn")
            ret = rknn.export_rknn(rknn_path)
            
            if ret == 0:
                logger.info(f"RKNN模型导出成功: {rknn_path}")
                return rknn_path
            else:
                return None
                
        except ImportError:
            logger.warning("RKNN工具包未安装")
            return None
        except Exception as e:
            logger.error(f"RKNN转换失败: {e}")
            return None
    
    def create_deployment_package(self,
                                   model_paths: Dict[str, str],
                                   output_path: str,
                                   include_runtime: bool = False) -> bool:
        """
        创建部署包
        
        Args:
            model_paths: 模型路径字典
            output_path: 输出包路径
            include_runtime: 是否包含运行时代码
        """
        import zipfile
        
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 添加模型文件
                for name, path in model_paths.items():
                    if os.path.exists(path):
                        zf.write(path, f"models/{os.path.basename(path)}")
                
                # 添加配置
                config_data = {
                    "model_name": self.config.model_name,
                    "input_shapes": self.config.input_shapes,
                    "output_names": self.config.output_names,
                    "target_device": self.config.target_device,
                    "fp16": self.config.fp16,
                    "int8": self.config.int8,
                    "device_info": self.device_info
                }
                
                zf.writestr("config.json", json.dumps(config_data, indent=2))
                
                # 添加运行时代码
                if include_runtime:
                    runtime_code = self._generate_runtime_code()
                    zf.writestr("inference.py", runtime_code)
                    zf.writestr("requirements.txt", self._generate_requirements())
            
            logger.info(f"部署包创建成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建部署包失败: {e}")
            return False
    
    def _generate_runtime_code(self) -> str:
        """生成运行时推理代码"""
        return '''#!/usr/bin/env python3
"""自动生成的推理代码"""

import os
import json
import numpy as np

class InferenceRuntime:
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.engine = None
        self._load_engine()
    
    def _load_engine(self):
        device = self.config["target_device"]
        
        if device in ["gpu", "jetson"]:
            # 尝试TensorRT
            trt_path = f"models/{self.config['model_name']}.engine"
            if os.path.exists(trt_path):
                self._load_tensorrt(trt_path)
                return
        
        # 回退到ONNX
        onnx_path = f"models/{self.config['model_name']}_optimized.onnx"
        self._load_onnx(onnx_path)
    
    def _load_tensorrt(self, path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        with open(path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.backend = "tensorrt"
    
    def _load_onnx(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path)
        self.backend = "onnx"
    
    def infer(self, inputs):
        if self.backend == "tensorrt":
            return self._infer_trt(inputs)
        else:
            return self._infer_onnx(inputs)
    
    def _infer_onnx(self, inputs):
        outputs = self.session.run(None, inputs)
        return {name: out for name, out in zip(
            [o.name for o in self.session.get_outputs()], outputs
        )}
    
    def _infer_trt(self, inputs):
        # TensorRT推理实现
        pass

if __name__ == "__main__":
    runtime = InferenceRuntime()
    # 测试推理
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    result = runtime.infer({"input": dummy_input})
    print("推理完成:", result.keys())
'''
    
    def _generate_requirements(self) -> str:
        """生成依赖文件"""
        return '''# 推理依赖
numpy>=1.19.0
onnxruntime-gpu>=1.10.0  # GPU版本
# onnxruntime>=1.10.0  # CPU版本
'''


# =============================================================================
# 性能基准测试
# =============================================================================
class Benchmark:
    """性能基准测试"""
    
    def __init__(self, warmup_runs: int = 10, test_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
    
    def run_benchmark(self,
                      engine: InferenceEngine,
                      inputs: Dict[str, np.ndarray],
                      batch_sizes: List[int] = [1, 4, 8]) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            engine: 推理引擎
            inputs: 输入数据 (batch_size=1)
            batch_sizes: 测试的批次大小
        
        Returns:
            基准测试结果
        """
        results = {
            "engine": engine.__class__.__name__,
            "batch_results": {}
        }
        
        for batch_size in batch_sizes:
            # 扩展批次
            batched_inputs = {}
            for name, data in inputs.items():
                batched_inputs[name] = np.repeat(data, batch_size, axis=0)
            
            # 预热
            for _ in range(self.warmup_runs):
                engine.infer(batched_inputs)
            
            # 测试
            latencies = []
            for _ in range(self.test_runs):
                engine.infer(batched_inputs)
                latencies.append(engine.get_latency())
            
            latencies = np.array(latencies)
            
            results["batch_results"][batch_size] = {
                "mean_latency_ms": float(np.mean(latencies)),
                "std_latency_ms": float(np.std(latencies)),
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "throughput_fps": float(batch_size * 1000 / np.mean(latencies))
            }
        
        return results
    
    def compare_engines(self,
                        engines: Dict[str, InferenceEngine],
                        inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        比较多个推理引擎
        
        Args:
            engines: 引擎字典 {name: engine}
            inputs: 输入数据
        """
        comparison = {}
        
        for name, engine in engines.items():
            logger.info(f"测试引擎: {name}")
            comparison[name] = self.run_benchmark(engine, inputs)
        
        return comparison
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成基准测试报告"""
        lines = ["=" * 60, "推理性能基准测试报告", "=" * 60, ""]
        
        for engine_name, engine_results in results.items():
            lines.append(f"\n引擎: {engine_name}")
            lines.append("-" * 40)
            
            for batch_size, metrics in engine_results.get("batch_results", {}).items():
                lines.append(f"\n批次大小: {batch_size}")
                lines.append(f"  平均延迟: {metrics['mean_latency_ms']:.2f} ms")
                lines.append(f"  标准差:   {metrics['std_latency_ms']:.2f} ms")
                lines.append(f"  P50延迟:  {metrics['p50_latency_ms']:.2f} ms")
                lines.append(f"  P95延迟:  {metrics['p95_latency_ms']:.2f} ms")
                lines.append(f"  P99延迟:  {metrics['p99_latency_ms']:.2f} ms")
                lines.append(f"  吞吐量:   {metrics['throughput_fps']:.1f} FPS")
        
        return "\n".join(lines)


# =============================================================================
# 完整部署流水线
# =============================================================================
class DeploymentPipeline:
    """完整部署流水线"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.onnx_converter = ONNXConverter(config)
        self.trt_converter = TensorRTConverter(config)
        self.edge_deployer = EdgeDeployer(config)
        self.benchmark = Benchmark()
    
    def run_pipeline(self,
                     model,
                     dummy_inputs: Dict[str, Any],
                     calibration_data: Optional[List] = None) -> Dict[str, Any]:
        """
        运行完整部署流水线
        
        Args:
            model: PyTorch模型
            dummy_inputs: 虚拟输入
            calibration_data: 校准数据 (INT8量化)
        
        Returns:
            部署结果
        """
        results = {
            "status": "success",
            "model_paths": {},
            "benchmark": None
        }
        
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: 转换到ONNX
        logger.info("Step 1: 转换到ONNX")
        onnx_path = os.path.join(output_dir, f"{self.config.model_name}.onnx")
        
        if not self.onnx_converter.convert_pytorch_to_onnx(
            model, dummy_inputs, onnx_path
        ):
            results["status"] = "failed"
            results["error"] = "ONNX转换失败"
            return results
        
        results["model_paths"]["onnx"] = onnx_path
        
        # Step 2: 优化ONNX
        logger.info("Step 2: 优化ONNX")
        optimized_path = os.path.join(output_dir, f"{self.config.model_name}_optimized.onnx")
        self.onnx_converter.optimize_onnx(onnx_path, optimized_path)
        results["model_paths"]["onnx_optimized"] = optimized_path
        
        # Step 3: 量化 (可选)
        if self.config.int8:
            logger.info("Step 3: INT8量化")
            quantized_path = os.path.join(output_dir, f"{self.config.model_name}_int8.onnx")
            self.onnx_converter.quantize_onnx(
                optimized_path, quantized_path,
                quantization_type="dynamic" if calibration_data is None else "static",
                calibration_data=calibration_data
            )
            results["model_paths"]["onnx_int8"] = quantized_path
        
        # Step 4: 构建TensorRT引擎 (如果可用)
        if self.trt_converter.trt_available:
            logger.info("Step 4: 构建TensorRT引擎")
            engine_path = os.path.join(output_dir, f"{self.config.model_name}.engine")
            
            if self.trt_converter.build_engine(
                optimized_path, engine_path,
                fp16=self.config.fp16,
                int8=self.config.int8
            ):
                results["model_paths"]["tensorrt"] = engine_path
        
        # Step 5: 基准测试
        logger.info("Step 5: 性能基准测试")
        engines = {}
        
        # ONNX引擎
        onnx_engine = ONNXRuntimeEngine()
        if onnx_engine.load(optimized_path):
            engines["onnx"] = onnx_engine
        
        # TensorRT引擎
        if "tensorrt" in results["model_paths"]:
            trt_engine = TensorRTEngine()
            if trt_engine.load(results["model_paths"]["tensorrt"]):
                engines["tensorrt"] = trt_engine
        
        if engines:
            # 准备测试输入
            test_inputs = {}
            for name, data in dummy_inputs.items():
                if hasattr(data, 'numpy'):
                    test_inputs[name] = data.numpy()
                else:
                    test_inputs[name] = np.array(data)
            
            benchmark_results = self.benchmark.compare_engines(engines, test_inputs)
            results["benchmark"] = benchmark_results
            
            # 生成报告
            report = self.benchmark.generate_report(benchmark_results)
            report_path = os.path.join(output_dir, "benchmark_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            results["benchmark_report"] = report_path
        
        # Step 6: 创建部署包
        logger.info("Step 6: 创建部署包")
        package_path = os.path.join(output_dir, f"{self.config.model_name}_deploy.zip")
        self.edge_deployer.create_deployment_package(
            results["model_paths"],
            package_path,
            include_runtime=True
        )
        results["deployment_package"] = package_path
        
        logger.info("部署流水线完成!")
        return results


# =============================================================================
# 占位符: 硬件接口
# =============================================================================
class HardwareInterface:
    """
    硬件接口占位符
    
    用于未来集成实际硬件设备
    """
    
    def __init__(self):
        self._connected = False
        self._device_info = {}
    
    def connect(self, device_config: Dict[str, Any]) -> bool:
        """
        连接硬件设备
        
        TODO: 实现实际硬件连接
        - LiDAR传感器连接
        - 音频采集设备连接
        - 气体传感器连接
        - 相机设备连接
        """
        logger.info("硬件接口占位符: connect()")
        self._connected = True
        self._device_info = device_config
        return True
    
    def disconnect(self) -> bool:
        """断开硬件连接"""
        logger.info("硬件接口占位符: disconnect()")
        self._connected = False
        return True
    
    def read_lidar_frame(self) -> Optional[np.ndarray]:
        """
        读取LiDAR帧
        
        TODO: 实现实际LiDAR数据读取
        - Velodyne
        - Ouster
        - Livox
        """
        logger.debug("硬件接口占位符: read_lidar_frame()")
        if not self._connected:
            return None
        return np.random.randn(16384, 4).astype(np.float32)
    
    def read_audio_buffer(self, duration: float = 1.0) -> Optional[np.ndarray]:
        """
        读取音频缓冲区
        
        TODO: 实现实际音频采集
        """
        logger.debug("硬件接口占位符: read_audio_buffer()")
        if not self._connected:
            return None
        sample_rate = 16000
        return np.random.randn(int(sample_rate * duration)).astype(np.float32)
    
    def read_gas_sensors(self) -> Optional[Dict[str, float]]:
        """
        读取气体传感器
        
        TODO: 实现实际传感器读取
        - SF6传感器
        - H2传感器
        - CO传感器
        - C2H2传感器
        """
        logger.debug("硬件接口占位符: read_gas_sensors()")
        if not self._connected:
            return None
        return {
            "SF6": np.random.uniform(800, 1200),
            "H2": np.random.uniform(50, 150),
            "CO": np.random.uniform(100, 300),
            "C2H2": np.random.uniform(1, 5)
        }
    
    def read_camera_frame(self) -> Optional[np.ndarray]:
        """
        读取相机帧
        
        TODO: 实现实际相机采集
        """
        logger.debug("硬件接口占位符: read_camera_frame()")
        if not self._connected:
            return None
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# =============================================================================
# 入口函数
# =============================================================================
def deploy_model(model,
                 dummy_inputs: Dict[str, Any],
                 config: Optional[DeploymentConfig] = None) -> Dict[str, Any]:
    """
    部署模型的便捷函数
    
    Args:
        model: PyTorch模型
        dummy_inputs: 虚拟输入
        config: 部署配置
    
    Returns:
        部署结果
    """
    if config is None:
        config = DeploymentConfig()
    
    pipeline = DeploymentPipeline(config)
    return pipeline.run_pipeline(model, dummy_inputs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例: 部署一个简单模型
    try:
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).flatten(1)
                return self.fc(x)
        
        model = SimpleModel()
        dummy_input = {"input": torch.randn(1, 3, 224, 224)}
        
        config = DeploymentConfig(
            model_name="simple_model",
            input_shapes={"input": [1, 3, 224, 224]},
            output_names=["output"],
            fp16=True,
            output_dir="deployed_models"
        )
        
        results = deploy_model(model, dummy_input, config)
        print(f"部署结果: {results['status']}")
        print(f"模型路径: {results['model_paths']}")
        
    except ImportError:
        logger.warning("PyTorch未安装,跳过示例")
