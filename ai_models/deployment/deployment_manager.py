#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统集成部署模块
ONNX/TensorRT转换、推理优化、边缘部署

功能:
1. PyTorch -> ONNX转换
2. ONNX优化 (图优化、算子融合)
3. TensorRT转换与优化
4. OpenVINO转换
5. 量化支持 (INT8/FP16)
6. 边缘设备部署 (Jetson/RK3588)
7. 推理性能基准测试

作者: AI巡检系统
版本: 1.0.0
"""

import os
import sys
import time
import logging
import json
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


# =============================================================================
# 配置类
# =============================================================================
@dataclass
class DeploymentConfig:
    """部署配置"""
    # 模型路径
    model_path: str = ""
    output_dir: str = "deployed_models"
    
    # 目标格式
    target_formats: List[str] = field(default_factory=lambda: ["onnx", "tensorrt"])
    
    # 量化配置
    quantization: str = "fp32"  # fp32, fp16, int8
    calibration_data_path: Optional[str] = None
    
    # TensorRT配置
    trt_max_batch_size: int = 1
    trt_max_workspace_size: int = 1 << 30  # 1GB
    trt_precision: str = "fp16"  # fp32, fp16, int8
    
    # 优化配置
    optimize_onnx: bool = True
    fuse_operations: bool = True
    
    # 边缘设备配置
    target_device: str = "gpu"  # gpu, cpu, jetson, rk3588
    
    # 性能测试配置
    benchmark_iterations: int = 100
    warmup_iterations: int = 10


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_name: str
    model_version: str
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    input_dtypes: Dict[str, str]
    output_dtypes: Dict[str, str]
    description: str = ""
    training_config: Dict = field(default_factory=dict)
    performance_metrics: Dict = field(default_factory=dict)


# =============================================================================
# ONNX转换与优化
# =============================================================================
class ONNXConverter:
    """ONNX模型转换器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def convert_pytorch_to_onnx(self,
                                 model: 'nn.Module',
                                 dummy_inputs: Dict[str, 'torch.Tensor'],
                                 output_path: str,
                                 input_names: Optional[List[str]] = None,
                                 output_names: Optional[List[str]] = None,
                                 dynamic_axes: Optional[Dict] = None,
                                 opset_version: int = 13) -> bool:
        """
        PyTorch模型转ONNX
        
        Args:
            model: PyTorch模型
            dummy_inputs: 虚拟输入字典
            output_path: 输出路径
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_axes: 动态轴配置
            opset_version: ONNX opset版本
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch未安装")
            return False
        
        try:
            model.eval()
            
            # 准备输入
            if len(dummy_inputs) == 1:
                args = list(dummy_inputs.values())[0]
            else:
                args = tuple(dummy_inputs.values())
            
            # 默认输入输出名称
            if input_names is None:
                input_names = list(dummy_inputs.keys())
            
            if output_names is None:
                output_names = ["output"]
            
            # 默认动态轴
            if dynamic_axes is None:
                dynamic_axes = {name: {0: "batch"} for name in input_names}
            
            # 导出
            torch.onnx.export(
                model,
                args,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True
            )
            
            logger.info(f"ONNX模型导出成功: {output_path}")
            
            # 验证
            if ONNX_AVAILABLE:
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX模型验证通过")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            return False
    
    def optimize_onnx(self, input_path: str, output_path: str) -> bool:
        """
        优化ONNX模型
        
        优化包括:
        - 常量折叠
        - 算子融合
        - 冗余节点消除
        - 形状推断
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX未安装")
            return False
        
        try:
            # 使用onnxruntime的图优化
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.optimized_model_filepath = output_path
            
            # 创建session触发优化
            _ = ort.InferenceSession(input_path, sess_options)
            
            logger.info(f"ONNX优化完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX优化失败: {e}")
            return False
    
    def quantize_onnx(self, 
                      input_path: str, 
                      output_path: str,
                      quantization_type: str = "dynamic",
                      calibration_data: Optional[np.ndarray] = None) -> bool:
        """
        ONNX模型量化
        
        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径
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
            else:
                # 静态量化需要校准数据
                if calibration_data is None:
                    logger.warning("静态量化需要校准数据,回退到动态量化")
                    quantize_dynamic(
                        input_path,
                        output_path,
                        weight_type=QuantType.QUInt8
                    )
                else:
                    # 创建校准数据读取器
                    class CalibrationDataReader:
                        def __init__(self, data):
                            self.data = data
                            self.index = 0
                        
                        def get_next(self):
                            if self.index >= len(self.data):
                                return None
                            result = {"input": self.data[self.index]}
                            self.index += 1
                            return result
                    
                    quantize_static(
                        input_path,
                        output_path,
                        CalibrationDataReader(calibration_data)
                    )
            
            logger.info(f"ONNX量化完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX量化失败: {e}")
            return False


# =============================================================================
# TensorRT转换
# =============================================================================
class TensorRTConverter:
    """TensorRT模型转换器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._logger = None
        
        if TRT_AVAILABLE:
            self._logger = trt.Logger(trt.Logger.WARNING)
    
    def convert_onnx_to_trt(self,
                            onnx_path: str,
                            trt_path: str,
                            precision: str = "fp16",
                            max_batch_size: int = 1,
                            max_workspace_size: int = 1 << 30,
                            calibration_data: Optional[np.ndarray] = None) -> bool:
        """
        ONNX模型转TensorRT
        
        Args:
            onnx_path: ONNX模型路径
            trt_path: TensorRT引擎输出路径
            precision: 精度 (fp32, fp16, int8)
            max_batch_size: 最大批次大小
            max_workspace_size: 最大工作空间
            calibration_data: INT8校准数据
        """
        if not TRT_AVAILABLE:
            logger.error("TensorRT未安装")
            return False
        
        try:
            builder = trt.Builder(self._logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self._logger)
            
            # 解析ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX解析错误: {parser.get_error(error)}")
                    return False
            
            # 配置
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            # 精度设置
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用FP16精度")
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                if calibration_data is not None:
                    config.int8_calibrator = self._create_calibrator(calibration_data)
                logger.info("启用INT8精度")
            
            # 构建引擎
            logger.info("开始构建TensorRT引擎...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("TensorRT引擎构建失败")
                return False
            
            # 序列化保存
            with open(trt_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT引擎保存成功: {trt_path}")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT转换失败: {e}")
            return False
    
    def _create_calibrator(self, calibration_data: np.ndarray):
        """创建INT8校准器"""
        
        class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, data, cache_file="calibration.cache"):
                super().__init__()
                self.data = data
                self.cache_file = cache_file
                self.current_index = 0
                self.batch_size = 1
                
                # 分配GPU内存
                import pycuda.driver as cuda
                import pycuda.autoinit
                self.device_input = cuda.mem_alloc(data[0].nbytes)
            
            def get_batch_size(self):
                return self.batch_size
            
            def get_batch(self, names):
                if self.current_index >= len(self.data):
                    return None
                
                import pycuda.driver as cuda
                cuda.memcpy_htod(self.device_input, self.data[self.current_index])
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
        
        return EntropyCalibrator(calibration_data)


# =============================================================================
# OpenVINO转换
# =============================================================================
class OpenVINOConverter:
    """OpenVINO模型转换器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def convert_onnx_to_openvino(self,
                                  onnx_path: str,
                                  output_dir: str,
                                  precision: str = "FP16") -> bool:
        """
        ONNX模型转OpenVINO IR格式
        """
        try:
            from openvino.tools import mo
            
            # 模型优化器转换
            mo.convert_model(
                onnx_path,
                output_dir=output_dir,
                compress_to_fp16=(precision == "FP16")
            )
            
            logger.info(f"OpenVINO模型转换成功: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"OpenVINO转换失败: {e}")
            return False


# =============================================================================
# 推理引擎封装
# =============================================================================
class InferenceEngine(ABC):
    """推理引擎基类"""
    
    @abstractmethod
    def load(self, model_path: str) -> bool:
        pass
    
    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def get_input_info(self) -> Dict[str, Dict]:
        pass
    
    @abstractmethod
    def get_output_info(self) -> Dict[str, Dict]:
        pass


class ONNXRuntimeEngine(InferenceEngine):
    """ONNX Runtime推理引擎"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.session = None
        self.input_info = {}
        self.output_info = {}
    
    def load(self, model_path: str) -> bool:
        if not ONNX_AVAILABLE:
            logger.error("ONNX Runtime未安装")
            return False
        
        try:
            # 选择执行提供者
            if self.device == "gpu":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # 获取输入输出信息
            for inp in self.session.get_inputs():
                self.input_info[inp.name] = {
                    "shape": inp.shape,
                    "dtype": inp.type
                }
            
            for out in self.session.get_outputs():
                self.output_info[out.name] = {
                    "shape": out.shape,
                    "dtype": out.type
                }
            
            logger.info(f"ONNX模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX模型加载失败: {e}")
            return False
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.session is None:
            raise RuntimeError("模型未加载")
        
        outputs = self.session.run(None, inputs)
        
        return {
            name: output 
            for name, output in zip(self.output_info.keys(), outputs)
        }
    
    def get_input_info(self) -> Dict[str, Dict]:
        return self.input_info
    
    def get_output_info(self) -> Dict[str, Dict]:
        return self.output_info


class TensorRTEngine(InferenceEngine):
    """TensorRT推理引擎"""
    
    def __init__(self):
        self.engine = None
        self.context = None
        self.input_info = {}
        self.output_info = {}
        self.bindings = []
        self.stream = None
    
    def load(self, model_path: str) -> bool:
        if not TRT_AVAILABLE:
            logger.error("TensorRT未安装")
            return False
        
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger_trt = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger_trt)
            
            with open(model_path, 'rb') as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            
            # 获取绑定信息
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                
                if self.engine.binding_is_input(i):
                    self.input_info[name] = {"shape": shape, "dtype": dtype}
                else:
                    self.output_info[name] = {"shape": shape, "dtype": dtype}
            
            logger.info(f"TensorRT引擎加载成功: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT引擎加载失败: {e}")
            return False
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.engine is None:
            raise RuntimeError("引擎未加载")
        
        import pycuda.driver as cuda
        
        # 分配设备内存
        d_inputs = {}
        d_outputs = {}
        h_outputs = {}
        bindings = []
        
        for name, info in self.input_info.items():
            data = inputs[name].astype(info["dtype"])
            d_inputs[name] = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(d_inputs[name], data)
            bindings.append(int(d_inputs[name]))
        
        for name, info in self.output_info.items():
            shape = tuple(max(1, s) for s in info["shape"])
            h_outputs[name] = np.empty(shape, dtype=info["dtype"])
            d_outputs[name] = cuda.mem_alloc(h_outputs[name].nbytes)
            bindings.append(int(d_outputs[name]))
        
        # 执行推理
        self.context.execute_async_v2(bindings, self.stream.handle)
        self.stream.synchronize()
        
        # 复制输出
        for name in self.output_info.keys():
            cuda.memcpy_dtoh(h_outputs[name], d_outputs[name])
        
        return h_outputs
    
    def get_input_info(self) -> Dict[str, Dict]:
        return self.input_info
    
    def get_output_info(self) -> Dict[str, Dict]:
        return self.output_info


# =============================================================================
# 边缘部署管理器
# =============================================================================
class EdgeDeploymentManager:
    """边缘设备部署管理器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.onnx_converter = ONNXConverter(config)
        self.trt_converter = TensorRTConverter(config)
        self.openvino_converter = OpenVINOConverter(config)
    
    def deploy_model(self,
                     model: Union['nn.Module', str],
                     model_name: str,
                     dummy_inputs: Optional[Dict[str, 'torch.Tensor']] = None,
                     input_names: Optional[List[str]] = None,
                     output_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        部署模型到目标格式
        
        Args:
            model: PyTorch模型或ONNX路径
            model_name: 模型名称
            dummy_inputs: 虚拟输入 (PyTorch模型需要)
            input_names: 输入名称
            output_names: 输出名称
        
        Returns:
            各格式的模型路径
        """
        output_dir = Path(self.config.output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        deployed_models = {}
        
        # 确定ONNX路径
        if isinstance(model, str) and model.endswith('.onnx'):
            onnx_path = model
        else:
            # PyTorch -> ONNX
            onnx_path = str(output_dir / f"{model_name}.onnx")
            if TORCH_AVAILABLE and dummy_inputs is not None:
                success = self.onnx_converter.convert_pytorch_to_onnx(
                    model, dummy_inputs, onnx_path,
                    input_names, output_names
                )
                if not success:
                    logger.error("ONNX转换失败")
                    return deployed_models
        
        deployed_models["onnx"] = onnx_path
        
        # ONNX优化
        if self.config.optimize_onnx:
            optimized_path = str(output_dir / f"{model_name}_optimized.onnx")
            if self.onnx_converter.optimize_onnx(onnx_path, optimized_path):
                deployed_models["onnx_optimized"] = optimized_path
                onnx_path = optimized_path
        
        # ONNX量化
        if self.config.quantization != "fp32":
            quantized_path = str(output_dir / f"{model_name}_quantized.onnx")
            if self.onnx_converter.quantize_onnx(onnx_path, quantized_path):
                deployed_models["onnx_quantized"] = quantized_path
        
        # TensorRT
        if "tensorrt" in self.config.target_formats and TRT_AVAILABLE:
            trt_path = str(output_dir / f"{model_name}.trt")
            if self.trt_converter.convert_onnx_to_trt(
                onnx_path, trt_path,
                precision=self.config.trt_precision,
                max_batch_size=self.config.trt_max_batch_size,
                max_workspace_size=self.config.trt_max_workspace_size
            ):
                deployed_models["tensorrt"] = trt_path
        
        # OpenVINO
        if "openvino" in self.config.target_formats and OPENVINO_AVAILABLE:
            openvino_dir = str(output_dir / "openvino")
            if self.openvino_converter.convert_onnx_to_openvino(
                onnx_path, openvino_dir
            ):
                deployed_models["openvino"] = openvino_dir
        
        # 保存元数据
        metadata = {
            "model_name": model_name,
            "deployed_formats": list(deployed_models.keys()),
            "config": self.config.__dict__,
            "paths": deployed_models
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"模型部署完成: {deployed_models}")
        return deployed_models


# =============================================================================
# 性能基准测试
# =============================================================================
class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def benchmark_onnx(self, 
                       model_path: str,
                       input_shapes: Dict[str, Tuple[int, ...]],
                       device: str = "cpu") -> Dict[str, float]:
        """ONNX模型基准测试"""
        engine = ONNXRuntimeEngine(device)
        if not engine.load(model_path):
            return {}
        
        # 生成测试数据
        inputs = {
            name: np.random.randn(*shape).astype(np.float32)
            for name, shape in input_shapes.items()
        }
        
        # 预热
        for _ in range(self.config.warmup_iterations):
            engine.infer(inputs)
        
        # 测试
        times = []
        for _ in range(self.config.benchmark_iterations):
            start = time.perf_counter()
            engine.infer(inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        return {
            "mean_latency_ms": np.mean(times),
            "std_latency_ms": np.std(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "throughput_fps": 1000 / np.mean(times)
        }
    
    def benchmark_tensorrt(self,
                           engine_path: str,
                           input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, float]:
        """TensorRT引擎基准测试"""
        if not TRT_AVAILABLE:
            logger.error("TensorRT未安装")
            return {}
        
        engine = TensorRTEngine()
        if not engine.load(engine_path):
            return {}
        
        # 生成测试数据
        inputs = {
            name: np.random.randn(*shape).astype(np.float32)
            for name, shape in input_shapes.items()
        }
        
        # 预热
        for _ in range(self.config.warmup_iterations):
            engine.infer(inputs)
        
        # 测试
        times = []
        for _ in range(self.config.benchmark_iterations):
            start = time.perf_counter()
            engine.infer(inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return {
            "mean_latency_ms": np.mean(times),
            "std_latency_ms": np.std(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "throughput_fps": 1000 / np.mean(times)
        }
    
    def compare_formats(self,
                        model_paths: Dict[str, str],
                        input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Dict]:
        """比较不同格式的性能"""
        results = {}
        
        for format_name, path in model_paths.items():
            logger.info(f"测试 {format_name}...")
            
            if format_name.startswith("onnx"):
                results[format_name] = self.benchmark_onnx(path, input_shapes)
            elif format_name == "tensorrt":
                results[format_name] = self.benchmark_tensorrt(path, input_shapes)
        
        # 打印对比结果
        logger.info("\n性能对比:")
        logger.info("-" * 60)
        for name, metrics in results.items():
            if metrics:
                logger.info(f"{name}:")
                logger.info(f"  延迟: {metrics['mean_latency_ms']:.2f} ± {metrics['std_latency_ms']:.2f} ms")
                logger.info(f"  吞吐: {metrics['throughput_fps']:.1f} FPS")
        
        return results


# =============================================================================
# Jetson部署辅助
# =============================================================================
class JetsonDeployer:
    """Jetson设备部署辅助类"""
    
    def __init__(self):
        self.device_info = self._get_device_info()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """获取Jetson设备信息"""
        info = {
            "platform": "unknown",
            "cuda_available": False,
            "tensorrt_available": TRT_AVAILABLE
        }
        
        # 检测Jetson平台
        if os.path.exists("/etc/nv_tegra_release"):
            with open("/etc/nv_tegra_release") as f:
                info["platform"] = "jetson"
                info["tegra_release"] = f.read().strip()
        
        if TORCH_AVAILABLE:
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
        
        return info
    
    def optimize_for_jetson(self, 
                            onnx_path: str,
                            output_path: str,
                            dla_core: int = -1) -> bool:
        """
        针对Jetson优化模型
        
        Args:
            onnx_path: ONNX模型路径
            output_path: 输出TensorRT引擎路径
            dla_core: DLA核心 (-1表示使用GPU)
        """
        if not TRT_AVAILABLE:
            logger.error("TensorRT未安装")
            return False
        
        try:
            logger_trt = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger_trt)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger_trt)
            
            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())
            
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 28  # 256MB for Jetson
            
            # 启用FP16
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            # DLA配置
            if dla_core >= 0:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = dla_core
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            
            engine = builder.build_engine(network, config)
            
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"Jetson优化引擎保存: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Jetson优化失败: {e}")
            return False
    
    def get_power_mode_recommendations(self) -> Dict[str, str]:
        """获取功耗模式建议"""
        return {
            "real_time": "MAXN (最大性能,最大功耗)",
            "balanced": "15W (平衡模式)",
            "power_saving": "10W (省电模式)",
            "command": "sudo nvpmodel -m <mode_id>"
        }


# =============================================================================
# 部署流水线
# =============================================================================
class DeploymentPipeline:
    """完整部署流水线"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_manager = EdgeDeploymentManager(config)
        self.benchmark = PerformanceBenchmark(config)
    
    def run(self,
            model: Union['nn.Module', str],
            model_name: str,
            dummy_inputs: Optional[Dict[str, 'torch.Tensor']] = None,
            input_names: Optional[List[str]] = None,
            output_names: Optional[List[str]] = None,
            run_benchmark: bool = True) -> Dict[str, Any]:
        """
        运行完整部署流水线
        
        Args:
            model: 模型
            model_name: 模型名称
            dummy_inputs: 虚拟输入
            input_names: 输入名称
            output_names: 输出名称
            run_benchmark: 是否运行基准测试
        
        Returns:
            部署结果
        """
        results = {
            "model_name": model_name,
            "deployed_models": {},
            "benchmark_results": {},
            "status": "success"
        }
        
        try:
            # 部署
            deployed = self.deployment_manager.deploy_model(
                model, model_name, dummy_inputs, input_names, output_names
            )
            results["deployed_models"] = deployed
            
            # 基准测试
            if run_benchmark and dummy_inputs is not None:
                input_shapes = {
                    name: tuple(tensor.shape) 
                    for name, tensor in dummy_inputs.items()
                }
                results["benchmark_results"] = self.benchmark.compare_formats(
                    deployed, input_shapes
                )
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"部署流水线失败: {e}")
        
        return results


# =============================================================================
# 便捷函数
# =============================================================================
def deploy_slam_model(checkpoint_path: str, output_dir: str = "deployed_models/slam"):
    """部署SLAM模型"""
    config = DeploymentConfig(
        output_dir=output_dir,
        target_formats=["onnx", "tensorrt"],
        trt_precision="fp16"
    )
    
    pipeline = DeploymentPipeline(config)
    
    # 加载模型
    if TORCH_AVAILABLE:
        from training.slam.slam_trainer import DeepLIO, SLAMTrainingConfig
        
        model_config = SLAMTrainingConfig()
        model = DeepLIO(model_config)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 虚拟输入
        dummy_inputs = {
            "source": torch.randn(1, 16384, 3),
            "target": torch.randn(1, 16384, 3)
        }
        
        return pipeline.run(
            model, "slam_deep_lio", dummy_inputs,
            input_names=["source", "target"],
            output_names=["pose"]
        )
    
    return None


def deploy_acoustic_model(checkpoint_path: str, output_dir: str = "deployed_models/acoustic"):
    """部署声学模型"""
    config = DeploymentConfig(
        output_dir=output_dir,
        target_formats=["onnx"],
        quantization="fp16"
    )
    
    pipeline = DeploymentPipeline(config)
    
    if TORCH_AVAILABLE:
        from training.acoustic.acoustic_trainer import AcousticAnomalyTransformer, AcousticTrainingConfig
        
        model_config = AcousticTrainingConfig()
        model = AcousticAnomalyTransformer(model_config)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 虚拟输入 (2秒音频 @ 16kHz)
        dummy_inputs = {
            "waveform": torch.randn(1, 32000)
        }
        
        return pipeline.run(
            model, "acoustic_transformer", dummy_inputs,
            input_names=["waveform"],
            output_names=["global_feat", "anomaly_logits", "anomaly_score"]
        )
    
    return None


def deploy_timeseries_model(checkpoint_path: str, output_dir: str = "deployed_models/timeseries"):
    """部署时序模型"""
    config = DeploymentConfig(
        output_dir=output_dir,
        target_formats=["onnx"],
        optimize_onnx=True
    )
    
    pipeline = DeploymentPipeline(config)
    
    if TORCH_AVAILABLE:
        from training.timeseries.timeseries_trainer import Informer, TimeSeriesTrainingConfig
        
        model_config = TimeSeriesTrainingConfig()
        model = Informer(model_config)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 虚拟输入
        dummy_inputs = {
            "input": torch.randn(1, 168, 8)
        }
        
        return pipeline.run(
            model, "timeseries_informer", dummy_inputs,
            input_names=["input"],
            output_names=["mu", "sigma"]
        )
    
    return None


# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例: 部署配置
    config = DeploymentConfig(
        output_dir="deployed_models",
        target_formats=["onnx", "tensorrt"],
        trt_precision="fp16",
        benchmark_iterations=100
    )
    
    # 创建部署管理器
    manager = EdgeDeploymentManager(config)
    
    logger.info("部署模块初始化完成")
    logger.info(f"ONNX可用: {ONNX_AVAILABLE}")
    logger.info(f"TensorRT可用: {TRT_AVAILABLE}")
    logger.info(f"OpenVINO可用: {OPENVINO_AVAILABLE}")
