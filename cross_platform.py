#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台训练与部署脚本
破夜绘明激光监测平台

功能:
1. Mac M系列 (MPS) 训练
2. ONNX模型导出
3. Windows推理验证

使用方法:
    # 训练 (Mac)
    python cross_platform.py --mode train --device mps --epochs 50
    
    # 导出ONNX
    python cross_platform.py --mode export --model-path checkpoints/best.pth
    
    # 推理验证 (Windows)
    python cross_platform.py --mode inference --onnx-path models/model.onnx
"""

import os
import sys
import json
import argparse
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装")

# ONNX Runtime
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


# =============================================================================
# 平台配置
# =============================================================================
@dataclass
class PlatformConfig:
    """平台配置"""
    system: str
    machine: str
    device: str
    device_name: str
    recommended_batch_size: int
    recommended_precision: str


def detect_platform() -> PlatformConfig:
    """检测运行平台"""
    system = platform.system()
    machine = platform.machine()
    
    # 默认CPU
    device = "cpu"
    device_name = "CPU"
    batch_size = 8
    precision = "float32"
    
    if TORCH_AVAILABLE:
        # Apple Silicon MPS
        if system == "Darwin" and machine == "arm64":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                device_name = f"Apple Silicon MPS ({machine})"
                batch_size = 16
                precision = "float32"
        
        # NVIDIA CUDA
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            batch_size = 32
            precision = "float16"
    
    return PlatformConfig(
        system=system,
        machine=machine,
        device=device,
        device_name=device_name,
        recommended_batch_size=batch_size,
        recommended_precision=precision
    )


def get_device(prefer: str = "auto") -> "torch.device":
    """获取PyTorch设备"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch未安装")
    
    if prefer == "auto":
        config = detect_platform()
        prefer = config.device
    
    if prefer == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# =============================================================================
# 模型导出
# =============================================================================
def export_to_onnx(model: nn.Module, input_shape: tuple, 
                   save_path: str, opset_version: int = 17) -> bool:
    """
    导出模型为ONNX格式
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状 (C, H, W)
        save_path: 保存路径
        opset_version: ONNX opset版本
    
    Returns:
        是否成功
    """
    logger.info(f"导出ONNX模型: {save_path}")
    
    model.eval()
    model = model.cpu()
    
    # 创建目录
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Dummy输入
    dummy_input = torch.randn(1, *input_shape)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            do_constant_folding=True
        )
        
        logger.info(f"✅ ONNX导出成功: {save_path}")
        
        # 验证
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX模型验证通过")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX导出失败: {e}")
        return False


# =============================================================================
# ONNX推理
# =============================================================================
class ONNXInference:
    """ONNX推理引擎"""
    
    def __init__(self, onnx_path: str, use_gpu: bool = False):
        """
        初始化推理引擎
        
        Args:
            onnx_path: ONNX模型路径
            use_gpu: 是否使用GPU
        """
        if not ORT_AVAILABLE:
            raise RuntimeError("ONNX Runtime未安装")
        
        self.onnx_path = onnx_path
        
        # 选择执行提供者
        providers = []
        
        if use_gpu:
            # TensorRT (最快)
            if 'TensorrtExecutionProvider' in ort.get_available_providers():
                providers.append(('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_fp16_enable': True
                }))
            
            # CUDA
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0
                }))
        
        providers.append('CPUExecutionProvider')
        
        # 创建会话
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"ONNX模型加载成功: {onnx_path}")
        logger.info(f"执行提供者: {self.session.get_providers()}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            input_data: 输入数据 [B, C, H, W]
        
        Returns:
            推理结果
        """
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_data.astype(np.float32)}
        )
        return output[0]
    
    def benchmark(self, input_shape: tuple, num_iterations: int = 100) -> Dict:
        """
        性能基准测试
        
        Args:
            input_shape: 输入形状
            num_iterations: 测试次数
        
        Returns:
            性能统计
        """
        import time
        
        # 准备输入
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 预热
        for _ in range(10):
            self.predict(test_input)
        
        # 测试
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.predict(test_input)
            times.append((time.perf_counter() - start) * 1000)
        
        times = np.array(times)
        
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(1000 / np.mean(times))
        }


# =============================================================================
# 命令行接口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="跨平台训练与部署")
    parser.add_argument("--mode", type=str, default="info",
                       choices=["info", "train", "export", "inference", "benchmark"],
                       help="运行模式")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "mps", "cuda", "cpu"],
                       help="设备")
    parser.add_argument("--model-path", type=str, help="模型路径")
    parser.add_argument("--onnx-path", type=str, help="ONNX路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    
    args = parser.parse_args()
    
    # 打印平台信息
    config = detect_platform()
    logger.info("="*50)
    logger.info("跨平台训练与部署工具")
    logger.info("="*50)
    logger.info(f"系统: {config.system} ({config.machine})")
    logger.info(f"设备: {config.device} - {config.device_name}")
    logger.info(f"推荐Batch: {config.recommended_batch_size}")
    logger.info(f"推荐精度: {config.recommended_precision}")
    
    if args.mode == "info":
        logger.info("\n平台检测完成")
    
    elif args.mode == "train":
        logger.info("\n开始训练...")
        logger.info("请使用 train_main.py 进行训练")
    
    elif args.mode == "export":
        if not args.model_path:
            logger.error("请指定 --model-path")
            return
        
        logger.info(f"\n导出ONNX: {args.model_path}")
        # 实际导出逻辑
    
    elif args.mode == "inference":
        if not args.onnx_path:
            logger.error("请指定 --onnx-path")
            return
        
        logger.info(f"\n推理测试: {args.onnx_path}")
        
        engine = ONNXInference(args.onnx_path, use_gpu=(config.device == "cuda"))
        
        # 测试推理
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = engine.predict(test_input)
        logger.info(f"输出形状: {output.shape}")
    
    elif args.mode == "benchmark":
        if not args.onnx_path:
            logger.error("请指定 --onnx-path")
            return
        
        logger.info(f"\n性能测试: {args.onnx_path}")
        
        engine = ONNXInference(args.onnx_path, use_gpu=(config.device == "cuda"))
        stats = engine.benchmark((1, 3, 224, 224))
        
        logger.info(f"平均: {stats['mean_ms']:.2f} ms")
        logger.info(f"FPS: {stats['fps']:.1f}")


if __name__ == "__main__":
    main()
