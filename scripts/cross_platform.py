#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台AI训练与部署脚本
破夜绘明激光监测平台

功能:
- Mac M系列芯片 (MPS加速) 训练
- 导出ONNX模型 (跨平台)
- Windows CUDA/TensorRT 推理

使用方法:
    # Mac上训练
    python cross_platform.py --mode train --device mps --epochs 50
    
    # 导出ONNX
    python cross_platform.py --mode export --model-path models/best.pth
    
    # Windows推理
    python cross_platform.py --mode inference --onnx-path models/model.onnx
"""

import os
import sys
import argparse
import platform
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """平台配置"""
    system: str           # Darwin, Windows, Linux
    device: str           # mps, cuda, cpu
    backend: str          # pytorch, onnx, tensorrt
    precision: str        # float32, float16
    batch_size: int
    is_training: bool


def detect_platform() -> PlatformConfig:
    """自动检测运行平台并返回最优配置"""
    system = platform.system()
    machine = platform.machine()
    
    logger.info(f"检测到系统: {system} ({machine})")
    
    if system == "Darwin":
        # macOS
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("✅ Apple Silicon MPS 可用")
                return PlatformConfig(
                    system="Darwin",
                    device="mps",
                    backend="pytorch",
                    precision="float32",  # MPS当前对FP16支持有限
                    batch_size=16,
                    is_training=True
                )
        except ImportError:
            pass
        
        return PlatformConfig(
            system="Darwin",
            device="cpu",
            backend="pytorch",
            precision="float32",
            batch_size=8,
            is_training=True
        )
    
    elif system == "Windows":
        # Windows - 优先使用CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ NVIDIA GPU 可用: {gpu_name}")
                return PlatformConfig(
                    system="Windows",
                    device="cuda",
                    backend="onnx",  # Windows推荐ONNX推理
                    precision="float16",
                    batch_size=32,
                    is_training=False
                )
        except ImportError:
            pass
        
        return PlatformConfig(
            system="Windows",
            device="cpu",
            backend="onnx",
            precision="float32",
            batch_size=4,
            is_training=False
        )
    
    else:
        # Linux
        return PlatformConfig(
            system="Linux",
            device="cpu",
            backend="pytorch",
            precision="float32",
            batch_size=8,
            is_training=True
        )


class CrossPlatformTrainer:
    """跨平台训练器"""
    
    def __init__(self, config: PlatformConfig):
        self.config = config
        self.device = self._get_torch_device()
        
    def _get_torch_device(self):
        """获取PyTorch设备"""
        import torch
        
        if self.config.device == "mps":
            return torch.device("mps")
        elif self.config.device == "cuda":
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def train(self, model, train_loader, val_loader=None, 
              epochs: int = 50, save_path: str = "models/best.pth"):
        """
        训练模型
        
        Args:
            model: PyTorch模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_path: 模型保存路径
        """
        import torch
        import torch.nn as nn
        
        logger.info(f"开始训练: device={self.config.device}, epochs={epochs}")
        
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            scheduler.step()
            
            train_acc = 100. * train_correct / train_total
            avg_loss = train_loss / len(train_loader)
            
            # 验证阶段
            if val_loader is not None:
                val_acc = self._validate(model, val_loader)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(model, save_path, epoch, val_acc)
                    logger.info(f"💾 保存最佳模型: val_acc={val_acc:.2f}%")
                
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"loss={avg_loss:.4f}, train_acc={train_acc:.2f}%, "
                           f"val_acc={val_acc:.2f}%")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"loss={avg_loss:.4f}, train_acc={train_acc:.2f}%")
        
        logger.info(f"✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
        return model
    
    def _validate(self, model, val_loader) -> float:
        """验证模型"""
        import torch
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return 100. * correct / total
    
    def _save_checkpoint(self, model, path: str, epoch: int, val_acc: float):
        """保存检查点"""
        import torch
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'platform': self.config.system,
            'device': self.config.device
        }, path)


class ONNXExporter:
    """ONNX导出器"""
    
    def __init__(self, opset_version: int = 17):
        self.opset_version = opset_version
    
    def export(self, model, input_shape: Tuple, 
               save_path: str, dynamic_batch: bool = True):
        """
        导出模型为ONNX格式
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状 (不含batch维度)
            save_path: ONNX保存路径
            dynamic_batch: 是否支持动态batch size
        """
        import torch
        
        logger.info(f"导出ONNX模型: {save_path}")
        
        model.eval()
        model = model.to("cpu")  # ONNX导出需要CPU
        
        # 创建dummy输入
        dummy_input = torch.randn(1, *input_shape)
        
        # 动态轴配置
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # 导出
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            opset_version=self.opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True
        )
        
        logger.info(f"✅ ONNX导出成功: {save_path}")
        
        # 验证ONNX
        self._verify_onnx(save_path, dummy_input)
        
        # 可选: 简化ONNX
        self._simplify_onnx(save_path)
        
        return save_path
    
    def _verify_onnx(self, path: str, dummy_input):
        """验证ONNX模型"""
        import onnx
        import onnxruntime as ort
        
        # 检查模型结构
        model = onnx.load(path)
        onnx.checker.check_model(model)
        logger.info("✅ ONNX模型验证通过")
        
        # 测试推理
        session = ort.InferenceSession(path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: dummy_input.numpy()})
        logger.info(f"✅ ONNX推理测试通过, 输出形状: {output[0].shape}")
    
    def _simplify_onnx(self, path: str):
        """简化ONNX模型"""
        try:
            import onnxsim
            import onnx
            
            model = onnx.load(path)
            simplified, ok = onnxsim.simplify(model)
            
            if ok:
                onnx.save(simplified, path)
                logger.info("✅ ONNX模型简化完成")
            else:
                logger.warning("⚠️ ONNX简化失败,保留原始模型")
        except ImportError:
            logger.info("ℹ️ onnxsim未安装,跳过简化步骤")


class CrossPlatformInference:
    """跨平台推理引擎"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.session = None
        self._load_model(use_gpu)
    
    def _load_model(self, use_gpu: bool):
        """加载ONNX模型"""
        import onnxruntime as ort
        
        # 选择执行提供者
        if use_gpu:
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_fp16_enable': True
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE'
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        actual_providers = self.session.get_providers()
        logger.info(f"✅ ONNX模型加载完成, 使用: {actual_providers}")
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """执行推理"""
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        return outputs[0]
    
    def benchmark(self, input_shape: Tuple, n_iterations: int = 100) -> Dict:
        """性能基准测试"""
        import time
        
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # 预热
        for _ in range(10):
            self.predict(dummy_input)
        
        # 测试
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.predict(dummy_input)
            times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # 转换为毫秒
        
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(1000 / np.mean(times))
        }


def main():
    parser = argparse.ArgumentParser(description="跨平台AI训练与部署")
    parser.add_argument("--mode", choices=["train", "export", "inference", "benchmark"],
                        default="train", help="运行模式")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"],
                        default="auto", help="计算设备")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--model-path", type=str, default="models/best.pth",
                        help="PyTorch模型路径")
    parser.add_argument("--onnx-path", type=str, default="models/model.onnx",
                        help="ONNX模型路径")
    parser.add_argument("--input-shape", type=str, default="3,640,640",
                        help="输入形状 (C,H,W)")
    
    args = parser.parse_args()
    
    # 解析输入形状
    input_shape = tuple(map(int, args.input_shape.split(",")))
    
    # 检测平台
    config = detect_platform()
    
    # 覆盖设备设置
    if args.device != "auto":
        config.device = args.device
    
    logger.info(f"运行配置: {config}")
    
    if args.mode == "train":
        # 训练模式 (Mac)
        import torch
        import torch.nn as nn
        
        # 示例: 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self, in_channels=3, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.classifier = nn.Linear(64, num_classes)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        model = SimpleModel()
        trainer = CrossPlatformTrainer(config)
        
        # 创建演示数据
        from torch.utils.data import DataLoader, TensorDataset
        train_data = torch.randn(100, *input_shape)
        train_labels = torch.randint(0, 10, (100,))
        train_loader = DataLoader(
            TensorDataset(train_data, train_labels),
            batch_size=config.batch_size,
            shuffle=True
        )
        
        trainer.train(model, train_loader, epochs=args.epochs, save_path=args.model_path)
        
    elif args.mode == "export":
        # 导出ONNX
        import torch
        
        checkpoint = torch.load(args.model_path, map_location="cpu")
        
        # 重建模型 (需要与训练时一致)
        class SimpleModel(torch.nn.Module):
            def __init__(self, in_channels=3, num_classes=10):
                super().__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, 32, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.AdaptiveAvgPool2d(1)
                )
                self.classifier = torch.nn.Linear(64, num_classes)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        model = SimpleModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        exporter = ONNXExporter()
        exporter.export(model, input_shape, args.onnx_path)
        
    elif args.mode == "inference":
        # 推理模式 (Windows)
        engine = CrossPlatformInference(args.onnx_path, use_gpu=config.device != "cpu")
        
        # 示例推理
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        result = engine.predict(dummy_input)
        logger.info(f"推理结果形状: {result.shape}")
        
    elif args.mode == "benchmark":
        # 性能测试
        engine = CrossPlatformInference(args.onnx_path, use_gpu=config.device != "cpu")
        stats = engine.benchmark(input_shape)
        
        logger.info("性能测试结果:")
        logger.info(f"  平均延迟: {stats['mean_ms']:.2f} ms")
        logger.info(f"  标准差: {stats['std_ms']:.2f} ms")
        logger.info(f"  最小延迟: {stats['min_ms']:.2f} ms")
        logger.info(f"  最大延迟: {stats['max_ms']:.2f} ms")
        logger.info(f"  FPS: {stats['fps']:.1f}")


if __name__ == "__main__":
    main()
