#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型压缩模块
量化/剪枝/知识蒸馏

功能:
1. 训练后量化 (PTQ)
2. 量化感知训练 (QAT)
3. 结构化剪枝
4. 非结构化剪枝
5. 知识蒸馏

作者: AI巡检系统
版本: 1.0.0
"""

import os
import logging
import copy
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """模型压缩配置"""
    # 量化配置
    quantization_bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    
    # 剪枝配置
    pruning_ratio: float = 0.5
    pruning_method: str = "magnitude"  # magnitude, l1, taylor
    structured: bool = True
    
    # 蒸馏配置
    temperature: float = 4.0
    alpha: float = 0.5  # 蒸馏损失权重
    
    # 训练配置
    num_epochs: int = 10
    learning_rate: float = 1e-4


if TORCH_AVAILABLE:
    # =========================================================================
    # 量化工具
    # =========================================================================
    class Quantizer:
        """量化器"""
        
        def __init__(self, bits: int = 8, symmetric: bool = True):
            self.bits = bits
            self.symmetric = symmetric
            self.qmin = -(2 ** (bits - 1)) if symmetric else 0
            self.qmax = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
        
        def compute_scale_zero_point(self, tensor: torch.Tensor,
                                     per_channel: bool = False,
                                     channel_axis: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
            """计算量化参数"""
            if per_channel:
                # 沿通道计算
                dims = list(range(tensor.dim()))
                dims.remove(channel_axis)
                
                min_val = tensor.amin(dim=dims, keepdim=True)
                max_val = tensor.amax(dim=dims, keepdim=True)
            else:
                min_val = tensor.min()
                max_val = tensor.max()
            
            if self.symmetric:
                abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
                scale = abs_max / self.qmax
                zero_point = torch.zeros_like(scale)
            else:
                scale = (max_val - min_val) / (self.qmax - self.qmin)
                zero_point = self.qmin - torch.round(min_val / scale)
            
            scale = torch.clamp(scale, min=1e-8)
            
            return scale, zero_point
        
        def quantize(self, tensor: torch.Tensor, scale: torch.Tensor,
                    zero_point: torch.Tensor) -> torch.Tensor:
            """量化"""
            q_tensor = torch.round(tensor / scale + zero_point)
            q_tensor = torch.clamp(q_tensor, self.qmin, self.qmax)
            return q_tensor
        
        def dequantize(self, q_tensor: torch.Tensor, scale: torch.Tensor,
                      zero_point: torch.Tensor) -> torch.Tensor:
            """反量化"""
            return (q_tensor - zero_point) * scale


    class QuantizedLinear(nn.Module):
        """量化线性层"""
        
        def __init__(self, linear: nn.Linear, bits: int = 8,
                     per_channel: bool = True):
            super().__init__()
            
            self.in_features = linear.in_features
            self.out_features = linear.out_features
            
            self.quantizer = Quantizer(bits)
            
            # 量化权重
            weight = linear.weight.data
            scale, zero_point = self.quantizer.compute_scale_zero_point(
                weight, per_channel=per_channel, channel_axis=0
            )
            
            self.register_buffer('weight_scale', scale)
            self.register_buffer('weight_zero_point', zero_point)
            self.register_buffer('weight_quantized', 
                               self.quantizer.quantize(weight, scale, zero_point).to(torch.int8))
            
            if linear.bias is not None:
                self.register_buffer('bias', linear.bias.data)
            else:
                self.bias = None
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 反量化权重
            weight = self.quantizer.dequantize(
                self.weight_quantized.float(),
                self.weight_scale,
                self.weight_zero_point
            )
            
            return F.linear(x, weight, self.bias)


    class PostTrainingQuantization:
        """训练后量化"""
        
        def __init__(self, config: CompressionConfig):
            self.config = config
            self.quantizer = Quantizer(config.quantization_bits, config.symmetric)
        
        def quantize_model(self, model: nn.Module,
                          calibration_data: Optional[DataLoader] = None) -> nn.Module:
            """
            量化模型
            
            Args:
                model: 原始模型
                calibration_data: 校准数据
            
            Returns:
                量化后的模型
            """
            quantized_model = copy.deepcopy(model)
            
            # 收集激活统计 (如果有校准数据)
            if calibration_data is not None:
                self._calibrate(quantized_model, calibration_data)
            
            # 量化所有Linear层
            self._quantize_layers(quantized_model)
            
            return quantized_model
        
        def _calibrate(self, model: nn.Module, data_loader: DataLoader):
            """收集激活统计"""
            model.eval()
            
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                
                with torch.no_grad():
                    model(x)
        
        def _quantize_layers(self, model: nn.Module, prefix: str = ''):
            """递归量化层"""
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(module, nn.Linear):
                    quantized = QuantizedLinear(
                        module,
                        bits=self.config.quantization_bits,
                        per_channel=self.config.per_channel
                    )
                    setattr(model, name, quantized)
                    logger.info(f"量化层: {full_name}")
                else:
                    self._quantize_layers(module, full_name)


    # =========================================================================
    # 剪枝工具
    # =========================================================================
    class Pruner:
        """剪枝器"""
        
        def __init__(self, config: CompressionConfig):
            self.config = config
        
        def compute_importance(self, module: nn.Module,
                              method: str = "magnitude") -> torch.Tensor:
            """
            计算参数重要性
            
            Args:
                module: 要剪枝的模块
                method: 重要性计算方法
            
            Returns:
                重要性分数
            """
            if not hasattr(module, 'weight'):
                return None
            
            weight = module.weight.data
            
            if method == "magnitude":
                # 权重绝对值
                importance = torch.abs(weight)
            
            elif method == "l1":
                # L1范数 (结构化剪枝)
                if self.config.structured:
                    importance = torch.norm(weight, p=1, dim=1)  # 按输出通道
                else:
                    importance = torch.abs(weight)
            
            elif method == "taylor":
                # Taylor展开 (需要梯度)
                if weight.grad is not None:
                    importance = torch.abs(weight * weight.grad)
                else:
                    importance = torch.abs(weight)
            
            else:
                importance = torch.abs(weight)
            
            return importance
        
        def create_mask(self, importance: torch.Tensor,
                       ratio: float) -> torch.Tensor:
            """
            创建剪枝掩码
            
            Args:
                importance: 重要性分数
                ratio: 剪枝比例
            
            Returns:
                二值掩码
            """
            if self.config.structured:
                # 结构化剪枝: 按通道
                num_to_prune = int(importance.numel() * ratio)
                threshold = torch.kthvalue(importance.flatten(), num_to_prune)[0]
                mask = (importance > threshold).float()
            else:
                # 非结构化剪枝: 按元素
                num_to_prune = int(importance.numel() * ratio)
                threshold = torch.kthvalue(importance.flatten(), num_to_prune)[0]
                mask = (importance > threshold).float()
            
            return mask
        
        def apply_mask(self, module: nn.Module, mask: torch.Tensor):
            """应用剪枝掩码"""
            if hasattr(module, 'weight'):
                if self.config.structured:
                    # 结构化: 整个通道置零
                    if mask.shape != module.weight.shape:
                        if mask.numel() == module.weight.numel():
                            mask = mask.reshape_as(module.weight)
                        else:
                            mask = mask.view(-1, 1).expand_as(module.weight)
                
                module.weight.data *= mask
        
        def prune_model(self, model: nn.Module) -> nn.Module:
            """
            剪枝模型
            
            Args:
                model: 原始模型
            
            Returns:
                剪枝后的模型
            """
            pruned_model = copy.deepcopy(model)
            
            for name, module in pruned_model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    importance = self.compute_importance(
                        module, self.config.pruning_method
                    )
                    
                    if importance is not None:
                        mask = self.create_mask(importance, self.config.pruning_ratio)
                        self.apply_mask(module, mask)
                        
                        # 计算稀疏度
                        sparsity = 1.0 - mask.float().mean().item()
                        logger.info(f"剪枝层 {name}: 稀疏度 {sparsity:.2%}")
            
            return pruned_model
        
        def iterative_pruning(self, model: nn.Module, train_loader: DataLoader,
                            criterion, optimizer, target_ratio: float,
                            num_iterations: int = 5) -> nn.Module:
            """
            迭代剪枝
            
            Args:
                model: 原始模型
                train_loader: 训练数据
                criterion: 损失函数
                optimizer: 优化器
                target_ratio: 目标剪枝比例
                num_iterations: 迭代次数
            """
            pruned_model = copy.deepcopy(model)
            
            ratio_per_iter = 1 - (1 - target_ratio) ** (1 / num_iterations)
            
            for iteration in range(num_iterations):
                logger.info(f"剪枝迭代 {iteration + 1}/{num_iterations}")
                
                # 剪枝
                self.config.pruning_ratio = ratio_per_iter
                pruned_model = self.prune_model(pruned_model)
                
                # 微调
                pruned_model.train()
                for epoch in range(self.config.num_epochs):
                    for batch in train_loader:
                        if isinstance(batch, (tuple, list)):
                            x, y = batch
                        else:
                            x = batch
                            y = None
                        
                        optimizer.zero_grad()
                        output = pruned_model(x)
                        
                        if y is not None:
                            loss = criterion(output, y)
                        else:
                            loss = criterion(output)
                        
                        loss.backward()
                        optimizer.step()
            
            return pruned_model


    # =========================================================================
    # 知识蒸馏
    # =========================================================================
    class KnowledgeDistillation:
        """知识蒸馏"""
        
        def __init__(self, config: CompressionConfig):
            self.config = config
        
        def distillation_loss(self, student_logits: torch.Tensor,
                             teacher_logits: torch.Tensor,
                             labels: torch.Tensor,
                             temperature: float = 4.0,
                             alpha: float = 0.5) -> torch.Tensor:
            """
            计算蒸馏损失
            
            Args:
                student_logits: 学生模型输出
                teacher_logits: 教师模型输出
                labels: 真实标签
                temperature: 温度参数
                alpha: 软标签损失权重
            
            Returns:
                总损失
            """
            # 硬标签损失
            hard_loss = F.cross_entropy(student_logits, labels)
            
            # 软标签损失 (KL散度)
            soft_student = F.log_softmax(student_logits / temperature, dim=-1)
            soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            soft_loss = soft_loss * (temperature ** 2)
            
            # 组合损失
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            return loss
        
        def train_student(self, teacher: nn.Module, student: nn.Module,
                         train_loader: DataLoader, optimizer,
                         num_epochs: int = 10) -> nn.Module:
            """
            训练学生模型
            
            Args:
                teacher: 教师模型
                student: 学生模型
                train_loader: 训练数据
                optimizer: 优化器
                num_epochs: 训练轮数
            
            Returns:
                训练后的学生模型
            """
            teacher.eval()
            
            for epoch in range(num_epochs):
                student.train()
                total_loss = 0
                
                for batch in train_loader:
                    if isinstance(batch, (tuple, list)):
                        x, y = batch
                    else:
                        x = batch
                        y = torch.zeros(x.size(0), dtype=torch.long)
                    
                    # 教师预测
                    with torch.no_grad():
                        teacher_logits = teacher(x)
                    
                    # 学生预测
                    student_logits = student(x)
                    
                    # 计算损失
                    loss = self.distillation_loss(
                        student_logits, teacher_logits, y,
                        self.config.temperature, self.config.alpha
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
            
            return student


    class FeatureDistillation:
        """特征蒸馏"""
        
        def __init__(self, config: CompressionConfig):
            self.config = config
            self.teacher_features = {}
            self.student_features = {}
        
        def _get_hook(self, features_dict: Dict, name: str) -> Callable:
            """创建钩子函数"""
            def hook(module, input, output):
                features_dict[name] = output
            return hook
        
        def register_hooks(self, teacher: nn.Module, student: nn.Module,
                          layer_pairs: List[Tuple[str, str]]):
            """
            注册特征提取钩子
            
            Args:
                teacher: 教师模型
                student: 学生模型
                layer_pairs: [(teacher_layer, student_layer), ...]
            """
            self.hooks = []
            
            for t_name, s_name in layer_pairs:
                # 教师钩子
                t_module = dict(teacher.named_modules())[t_name]
                t_hook = t_module.register_forward_hook(
                    self._get_hook(self.teacher_features, t_name)
                )
                self.hooks.append(t_hook)
                
                # 学生钩子
                s_module = dict(student.named_modules())[s_name]
                s_hook = s_module.register_forward_hook(
                    self._get_hook(self.student_features, s_name)
                )
                self.hooks.append(s_hook)
        
        def feature_loss(self, layer_pairs: List[Tuple[str, str]]) -> torch.Tensor:
            """计算特征蒸馏损失"""
            loss = 0
            
            for t_name, s_name in layer_pairs:
                t_feat = self.teacher_features.get(t_name)
                s_feat = self.student_features.get(s_name)
                
                if t_feat is not None and s_feat is not None:
                    # 特征对齐
                    if t_feat.shape != s_feat.shape:
                        # 使用投影层对齐
                        s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
                    
                    loss += F.mse_loss(s_feat, t_feat.detach())
            
            return loss
        
        def remove_hooks(self):
            """移除钩子"""
            for hook in self.hooks:
                hook.remove()


    # =========================================================================
    # 综合压缩流水线
    # =========================================================================
    class CompressionPipeline:
        """模型压缩流水线"""
        
        def __init__(self, config: CompressionConfig):
            self.config = config
            self.ptq = PostTrainingQuantization(config)
            self.pruner = Pruner(config)
            self.distiller = KnowledgeDistillation(config)
        
        def compress(self, model: nn.Module, train_loader: DataLoader,
                    val_loader: Optional[DataLoader] = None,
                    steps: List[str] = ["prune", "quantize"]) -> nn.Module:
            """
            执行模型压缩
            
            Args:
                model: 原始模型
                train_loader: 训练数据
                val_loader: 验证数据
                steps: 压缩步骤 ["prune", "quantize", "distill"]
            
            Returns:
                压缩后的模型
            """
            compressed = copy.deepcopy(model)
            
            for step in steps:
                logger.info(f"执行压缩步骤: {step}")
                
                if step == "prune":
                    compressed = self.pruner.prune_model(compressed)
                
                elif step == "quantize":
                    compressed = self.ptq.quantize_model(compressed, train_loader)
                
                elif step == "distill":
                    # 创建小型学生模型
                    student = self._create_student_model(model)
                    optimizer = torch.optim.Adam(
                        student.parameters(), lr=self.config.learning_rate
                    )
                    compressed = self.distiller.train_student(
                        compressed, student, train_loader, optimizer,
                        self.config.num_epochs
                    )
            
            # 评估压缩效果
            self._evaluate_compression(model, compressed, val_loader)
            
            return compressed
        
        def _create_student_model(self, teacher: nn.Module) -> nn.Module:
            """创建学生模型 (简化版)"""
            # 这里简单地复制教师模型结构
            # 实际应用中应该设计更小的学生模型
            return copy.deepcopy(teacher)
        
        def _evaluate_compression(self, original: nn.Module,
                                 compressed: nn.Module,
                                 val_loader: Optional[DataLoader]):
            """评估压缩效果"""
            # 计算参数量
            orig_params = sum(p.numel() for p in original.parameters())
            comp_params = sum(p.numel() for p in compressed.parameters())
            
            # 计算模型大小
            orig_size = sum(p.numel() * p.element_size() for p in original.parameters())
            comp_size = sum(p.numel() * p.element_size() for p in compressed.parameters())
            
            logger.info(f"原始模型参数量: {orig_params:,}")
            logger.info(f"压缩模型参数量: {comp_params:,}")
            logger.info(f"参数压缩比: {orig_params / comp_params:.2f}x")
            logger.info(f"原始模型大小: {orig_size / 1024 / 1024:.2f} MB")
            logger.info(f"压缩模型大小: {comp_size / 1024 / 1024:.2f} MB")
            logger.info(f"大小压缩比: {orig_size / comp_size:.2f}x")
            
            # 评估精度 (如果有验证数据)
            if val_loader is not None:
                original.eval()
                compressed.eval()
                
                orig_correct = 0
                comp_correct = 0
                total = 0
                
                for batch in val_loader:
                    if isinstance(batch, (tuple, list)):
                        x, y = batch
                    else:
                        continue
                    
                    with torch.no_grad():
                        orig_pred = original(x).argmax(dim=1)
                        comp_pred = compressed(x).argmax(dim=1)
                    
                    orig_correct += (orig_pred == y).sum().item()
                    comp_correct += (comp_pred == y).sum().item()
                    total += y.size(0)
                
                logger.info(f"原始模型准确率: {orig_correct / total:.4f}")
                logger.info(f"压缩模型准确率: {comp_correct / total:.4f}")


# =============================================================================
# 入口函数
# =============================================================================
def compress_model(model: nn.Module, train_loader: DataLoader,
                  config: Optional[CompressionConfig] = None,
                  steps: List[str] = ["prune", "quantize"]) -> nn.Module:
    """
    压缩模型的便捷函数
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return model
    
    if config is None:
        config = CompressionConfig()
    
    pipeline = CompressionPipeline(config)
    return pipeline.compress(model, train_loader, steps=steps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if TORCH_AVAILABLE:
        # 示例
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 10)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        
        model = SimpleModel()
        
        # 创建虚拟数据
        train_data = [(torch.randn(32, 784), torch.randint(0, 10, (32,))) 
                     for _ in range(100)]
        train_loader = DataLoader(train_data, batch_size=32)
        
        # 压缩模型
        config = CompressionConfig(
            pruning_ratio=0.3,
            quantization_bits=8
        )
        
        compressed = compress_model(model, train_loader, config, 
                                   steps=["prune", "quantize"])
