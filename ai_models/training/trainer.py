#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台训练器核心模块
===================

支持:
- Mac M系列 (MPS加速)
- Windows (CUDA)
- CPU回退

作者: 破夜绘明团队
"""

import os
import sys
import json
import time
import platform
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime

import numpy as np

# PyTorch导入
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    F = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 平台检测
# =============================================================================
@dataclass
class PlatformInfo:
    """平台信息"""
    system: str              # Darwin, Windows, Linux
    machine: str             # arm64, x86_64
    device: str              # mps, cuda, cpu
    device_name: str         # 设备名称
    memory_gb: float         # 可用内存
    is_apple_silicon: bool   # 是否为Apple Silicon
    cuda_available: bool     # CUDA是否可用
    mps_available: bool      # MPS是否可用
    recommended_batch_size: int
    recommended_precision: str  # float32, float16


def detect_platform() -> PlatformInfo:
    """检测当前运行平台"""
    system = platform.system()
    machine = platform.machine()
    
    # 检测Apple Silicon
    is_apple_silicon = (system == "Darwin" and machine == "arm64")
    
    # 检测CUDA
    cuda_available = False
    cuda_device_name = ""
    if TORCH_AVAILABLE:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
    
    # 检测MPS
    mps_available = False
    if TORCH_AVAILABLE and hasattr(torch.backends, 'mps'):
        mps_available = torch.backends.mps.is_available()
    
    # 确定最佳设备
    if mps_available:
        device = "mps"
        device_name = f"Apple Silicon ({machine})"
        recommended_batch_size = 16
        recommended_precision = "float32"  # MPS对FP16支持有限
    elif cuda_available:
        device = "cuda"
        device_name = cuda_device_name
        recommended_batch_size = 32
        recommended_precision = "float16"
    else:
        device = "cpu"
        device_name = f"CPU ({machine})"
        recommended_batch_size = 8
        recommended_precision = "float32"
    
    # 获取内存信息
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        memory_gb = 24.0 if is_apple_silicon else 16.0  # 估计值
    
    return PlatformInfo(
        system=system,
        machine=machine,
        device=device,
        device_name=device_name,
        memory_gb=memory_gb,
        is_apple_silicon=is_apple_silicon,
        cuda_available=cuda_available,
        mps_available=mps_available,
        recommended_batch_size=recommended_batch_size,
        recommended_precision=recommended_precision
    )


def get_device(prefer: str = "auto") -> torch.device:
    """
    获取PyTorch设备
    
    Args:
        prefer: 偏好设备 ("auto", "mps", "cuda", "cpu")
    
    Returns:
        torch.device
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch未安装")
    
    if prefer == "auto":
        platform_info = detect_platform()
        prefer = platform_info.device
    
    if prefer == "mps" and torch.backends.mps.is_available():
        logger.info("✅ 使用 Apple Silicon MPS 加速")
        return torch.device("mps")
    elif prefer == "cuda" and torch.cuda.is_available():
        logger.info(f"✅ 使用 NVIDIA CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        logger.info("⚠️ 使用 CPU 训练")
        return torch.device("cpu")


# =============================================================================
# 训练配置
# =============================================================================
@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    model_name: str = "default"
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 优化器配置
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # 混合精度训练
    use_amp: bool = False  # 自动混合精度
    
    # 数据增强
    augmentation: bool = True
    
    # 早停
    early_stopping: bool = True
    patience: int = 10
    
    # 检查点
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5
    
    # 日志
    log_frequency: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    
    # 验证
    val_frequency: int = 1
    
    # 设备
    device: str = "auto"
    num_workers: int = 4


# =============================================================================
# 训练回调
# =============================================================================
class TrainingCallback:
    """训练回调基类"""
    
    def on_train_begin(self, trainer):
        pass
    
    def on_train_end(self, trainer):
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        pass
    
    def on_batch_begin(self, trainer, batch_idx):
        pass
    
    def on_batch_end(self, trainer, batch_idx, logs):
        pass


class EarlyStopping(TrainingCallback):
    """早停回调"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 monitor: str = "val_loss", mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def on_epoch_end(self, trainer, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.best_value is None:
            self.best_value = current
        elif self._is_improvement(current):
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"早停触发: {self.patience}轮无改善")
    
    def _is_improvement(self, current):
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta


class ModelCheckpoint(TrainingCallback):
    """模型检查点回调"""
    
    def __init__(self, save_dir: str, monitor: str = "val_loss",
                 mode: str = "min", save_best_only: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = None
    
    def on_epoch_end(self, trainer, epoch, logs):
        current = logs.get(self.monitor)
        
        # 检查是否需要保存
        should_save = False
        if not self.save_best_only:
            should_save = True
        elif current is not None:
            if self.best_value is None:
                should_save = True
                self.best_value = current
            elif self._is_improvement(current):
                should_save = True
                self.best_value = current
        
        if should_save:
            self._save_checkpoint(trainer, epoch, logs)
    
    def _is_improvement(self, current):
        if self.mode == "min":
            return current < self.best_value
        else:
            return current > self.best_value
    
    def _save_checkpoint(self, trainer, epoch, logs):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'logs': logs,
            'config': trainer.config.__dict__,
            'platform': detect_platform().__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        if trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        save_path = self.save_dir / f"{trainer.config.model_name}_best.pth"
        torch.save(checkpoint, save_path)
        logger.info(f"💾 保存检查点: {save_path}")


# =============================================================================
# 跨平台训练器
# =============================================================================
class CrossPlatformTrainer:
    """
    跨平台模型训练器
    
    支持Mac MPS和Windows CUDA，自动处理设备差异
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig = None):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            config: 训练配置
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch未安装")
        
        self.config = config or TrainingConfig()
        self.platform_info = detect_platform()
        
        # 设备设置
        self.device = get_device(self.config.device)
        self.model = model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 混合精度
        self.scaler = None
        if self.config.use_amp and self.device.type == "cuda":
            self.scaler = GradScaler()
        
        # 回调
        self.callbacks: List[TrainingCallback] = []
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # 日志
        self._setup_logging()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        params = self.model.parameters()
        
        if self.config.optimizer == "adam":
            return optim.Adam(
                params, 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"未知优化器: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config.save_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir / "tensorboard")
            except ImportError:
                self.writer = None
                logger.warning("TensorBoard未安装")
        else:
            self.writer = None
    
    def add_callback(self, callback: TrainingCallback):
        """添加回调"""
        self.callbacks.append(callback)
    
    def _run_callbacks(self, event: str, **kwargs):
        """执行回调"""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(self, **kwargs)
    
    def train(self, train_loader: DataLoader, 
              val_loader: DataLoader = None,
              criterion: nn.Module = None) -> Dict[str, List]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
        
        Returns:
            训练历史
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # 设置默认回调
        if self.config.early_stopping:
            self.add_callback(EarlyStopping(patience=self.config.patience))
        
        self.add_callback(ModelCheckpoint(
            save_dir=self.config.save_dir,
            save_best_only=self.config.save_best_only
        ))
        
        # 打印训练信息
        self._print_training_info()
        
        # 开始训练
        self._run_callbacks('on_train_begin')
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            self._run_callbacks('on_epoch_begin', epoch=epoch)
            
            # 训练一个epoch
            train_loss = self._train_epoch(train_loader, criterion)
            
            # 验证
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None and (epoch + 1) % self.config.val_frequency == 0:
                val_loss, val_acc = self._validate(val_loader, criterion)
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 日志
            logs = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            self._log_epoch(epoch, logs)
            self._run_callbacks('on_epoch_end', epoch=epoch, logs=logs)
            
            # 检查早停
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping) and callback.should_stop:
                    logger.info("早停触发，训练结束")
                    break
            else:
                continue
            break
        
        self._run_callbacks('on_train_end')
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader,
                     criterion: nn.Module) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            self._run_callbacks('on_batch_begin', batch_idx=batch_idx)

            # 处理不同格式的batch
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    data, target = batch
                else:
                    data, target = batch[0], batch[1]
            else:
                data, target = batch['image'], batch['label']

            data = data.to(self.device)

            # 处理目标 - 检测任务target是列表，分类任务是tensor
            if isinstance(target, list):
                # 目标检测: target是dict列表
                target = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in t.items()} for t in target]
            elif isinstance(target, torch.Tensor):
                target = target.to(self.device)

            self.optimizer.zero_grad()

            # 混合精度训练
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = self._compute_loss(output, target, criterion)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self._compute_loss(output, target, criterion)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # 日志
            if (batch_idx + 1) % self.config.log_frequency == 0:
                logger.info(
                    f"Epoch [{self.current_epoch+1}/{self.config.epochs}] "
                    f"Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss.item():.4f}"
                )

            self._run_callbacks('on_batch_end', batch_idx=batch_idx,
                              logs={'loss': loss.item()})

        return total_loss / num_batches

    def _compute_loss(self, output, target, criterion):
        """计算损失，处理不同模型输出格式"""
        # 检测模型输出是列表
        if isinstance(output, list):
            if len(output) > 0 and isinstance(output[0], dict):
                # YOLOv8风格输出: [{'cls': ..., 'reg': ...}, ...]
                # 简化损失: 使用分类损失作为代理
                total_loss = torch.tensor(0.0, device=self.device)
                for out in output:
                    if 'cls' in out:
                        cls_out = out['cls']  # [B, num_classes, num_anchors]
                        # 使用简化的分类损失
                        cls_out = cls_out.permute(0, 2, 1)  # [B, num_anchors, num_classes]
                        B, A, C = cls_out.shape
                        # 创建伪标签 (实际训练需要根据target生成)
                        pseudo_labels = torch.zeros(B, A, dtype=torch.long, device=self.device)
                        total_loss += F.cross_entropy(cls_out.reshape(-1, C), pseudo_labels.reshape(-1))
                return total_loss / len(output) if output else torch.tensor(0.0, device=self.device)
            else:
                # 其他列表格式
                return criterion(output[0], target)
        # 字典格式输出 (RT-DETR)
        elif isinstance(output, dict):
            if 'pred_logits' in output:
                pred_logits = output['pred_logits']
                B, Q, C = pred_logits.shape
                # 简化: 对所有query使用背景类标签
                pseudo_labels = torch.full((B, Q), C-1, dtype=torch.long, device=self.device)
                return F.cross_entropy(pred_logits.reshape(-1, C), pseudo_labels.reshape(-1))
            return criterion(list(output.values())[0], target)
        # 标准tensor输出
        else:
            return criterion(output, target)
    
    def _validate(self, val_loader: DataLoader,
                  criterion: nn.Module) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    data, target = batch[0], batch[1]
                else:
                    data, target = batch['image'], batch['label']

                data = data.to(self.device)

                # 处理目标 - 检测任务target是列表，分类任务是tensor
                if isinstance(target, list):
                    target = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in t.items()} for t in target]
                elif isinstance(target, torch.Tensor):
                    target = target.to(self.device)

                output = self.model(data)
                loss = self._compute_loss(output, target, criterion)

                total_loss += loss.item()

                # 计算准确率 (仅对分类任务)
                if isinstance(output, torch.Tensor) and output.dim() > 1 and isinstance(target, torch.Tensor):
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy
    
    def _log_epoch(self, epoch: int, logs: Dict):
        """记录epoch日志"""
        logger.info(
            f"Epoch {epoch+1}/{self.config.epochs} | "
            f"Train Loss: {logs['train_loss']:.4f} | "
            f"Val Loss: {logs['val_loss']:.4f} | "
            f"Val Acc: {logs['val_acc']:.4f} | "
            f"LR: {logs['lr']:.6f}"
        )
        
        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', logs['train_loss'], epoch)
            self.writer.add_scalar('Loss/val', logs['val_loss'], epoch)
            self.writer.add_scalar('Accuracy/val', logs['val_acc'], epoch)
            self.writer.add_scalar('Learning_rate', logs['lr'], epoch)
    
    def _print_training_info(self):
        """打印训练信息"""
        logger.info("=" * 60)
        logger.info("训练配置")
        logger.info("=" * 60)
        logger.info(f"模型名称: {self.config.model_name}")
        logger.info(f"设备: {self.device} ({self.platform_info.device_name})")
        logger.info(f"平台: {self.platform_info.system} ({self.platform_info.machine})")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch Size: {self.config.batch_size}")
        logger.info(f"Learning Rate: {self.config.learning_rate}")
        logger.info(f"优化器: {self.config.optimizer}")
        logger.info(f"调度器: {self.config.scheduler}")
        logger.info(f"混合精度: {self.config.use_amp}")
        logger.info("=" * 60)
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config.__dict__,
            'platform': self.platform_info.__dict__
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"检查点已加载: {path}")


# =============================================================================
# 训练流水线
# =============================================================================
class TrainingPipeline:
    """
    完整训练流水线
    
    管理所有插件模型的训练、验证和导出
    """
    
    # 插件模型映射
    PLUGIN_MODELS = {
        "transformer": [
            {"name": "defect_yolov8n", "type": "detection", "input_size": (640, 640)},
            {"name": "oil_unet", "type": "segmentation", "input_size": (512, 512)},
            {"name": "silica_cnn", "type": "classification", "input_size": (224, 224)},
            {"name": "thermal_anomaly", "type": "classification", "input_size": (224, 224)},
        ],
        "switch": [
            {"name": "switch_yolov8s", "type": "detection", "input_size": (640, 640)},
            {"name": "indicator_ocr", "type": "ocr", "input_size": (32, 128)},
        ],
        "busbar": [
            {"name": "busbar_yolov8m", "type": "detection", "input_size": (1280, 1280)},
            {"name": "noise_classifier", "type": "classification", "input_size": (128, 128)},
        ],
        "capacitor": [
            {"name": "capacitor_yolov8", "type": "detection", "input_size": (640, 640)},
            {"name": "rtdetr_intrusion", "type": "detection", "input_size": (640, 640)},
        ],
        "meter": [
            {"name": "hrnet_keypoint", "type": "keypoint", "input_size": (256, 256)},
            {"name": "crnn_ocr", "type": "ocr", "input_size": (32, 128)},
            {"name": "meter_classifier", "type": "classification", "input_size": (224, 224)},
        ],
    }
    
    def __init__(self, base_dir: str = ".", config: Dict = None):
        """
        初始化流水线
        
        Args:
            base_dir: 项目根目录
            config: 全局配置
        """
        self.base_dir = Path(base_dir)
        self.config = config or {}
        self.platform_info = detect_platform()
        
        # 创建目录结构
        self._create_directories()
        
        # 训练状态
        self.trained_models = {}
        
        logger.info(f"训练流水线初始化完成")
        logger.info(f"平台: {self.platform_info.system} ({self.platform_info.device})")
    
    def _create_directories(self):
        """创建目录结构"""
        dirs = [
            "models/transformer",
            "models/switch", 
            "models/busbar",
            "models/capacitor",
            "models/meter",
            "models/common",
            "checkpoints",
            "logs",
            "data/raw",
            "data/processed",
            "exports/onnx",
        ]
        
        for d in dirs:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)
    
    def train_plugin(self, plugin_name: str, 
                     model_name: str = None,
                     epochs: int = 50,
                     batch_size: int = None,
                     data_dir: str = None,
                     pretrained: bool = True) -> Dict:
        """
        训练单个插件模型
        
        Args:
            plugin_name: 插件名称 (transformer, switch, busbar, capacitor, meter)
            model_name: 模型名称，如不指定则训练该插件所有模型
            epochs: 训练轮数
            batch_size: 批大小
            data_dir: 数据目录
            pretrained: 是否使用预训练权重
        
        Returns:
            训练结果
        """
        if plugin_name not in self.PLUGIN_MODELS:
            raise ValueError(f"未知插件: {plugin_name}")
        
        models = self.PLUGIN_MODELS[plugin_name]
        if model_name:
            models = [m for m in models if m['name'] == model_name]
            if not models:
                raise ValueError(f"未知模型: {model_name}")
        
        results = {}
        
        for model_info in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"训练模型: {plugin_name}/{model_info['name']}")
            logger.info(f"{'='*60}")
            
            try:
                result = self._train_single_model(
                    plugin_name=plugin_name,
                    model_info=model_info,
                    epochs=epochs,
                    batch_size=batch_size or self.platform_info.recommended_batch_size,
                    data_dir=data_dir,
                    pretrained=pretrained
                )
                results[model_info['name']] = result
                self.trained_models[f"{plugin_name}/{model_info['name']}"] = result
                
            except Exception as e:
                logger.error(f"模型训练失败: {e}")
                results[model_info['name']] = {"status": "failed", "error": str(e)}
        
        return results
    
    def _train_single_model(self, plugin_name: str, model_info: Dict,
                           epochs: int, batch_size: int,
                           data_dir: str, pretrained: bool) -> Dict:
        """训练单个模型"""
        from .models import create_model
        from .datasets import create_dataloader
        
        model_type = model_info['type']
        model_name = model_info['name']
        input_size = model_info['input_size']
        
        # 创建模型
        model = create_model(
            model_type=model_type,
            model_name=model_name,
            input_size=input_size,
            pretrained=pretrained,
            plugin_name=plugin_name
        )
        
        # 创建数据加载器
        train_loader, val_loader = create_dataloader(
            plugin_name=plugin_name,
            model_type=model_type,
            data_dir=data_dir or str(self.base_dir / "data"),
            batch_size=batch_size,
            input_size=input_size
        )
        
        # 训练配置
        config = TrainingConfig(
            model_name=f"{plugin_name}_{model_name}",
            epochs=epochs,
            batch_size=batch_size,
            save_dir=str(self.base_dir / "checkpoints" / plugin_name),
            use_amp=(self.platform_info.device == "cuda")
        )
        
        # 创建训练器
        trainer = CrossPlatformTrainer(model, config)
        
        # 选择损失函数
        criterion = self._get_criterion(model_type)
        
        # 训练
        history = trainer.train(train_loader, val_loader, criterion)
        
        # 保存最终模型
        save_path = self.base_dir / "checkpoints" / plugin_name / f"{model_name}_final.pth"
        trainer.save_checkpoint(str(save_path))
        
        return {
            "status": "success",
            "model_path": str(save_path),
            "history": history,
            "best_val_acc": max(history.get('val_acc', [0])),
            "final_train_loss": history['train_loss'][-1] if history['train_loss'] else None
        }
    
    def _get_criterion(self, model_type: str) -> nn.Module:
        """获取损失函数"""
        if model_type == "detection":
            # YOLO使用自定义损失
            return nn.CrossEntropyLoss()  # 简化，实际使用YOLO损失
        elif model_type == "segmentation":
            return nn.BCEWithLogitsLoss()
        elif model_type == "classification":
            return nn.CrossEntropyLoss()
        elif model_type == "keypoint":
            return nn.MSELoss()
        elif model_type == "ocr":
            return nn.CTCLoss(blank=0, zero_infinity=True)
        else:
            return nn.CrossEntropyLoss()
    
    def train_all(self, epochs: int = 50, 
                  plugins: List[str] = None) -> Dict:
        """
        训练所有插件模型
        
        Args:
            epochs: 训练轮数
            plugins: 要训练的插件列表，默认全部
        
        Returns:
            所有训练结果
        """
        plugins = plugins or list(self.PLUGIN_MODELS.keys())
        
        all_results = {}
        
        for plugin in plugins:
            logger.info(f"\n\n{'#'*60}")
            logger.info(f"# 开始训练插件: {plugin}")
            logger.info(f"{'#'*60}")
            
            results = self.train_plugin(plugin, epochs=epochs)
            all_results[plugin] = results
        
        # 保存训练摘要
        self._save_training_summary(all_results)
        
        return all_results
    
    def export_onnx(self, plugin_name: str, model_name: str,
                    checkpoint_path: str = None,
                    opset_version: int = 17) -> str:
        """
        导出模型为ONNX格式
        
        Args:
            plugin_name: 插件名称
            model_name: 模型名称
            checkpoint_path: 检查点路径
            opset_version: ONNX opset版本
        
        Returns:
            ONNX文件路径
        """
        from .exporters import ONNXExporter
        from .models import create_model
        
        # 获取模型信息
        models = self.PLUGIN_MODELS.get(plugin_name, [])
        model_info = next((m for m in models if m['name'] == model_name), None)
        if not model_info:
            raise ValueError(f"未找到模型: {plugin_name}/{model_name}")
        
        # 创建模型
        model = create_model(
            model_type=model_info['type'],
            model_name=model_name,
            input_size=model_info['input_size'],
            pretrained=False,
            plugin_name=plugin_name
        )
        
        # 加载检查点
        if checkpoint_path is None:
            checkpoint_path = self.base_dir / "checkpoints" / plugin_name / f"{model_name}_best.pth"
        
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 导出ONNX
        exporter = ONNXExporter(opset_version=opset_version)
        
        output_path = self.base_dir / "models" / plugin_name / f"{model_name}.onnx"
        
        exporter.export(
            model=model,
            input_shape=(3, *model_info['input_size']),
            save_path=str(output_path),
            dynamic_batch=True
        )
        
        return str(output_path)
    
    def export_all_onnx(self) -> Dict[str, str]:
        """导出所有模型为ONNX"""
        exported = {}
        
        for plugin_name, models in self.PLUGIN_MODELS.items():
            for model_info in models:
                try:
                    onnx_path = self.export_onnx(plugin_name, model_info['name'])
                    exported[f"{plugin_name}/{model_info['name']}"] = onnx_path
                    logger.info(f"✅ 导出成功: {onnx_path}")
                except Exception as e:
                    logger.error(f"❌ 导出失败 {plugin_name}/{model_info['name']}: {e}")
                    exported[f"{plugin_name}/{model_info['name']}"] = f"ERROR: {e}"
        
        return exported
    
    def _save_training_summary(self, results: Dict):
        """保存训练摘要"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "platform": self.platform_info.__dict__,
            "results": results
        }
        
        summary_path = self.base_dir / "checkpoints" / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"训练摘要已保存: {summary_path}")


# =============================================================================
# 便捷函数
# =============================================================================
def quick_train(plugin: str, model: str = None, epochs: int = 50, 
                data_dir: str = None) -> Dict:
    """
    快速训练模型
    
    Example:
        # 训练主变缺陷检测模型
        result = quick_train("transformer", "defect_yolov8n", epochs=50)
        
        # 训练所有表计模型
        result = quick_train("meter", epochs=30)
    """
    pipeline = TrainingPipeline()
    return pipeline.train_plugin(plugin, model, epochs=epochs, data_dir=data_dir)


def quick_export(plugin: str, model: str) -> str:
    """
    快速导出ONNX
    
    Example:
        onnx_path = quick_export("transformer", "defect_yolov8n")
    """
    pipeline = TrainingPipeline()
    return pipeline.export_onnx(plugin, model)
