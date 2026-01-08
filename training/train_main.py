#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 模型训练主入口
==============================================

目录结构:
    training/
    ├── train_main.py          # 本文件
    ├── train_mac.sh           # Mac训练脚本
    ├── configs/               # 训练配置
    ├── logs/                  # 训练日志
    ├── checkpoints/           # 检查点文件
    │   ├── transformer/
    │   ├── switch/
    │   ├── busbar/
    │   ├── capacitor/
    │   └── meter/
    ├── exports/               # ONNX临时导出
    ├── data/                  # 训练数据
    └── results/               # 训练结果

使用方法:
    # 从项目根目录运行
    python training/train_main.py --mode demo

    # 训练单个插件
    python training/train_main.py --mode plugin --plugin transformer --epochs 30

    # 训练所有模型
    python training/train_main.py --mode all --epochs 50

作者: 破夜绘明团队
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# 获取脚本所在目录和项目根目录
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# 确保日志目录存在
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志 - 保存到 training/logs/
log_filename = LOG_DIR / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_filename), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# PyTorch导入
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    logger.info(f"✅ PyTorch {torch.__version__} 已加载")
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = object
    DataLoader = None
    logger.error(f"❌ PyTorch未安装: {e}")

# ONNX导入
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


# =============================================================================
# 平台检测
# =============================================================================
def detect_platform():
    """检测运行平台"""
    import platform
    
    system = platform.system()
    machine = platform.machine()
    
    info = {
        "system": system,
        "machine": machine,
        "device": "cpu",
        "device_name": "CPU"
    }
    
    if TORCH_AVAILABLE:
        # 检测MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["device"] = "mps"
            info["device_name"] = f"Apple Silicon MPS ({machine})"
        # 检测CUDA
        elif torch.cuda.is_available():
            info["device"] = "cuda"
            info["device_name"] = torch.cuda.get_device_name(0)
    
    return info


def get_device():
    """获取最佳设备"""
    if not TORCH_AVAILABLE:
        return None
    
    platform_info = detect_platform()
    device_type = platform_info["device"]
    
    if device_type == "mps":
        logger.info(f"✅ 使用 Apple Silicon MPS 加速")
        return torch.device("mps")
    elif device_type == "cuda":
        logger.info(f"✅ 使用 NVIDIA CUDA: {platform_info['device_name']}")
        return torch.device("cuda")
    else:
        logger.info("⚠️ 使用 CPU 训练")
        return torch.device("cpu")


# =============================================================================
# 模型定义 (简化版，确保能运行)
# =============================================================================
if TORCH_AVAILABLE:
    class SimpleDetectionModel(nn.Module):
        """简化的检测模型"""
        def __init__(self, num_classes=10, input_size=(640, 640)):
            super().__init__()
            self.num_classes = num_classes

            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )

            self.head = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            x = self.head(x)
            return x


    class SimpleSegmentationModel(nn.Module):
        """简化的分割模型"""
        def __init__(self, num_classes=2):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_classes, 1)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


    class SimpleClassificationModel(nn.Module):
        """简化的分类模型"""
        def __init__(self, num_classes=4):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x


# =============================================================================
# 数据集 (修复版)
# =============================================================================
if TORCH_AVAILABLE:
    class SimulatedDataset(Dataset):
        """模拟数据集 - 确保生成有效数据"""

        def __init__(self, num_samples=500, input_size=(224, 224),
                     num_classes=10, task="classification"):
            self.num_samples = num_samples
            self.input_size = input_size
            self.num_classes = num_classes
            self.task = task

            # 预生成所有数据以确保一致性
            logger.info(f"生成模拟数据集: {num_samples} 样本, {task}任务, {num_classes}类")

            self.images = []
            self.labels = []

            for i in range(num_samples):
                # 生成带模式的图像 (不是纯随机噪声)
                img = self._generate_patterned_image(i)
                label = i % num_classes

                self.images.append(img)
                self.labels.append(label)

            logger.info(f"✅ 数据集生成完成")

        def _generate_patterned_image(self, seed):
            """生成带模式的图像"""
            np.random.seed(seed)

            # 基础图像
            img = np.random.randint(50, 200, (*self.input_size, 3), dtype=np.uint8)

            # 添加一些模式使其有区分度
            class_id = seed % self.num_classes

            # 根据类别添加不同的形状
            h, w = self.input_size
            center_x, center_y = w // 2, h // 2
            radius = min(h, w) // 4

            # 简单的圆形/方形模式
            for y in range(h):
                for x in range(w):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius * (0.5 + 0.1 * class_id):
                        img[y, x] = [50 + class_id * 20, 100, 150]

            # 归一化到 [0, 1]
            img = img.astype(np.float32) / 255.0

            # 标准化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std

            # 转换为 CHW 格式
            img = img.transpose(2, 0, 1)

            return img.astype(np.float32)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = torch.from_numpy(self.images[idx])
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label


# =============================================================================
# 训练器 (修复版)
# =============================================================================
class Trainer:
    """简化的训练器"""
    
    def __init__(self, model, device, save_dir):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs, model_name):
        """完整训练流程"""
        best_acc = 0
        
        logger.info(f"开始训练: {model_name}")
        logger.info(f"设备: {self.device}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"保存目录: {self.save_dir}")
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            )
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(model_name, epoch, val_acc, is_best=True)
        
        # 保存最终模型
        self.save_checkpoint(model_name, epochs, val_acc, is_best=False)
        
        logger.info(f"✅ 训练完成! 最佳准确率: {best_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, model_name, epoch, accuracy, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'history': self.history,
        }
        
        suffix = "best" if is_best else "final"
        save_path = self.save_dir / f"{model_name}_{suffix}.pth"
        
        torch.save(checkpoint, save_path)
        
        # 验证文件已保存
        if save_path.exists():
            size_kb = save_path.stat().st_size / 1024
            logger.info(f"💾 保存检查点: {save_path} ({size_kb:.1f} KB)")
        else:
            logger.error(f"❌ 保存失败: {save_path}")


# =============================================================================
# ONNX导出器 (修复版)
# =============================================================================
class ONNXExporter:
    """ONNX导出器"""
    
    @staticmethod
    def export(model, input_size, save_path, model_name):
        """导出模型为ONNX"""
        logger.info(f"导出ONNX: {model_name}")
        
        model.eval()
        model = model.cpu()
        
        # 创建目录
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Dummy输入
        dummy_input = torch.randn(1, 3, *input_size)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                opset_version=17,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证文件
            if Path(save_path).exists():
                size_kb = Path(save_path).stat().st_size / 1024
                logger.info(f"✅ ONNX导出成功: {save_path} ({size_kb:.1f} KB)")
                
                # 验证ONNX模型
                if ONNX_AVAILABLE:
                    onnx_model = onnx.load(save_path)
                    onnx.checker.check_model(onnx_model)
                    logger.info(f"✅ ONNX模型验证通过")
                
                return True
            else:
                logger.error(f"❌ ONNX文件未创建")
                return False
                
        except Exception as e:
            logger.error(f"❌ ONNX导出失败: {e}")
            return False


# =============================================================================
# 模型配置
# =============================================================================
PLUGIN_CONFIGS = {
    "transformer": {
        "description": "A组 - 主变巡视",
        "models": [
            {"name": "defect_yolov8n", "type": "detection", "input_size": (640, 640), "num_classes": 6},
            {"name": "oil_unet", "type": "segmentation", "input_size": (512, 512), "num_classes": 2},
            {"name": "silica_cnn", "type": "classification", "input_size": (224, 224), "num_classes": 4},
            {"name": "thermal_anomaly", "type": "classification", "input_size": (224, 224), "num_classes": 3},
        ]
    },
    "switch": {
        "description": "B组 - 开关间隔",
        "models": [
            {"name": "switch_yolov8s", "type": "detection", "input_size": (640, 640), "num_classes": 8},
            {"name": "indicator_ocr", "type": "classification", "input_size": (32, 128), "num_classes": 50},
        ]
    },
    "busbar": {
        "description": "C组 - 母线巡视",
        "models": [
            {"name": "busbar_yolov8m", "type": "detection", "input_size": (640, 640), "num_classes": 8},
            {"name": "noise_classifier", "type": "classification", "input_size": (128, 128), "num_classes": 5},
        ]
    },
    "capacitor": {
        "description": "D组 - 电容器",
        "models": [
            {"name": "capacitor_yolov8", "type": "detection", "input_size": (640, 640), "num_classes": 6},
            {"name": "rtdetr_intrusion", "type": "detection", "input_size": (640, 640), "num_classes": 4},
        ]
    },
    "meter": {
        "description": "E组 - 表计读数",
        "models": [
            {"name": "hrnet_keypoint", "type": "detection", "input_size": (256, 256), "num_classes": 8},
            {"name": "crnn_ocr", "type": "classification", "input_size": (32, 128), "num_classes": 37},
            {"name": "meter_classifier", "type": "classification", "input_size": (224, 224), "num_classes": 5},
        ]
    },
}


# =============================================================================
# 训练管理器
# =============================================================================
class TrainingManager:
    """训练管理器 - 统一管理训练文件输出"""

    def __init__(self, base_dir=None):
        # 使用 training/ 目录作为基础目录
        self.base_dir = Path(base_dir) if base_dir else SCRIPT_DIR
        self.project_root = PROJECT_ROOT
        self.device = get_device() if TORCH_AVAILABLE else None

        # 创建 training/ 下的目录结构
        self.dirs = {
            "checkpoints": self.base_dir / "checkpoints",      # 检查点
            "exports": self.base_dir / "exports",              # ONNX临时导出
            "logs": self.base_dir / "logs",                    # 日志
            "results": self.base_dir / "results",              # 训练结果
            "data": self.base_dir / "data",                    # 训练数据
            "models_deploy": self.project_root / "models",     # 部署模型目录
        }

        for name, d in self.dirs.items():
            d.mkdir(parents=True, exist_ok=True)
            if name != "models_deploy":
                logger.info(f"📁 训练目录: {d}")

        self.results = {}
    
    def create_model(self, model_config):
        """创建模型"""
        model_type = model_config["type"]
        num_classes = model_config["num_classes"]
        input_size = model_config["input_size"]
        
        if model_type == "detection":
            return SimpleDetectionModel(num_classes, input_size)
        elif model_type == "segmentation":
            return SimpleSegmentationModel(num_classes)
        else:
            return SimpleClassificationModel(num_classes)
    
    def train_model(self, plugin_name, model_config, epochs=10, batch_size=16):
        """训练单个模型"""
        model_name = model_config["name"]
        input_size = model_config["input_size"]
        num_classes = model_config["num_classes"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"训练: {plugin_name}/{model_name}")
        logger.info(f"类型: {model_config['type']}, 输入: {input_size}, 类别: {num_classes}")
        logger.info(f"{'='*60}")
        
        # 创建模型
        model = self.create_model(model_config)
        
        # 创建数据集
        train_dataset = SimulatedDataset(
            num_samples=500, 
            input_size=input_size, 
            num_classes=num_classes
        )
        val_dataset = SimulatedDataset(
            num_samples=100, 
            input_size=input_size, 
            num_classes=num_classes
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"训练集: {len(train_dataset)} 样本")
        logger.info(f"验证集: {len(val_dataset)} 样本")
        
        # 创建训练器
        save_dir = self.dirs["checkpoints"] / plugin_name
        trainer = Trainer(model, self.device, save_dir)

        # 训练
        history = trainer.train(train_loader, val_loader, epochs, model_name)

        # 导出ONNX到临时目录
        export_dir = self.dirs["exports"] / plugin_name
        export_path = export_dir / f"{model_name}.onnx"
        ONNXExporter.export(model, input_size, str(export_path), model_name)

        # 复制到部署目录
        deploy_dir = self.dirs["models_deploy"] / plugin_name
        deploy_dir.mkdir(parents=True, exist_ok=True)
        deploy_path = deploy_dir / f"{model_name}.onnx"

        if export_path.exists():
            import shutil
            shutil.copy2(str(export_path), str(deploy_path))
            logger.info(f"📦 已部署到: {deploy_path}")

        return {
            "status": "success",
            "history": history,
            "checkpoint": str(save_dir / f"{model_name}_best.pth"),
            "onnx_export": str(export_path),
            "onnx_deploy": str(deploy_path)
        }
    
    def train_plugin(self, plugin_name, epochs=10):
        """训练插件所有模型"""
        if plugin_name not in PLUGIN_CONFIGS:
            logger.error(f"未知插件: {plugin_name}")
            return
        
        config = PLUGIN_CONFIGS[plugin_name]
        logger.info(f"\n{'#'*60}")
        logger.info(f"# 训练插件: {plugin_name} - {config['description']}")
        logger.info(f"{'#'*60}")
        
        results = {}
        for model_config in config["models"]:
            try:
                result = self.train_model(plugin_name, model_config, epochs)
                results[model_config["name"]] = result
            except Exception as e:
                logger.error(f"训练失败 {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                results[model_config["name"]] = {"status": "failed", "error": str(e)}
        
        self.results[plugin_name] = results
        return results
    
    def train_all(self, epochs=10):
        """训练所有插件"""
        for plugin_name in PLUGIN_CONFIGS:
            self.train_plugin(plugin_name, epochs)

        # 保存摘要到 results/ 目录
        summary_path = self.dirs["results"] / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"\n✅ 训练摘要已保存: {summary_path}")

        return self.results

    def run_demo(self, epochs=3):
        """演示模式"""
        logger.info("\n" + "="*60)
        logger.info("演示模式 - 快速训练测试")
        logger.info("="*60)

        # 只训练一个模型作为演示
        plugin_name = "transformer"
        model_config = PLUGIN_CONFIGS[plugin_name]["models"][2]  # silica_cnn

        result = self.train_model(plugin_name, model_config, epochs=epochs, batch_size=32)

        logger.info("\n" + "="*60)
        logger.info("演示完成!")
        logger.info("="*60)
        logger.info(f"检查点: {result['checkpoint']}")
        logger.info(f"ONNX导出: {result['onnx_export']}")
        logger.info(f"ONNX部署: {result['onnx_deploy']}")

        # 验证文件
        if Path(result['checkpoint']).exists():
            logger.info(f"✅ 检查点文件存在: {Path(result['checkpoint']).stat().st_size / 1024:.1f} KB")
        if Path(result['onnx_deploy']).exists():
            logger.info(f"✅ ONNX部署文件存在: {Path(result['onnx_deploy']).stat().st_size / 1024:.1f} KB")

        return result


# =============================================================================
# 命令行接口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="破夜绘明激光监测平台 - 模型训练")
    parser.add_argument("--mode", type=str, default="demo",
                       choices=["demo", "plugin", "all", "info"],
                       help="运行模式")
    parser.add_argument("--plugin", type=str, default=None,
                       choices=list(PLUGIN_CONFIGS.keys()),
                       help="指定插件")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    
    args = parser.parse_args()
    
    # 打印系统信息
    logger.info("="*60)
    logger.info("破夜绘明激光监测平台 - 模型训练系统")
    logger.info("="*60)
    
    platform_info = detect_platform()
    logger.info(f"系统: {platform_info['system']} ({platform_info['machine']})")
    logger.info(f"设备: {platform_info['device']} - {platform_info['device_name']}")
    logger.info(f"PyTorch: {'可用' if TORCH_AVAILABLE else '不可用'}")
    logger.info(f"ONNX: {'可用' if ONNX_AVAILABLE else '不可用'}")
    logger.info(f"ONNX Runtime: {'可用' if ORT_AVAILABLE else '不可用'}")
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装，无法训练")
        return
    
    # 创建管理器
    manager = TrainingManager()
    
    if args.mode == "info":
        logger.info("\n可训练的模型:")
        for plugin, config in PLUGIN_CONFIGS.items():
            logger.info(f"\n{plugin} - {config['description']}:")
            for m in config["models"]:
                logger.info(f"  - {m['name']} ({m['type']}, {m['input_size']})")
    
    elif args.mode == "demo":
        manager.run_demo(epochs=args.epochs)
    
    elif args.mode == "plugin":
        if not args.plugin:
            logger.error("请使用 --plugin 指定插件")
            return
        manager.train_plugin(args.plugin, epochs=args.epochs)
    
    elif args.mode == "all":
        manager.train_all(epochs=args.epochs)
    
    logger.info("\n✅ 完成!")


if __name__ == "__main__":
    main()
