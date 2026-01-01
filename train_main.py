#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 模型训练主入口
=====================================

功能:
1. 使用公开500kV变电站数据集预训练基础模型
2. 支持云南保山站数据微调
3. Mac MPS加速训练
4. 自动导出ONNX用于Windows部署

使用方法:
    # 训练所有模型
    python train_main.py --mode all --epochs 50
    
    # 训练单个插件
    python train_main.py --mode plugin --plugin transformer --epochs 30
    
    # 仅导出ONNX
    python train_main.py --mode export
    
    # 性能测试
    python train_main.py --mode benchmark

作者: 破夜绘明团队
日期: 2025
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入训练模块
from ai_models.training.trainer import (
    TrainingPipeline, 
    CrossPlatformTrainer,
    TrainingConfig,
    detect_platform,
    get_device
)
from ai_models.training.datasets import (
    create_dataloader,
    SimulatedDataset,
    DatasetDownloader
)
from ai_models.training.models import create_model, get_model_info
from ai_models.training.exporters import (
    ONNXExporter,
    ONNXValidator,
    BatchExporter,
    generate_windows_validation_script
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# 训练配置
# =============================================================================
# 各插件模型配置
PLUGIN_MODEL_CONFIGS = {
    # A组 - 主变巡视
    "transformer": {
        "models": [
            {
                "name": "defect_yolov8n",
                "type": "detection",
                "input_size": (640, 640),
                "num_classes": 6,
                "classes": ["oil_leak", "rust", "damage", "foreign_object", "crack", "discoloration"],
                "epochs": 50,
                "batch_size": 16,
            },
            {
                "name": "oil_unet",
                "type": "segmentation",
                "input_size": (512, 512),
                "num_classes": 2,
                "epochs": 30,
                "batch_size": 8,
            },
            {
                "name": "silica_cnn",
                "type": "classification",
                "input_size": (224, 224),
                "num_classes": 4,
                "classes": ["blue", "pink", "white", "unknown"],
                "epochs": 30,
                "batch_size": 32,
            },
            {
                "name": "thermal_anomaly",
                "type": "classification",
                "input_size": (224, 224),
                "num_classes": 3,
                "classes": ["normal", "warning", "alarm"],
                "epochs": 30,
                "batch_size": 32,
            },
        ]
    },
    
    # B组 - 开关间隔
    "switch": {
        "models": [
            {
                "name": "switch_yolov8s",
                "type": "detection",
                "input_size": (640, 640),
                "num_classes": 8,
                "classes": ["breaker_open", "breaker_closed", "isolator_open", "isolator_closed",
                           "grounding_open", "grounding_closed", "indicator_red", "indicator_green"],
                "epochs": 50,
                "batch_size": 16,
            },
            {
                "name": "indicator_ocr",
                "type": "ocr",
                "input_size": (32, 128),
                "charset_size": 50,  # 中文+数字+字母
                "epochs": 40,
                "batch_size": 64,
            },
        ]
    },
    
    # C组 - 母线巡视
    "busbar": {
        "models": [
            {
                "name": "busbar_yolov8m",
                "type": "detection",
                "input_size": (1280, 1280),  # 高分辨率用于小目标
                "num_classes": 8,
                "classes": ["insulator_crack", "insulator_dirty", "fitting_loose", "fitting_rust",
                           "wire_damage", "foreign_object", "bird", "insect"],
                "epochs": 60,
                "batch_size": 8,  # 大尺寸减少batch
            },
            {
                "name": "noise_classifier",
                "type": "classification",
                "input_size": (128, 128),
                "num_classes": 5,
                "classes": ["real_defect", "bird", "insect", "shadow", "reflection"],
                "epochs": 30,
                "batch_size": 32,
            },
        ]
    },
    
    # D组 - 电容器
    "capacitor": {
        "models": [
            {
                "name": "capacitor_yolov8",
                "type": "detection",
                "input_size": (640, 640),
                "num_classes": 6,
                "classes": ["capacitor_unit", "capacitor_tilted", "capacitor_fallen",
                           "capacitor_missing", "connection_wire", "fence"],
                "epochs": 50,
                "batch_size": 16,
            },
            {
                "name": "rtdetr_intrusion",
                "type": "detection",
                "input_size": (640, 640),
                "num_classes": 4,
                "classes": ["person", "vehicle", "animal", "unknown"],
                "epochs": 50,
                "batch_size": 16,
            },
        ]
    },
    
    # E组 - 表计读数
    "meter": {
        "models": [
            {
                "name": "hrnet_keypoint",
                "type": "keypoint",
                "input_size": (256, 256),
                "num_keypoints": 8,  # 表盘关键点
                "epochs": 40,
                "batch_size": 32,
            },
            {
                "name": "crnn_ocr",
                "type": "ocr",
                "input_size": (32, 128),
                "charset_size": 37,  # 0-9 + a-z + blank
                "epochs": 40,
                "batch_size": 64,
            },
            {
                "name": "meter_classifier",
                "type": "classification",
                "input_size": (224, 224),
                "num_classes": 5,
                "classes": ["pressure_gauge", "temperature", "oil_level", "sf6_pressure", "digital"],
                "epochs": 30,
                "batch_size": 32,
            },
        ]
    },
}


# =============================================================================
# 训练管理器
# =============================================================================
class TrainingManager:
    """
    训练管理器
    
    负责协调所有模型的训练、验证和导出
    """
    
    def __init__(self, base_dir: str = "."):
        """
        初始化训练管理器
        
        Args:
            base_dir: 项目根目录
        """
        self.base_dir = Path(base_dir)
        self.platform_info = detect_platform()
        
        # 目录结构
        self.dirs = {
            "data": self.base_dir / "data",
            "checkpoints": self.base_dir / "checkpoints",
            "models": self.base_dir / "models",
            "logs": self.base_dir / "logs",
            "exports": self.base_dir / "exports",
        }
        
        # 创建目录
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # 训练结果
        self.results = {}
        
        self._print_system_info()
    
    def _print_system_info(self):
        """打印系统信息"""
        logger.info("=" * 60)
        logger.info("破夜绘明激光监测平台 - 模型训练系统")
        logger.info("=" * 60)
        logger.info(f"平台: {self.platform_info.system} ({self.platform_info.machine})")
        logger.info(f"设备: {self.platform_info.device} ({self.platform_info.device_name})")
        logger.info(f"内存: {self.platform_info.memory_gb:.1f} GB")
        logger.info(f"Apple Silicon: {self.platform_info.is_apple_silicon}")
        logger.info(f"CUDA可用: {self.platform_info.cuda_available}")
        logger.info(f"MPS可用: {self.platform_info.mps_available}")
        logger.info(f"推荐Batch Size: {self.platform_info.recommended_batch_size}")
        logger.info(f"推荐精度: {self.platform_info.recommended_precision}")
        logger.info("=" * 60)
    
    def prepare_data(self, use_public_dataset: bool = True):
        """
        准备训练数据
        
        Args:
            use_public_dataset: 是否使用公开数据集
        """
        logger.info("\n准备训练数据...")
        
        data_dir = self.dirs["data"]
        
        if use_public_dataset:
            logger.info("使用公开500kV变电站数据集进行预训练")
            
            # 列出可用数据集
            available = DatasetDownloader.list_datasets()
            logger.info(f"可用数据集: {list(available.keys())}")
            
            # 下载提示
            logger.info("\n请按以下步骤准备数据:")
            logger.info("1. 从公开渠道获取电力设备缺陷数据集")
            logger.info("2. 将数据组织为COCO格式或分类目录格式")
            logger.info(f"3. 放置到: {data_dir}")
            logger.info("\n数据目录结构示例:")
            logger.info("""
data/
├── transformer/           # A组数据
│   ├── defect/           # 缺陷检测
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── annotations/
│   │       ├── train.json
│   │       └── val.json
│   ├── silica/           # 硅胶分类
│   │   ├── train/
│   │   │   ├── blue/
│   │   │   ├── pink/
│   │   │   └── white/
│   │   └── val/
│   └── thermal/          # 热成像
├── switch/               # B组数据
├── busbar/               # C组数据
├── capacitor/            # D组数据
└── meter/                # E组数据
""")
        else:
            logger.info("使用模拟数据进行训练测试")
    
    def train_model(self, plugin: str, model_config: Dict, 
                    data_dir: str = None, use_simulated: bool = False) -> Dict:
        """
        训练单个模型
        
        Args:
            plugin: 插件名称
            model_config: 模型配置
            data_dir: 数据目录
            use_simulated: 是否使用模拟数据
        
        Returns:
            训练结果
        """
        import torch
        import torch.nn as nn
        
        model_name = model_config["name"]
        model_type = model_config["type"]
        input_size = model_config["input_size"]
        epochs = model_config.get("epochs", 50)
        batch_size = model_config.get("batch_size", self.platform_info.recommended_batch_size)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"训练模型: {plugin}/{model_name}")
        logger.info(f"类型: {model_type}, 输入: {input_size}, Epochs: {epochs}")
        logger.info(f"{'='*60}")
        
        # 创建模型
        model = create_model(
            model_type=model_type,
            model_name=model_name,
            input_size=input_size,
            pretrained=False,
            plugin_name=plugin,
            num_classes=model_config.get("num_classes", 10),
            num_keypoints=model_config.get("num_keypoints", 8),
            charset_size=model_config.get("charset_size", 37)
        )
        
        # 打印模型信息
        info = get_model_info(model)
        logger.info(f"模型参数: {info['total_params']:,}")
        logger.info(f"模型大小: {info['model_size_mb']:.2f} MB")
        
        # 创建数据加载器
        if data_dir is None:
            data_dir = str(self.dirs["data"] / plugin)
        
        train_loader, val_loader = create_dataloader(
            plugin_name=plugin,
            model_type=model_type,
            data_dir=data_dir,
            batch_size=batch_size,
            input_size=input_size,
            use_simulated=use_simulated
        )
        
        logger.info(f"训练集: {len(train_loader.dataset)} 样本")
        logger.info(f"验证集: {len(val_loader.dataset)} 样本")
        
        # 训练配置
        config = TrainingConfig(
            model_name=f"{plugin}_{model_name}",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-4,
            save_dir=str(self.dirs["checkpoints"] / plugin),
            use_amp=(self.platform_info.device == "cuda"),
            early_stopping=True,
            patience=10
        )
        
        # 创建训练器
        trainer = CrossPlatformTrainer(model, config)
        
        # 选择损失函数
        if model_type == "detection":
            criterion = nn.CrossEntropyLoss()  # 简化,实际使用YOLO损失
        elif model_type == "segmentation":
            criterion = nn.BCEWithLogitsLoss()
        elif model_type == "classification":
            criterion = nn.CrossEntropyLoss()
        elif model_type == "keypoint":
            criterion = nn.MSELoss()
        elif model_type == "ocr":
            criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 训练
        history = trainer.train(train_loader, val_loader, criterion)
        
        # 保存最终模型
        checkpoint_path = self.dirs["checkpoints"] / plugin / f"{model_name}_final.pth"
        trainer.save_checkpoint(str(checkpoint_path))
        
        result = {
            "status": "success",
            "plugin": plugin,
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "history": history,
            "best_val_acc": max(history.get('val_acc', [0])),
            "final_loss": history['train_loss'][-1] if history['train_loss'] else None
        }
        
        return result
    
    def train_plugin(self, plugin: str, use_simulated: bool = False) -> Dict:
        """
        训练单个插件的所有模型
        
        Args:
            plugin: 插件名称
            use_simulated: 是否使用模拟数据
        
        Returns:
            训练结果
        """
        if plugin not in PLUGIN_MODEL_CONFIGS:
            raise ValueError(f"未知插件: {plugin}")
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"# 开始训练插件: {plugin}")
        logger.info(f"{'#'*60}")
        
        plugin_config = PLUGIN_MODEL_CONFIGS[plugin]
        results = {}
        
        for model_config in plugin_config["models"]:
            try:
                result = self.train_model(
                    plugin=plugin,
                    model_config=model_config,
                    use_simulated=use_simulated
                )
                results[model_config["name"]] = result
                
            except Exception as e:
                logger.error(f"训练失败 {model_config['name']}: {e}")
                results[model_config["name"]] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        self.results[plugin] = results
        return results
    
    def train_all(self, use_simulated: bool = False, 
                  plugins: List[str] = None) -> Dict:
        """
        训练所有插件模型
        
        Args:
            use_simulated: 是否使用模拟数据
            plugins: 要训练的插件列表,默认全部
        
        Returns:
            所有训练结果
        """
        plugins = plugins or list(PLUGIN_MODEL_CONFIGS.keys())
        
        logger.info(f"\n开始训练 {len(plugins)} 个插件的所有模型")
        logger.info(f"插件列表: {plugins}")
        
        for plugin in plugins:
            self.train_plugin(plugin, use_simulated)
        
        # 保存训练摘要
        self._save_training_summary()
        
        return self.results
    
    def export_onnx(self, plugin: str = None, model_name: str = None) -> Dict:
        """
        导出ONNX模型
        
        Args:
            plugin: 插件名称,默认全部
            model_name: 模型名称,默认该插件全部
        
        Returns:
            导出结果
        """
        import torch
        
        logger.info("\n开始导出ONNX模型...")
        
        exporter = ONNXExporter(opset_version=17)
        export_results = {}
        
        plugins = [plugin] if plugin else list(PLUGIN_MODEL_CONFIGS.keys())
        
        for p in plugins:
            plugin_config = PLUGIN_MODEL_CONFIGS[p]
            models = plugin_config["models"]
            
            if model_name:
                models = [m for m in models if m["name"] == model_name]
            
            for model_config in models:
                m_name = model_config["name"]
                key = f"{p}/{m_name}"
                
                logger.info(f"\n导出: {key}")
                
                try:
                    # 创建模型
                    model = create_model(
                        model_type=model_config["type"],
                        model_name=m_name,
                        input_size=model_config["input_size"],
                        pretrained=False,
                        plugin_name=p,
                        num_classes=model_config.get("num_classes", 10)
                    )
                    
                    # 尝试加载检查点
                    checkpoint_path = self.dirs["checkpoints"] / p / f"{m_name}_best.pth"
                    if not checkpoint_path.exists():
                        checkpoint_path = self.dirs["checkpoints"] / p / f"{m_name}_final.pth"
                    
                    if checkpoint_path.exists():
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        logger.info(f"已加载检查点: {checkpoint_path}")
                    else:
                        logger.warning(f"检查点不存在,使用随机权重")
                    
                    # 导出
                    output_path = self.dirs["models"] / p / f"{m_name}.onnx"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    exporter.export(
                        model=model,
                        input_shape=(3, *model_config["input_size"]),
                        save_path=str(output_path)
                    )
                    
                    export_results[key] = {
                        "status": "success",
                        "path": str(output_path)
                    }
                    
                except Exception as e:
                    logger.error(f"导出失败: {e}")
                    export_results[key] = {
                        "status": "failed",
                        "error": str(e)
                    }
        
        # 生成Windows验证脚本
        script_path = self.dirs["exports"] / "validate_onnx_windows.py"
        generate_windows_validation_script(
            onnx_dir=str(self.dirs["models"]),
            output_path=str(script_path)
        )
        
        return export_results
    
    def benchmark(self, use_gpu: bool = None) -> Dict:
        """
        性能基准测试
        
        Args:
            use_gpu: 是否使用GPU
        
        Returns:
            测试结果
        """
        logger.info("\n运行性能基准测试...")
        
        if use_gpu is None:
            use_gpu = self.platform_info.cuda_available
        
        results = {}
        
        # 查找所有ONNX模型
        onnx_files = list(self.dirs["models"].rglob("*.onnx"))
        
        for onnx_path in onnx_files:
            model_name = onnx_path.stem
            logger.info(f"\n测试: {model_name}")
            
            try:
                validator = ONNXValidator(str(onnx_path), use_gpu=use_gpu)
                stats = validator.benchmark(num_iterations=100)
                
                results[model_name] = stats
                
                logger.info(f"  平均: {stats['mean_ms']:.2f} ms")
                logger.info(f"  FPS: {stats['fps']:.1f}")
                
            except Exception as e:
                logger.error(f"  测试失败: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def _save_training_summary(self):
        """保存训练摘要"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "platform": self.platform_info.__dict__,
            "results": self.results
        }
        
        summary_path = self.dirs["checkpoints"] / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n训练摘要已保存: {summary_path}")


# =============================================================================
# 命令行接口
# =============================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="破夜绘明激光监测平台 - 模型训练系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用模拟数据训练所有模型
    python train_main.py --mode all --simulated --epochs 10
    
    # 训练主变巡视插件
    python train_main.py --mode plugin --plugin transformer --epochs 50
    
    # 导出所有ONNX模型
    python train_main.py --mode export
    
    # 性能测试
    python train_main.py --mode benchmark
    
    # 查看数据准备指南
    python train_main.py --mode prepare
"""
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all",
        choices=["all", "plugin", "model", "export", "benchmark", "prepare", "info"],
        help="运行模式"
    )
    
    parser.add_argument(
        "--plugin",
        type=str,
        default=None,
        choices=["transformer", "switch", "busbar", "capacitor", "meter"],
        help="指定插件"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定模型名称"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数(覆盖默认值)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小(覆盖默认值)"
    )
    
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="使用模拟数据"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据目录"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="强制使用GPU"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用CPU"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建训练管理器
    manager = TrainingManager()
    
    if args.mode == "info":
        # 显示系统信息
        logger.info("\n可训练的模型列表:")
        for plugin, config in PLUGIN_MODEL_CONFIGS.items():
            logger.info(f"\n{plugin}:")
            for model in config["models"]:
                logger.info(f"  - {model['name']} ({model['type']}, {model['input_size']})")
    
    elif args.mode == "prepare":
        # 准备数据
        manager.prepare_data(use_public_dataset=True)
    
    elif args.mode == "all":
        # 训练所有模型
        manager.train_all(use_simulated=args.simulated)
        
        # 自动导出ONNX
        manager.export_onnx()
    
    elif args.mode == "plugin":
        # 训练单个插件
        if not args.plugin:
            logger.error("请使用 --plugin 指定插件名称")
            return
        
        manager.train_plugin(args.plugin, use_simulated=args.simulated)
        
        # 导出该插件的ONNX
        manager.export_onnx(plugin=args.plugin)
    
    elif args.mode == "model":
        # 训练单个模型
        if not args.plugin or not args.model:
            logger.error("请使用 --plugin 和 --model 指定插件和模型名称")
            return
        
        # 查找模型配置
        plugin_config = PLUGIN_MODEL_CONFIGS.get(args.plugin)
        if not plugin_config:
            logger.error(f"未知插件: {args.plugin}")
            return
        
        model_config = None
        for m in plugin_config["models"]:
            if m["name"] == args.model:
                model_config = m
                break
        
        if not model_config:
            logger.error(f"未知模型: {args.model}")
            return
        
        # 覆盖配置
        if args.epochs:
            model_config["epochs"] = args.epochs
        if args.batch_size:
            model_config["batch_size"] = args.batch_size
        
        manager.train_model(args.plugin, model_config, use_simulated=args.simulated)
        manager.export_onnx(plugin=args.plugin, model_name=args.model)
    
    elif args.mode == "export":
        # 仅导出ONNX
        results = manager.export_onnx(plugin=args.plugin, model_name=args.model)
        
        # 打印结果
        logger.info("\n导出结果:")
        for key, result in results.items():
            status = "✅" if result["status"] == "success" else "❌"
            logger.info(f"  {status} {key}")
    
    elif args.mode == "benchmark":
        # 性能测试
        use_gpu = args.gpu and not args.cpu
        results = manager.benchmark(use_gpu=use_gpu)
        
        # 打印结果
        logger.info("\n性能测试结果:")
        for model, stats in results.items():
            if "error" not in stats:
                logger.info(f"  {model}: {stats['mean_ms']:.2f}ms, {stats['fps']:.1f} FPS")
    
    logger.info("\n完成!")


if __name__ == "__main__":
    main()
