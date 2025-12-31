#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI巡检系统 - 完整集成脚本
整合所有模型训练、部署、研究模块

功能:
1. 模型训练流水线
2. 系统部署流水线
3. 高级研究功能
4. 完整演示

作者: AI巡检系统
版本: 1.0.0
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# 设置路径 - 支持从项目根目录或ai_models目录导入
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

# 确保项目根目录和ai_models目录都在路径中
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

logger = logging.getLogger(__name__)


# =============================================================================
# 导入模块 (带错误处理)
# =============================================================================
def safe_import(module_path: str, class_name: str):
    """安全导入模块，支持多种导入路径"""
    # 尝试的导入路径列表
    import_paths = [
        module_path,  # 相对路径: training.slam.slam_trainer
        f"ai_models.{module_path}",  # 完整路径: ai_models.training.slam.slam_trainer
    ]

    for path in import_paths:
        try:
            module = __import__(path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue

    logger.warning(f"无法导入 {module_path}.{class_name}")
    return None


# 训练模块
SLAMTrainer = safe_import('training.slam.slam_trainer', 'SLAMTrainer')
AcousticTrainer = safe_import('training.acoustic.acoustic_trainer', 'AcousticTrainer')
TimeSeriesTrainer = safe_import('training.timeseries.timeseries_trainer', 'TimeSeriesTrainer')
FusionTrainer = safe_import('training.fusion.fusion_trainer', 'FusionTrainer')

# 部署模块
DeploymentPipeline = safe_import('deployment.deployment_pipeline', 'DeploymentPipeline')
DeploymentConfig = safe_import('deployment.deployment_pipeline', 'DeploymentConfig')

# 研究模块
GraphSLAM = safe_import('research.graph_slam.graph_slam', 'GraphSLAM')
FewShotTrainer = safe_import('research.few_shot.few_shot_learning', 'FewShotTrainer')
UncertaintyAwarePredictor = safe_import('research.uncertainty.uncertainty_quantification', 'UncertaintyAwarePredictor')
CompressionPipeline = safe_import('research.compression.model_compression', 'CompressionPipeline')
ActiveLearningTrainer = safe_import('research.active_learning.active_learning', 'ActiveLearningTrainer')
AttentionFusionNetwork = safe_import('research.attention_fusion.attention_fusion', 'AttentionFusionNetwork')


# =============================================================================
# 训练流水线
# =============================================================================
class TrainingPipeline:
    """训练流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get('output_dir', 'trained_models')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_slam(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """训练SLAM模型"""
        logger.info("=" * 50)
        logger.info("训练SLAM模型")
        logger.info("=" * 50)
        
        if SLAMTrainer is None:
            logger.error("SLAM训练模块未加载")
            return {"status": "failed", "error": "模块未加载"}
        
        try:
            from training.slam.slam_trainer import SLAMConfig
            
            config = SLAMConfig(
                data_dir=data_dir or "data/slam",
                num_epochs=self.config.get('slam_epochs', 50),
                save_dir=os.path.join(self.output_dir, "slam")
            )
            
            trainer = SLAMTrainer(config)
            trainer.train()
            
            # 导出ONNX
            onnx_path = trainer.export_onnx(
                os.path.join(self.output_dir, "slam", "model.onnx")
            )
            
            return {
                "status": "success",
                "checkpoint": os.path.join(self.output_dir, "slam", "best.pth"),
                "onnx": onnx_path
            }
        except Exception as e:
            logger.error(f"SLAM训练失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def train_acoustic(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """训练声学模型"""
        logger.info("=" * 50)
        logger.info("训练声学异常检测模型")
        logger.info("=" * 50)
        
        if AcousticTrainer is None:
            logger.error("声学训练模块未加载")
            return {"status": "failed", "error": "模块未加载"}
        
        try:
            from training.acoustic.acoustic_trainer import AcousticConfig
            
            config = AcousticConfig(
                data_dir=data_dir or "data/acoustic",
                num_epochs=self.config.get('acoustic_epochs', 100),
                save_dir=os.path.join(self.output_dir, "acoustic")
            )
            
            trainer = AcousticTrainer(config)
            trainer.train()
            
            onnx_path = trainer.export_onnx(
                os.path.join(self.output_dir, "acoustic", "model.onnx")
            )
            
            return {
                "status": "success",
                "checkpoint": os.path.join(self.output_dir, "acoustic", "best.pth"),
                "onnx": onnx_path
            }
        except Exception as e:
            logger.error(f"声学训练失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def train_timeseries(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """训练时序预测模型"""
        logger.info("=" * 50)
        logger.info("训练时序预测模型")
        logger.info("=" * 50)
        
        if TimeSeriesTrainer is None:
            logger.error("时序训练模块未加载")
            return {"status": "failed", "error": "模块未加载"}
        
        try:
            from training.timeseries.timeseries_trainer import TimeSeriesConfig
            
            config = TimeSeriesConfig(
                data_dir=data_dir or "data/timeseries",
                num_epochs=self.config.get('timeseries_epochs', 100),
                save_dir=os.path.join(self.output_dir, "timeseries")
            )
            
            trainer = TimeSeriesTrainer(config)
            trainer.train()
            
            onnx_path = trainer.export_onnx(
                os.path.join(self.output_dir, "timeseries", "model.onnx")
            )
            
            return {
                "status": "success",
                "checkpoint": os.path.join(self.output_dir, "timeseries", "best.pth"),
                "onnx": onnx_path
            }
        except Exception as e:
            logger.error(f"时序训练失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def train_fusion(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """训练多模态融合模型"""
        logger.info("=" * 50)
        logger.info("训练多模态融合模型")
        logger.info("=" * 50)
        
        if FusionTrainer is None:
            logger.error("融合训练模块未加载")
            return {"status": "failed", "error": "模块未加载"}
        
        try:
            from training.fusion.fusion_trainer import FusionConfig
            
            config = FusionConfig(
                data_dir=data_dir or "data/fusion",
                num_epochs=self.config.get('fusion_epochs', 100),
                save_dir=os.path.join(self.output_dir, "fusion")
            )
            
            trainer = FusionTrainer(config)
            trainer.train()
            
            onnx_path = trainer.export_onnx(
                os.path.join(self.output_dir, "fusion", "model.onnx")
            )
            
            return {
                "status": "success",
                "checkpoint": os.path.join(self.output_dir, "fusion", "best.pth"),
                "onnx": onnx_path
            }
        except Exception as e:
            logger.error(f"融合训练失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def train_all(self) -> Dict[str, Dict[str, Any]]:
        """训练所有模型"""
        results = {}
        
        results['slam'] = self.train_slam()
        results['acoustic'] = self.train_acoustic()
        results['timeseries'] = self.train_timeseries()
        results['fusion'] = self.train_fusion()
        
        # 汇总
        success_count = sum(1 for r in results.values() if r.get('status') == 'success')
        logger.info(f"\n训练完成: {success_count}/{len(results)} 成功")
        
        return results


# =============================================================================
# 部署流水线
# =============================================================================
class SystemDeployment:
    """系统部署"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get('deploy_dir', 'deployed_models')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def deploy_model(self, model, model_name: str, 
                    dummy_input: Dict[str, Any]) -> Dict[str, Any]:
        """部署单个模型"""
        if DeploymentPipeline is None:
            logger.error("部署模块未加载")
            return {"status": "failed", "error": "模块未加载"}
        
        try:
            deploy_config = DeploymentConfig(
                model_name=model_name,
                fp16=self.config.get('fp16', True),
                int8=self.config.get('int8', False),
                output_dir=os.path.join(self.output_dir, model_name)
            )
            
            pipeline = DeploymentPipeline(deploy_config)
            results = pipeline.run_pipeline(model, dummy_input)
            
            return results
        except Exception as e:
            logger.error(f"部署失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def deploy_all_from_checkpoints(self, checkpoint_dir: str) -> Dict[str, Any]:
        """从检查点部署所有模型"""
        results = {}
        
        # 需要PyTorch加载检查点
        try:
            import torch
        except ImportError:
            logger.error("PyTorch未安装,无法加载检查点")
            return {"status": "failed", "error": "PyTorch未安装"}
        
        # 部署各模型
        model_configs = [
            ("slam", "training.slam.slam_trainer", "DeepLIO", (16384, 3)),
            ("acoustic", "training.acoustic.acoustic_trainer", "AcousticAnomalyTransformer", (32000,)),
            ("timeseries", "training.timeseries.timeseries_trainer", "Informer", (168, 8)),
            ("fusion", "training.fusion.fusion_trainer", "MultimodalFusionNetwork", None),
        ]
        
        for model_name, module_path, class_name, input_shape in model_configs:
            checkpoint_path = os.path.join(checkpoint_dir, model_name, "best.pth")
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"检查点不存在: {checkpoint_path}")
                results[model_name] = {"status": "skipped", "error": "检查点不存在"}
                continue
            
            logger.info(f"部署模型: {model_name}")
            
            try:
                # 动态导入模型类
                ModelClass = safe_import(module_path, class_name)
                if ModelClass is None:
                    results[model_name] = {"status": "failed", "error": "模型类未找到"}
                    continue
                
                # 加载模型
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model = ModelClass()
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # 创建虚拟输入
                if input_shape:
                    dummy_input = {"input": torch.randn(1, *input_shape)}
                else:
                    # 融合模型的多模态输入
                    dummy_input = {
                        "visual": torch.randn(1, 512),
                        "pointcloud": torch.randn(1, 256),
                        "audio": torch.randn(1, 128),
                        "thermal": torch.randn(1, 128),
                        "timeseries": torch.randn(1, 64)
                    }
                
                # 部署
                results[model_name] = self.deploy_model(model, model_name, dummy_input)
                
            except Exception as e:
                logger.error(f"部署 {model_name} 失败: {e}")
                results[model_name] = {"status": "failed", "error": str(e)}
        
        return results


# =============================================================================
# 研究功能
# =============================================================================
class ResearchFeatures:
    """研究功能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def run_graph_slam_demo(self) -> Dict[str, Any]:
        """图优化SLAM演示"""
        logger.info("=" * 50)
        logger.info("图优化SLAM演示")
        logger.info("=" * 50)
        
        if GraphSLAM is None:
            logger.error("GraphSLAM模块未加载")
            return {"status": "failed", "error": "模块未加载"}
        
        try:
            from research.graph_slam.graph_slam import GraphSLAMConfig, Pose3D
            
            config = GraphSLAMConfig()
            slam = GraphSLAM(config)
            
            # 模拟数据
            for i in range(100):
                translation = np.array([i * 0.1, np.sin(i * 0.1), 0.0])
                pose = Pose3D(translation, np.array([1.0, 0.0, 0.0, 0.0]))
                point_cloud = np.random.randn(1000, 3) + translation
                
                result = slam.process_frame(pose, point_cloud, timestamp=i * 0.1)
                
                if result["loop_closure"]:
                    logger.info(f"检测到回环: 帧{result['frame_id']} -> 帧{result['loop_closure']['to_id']}")
            
            # 获取结果
            trajectory = slam.get_trajectory()
            global_map = slam.get_map()
            
            return {
                "status": "success",
                "trajectory_length": len(trajectory),
                "map_points": len(global_map) if global_map is not None else 0
            }
        except Exception as e:
            logger.error(f"GraphSLAM演示失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_few_shot_demo(self) -> Dict[str, Any]:
        """小样本学习演示"""
        logger.info("=" * 50)
        logger.info("小样本学习演示")
        logger.info("=" * 50)
        
        if FewShotTrainer is None:
            logger.error("FewShot模块未加载")
            return {"status": "failed", "error": "模块未加载"}
        
        try:
            from research.few_shot.few_shot_learning import FewShotConfig
            
            config = FewShotConfig(
                n_way=5,
                k_shot=5,
                num_episodes=100  # 演示用较少
            )
            
            trainer = FewShotTrainer(config, model_type="prototypical")
            # 这里只是初始化演示,实际训练需要数据
            
            return {
                "status": "success",
                "model_type": "prototypical",
                "config": {"n_way": 5, "k_shot": 5}
            }
        except Exception as e:
            logger.error(f"FewShot演示失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_uncertainty_demo(self) -> Dict[str, Any]:
        """不确定性量化演示"""
        logger.info("=" * 50)
        logger.info("不确定性量化演示")
        logger.info("=" * 50)
        
        try:
            import torch
            import torch.nn as nn
            
            from research.uncertainty.uncertainty_quantification import (
                UncertaintyConfig, create_uncertainty_model
            )
            
            # 创建简单模型
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(10, 64)
                    self.fc2 = nn.Linear(64, 5)
                
                def forward(self, x):
                    return self.fc2(torch.relu(self.fc1(x)))
            
            model = SimpleModel()
            config = UncertaintyConfig(mc_samples=30)
            config.method = "mc_dropout"
            
            predictor = create_uncertainty_model(model, config)
            
            # 测试
            x = torch.randn(16, 10)
            results = predictor.predict_with_uncertainty(x, config.num_samples)
            
            return {
                "status": "success",
                "mean_shape": list(results['mean'].shape),
                "std_shape": list(results['std'].shape)
            }
        except Exception as e:
            logger.error(f"不确定性演示失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_compression_demo(self) -> Dict[str, Any]:
        """模型压缩演示"""
        logger.info("=" * 50)
        logger.info("模型压缩演示")
        logger.info("=" * 50)
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            from research.compression.model_compression import (
                CompressionConfig, compress_model
            )
            
            # 创建简单模型
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(784, 512)
                    self.fc2 = nn.Linear(512, 256)
                    self.fc3 = nn.Linear(256, 10)
                
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    return self.fc3(x)
            
            model = SimpleModel()
            
            # 虚拟数据
            data = TensorDataset(
                torch.randn(100, 784),
                torch.randint(0, 10, (100,))
            )
            loader = DataLoader(data, batch_size=32)
            
            config = CompressionConfig(pruning_ratio=0.3)
            compressed = compress_model(model, loader, config, steps=["prune"])
            
            # 计算参数量
            orig_params = sum(p.numel() for p in model.parameters())
            comp_params = sum(p.numel() for p in compressed.parameters())
            
            return {
                "status": "success",
                "original_params": orig_params,
                "compressed_params": comp_params
            }
        except Exception as e:
            logger.error(f"压缩演示失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_active_learning_demo(self) -> Dict[str, Any]:
        """主动学习演示"""
        logger.info("=" * 50)
        logger.info("主动学习演示")
        logger.info("=" * 50)
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader
            
            from research.active_learning.active_learning import (
                ActiveLearningConfig, ActiveLearningTrainer
            )
            
            # 简单模型
            class SimpleModel(nn.Module):
                def __init__(self, input_dim=10, num_classes=5):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, 64)
                    self.fc2 = nn.Linear(64, num_classes)
                
                def forward(self, x):
                    return self.fc2(torch.relu(self.fc1(x)))
            
            # 虚拟数据集
            class DummyDataset(Dataset):
                def __init__(self, size=500):
                    self.data = torch.randn(size, 10)
                    self.labels = torch.randint(0, 5, (size,))
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx], self.labels[idx]
            
            dataset = DummyDataset(500)
            test_dataset = DummyDataset(100)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            config = ActiveLearningConfig(
                strategy="uncertainty",
                batch_size=20,
                total_budget=200,
                epochs_per_round=5
            )
            
            trainer = ActiveLearningTrainer(
                config=config,
                model_class=SimpleModel,
                model_kwargs={"input_dim": 10, "num_classes": 5},
                dataset=dataset,
                criterion=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs={"lr": 1e-3}
            )
            
            history = trainer.run(test_loader, num_rounds=3)
            
            return {
                "status": "success",
                "rounds": len(history),
                "final_accuracy": history[-1]['accuracy'] if history else 0
            }
        except Exception as e:
            logger.error(f"主动学习演示失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_attention_fusion_demo(self) -> Dict[str, Any]:
        """注意力融合演示"""
        logger.info("=" * 50)
        logger.info("注意力融合演示")
        logger.info("=" * 50)
        
        try:
            import torch
            
            from research.attention_fusion.attention_fusion import (
                AttentionFusionConfig, create_attention_fusion
            )
            
            config = AttentionFusionConfig(
                modality_dims={
                    "visual": 512,
                    "pointcloud": 256,
                    "audio": 128
                },
                fusion_dim=256,
                fusion_type="cross_attention"
            )
            
            model = create_attention_fusion(config)
            
            # 测试
            modalities = {
                "visual": torch.randn(4, 512),
                "pointcloud": torch.randn(4, 256),
                "audio": torch.randn(4, 128)
            }
            
            outputs = model(modalities)
            
            return {
                "status": "success",
                "fused_shape": list(outputs['fused'].shape),
                "logits_shape": list(outputs['logits'].shape)
            }
        except Exception as e:
            logger.error(f"注意力融合演示失败: {e}")
            return {"status": "failed", "error": str(e)}


# =============================================================================
# 主程序
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="AI巡检系统集成")
    
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['train', 'deploy', 'research', 'demo', 'all'],
                       help='运行模式')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--data', type=str, default='data',
                       help='数据目录')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--fp16', action='store_true',
                       help='使用FP16')
    parser.add_argument('--int8', action='store_true',
                       help='使用INT8量化')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = {
        'output_dir': os.path.join(args.output, 'trained_models'),
        'deploy_dir': os.path.join(args.output, 'deployed_models'),
        'data_dir': args.data,
        'slam_epochs': args.epochs,
        'acoustic_epochs': args.epochs,
        'timeseries_epochs': args.epochs,
        'fusion_epochs': args.epochs,
        'fp16': args.fp16,
        'int8': args.int8
    }
    
    results = {}
    
    # 训练
    if args.mode in ['train', 'all']:
        logger.info("\n" + "=" * 60)
        logger.info("开始训练流水线")
        logger.info("=" * 60)
        
        pipeline = TrainingPipeline(config)
        results['training'] = pipeline.train_all()
    
    # 部署
    if args.mode in ['deploy', 'all']:
        logger.info("\n" + "=" * 60)
        logger.info("开始部署流水线")
        logger.info("=" * 60)
        
        deployment = SystemDeployment(config)
        checkpoint_dir = config['output_dir']
        results['deployment'] = deployment.deploy_all_from_checkpoints(checkpoint_dir)
    
    # 研究功能
    if args.mode in ['research', 'all']:
        logger.info("\n" + "=" * 60)
        logger.info("运行研究功能演示")
        logger.info("=" * 60)
        
        research = ResearchFeatures(config)
        results['research'] = {
            'graph_slam': research.run_graph_slam_demo(),
            'few_shot': research.run_few_shot_demo(),
            'uncertainty': research.run_uncertainty_demo(),
            'compression': research.run_compression_demo(),
            'active_learning': research.run_active_learning_demo(),
            'attention_fusion': research.run_attention_fusion_demo()
        }
    
    # 演示模式
    if args.mode == 'demo':
        logger.info("\n" + "=" * 60)
        logger.info("运行完整演示")
        logger.info("=" * 60)
        
        research = ResearchFeatures(config)
        
        demos = [
            ('图优化SLAM', research.run_graph_slam_demo),
            ('不确定性量化', research.run_uncertainty_demo),
            ('模型压缩', research.run_compression_demo),
            ('主动学习', research.run_active_learning_demo),
            ('注意力融合', research.run_attention_fusion_demo),
        ]
        
        results['demo'] = {}
        for name, func in demos:
            logger.info(f"\n运行: {name}")
            results['demo'][name] = func()
    
    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("运行结果汇总")
    logger.info("=" * 60)
    
    import json
    print(json.dumps(results, indent=2, default=str))
    
    return results


if __name__ == "__main__":
    main()
