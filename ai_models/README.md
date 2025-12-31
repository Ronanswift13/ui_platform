# AI巡检系统 - 模型开发与部署

## 概述

本项目提供完整的AI巡检系统模型开发、训练、部署和研究功能,包括:

- **模型训练**: SLAM、声学异常检测、时序预测、多模态融合
- **系统部署**: ONNX转换、TensorRT优化、边缘部署
- **高级研究**: 图优化SLAM、小样本学习、不确定性量化、模型压缩、主动学习、注意力融合

## 目录结构

```
ai_models/                       # 模型开发与部署目录
├── __init__.py                  # 模块入口
├── training/                    # 训练模块
│   ├── __init__.py
│   ├── slam/                    # SLAM模型训练
│   │   ├── __init__.py
│   │   └── slam_trainer.py      # 点云配准、回环检测、语义分割
│   ├── acoustic/                # 声学模型训练
│   │   ├── __init__.py
│   │   └── acoustic_trainer.py  # 异常检测Transformer、VAE
│   ├── timeseries/              # 时序模型训练
│   │   ├── __init__.py
│   │   └── timeseries_trainer.py # LSTM、Informer预测
│   └── fusion/                  # 融合模型训练
│       ├── __init__.py
│       └── fusion_trainer.py    # 多模态融合网络
├── deployment/                  # 部署模块
│   ├── __init__.py
│   ├── deployment_pipeline.py   # ONNX/TensorRT转换、边缘部署
│   └── deployment_manager.py    # 部署管理器
├── research/                    # 研究模块
│   ├── __init__.py
│   ├── graph_slam/              # 图优化SLAM
│   │   ├── __init__.py
│   │   └── graph_slam.py        # 回环检测+位姿图优化
│   ├── few_shot/                # 小样本学习
│   │   ├── __init__.py
│   │   └── few_shot_learning.py # 原型网络、MAML等
│   ├── uncertainty/             # 不确定性量化
│   │   ├── __init__.py
│   │   └── uncertainty_quantification.py # MC Dropout、贝叶斯网络
│   ├── compression/             # 模型压缩
│   │   ├── __init__.py
│   │   └── model_compression.py # 量化、剪枝、蒸馏
│   ├── active_learning/         # 主动学习
│   │   ├── __init__.py
│   │   └── active_learning.py   # 不确定性采样、BADGE
│   └── attention_fusion/        # 注意力融合
│       ├── __init__.py
│       └── attention_fusion.py  # 跨模态注意力、门控融合
└── integration.py               # 集成脚本
```

## 快速开始

### 环境要求

```bash
# 核心依赖
pip install torch>=1.9.0 numpy scipy scikit-learn

# 可选依赖
pip install onnx onnxruntime-gpu tensorrt  # 部署
pip install soundfile librosa              # 音频处理
pip install open3d                         # 点云可视化
```

### 运行演示

```bash
# 从项目根目录运行
cd 破夜绘明激光监测平台

# 运行完整演示
python -m ai_models.integration --mode demo

# 或者进入ai_models目录运行
cd ai_models
python integration.py --mode demo

# 训练所有模型
python integration.py --mode train --epochs 50 --output ./output

# 部署模型
python integration.py --mode deploy --fp16

# 运行研究功能
python integration.py --mode research
```

### 从项目中导入

```python
# 从项目根目录导入
from ai_models.training.slam.slam_trainer import SLAMTrainer, SLAMConfig
from ai_models.deployment.deployment_pipeline import DeploymentPipeline
from ai_models.research.graph_slam.graph_slam import GraphSLAM
```

## 模型详情

### 1. SLAM模型 (`training/slam/`)

**网络架构:**
- **DeepLIO**: 深度学习LiDAR里程计,直接回归6-DOF位姿
- **DCP**: 深度点云配准,使用Transformer交叉注意力
- **PointNetVLAD**: 回环检测,生成全局描述子
- **RandLANet**: 点云语义分割

**特性:**
- 支持KITTI格式数据
- 体素降采样和数据增强
- ONNX导出支持

```python
from training.slam.slam_trainer import SLAMTrainer, SLAMConfig

config = SLAMConfig(
    data_dir="data/slam",
    num_epochs=100,
    model_type="deeplio"
)
trainer = SLAMTrainer(config)
trainer.train()
trainer.export_onnx("model.onnx")
```

### 2. 声学异常检测 (`training/acoustic/`)

**网络架构:**
- **AcousticAnomalyTransformer**: 编码器-解码器重建 + 分类
- **AcousticVAE**: 变分自编码器异常检测
- **ContrastiveAcousticModel**: SimCLR对比学习

**异常类型:**
- 正常、局部放电、电晕放电、机械故障、变压器异常

```python
from training.acoustic.acoustic_trainer import AcousticTrainer, AcousticConfig

config = AcousticConfig(
    data_dir="data/acoustic",
    sample_rate=16000,
    model_type="transformer"
)
trainer = AcousticTrainer(config)
trainer.train()
```

### 3. 时序预测 (`training/timeseries/`)

**网络架构:**
- **LSTMPredictor**: 自回归LSTM预测
- **TransformerPredictor**: Transformer编码器-解码器
- **Informer**: ProbSparse注意力,高效长序列预测

**预测目标:**
- SF6、H2、CO、C2H2等气体浓度

```python
from training.timeseries.timeseries_trainer import TimeSeriesTrainer, TimeSeriesConfig

config = TimeSeriesConfig(
    input_length=168,  # 7天
    prediction_length=24,  # 1天
    model_type="informer"
)
trainer = TimeSeriesTrainer(config)
trainer.train()
```

### 4. 多模态融合 (`training/fusion/`)

**融合策略:**
- **cross_attention**: 跨模态Transformer注意力
- **hierarchical**: 层次化融合 (空间→时序→全局)
- **early**: 早期特征拼接
- **late**: 晚期决策融合

```python
from training.fusion.fusion_trainer import FusionTrainer, FusionConfig

config = FusionConfig(
    fusion_strategy="cross_attention",
    modality_dims={"visual": 512, "audio": 128, ...}
)
trainer = FusionTrainer(config)
trainer.train()
```

## 部署指南

### ONNX转换

```python
from deployment.deployment_pipeline import ONNXConverter, DeploymentConfig

config = DeploymentConfig(
    model_name="slam_model",
    fp16=True,
    dynamic_batch=True
)
converter = ONNXConverter(config)
converter.convert_pytorch_to_onnx(model, dummy_inputs, "model.onnx")
```

### TensorRT优化

```python
from deployment.deployment_pipeline import TensorRTConverter

converter = TensorRTConverter(config)
converter.build_engine("model.onnx", "model.engine", fp16=True)
```

### 边缘部署

```python
from deployment.deployment_pipeline import EdgeDeployer

deployer = EdgeDeployer(config)
model_paths = deployer.prepare_for_device("model.onnx", "deployed/")
deployer.create_deployment_package(model_paths, "deploy.zip", include_runtime=True)
```

## 研究功能

### 图优化SLAM

```python
from research.graph_slam.graph_slam import GraphSLAM, Pose3D

slam = GraphSLAM()
for frame in frames:
    result = slam.process_frame(pose, point_cloud)
    if result["loop_closure"]:
        print(f"回环检测: {result['loop_closure']}")

trajectory = slam.get_trajectory()
global_map = slam.get_map()
```

### 小样本学习

```python
from research.few_shot.few_shot_learning import FewShotTrainer, FewShotConfig

config = FewShotConfig(n_way=5, k_shot=5)
trainer = FewShotTrainer(config, model_type="prototypical")
trainer.train()
```

### 不确定性量化

```python
from research.uncertainty.uncertainty_quantification import create_uncertainty_model

predictor = create_uncertainty_model(model, method="mc_dropout")
results = predictor.predict(x)
print(f"预测: {results['mean']}, 不确定性: {results['std']}")
```

### 模型压缩

```python
from research.compression.model_compression import compress_model, CompressionConfig

config = CompressionConfig(pruning_ratio=0.5, quantization_bits=8)
compressed = compress_model(model, train_loader, config, steps=["prune", "quantize"])
```

### 主动学习

```python
from research.active_learning.active_learning import ActiveLearningTrainer, ActiveLearningConfig

config = ActiveLearningConfig(strategy="uncertainty", batch_size=100)
trainer = ActiveLearningTrainer(config, model_class, dataset)
history = trainer.run(test_loader, num_rounds=10)
```

### 注意力融合

```python
from research.attention_fusion.attention_fusion import create_attention_fusion, AttentionFusionConfig

config = AttentionFusionConfig(fusion_type="cross_attention")
model = create_attention_fusion(config)
outputs = model(modalities)
```

## 性能指标

| 模型 | 任务 | 指标 | 值 |
|------|------|------|-----|
| DeepLIO | 里程计 | 平移误差 | <1% |
| AcousticTransformer | 异常检测 | F1-Score | >95% |
| Informer | 时序预测 | MAE | <5% |
| FusionNetwork | 故障分类 | 准确率 | >98% |

## 硬件支持

- **GPU**: NVIDIA GPU (CUDA 11.0+)
- **边缘设备**: Jetson Nano/Xavier, RK3588
- **推理加速**: TensorRT, ONNX Runtime

## 许可证

MIT License

## 联系方式

如有问题,请提交Issue或联系开发团队。
