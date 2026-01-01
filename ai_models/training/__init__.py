"""
破夜绘明激光监测平台 - AI模型训练框架
====================================

跨平台训练系统，支持:
- Mac M系列芯片 (MPS加速) 开发训练
- Windows (CUDA/TensorRT) 部署推理
- 自动导出ONNX格式

训练流程:
1. 使用公开500kV变电站数据集预训练基础模型
2. 在云南保山站数据上进行迁移学习微调
3. 导出ONNX模型用于Windows部署

使用方法:
    from ai_models.training import TrainingPipeline
    
    pipeline = TrainingPipeline()
    pipeline.train_all(epochs=50)
    pipeline.export_all_onnx()
"""

from .trainer import (
    TrainingPipeline,
    CrossPlatformTrainer,
    get_device,
    detect_platform
)

from .datasets import (
    SubstationDataset,
    TransformerDataset,
    SwitchDataset,
    BusbarDataset,
    CapacitorDataset,
    MeterDataset,
    create_dataloader
)

from .exporters import (
    ONNXExporter,
    export_to_onnx,
    verify_onnx_model
)

__all__ = [
    # 训练器
    "TrainingPipeline",
    "CrossPlatformTrainer",
    "get_device",
    "detect_platform",
    # 数据集
    "SubstationDataset",
    "TransformerDataset",
    "SwitchDataset", 
    "BusbarDataset",
    "CapacitorDataset",
    "MeterDataset",
    "create_dataloader",
    # 导出器
    "ONNXExporter",
    "export_to_onnx",
    "verify_onnx_model",
]

__version__ = "1.0.0"
