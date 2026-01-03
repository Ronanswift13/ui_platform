#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 训练模块
Multi-Voltage Level Substation Equipment Training System

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training

支持的电压等级:
- 特高压 (UHV): 1000kV交流, ±800kV直流
- 超高压 (EHV): 500kV, 330kV, 750kV
- 高压 (HV): 220kV, 110kV
- 中压 (MV): 35kV, 66kV
- 低压 (LV): 10kV, 6kV, 380V

支持的插件:
- transformer: 主变压器巡检
- switch: 开关间隔检测
- busbar: 母线巡检
- capacitor: 电容器巡检
- meter: 表计读数

目录结构:
    training/
    ├── __init__.py                # 本文件
    ├── train_main.py              # 主训练脚本
    ├── train_mac.sh               # Mac训练脚本
    ├── prepare_training_data.py   # 数据准备脚本
    ├── evaluate_training.py       # 训练评估脚本
    ├── data_augmentation.py       # 数据增强模块
    ├── model_integration.py       # 模型集成模块
    ├── quick_setup.py             # 快速设置脚本
    ├── setup_all_data.py          # 数据设置脚本
    ├── configs/                   # 训练配置
    │   ├── training_config.yaml   # 训练参数配置
    │   └── datasets_download.yaml # 数据集下载配置
    ├── checkpoints/               # 模型检查点
    │   ├── transformer/           # A组 - 主变巡视
    │   ├── switch/                # B组 - 开关间隔
    │   ├── busbar/                # C组 - 母线巡视
    │   ├── capacitor/             # D组 - 电容器
    │   └── meter/                 # E组 - 表计读数
    ├── exports/                   # ONNX导出
    ├── logs/                      # 训练日志
    ├── data/                      # 训练数据
    │   ├── raw/                   # 原始数据
    │   ├── processed/             # 处理后数据
    │   ├── placeholder/           # 占位符数据
    │   ├── augmented/             # 增强数据
    │   ├── loaders/               # 数据加载器
    │   └── voltage_loaders.py     # 电压等级数据加载器
    └── results/                   # 训练结果

使用方法:
    # 从项目根目录运行
    python train.py --mode demo
    python train.py --mode plugin --plugin transformer --epochs 30

    # 或直接运行训练模块
    python -m training.train_main --mode demo

作者: 破夜绘明激光监测平台开发团队
版本: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "破夜绘明激光监测平台开发团队"

from pathlib import Path

# =============================================================================
# 路径配置
# =============================================================================
BASE_TRAINING_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
CHECKPOINTS_PATH = BASE_TRAINING_PATH / "checkpoints"
DATA_PATH = BASE_TRAINING_PATH / "data"
EXPORTS_PATH = BASE_TRAINING_PATH / "exports"
LOGS_PATH = BASE_TRAINING_PATH / "logs"
RESULTS_PATH = BASE_TRAINING_PATH / "results"
CONFIGS_PATH = BASE_TRAINING_PATH / "configs"

# =============================================================================
# 电压等级定义
# =============================================================================
VOLTAGE_CATEGORIES = {
    "UHV": {
        "name": "特高压",
        "name_en": "Ultra High Voltage",
        "levels": ["1000kV_AC", "800kV_DC"],
        "description": "交流1000kV及以上、直流±800kV及以上"
    },
    "EHV": {
        "name": "超高压",
        "name_en": "Extra High Voltage",
        "levels": ["500kV", "330kV", "750kV"],
        "description": "交流330kV-750kV、直流±500kV"
    },
    "HV": {
        "name": "高压",
        "name_en": "High Voltage",
        "levels": ["220kV", "110kV"],
        "description": "110kV、220kV"
    },
    "MV": {
        "name": "中压",
        "name_en": "Medium Voltage",
        "levels": ["35kV", "66kV"],
        "description": "35kV、66kV"
    },
    "LV": {
        "name": "低压",
        "name_en": "Low Voltage",
        "levels": ["10kV", "6kV", "380V"],
        "description": "10kV及以下"
    }
}

# =============================================================================
# 插件定义
# =============================================================================
PLUGINS = {
    "transformer": {
        "name": "主变压器巡检",
        "name_en": "Transformer Inspection",
        "description": "检测油泄漏、锈蚀、硅胶颜色、油位、套管裂纹等"
    },
    "switch": {
        "name": "开关间隔检测",
        "name_en": "Switch Compartment Detection",
        "description": "检测断路器/隔离开关/接地开关状态、指示灯颜色"
    },
    "busbar": {
        "name": "母线巡检",
        "name_en": "Busbar Inspection",
        "description": "检测绝缘子缺陷、金具松动、鸟巢、异物等"
    },
    "capacitor": {
        "name": "电容器巡检",
        "name_en": "Capacitor Inspection",
        "description": "检测电容器倾斜、掉落、缺失、熔丝熔断等"
    },
    "meter": {
        "name": "表计读数",
        "name_en": "Meter Reading",
        "description": "SF6压力表、油温表、油位计等表计识别与读数"
    }
}

# =============================================================================
# 公开数据集来源
# =============================================================================
PUBLIC_DATASETS = {
    "insulator_defect": {
        "name": "Insulator-Defect Detection Dataset",
        "url": "https://datasetninja.com/insulator-defect-detection",
        "size": "2.43GB",
        "images": 1600
    },
    "cplid": {
        "name": "Chinese Power Line Insulator Dataset",
        "url": "https://github.com/InsulatorData/InsulatorDataSet",
        "size": "~500MB",
        "images": 848
    },
    "mpid": {
        "name": "Merged Public Insulator Dataset",
        "url": "https://github.com/phd-benel/MPID",
        "size": "~1GB",
        "images": 6000
    },
    "ufpr_amr": {
        "name": "UFPR-AMR Dataset (Meter Reading)",
        "url": "https://github.com/raysonlaroca/ufpr-amr-dataset",
        "size": "~200MB",
        "images": 2000
    },
    "transformer_thermal": {
        "name": "Transformer Thermal Images",
        "url": "https://data.mendeley.com/datasets/8mg8mkc7k5/3",
        "size": "~50MB",
        "images": 255
    }
}

# =============================================================================
# 辅助函数
# =============================================================================
def get_training_path() -> Path:
    """获取训练根路径"""
    return BASE_TRAINING_PATH

def get_checkpoint_path(plugin: str, voltage_level: str = "") -> Path:
    """获取检查点路径"""
    path = CHECKPOINTS_PATH / plugin
    if voltage_level:
        path = path / voltage_level
    return path

def get_data_path(voltage_level: str = "", plugin: str = "") -> Path:
    """获取数据路径"""
    path = DATA_PATH / "processed"
    if voltage_level:
        path = path / voltage_level
    if plugin:
        path = path / plugin
    return path

def list_voltage_levels() -> list:
    """列出所有电压等级"""
    levels = []
    for category, info in VOLTAGE_CATEGORIES.items():
        for level in info["levels"]:
            levels.append(f"{category}_{level}")
    return levels

def list_plugins() -> list:
    """列出所有插件"""
    return list(PLUGINS.keys())

def get_voltage_category(voltage_level: str) -> str:
    """根据电压等级获取类别"""
    for category, info in VOLTAGE_CATEGORIES.items():
        if voltage_level in info["levels"] or voltage_level.startswith(category):
            return category
    return "HV"  # 默认返回高压

# =============================================================================
# 导入子模块
# =============================================================================
try:
    from .prepare_training_data import (
        TrainingDataPreparer,
        DatasetDownloader,
        PlaceholderGenerator,
        DataOrganizer
    )
except ImportError:
    TrainingDataPreparer = None
    DatasetDownloader = None
    PlaceholderGenerator = None
    DataOrganizer = None

try:
    from .train_main import (
        TrainingConfig,
        VoltageDatasetManager,
        YOLOv8Trainer,
        BatchTrainingManager,
        FewShotTrainer
    )
except ImportError:
    TrainingConfig = None
    VoltageDatasetManager = None
    YOLOv8Trainer = None
    BatchTrainingManager = None
    FewShotTrainer = None

try:
    from .evaluate_training import (
        ModelEvaluator,
        BatchEvaluator,
        ModelComparator
    )
except ImportError:
    ModelEvaluator = None
    BatchEvaluator = None
    ModelComparator = None

try:
    from .data_augmentation import (
        DataAugmentor,
        AugmentationCompose,
        ColorJitter,
        RandomFlip,
        RandomRotate,
        WeatherSimulation,
        InfraredSimulation,
        LightingVariation,
        get_default_augmentation
    )
except ImportError:
    DataAugmentor = None
    AugmentationCompose = None
    ColorJitter = None
    RandomFlip = None
    RandomRotate = None
    WeatherSimulation = None
    InfraredSimulation = None
    LightingVariation = None
    get_default_augmentation = None

try:
    from .model_integration import (
        ModelInfo,
        ModelRegistry,
        ModelDeployer,
        ModelVersionManager,
        PlatformIntegrator
    )
except ImportError:
    ModelInfo = None
    ModelRegistry = None
    ModelDeployer = None
    ModelVersionManager = None
    PlatformIntegrator = None

try:
    from .data.voltage_loaders import (
        DataLoaderFactory,
        BaseVoltageDataLoader,
        UHV1000kVACLoader,
        EHV500kVLoader,
        HV220kVLoader,
        HV110kVLoader,
        MV35kVLoader,
        LV10kVLoader
    )
except ImportError:
    DataLoaderFactory = None
    BaseVoltageDataLoader = None
    UHV1000kVACLoader = None
    EHV500kVLoader = None
    HV220kVLoader = None
    HV110kVLoader = None
    MV35kVLoader = None
    LV10kVLoader = None

# =============================================================================
# 导出
# =============================================================================
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    # 路径
    "BASE_TRAINING_PATH",
    "CHECKPOINTS_PATH",
    "DATA_PATH",
    "EXPORTS_PATH",
    "LOGS_PATH",
    "RESULTS_PATH",
    "CONFIGS_PATH",
    # 常量
    "VOLTAGE_CATEGORIES",
    "PLUGINS",
    "PUBLIC_DATASETS",
    # 函数
    "get_training_path",
    "get_checkpoint_path",
    "get_data_path",
    "list_voltage_levels",
    "list_plugins",
    "get_voltage_category",
    # 数据准备类
    "TrainingDataPreparer",
    "DatasetDownloader",
    "PlaceholderGenerator",
    "DataOrganizer",
    # 训练类
    "TrainingConfig",
    "VoltageDatasetManager",
    "YOLOv8Trainer",
    "BatchTrainingManager",
    "FewShotTrainer",
    # 评估类
    "ModelEvaluator",
    "BatchEvaluator",
    "ModelComparator",
    # 数据增强类
    "DataAugmentor",
    "AugmentationCompose",
    "ColorJitter",
    "RandomFlip",
    "RandomRotate",
    "WeatherSimulation",
    "InfraredSimulation",
    "LightingVariation",
    "get_default_augmentation",
    # 模型集成类
    "ModelInfo",
    "ModelRegistry",
    "ModelDeployer",
    "ModelVersionManager",
    "PlatformIntegrator",
    # 数据加载器类
    "DataLoaderFactory",
    "BaseVoltageDataLoader",
    "UHV1000kVACLoader",
    "EHV500kVLoader",
    "HV220kVLoader",
    "HV110kVLoader",
    "MV35kVLoader",
    "LV10kVLoader"
]
