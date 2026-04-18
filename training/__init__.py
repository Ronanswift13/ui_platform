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

支持的插件 (v1 - 站内设备):
- transformer: 主变压器巡检
- switch: 开关间隔检测
- busbar: 母线巡检
- capacitor: 电容器巡检
- meter: 表计读数

支持的视觉插件 (v2 - 统一训练库):
- busbar_inspection: 母线巡检 (detection)
- capacitor_inspection: 电容器巡检 (detection + classification)
- switch_inspection: 开关间隔检测 (detection + classification)
- transformer_inspection: 主变压器巡检 (detection + classification)
- animal_detection: 动物入侵检测 (detection + classification + thermal)
- bird_monitoring: 鸟类监测 (detection + classification)
- fire_detection: 火灾检测 (detection + classification + thermal)
- meter_reading: 表计读数识别 (ocr + detection)
- temperature_monitoring: 温度监测 (thermal_anomaly + classification)
- thermal: 热像分析 (thermal_anomaly) [占位]
- hyperspectral_detection: 高光谱检测 (hyperspectral_classification + anomaly)

目录结构:
    training/
    ├── __init__.py                # 本文件
    ├── train_main.py              # 主训练脚本 (v1)
    ├── training_api.py            # 训练 API (v1)
    ├── training_api_v2.py         # 统一训练 API (v2)
    │
    │ ── v2 统一训练库 ──
    ├── registry/                  # 插件训练映射注册表
    │   └── plugin_training_mapping.json
    ├── schemas/                   # Schema 定义与校验
    │   ├── dataset_manifest.py    # 数据集 manifest 结构
    │   ├── label_schemas.py       # 按 task_type 的标签格式
    │   └── training_config_schema.py
    ├── pipelines/                 # 五阶段训练流水线
    │   ├── ingestion/             # 数据摄入 + manifest 校验
    │   ├── preprocessing/         # 按 task_type 预处理
    │   ├── training/              # 按 task_type 训练
    │   ├── evaluation/            # 统一评估
    │   └── export/                # ONNX / TensorRT 导出
    ├── plugin_configs/            # 11 个视觉插件训练模板
    ├── datasets/visual_defect/    # 统一数据集存储
    │
    │ ── v1 兼容层 ──
    ├── train_mac.sh               # Mac训练脚本
    ├── prepare_training_data.py   # 数据准备脚本
    ├── evaluate_training.py       # 训练评估脚本
    ├── data_augmentation.py       # 数据增强模块
    ├── model_integration.py       # 模型集成模块
    ├── configs/                   # 训练配置
    ├── checkpoints/               # 模型检查点 (按 plugin_id + task_type)
    ├── exports/                   # ONNX导出 (按 plugin_id + task_type)
    ├── data/                      # 训练数据 (v1 格式)
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
# v2 统一训练库导入
# =============================================================================
try:
    from .registry import (
        get_registry,
        get_plugin_config,
        resolve_alias,
        list_visual_plugins,
        get_task_type_info,
    )
except ImportError:
    get_registry = None
    get_plugin_config = None
    resolve_alias = None
    list_visual_plugins = None
    get_task_type_info = None

try:
    from .schemas import (
        DatasetManifest,
        validate_manifest,
        get_schema_for_task_type,
        TrainingConfigSchema,
        validate_training_config,
    )
except ImportError:
    DatasetManifest = None
    validate_manifest = None
    get_schema_for_task_type = None
    TrainingConfigSchema = None
    validate_training_config = None

try:
    from .pipelines.ingestion import ManifestValidator, DataRouter
except ImportError:
    ManifestValidator = None
    DataRouter = None

try:
    from .pipelines.export import ModelExporter
except ImportError:
    ModelExporter = None

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
    "LV10kVLoader",
    # v2 注册表
    "get_registry",
    "get_plugin_config",
    "resolve_alias",
    "list_visual_plugins",
    "get_task_type_info",
    # v2 Schema
    "DatasetManifest",
    "validate_manifest",
    "get_schema_for_task_type",
    "TrainingConfigSchema",
    "validate_training_config",
    # v2 流水线
    "ManifestValidator",
    "DataRouter",
    "ModelExporter",
]
