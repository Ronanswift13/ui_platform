#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 一键数据准备脚本
One-Click Training Data Preparation

此脚本将按照目录结构自动准备所有电压等级的训练数据:
- transformer: 主变压器巡检
- switch: 开关间隔检测
- busbar: 母线巡检
- capacitor: 电容器巡检
- meter: 表计读数

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training
"""

import os
import sys
import json
import yaml
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "data_preparation.log")
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# 路径配置
# =============================================================================
BASE_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
CHECKPOINTS_PATH = BASE_PATH / "checkpoints"
DATA_PATH = BASE_PATH / "data"
CONFIGS_PATH = BASE_PATH / "configs"
EXPORTS_PATH = BASE_PATH / "exports"
LOGS_PATH = BASE_PATH / "logs"
RESULTS_PATH = BASE_PATH / "results"

# 电压等级列表
VOLTAGE_LEVELS = [
    "UHV_1000kV_AC",  # 特高压交流
    "UHV_800kV_DC",   # 特高压直流
    "EHV_500kV",      # 超高压500kV
    "EHV_330kV",      # 超高压330kV
    "HV_220kV",       # 高压220kV
    "HV_110kV",       # 高压110kV
    "MV_35kV",        # 中压35kV
    "LV_10kV",        # 低压10kV
]

# 插件列表
PLUGINS = ["transformer", "switch", "busbar", "capacitor", "meter", "few_shot", "test"]


def create_base_structure():
    """创建基础目录结构"""
    logger.info("=" * 60)
    logger.info("创建基础目录结构")
    logger.info("=" * 60)
    
    directories = [
        BASE_PATH,
        CHECKPOINTS_PATH,
        DATA_PATH,
        DATA_PATH / "raw",
        DATA_PATH / "processed",
        DATA_PATH / "placeholder",
        DATA_PATH / "augmented",
        CONFIGS_PATH,
        EXPORTS_PATH,
        LOGS_PATH,
        RESULTS_PATH,
    ]
    
    # 为每个插件创建checkpoint目录
    for plugin in PLUGINS:
        directories.append(CHECKPOINTS_PATH / plugin)
        # 为每个电压等级创建子目录
        for voltage in VOLTAGE_LEVELS:
            directories.append(CHECKPOINTS_PATH / plugin / voltage)
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"创建目录: {dir_path}")
    
    logger.info(f"基础目录结构创建完成: {BASE_PATH}")
    return True


def prepare_transformer_data():
    """准备主变压器训练数据"""
    logger.info("=" * 60)
    logger.info("准备主变压器训练数据")
    logger.info("=" * 60)
    
    try:
        from data.loaders.transformer_loader import prepare_all_transformer_data
        return prepare_all_transformer_data()
    except ImportError as e:
        logger.warning(f"导入transformer_loader失败: {e}")
        return _prepare_plugin_data_fallback("transformer")


def prepare_switch_data():
    """准备开关间隔训练数据"""
    logger.info("=" * 60)
    logger.info("准备开关间隔训练数据")
    logger.info("=" * 60)
    
    try:
        from data.loaders.switch_loader import prepare_all_switch_data
        return prepare_all_switch_data()
    except ImportError as e:
        logger.warning(f"导入switch_loader失败: {e}")
        return _prepare_plugin_data_fallback("switch")


def prepare_busbar_data():
    """准备母线巡检训练数据"""
    logger.info("=" * 60)
    logger.info("准备母线巡检训练数据")
    logger.info("=" * 60)
    
    try:
        from data.loaders.busbar_loader import prepare_all_busbar_data
        return prepare_all_busbar_data()
    except ImportError as e:
        logger.warning(f"导入busbar_loader失败: {e}")
        return _prepare_plugin_data_fallback("busbar")


def prepare_capacitor_data():
    """准备电容器巡检训练数据"""
    logger.info("=" * 60)
    logger.info("准备电容器巡检训练数据")
    logger.info("=" * 60)
    
    try:
        from data.loaders.capacitor_meter_loader import prepare_all_capacitor_data
        return prepare_all_capacitor_data()
    except ImportError as e:
        logger.warning(f"导入capacitor_loader失败: {e}")
        return _prepare_plugin_data_fallback("capacitor")


def prepare_meter_data():
    """准备表计读数训练数据"""
    logger.info("=" * 60)
    logger.info("准备表计读数训练数据")
    logger.info("=" * 60)
    
    try:
        from data.loaders.capacitor_meter_loader import prepare_all_meter_data
        return prepare_all_meter_data()
    except ImportError as e:
        logger.warning(f"导入meter_loader失败: {e}")
        return _prepare_plugin_data_fallback("meter")


def _prepare_plugin_data_fallback(plugin: str) -> Dict[str, Any]:
    """备用数据准备方法"""
    results = {}
    
    for voltage in VOLTAGE_LEVELS:
        data_path = DATA_PATH / "processed" / voltage / plugin
        
        # 创建目录结构
        for split in ["train", "val", "test"]:
            (data_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (data_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # 生成基础配置
        yaml_content = {
            "path": str(data_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 5,
            "names": {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3", 4: "class_4"},
            "voltage_level": voltage,
            "plugin": plugin,
        }
        
        yaml_path = data_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        results[voltage] = {
            "data_path": str(data_path),
            "data_yaml": str(yaml_path),
            "status": "placeholder"
        }
    
    return results


def generate_summary_report(results: Dict[str, Any]) -> Path:
    """生成数据准备汇总报告"""
    logger.info("=" * 60)
    logger.info("生成汇总报告")
    logger.info("=" * 60)
    
    report_path = BASE_PATH / "DATA_PREPARATION_REPORT.md"
    
    report_content = f"""# 训练数据准备报告

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 训练路径
```
{BASE_PATH}
```

## 目录结构
```
training/
├── checkpoints/          # 模型检查点
│   ├── busbar/           # 母线巡检模型
│   ├── capacitor/        # 电容器巡检模型
│   ├── few_shot/         # Few-shot学习模型
│   ├── meter/            # 表计读数模型
│   ├── switch/           # 开关间隔模型
│   ├── test/             # 测试模型
│   └── transformer/      # 主变压器模型
├── configs/              # 配置文件
├── data/                 # 训练数据
│   ├── raw/              # 原始下载数据
│   ├── processed/        # 处理后数据
│   ├── placeholder/      # 占位符数据
│   └── augmented/        # 增强数据
├── exports/              # 导出的ONNX模型
├── logs/                 # 训练日志
└── results/              # 训练结果
```

## 电压等级覆盖

| 类别 | 电压等级 | 说明 |
|------|----------|------|
| 特高压(UHV) | 1000kV AC | 交流特高压 |
| 特高压(UHV) | ±800kV DC | 直流特高压 |
| 超高压(EHV) | 500kV | 超高压 |
| 超高压(EHV) | 330kV | 超高压 |
| 高压(HV) | 220kV | 高压 |
| 高压(HV) | 110kV | 高压 |
| 中压(MV) | 35kV | 中压 |
| 低压(LV) | 10kV | 低压 |

## 插件数据准备状态

"""
    
    # 统计各插件状态
    for plugin, plugin_results in results.items():
        report_content += f"### {plugin}\n\n"
        report_content += "| 电压等级 | 状态 | 类别数 | 数据路径 |\n"
        report_content += "|----------|------|--------|----------|\n"
        
        if isinstance(plugin_results, dict):
            for voltage, info in plugin_results.items():
                if isinstance(info, dict):
                    status = "✅ 就绪" if info.get("status") != "error" else "❌ 错误"
                    num_classes = info.get("num_classes", info.get("nc", "N/A"))
                    data_path = info.get("data_path", "N/A")
                    report_content += f"| {voltage} | {status} | {num_classes} | {data_path} |\n"
        
        report_content += "\n"
    
    report_content += """
## 公开数据集下载

### 绝缘子缺陷数据集
```bash
# CPLID - 中国电力线绝缘子数据集
git clone https://github.com/InsulatorData/InsulatorDataSet.git data/raw/cplid/

# MPID - 合并公开绝缘子数据集
git clone https://github.com/phd-benel/MPID.git data/raw/mpid/

# Insulator-Defect Detection Dataset
# 访问: https://datasetninja.com/insulator-defect-detection
```

### 电表读数数据集
```bash
# UFPR-AMR数据集
git clone https://github.com/raysonlaroca/ufpr-amr-dataset.git data/raw/ufpr_amr/

# Copel-AMR数据集
git clone https://github.com/raysonlaroca/copel-amr-dataset.git data/raw/copel_amr/
```

### 变压器热成像数据集
- URL: https://data.mendeley.com/datasets/8mg8mkc7k5/3
- 255张热成像图像

## 开始训练

```bash
# 训练单个模型
python train_main.py train --voltage HV_220kV --plugin transformer

# 训练所有模型
python train_main.py train --all

# 评估模型
python evaluate_training.py batch
```

## 下一步操作

1. 下载公开数据集到 `data/raw/` 目录
2. 将数据转换为YOLO格式并放入 `data/processed/` 对应目录
3. 运行训练脚本
4. 导出ONNX模型用于部署

## 联系支持

如有问题，请查看各数据目录下的README.md文件。
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"汇总报告: {report_path}")
    return report_path


def copy_training_files_to_target():
    """复制训练文件到目标目录"""
    logger.info("=" * 60)
    logger.info("复制训练文件")
    logger.info("=" * 60)
    
    # 获取当前脚本目录
    current_dir = Path(__file__).parent
    
    # 需要复制的文件
    files_to_copy = [
        "train_main.py",
        "evaluate_training.py",
        "prepare_training_data.py",
        "train_mac.sh",
        "__init__.py",
    ]
    
    for filename in files_to_copy:
        src = current_dir / filename
        dst = BASE_PATH / filename
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"复制: {src} -> {dst}")
    
    # 复制配置文件
    configs_src = current_dir / "configs"
    if configs_src.exists():
        for config_file in configs_src.glob("*.yaml"):
            dst = CONFIGS_PATH / config_file.name
            shutil.copy2(config_file, dst)
            logger.info(f"复制配置: {config_file} -> {dst}")
    
    # 复制数据加载器
    loaders_src = current_dir / "data" / "loaders"
    loaders_dst = DATA_PATH / "loaders"
    loaders_dst.mkdir(parents=True, exist_ok=True)
    
    if loaders_src.exists():
        for loader_file in loaders_src.glob("*.py"):
            dst = loaders_dst / loader_file.name
            shutil.copy2(loader_file, dst)
            logger.info(f"复制加载器: {loader_file} -> {dst}")
    
    # 创建__init__.py
    init_content = '"""训练数据加载器模块"""\n'
    with open(loaders_dst / "__init__.py", 'w') as f:
        f.write(init_content)


def main():
    """主函数 - 一键准备所有训练数据"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     破夜绘明激光监测平台 - 训练数据准备工具                     ║
║     Multi-Voltage Level Training Data Preparation            ║
╠══════════════════════════════════════════════════════════════╣
║  支持电压等级:                                                ║
║  • 特高压(UHV): 1000kV AC, ±800kV DC                         ║
║  • 超高压(EHV): 500kV, 330kV                                 ║
║  • 高压(HV): 220kV, 110kV                                    ║
║  • 中压(MV): 35kV                                            ║
║  • 低压(LV): 10kV                                            ║
╠══════════════════════════════════════════════════════════════╣
║  插件功能:                                                    ║
║  • transformer: 主变压器巡检                                  ║
║  • switch: 开关间隔检测                                       ║
║  • busbar: 母线巡检                                          ║
║  • capacitor: 电容器巡检                                      ║
║  • meter: 表计读数                                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("开始数据准备...")
    logger.info(f"目标路径: {BASE_PATH}")
    
    all_results = {}
    
    try:
        # 1. 创建基础目录结构
        create_base_structure()
        
        # 2. 复制训练文件
        copy_training_files_to_target()
        
        # 3. 准备各插件数据
        all_results["transformer"] = prepare_transformer_data()
        all_results["switch"] = prepare_switch_data()
        all_results["busbar"] = prepare_busbar_data()
        all_results["capacitor"] = prepare_capacitor_data()
        all_results["meter"] = prepare_meter_data()
        
        # 4. 生成汇总报告
        report_path = generate_summary_report(all_results)
        
        # 5. 保存JSON结果
        results_json_path = BASE_PATH / "data_preparation_results.json"
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "base_path": str(BASE_PATH),
                "results": {k: str(v) if not isinstance(v, dict) else v for k, v in all_results.items()}
            }, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    数据准备完成!                              ║
╠══════════════════════════════════════════════════════════════╣
║  训练目录: {str(BASE_PATH):<47} ║
║  汇总报告: DATA_PREPARATION_REPORT.md                        ║
╠══════════════════════════════════════════════════════════════╣
║  下一步操作:                                                  ║
║  1. 下载公开数据集到 data/raw/                               ║
║  2. 将数据转换为YOLO格式                                     ║
║  3. 运行: python train_main.py train --all                   ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        return all_results
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
