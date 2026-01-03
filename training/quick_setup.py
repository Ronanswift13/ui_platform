#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 快速初始化脚本
一键创建完整的训练目录结构和配置文件

使用方法:
    python quick_setup.py

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training
"""

import os
import sys
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime

# =============================================================================
# 路径配置
# =============================================================================
BASE_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")

# 目录结构
DIRECTORIES = [
    "checkpoints/busbar",
    "checkpoints/capacitor",
    "checkpoints/few_shot",
    "checkpoints/meter",
    "checkpoints/switch",
    "checkpoints/test",
    "checkpoints/transformer",
    "configs",
    "data/raw",
    "data/processed",
    "data/placeholder",
    "data/augmented",
    "exports",
    "logs",
    "results",
    "adapters"
]

# 电压等级配置
VOLTAGE_LEVELS = {
    "UHV_1000kV_AC": {
        "name": "1000kV交流特高压",
        "category": "UHV",
        "thermal": {"normal": 70, "warning": 85, "alarm": 100}
    },
    "UHV_800kV_DC": {
        "name": "±800kV直流特高压",
        "category": "UHV",
        "thermal": {"normal": 70, "warning": 85, "alarm": 100}
    },
    "EHV_500kV": {
        "name": "500kV超高压",
        "category": "EHV",
        "thermal": {"normal": 65, "warning": 80, "alarm": 95}
    },
    "EHV_330kV": {
        "name": "330kV超高压",
        "category": "EHV",
        "thermal": {"normal": 63, "warning": 78, "alarm": 92}
    },
    "HV_220kV": {
        "name": "220kV高压",
        "category": "HV",
        "thermal": {"normal": 60, "warning": 75, "alarm": 85}
    },
    "HV_110kV": {
        "name": "110kV高压",
        "category": "HV",
        "thermal": {"normal": 55, "warning": 70, "alarm": 80}
    },
    "MV_35kV": {
        "name": "35kV中压",
        "category": "MV",
        "thermal": {"normal": 50, "warning": 65, "alarm": 75}
    },
    "LV_10kV": {
        "name": "10kV低压",
        "category": "LV",
        "thermal": {"normal": 45, "warning": 60, "alarm": 70}
    }
}

# 插件配置
PLUGINS = {
    "transformer": {
        "name": "主变压器巡检",
        "classes": {
            "UHV": ["oil_leak", "rust", "surface_damage", "foreign_object", 
                   "silica_gel_normal", "silica_gel_abnormal", "oil_level_normal", 
                   "oil_level_abnormal", "bushing_crack", "porcelain_contamination",
                   "partial_discharge", "core_ground_current", "winding_deformation"],
            "EHV": ["oil_leak", "rust", "surface_damage", "foreign_object",
                   "silica_gel_normal", "silica_gel_abnormal", "oil_level_normal",
                   "oil_level_abnormal", "bushing_crack", "porcelain_contamination"],
            "HV": ["oil_leak", "rust", "surface_damage", "foreign_object",
                  "silica_gel_normal", "silica_gel_abnormal", "oil_level_normal", "oil_level_abnormal"],
            "MV": ["oil_leak", "rust", "silica_gel_normal", "silica_gel_abnormal",
                  "oil_level_normal", "oil_level_abnormal", "conductor_rust"],
            "LV": ["rust", "oil_leak", "radiator_damage", "box_transformer"]
        }
    },
    "switch": {
        "name": "开关间隔检测",
        "classes": {
            "UHV": ["breaker_open", "breaker_closed", "isolator_open", "isolator_closed",
                   "grounding_open", "grounding_closed", "indicator_red", "indicator_green",
                   "gis_position", "gis_sf6_density"],
            "EHV": ["breaker_open", "breaker_closed", "isolator_open", "isolator_closed",
                   "grounding_open", "grounding_closed", "indicator_red", "indicator_green", "gis_position"],
            "HV": ["breaker_open", "breaker_closed", "isolator_open", "isolator_closed",
                  "grounding_open", "grounding_closed", "indicator_red", "indicator_green"],
            "MV": ["breaker_open", "breaker_closed", "isolator_open", "isolator_closed",
                  "indicator_red", "indicator_green", "cabinet_door"],
            "LV": ["cabinet_switch", "air_breaker", "indicator_light", "label"]
        }
    },
    "busbar": {
        "name": "母线巡检",
        "classes": {
            "UHV": ["insulator_crack", "insulator_dirty", "insulator_flashover",
                   "fitting_loose", "fitting_rust", "wire_damage", "foreign_object",
                   "bird", "pin_missing", "spacer_damage", "corona_discharge"],
            "EHV": ["insulator_crack", "insulator_dirty", "insulator_flashover",
                   "fitting_loose", "fitting_rust", "wire_damage", "foreign_object",
                   "bird", "pin_missing", "spacer_damage"],
            "HV": ["insulator_crack", "insulator_dirty", "fitting_loose", "fitting_rust",
                  "wire_damage", "foreign_object", "bird", "bird_nest"],
            "MV": ["insulator_crack", "insulator_dirty", "fitting_loose", 
                  "wire_damage", "foreign_object"],
            "LV": ["bolt_loose", "wire_damage", "foreign_object"]
        }
    },
    "capacitor": {
        "name": "电容器巡检",
        "classes": {
            "UHV": ["capacitor_unit", "capacitor_tilted", "capacitor_fallen",
                   "capacitor_missing", "person", "vehicle", "fuse_blown", "silver_contact_damage"],
            "EHV": ["capacitor_unit", "capacitor_tilted", "capacitor_fallen",
                   "capacitor_missing", "person", "vehicle", "fuse_blown"],
            "HV": ["capacitor_unit", "capacitor_tilted", "capacitor_fallen",
                  "capacitor_missing", "person", "vehicle"],
            "MV": ["capacitor_unit", "capacitor_tilted", "capacitor_missing", "safety_distance"],
            "LV": ["capacitor_bulge", "capacitor_leak"]
        }
    },
    "meter": {
        "name": "表计读数",
        "classes": {
            "UHV": ["sf6_pressure_gauge", "sf6_density_relay", "oil_temp_gauge",
                   "oil_level_gauge", "gas_relay", "digital_display", "pointer_gauge"],
            "EHV": ["sf6_pressure_gauge", "sf6_density_relay", "gas_relay",
                   "oil_temp_gauge", "oil_level_gauge"],
            "HV": ["sf6_pressure_gauge", "oil_temp_gauge", "oil_level_gauge"],
            "MV": ["digital_meter", "analog_meter", "transformer_monitor"],
            "LV": ["energy_meter", "water_meter", "pointer_meter", "digital_meter"]
        }
    }
}


def create_directory_structure():
    """创建目录结构"""
    print("=" * 60)
    print("创建目录结构...")
    print("=" * 60)
    
    for dir_path in DIRECTORIES:
        full_path = BASE_PATH / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    # 创建电压等级目录
    for voltage_level in VOLTAGE_LEVELS.keys():
        for plugin in PLUGINS.keys():
            # processed目录
            (BASE_PATH / "data" / "processed" / voltage_level / plugin / "images" / "train").mkdir(parents=True, exist_ok=True)
            (BASE_PATH / "data" / "processed" / voltage_level / plugin / "images" / "val").mkdir(parents=True, exist_ok=True)
            (BASE_PATH / "data" / "processed" / voltage_level / plugin / "images" / "test").mkdir(parents=True, exist_ok=True)
            (BASE_PATH / "data" / "processed" / voltage_level / plugin / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (BASE_PATH / "data" / "processed" / voltage_level / plugin / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (BASE_PATH / "data" / "processed" / voltage_level / plugin / "labels" / "test").mkdir(parents=True, exist_ok=True)
            
            # placeholder目录
            (BASE_PATH / "data" / "placeholder" / voltage_level / plugin / "images").mkdir(parents=True, exist_ok=True)
            (BASE_PATH / "data" / "placeholder" / voltage_level / plugin / "labels").mkdir(parents=True, exist_ok=True)
            
            # checkpoints目录
            (BASE_PATH / "checkpoints" / plugin / voltage_level).mkdir(parents=True, exist_ok=True)
            
            # exports目录
            (BASE_PATH / "exports" / plugin / voltage_level).mkdir(parents=True, exist_ok=True)
    
    print(f"\n  共创建 {len(VOLTAGE_LEVELS) * len(PLUGINS)} 个电压等级-插件组合目录")


def generate_data_yaml_files():
    """生成所有data.yaml文件"""
    print("\n" + "=" * 60)
    print("生成data.yaml配置文件...")
    print("=" * 60)
    
    for voltage_level, v_config in VOLTAGE_LEVELS.items():
        category = v_config["category"]
        
        for plugin, p_config in PLUGINS.items():
            classes = p_config["classes"].get(category, [])
            
            data_path = BASE_PATH / "data" / "processed" / voltage_level / plugin
            yaml_content = {
                "path": str(data_path),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: cls for i, cls in enumerate(classes)},
                "nc": len(classes),
                "voltage_level": voltage_level,
                "plugin": plugin,
                "thermal_thresholds": v_config["thermal"]
            }
            
            yaml_path = data_path / "data.yaml"
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
            
            # 生成classes.txt
            classes_path = data_path / "classes.txt"
            with open(classes_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(classes))
    
    print(f"  ✓ 生成 {len(VOLTAGE_LEVELS) * len(PLUGINS)} 个data.yaml文件")


def generate_placeholder_readme():
    """生成占位符目录的README"""
    print("\n" + "=" * 60)
    print("生成占位符说明文件...")
    print("=" * 60)
    
    for voltage_level, v_config in VOLTAGE_LEVELS.items():
        category = v_config["category"]
        
        for plugin, p_config in PLUGINS.items():
            classes = p_config["classes"].get(category, [])
            
            readme_content = f"""# 占位符数据目录 - {voltage_level} / {plugin}

## 说明
此目录用于存放现场采集的训练数据。

## 电压等级信息
- 名称: {v_config["name"]}
- 类别: {category}
- 热成像阈值: 正常<{v_config["thermal"]["normal"]}°C, 预警<{v_config["thermal"]["warning"]}°C, 报警>{v_config["thermal"]["alarm"]}°C

## 检测类别 ({len(classes)}个)
{chr(10).join(f'- {i}: {cls}' for i, cls in enumerate(classes))}

## 数据要求
1. 图像格式: JPG/PNG, 建议分辨率 1920x1080 或更高
2. 标注格式: YOLO格式 (class_id x_center y_center width height)
3. 建议样本数: 每类至少100张图像

## 采集建议
- 多角度拍摄设备
- 包含正常和异常状态
- 不同光照条件 (白天/夜间)
- 不同天气条件 (晴天/阴天/雨天)

## 目录结构
```
{plugin}/
├── images/        # 放置图像文件
└── labels/        # 放置YOLO格式标注文件
```

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            
            readme_path = BASE_PATH / "data" / "placeholder" / voltage_level / plugin / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
    
    print(f"  ✓ 生成 {len(VOLTAGE_LEVELS) * len(PLUGINS)} 个README文件")


def generate_summary():
    """生成汇总信息"""
    print("\n" + "=" * 60)
    print("初始化完成!")
    print("=" * 60)
    
    print(f"\n训练根目录: {BASE_PATH}")
    print(f"\n支持的电压等级: {len(VOLTAGE_LEVELS)} 个")
    for level, config in VOLTAGE_LEVELS.items():
        print(f"  - {level}: {config['name']}")
    
    print(f"\n支持的插件: {len(PLUGINS)} 个")
    for plugin, config in PLUGINS.items():
        print(f"  - {plugin}: {config['name']}")
    
    total_classes = 0
    for v_level, v_config in VOLTAGE_LEVELS.items():
        category = v_config["category"]
        for plugin, p_config in PLUGINS.items():
            total_classes += len(p_config["classes"].get(category, []))
    
    print(f"\n总检测类别数: {total_classes}")
    print(f"总配置组合数: {len(VOLTAGE_LEVELS) * len(PLUGINS)}")
    
    print("\n下一步操作:")
    print("  1. 下载公开数据集: python prepare_training_data.py")
    print("  2. 或直接开始训练: python train_main.py train --all")
    print("  3. Mac用户可使用: ./train_mac.sh full")


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     破夜绘明激光监测平台 - 训练环境快速初始化                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 检查目标路径
    if not BASE_PATH.parent.exists():
        print(f"警告: 父目录不存在 - {BASE_PATH.parent}")
        print("正在创建完整路径...")
    
    # 执行初始化
    create_directory_structure()
    generate_data_yaml_files()
    generate_placeholder_readme()
    generate_summary()


if __name__ == "__main__":
    main()
