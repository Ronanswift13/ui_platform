#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 主变压器训练数据加载器
Transformer Plugin Training Data Loader

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training/checkpoints/transformer
数据路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training/data
"""

import os
import json
import yaml
import shutil
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 路径配置
# =============================================================================
BASE_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
DATA_PATH = BASE_PATH / "data"
CHECKPOINT_PATH = BASE_PATH / "checkpoints" / "transformer"

# =============================================================================
# 检测类别定义 - 按电压等级
# =============================================================================
TRANSFORMER_CLASSES = {
    # 特高压 1000kV
    "UHV_1000kV_AC": {
        "classes": [
            "oil_leak",              # 油泄漏
            "rust",                  # 锈蚀
            "surface_damage",        # 表面破损
            "foreign_object",        # 异物
            "silica_gel_normal",     # 硅胶颜色正常(蓝色)
            "silica_gel_abnormal",   # 硅胶颜色异常(粉红/白色)
            "oil_level_normal",      # 油位正常
            "oil_level_abnormal",    # 油位异常
            "bushing_crack",         # 套管裂纹
            "porcelain_contamination", # 瓷套污损
            "partial_discharge",     # 局部放电
            "core_ground_current",   # 铁芯接地电流异常
            "winding_deformation",   # 绕组变形
        ],
        "thermal_thresholds": {"normal": 70, "warning": 85, "alarm": 100},
        "equipment_specs": {
            "capacity_mva": "1000-3000 (单相)",
            "oil_tank_diameter_m": "5-8",
            "bushing_height_m": "8-12",
        }
    },
    
    # 特高压 ±800kV直流
    "UHV_800kV_DC": {
        "classes": [
            "converter_transformer",  # 换流变压器
            "oil_leak",
            "bushing_crack",
            "cooling_system_abnormal", # 冷却系统异常
            "tap_changer_abnormal",   # 有载调压开关异常
            "valve_winding",          # 阀侧绕组
            "dc_bias",                # 直流偏磁
        ],
        "thermal_thresholds": {"normal": 70, "warning": 85, "alarm": 100},
        "equipment_specs": {
            "capacity_mva": "300-600",
            "type": "换流变压器",
        }
    },
    
    # 超高压 500kV
    "EHV_500kV": {
        "classes": [
            "oil_leak",
            "rust", 
            "surface_damage",
            "foreign_object",
            "silica_gel_normal",
            "silica_gel_abnormal",
            "oil_level_normal",
            "oil_level_abnormal",
            "bushing_crack",
            "porcelain_contamination",
        ],
        "thermal_thresholds": {"normal": 65, "warning": 80, "alarm": 95},
        "equipment_specs": {
            "capacity_mva": "500-1000",
            "oil_tank_diameter_m": "4-6",
            "bushing_height_m": "6-8",
        }
    },
    
    # 超高压 330kV
    "EHV_330kV": {
        "classes": [
            "oil_leak",
            "rust",
            "surface_damage",
            "silica_gel_normal",
            "silica_gel_abnormal",
            "oil_level_normal",
            "oil_level_abnormal",
        ],
        "thermal_thresholds": {"normal": 63, "warning": 78, "alarm": 92},
        "equipment_specs": {
            "capacity_mva": "180-500",
            "oil_tank_diameter_m": "3.5-5",
        }
    },
    
    # 高压 220kV
    "HV_220kV": {
        "classes": [
            "oil_leak",
            "rust",
            "surface_damage",
            "foreign_object",
            "silica_gel_normal",
            "silica_gel_abnormal",
            "oil_level_normal",
            "oil_level_abnormal",
        ],
        "thermal_thresholds": {"normal": 60, "warning": 75, "alarm": 85},
        "equipment_specs": {
            "capacity_mva": "50-180",
            "oil_tank_diameter_m": "2.5-4",
        }
    },
    
    # 高压 110kV
    "HV_110kV": {
        "classes": [
            "oil_leak",
            "rust",
            "surface_damage",
            "silica_gel_normal",
            "silica_gel_abnormal",
            "oil_level_normal",
            "oil_level_abnormal",
        ],
        "thermal_thresholds": {"normal": 55, "warning": 70, "alarm": 80},
        "equipment_specs": {
            "capacity_mva": "20-63",
            "oil_tank_diameter_m": "2-3",
        }
    },
    
    # 中压 35kV
    "MV_35kV": {
        "classes": [
            "oil_leak",
            "rust",
            "silica_gel_normal",
            "silica_gel_abnormal",
            "oil_level_normal",
            "oil_level_abnormal",
            "conductor_rust",  # 导电接头锈蚀
        ],
        "thermal_thresholds": {"normal": 50, "warning": 65, "alarm": 75},
        "equipment_specs": {
            "capacity_mva": "2-20",
            "oil_tank_diameter_m": "1-2",
        }
    },
    
    # 低压 10kV
    "LV_10kV": {
        "classes": [
            "rust",
            "oil_leak",
            "radiator_damage",    # 散热片损伤
            "box_transformer",    # 箱式变压器
            "terminal_damage",    # 接线端子损坏
        ],
        "thermal_thresholds": {"normal": 45, "warning": 60, "alarm": 70},
        "equipment_specs": {
            "capacity_mva": "0.1-2",
            "type": "干式/油浸式",
        }
    },
}


# =============================================================================
# 公开数据集配置
# =============================================================================
PUBLIC_DATASETS = {
    "transformer_thermal_mendeley": {
        "name": "Transformer Thermal Images Dataset",
        "url": "https://data.mendeley.com/datasets/8mg8mkc7k5/3",
        "description": "255张变压器热成像图像，9个类别(1正常+8故障)",
        "images": 255,
        "classes": ["healthy", "fault_1", "fault_2", "fault_3", "fault_4", 
                   "fault_5", "fault_6", "fault_7", "fault_8"],
        "format": "images",
        "applicable_voltage": ["all"],
        "download_method": "manual",
    },
    "power_equipment_ir": {
        "name": "Power Equipment Infrared Dataset",
        "url": "研究论文数据集",
        "description": "2560张电力设备红外图像，包含变压器类别",
        "images": 2560,
        "classes": ["transformer", "switch", "reactor", "current_transformer"],
        "format": "VOC",
        "applicable_voltage": ["HV_220kV", "EHV_500kV"],
        "download_method": "manual",
    },
    "substation_defect": {
        "name": "Substation Equipment Defect Dataset",
        "url": "State Grid Dataset",
        "description": "变电站设备缺陷数据集，包含油泄漏、锈蚀等",
        "images": 12968,
        "classes": ["oil_leak", "rust", "damaged_breather", "silicone_discoloration"],
        "format": "YOLO",
        "applicable_voltage": ["HV_220kV", "EHV_500kV"],
        "download_method": "manual",
    }
}


# =============================================================================
# 数据加载器类
# =============================================================================
class TransformerDataLoader:
    """主变压器训练数据加载器"""
    
    def __init__(self, voltage_level: str):
        self.voltage_level = voltage_level
        self.config = TRANSFORMER_CLASSES.get(voltage_level, TRANSFORMER_CLASSES["HV_220kV"])
        self.classes = self.config["classes"]
        self.data_path = DATA_PATH / "processed" / voltage_level / "transformer"
        self.checkpoint_path = CHECKPOINT_PATH / voltage_level
        
    def setup_directories(self):
        """创建目录结构"""
        directories = [
            self.data_path / "images" / "train",
            self.data_path / "images" / "val",
            self.data_path / "images" / "test",
            self.data_path / "labels" / "train",
            self.data_path / "labels" / "val",
            self.data_path / "labels" / "test",
            self.checkpoint_path,
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"创建目录结构: {self.data_path}")
        
    def generate_data_yaml(self) -> Path:
        """生成YOLO训练配置文件"""
        yaml_content = {
            "path": str(self.data_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": {i: cls for i, cls in enumerate(self.classes)},
            
            # 额外信息
            "voltage_level": self.voltage_level,
            "thermal_thresholds": self.config["thermal_thresholds"],
            "equipment_specs": self.config.get("equipment_specs", {}),
        }
        
        yaml_path = self.data_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"生成data.yaml: {yaml_path}")
        return yaml_path
    
    def generate_classes_file(self) -> Path:
        """生成类别文件"""
        classes_path = self.data_path / "classes.txt"
        with open(classes_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.classes))
        return classes_path
    
    def generate_placeholder_samples(self, num_samples: int = 10) -> Dict[str, List[Path]]:
        """
        生成占位符样本
        用于缺少真实数据时的结构占位
        """
        generated = {"images": [], "labels": []}
        
        for split in ["train", "val", "test"]:
            split_samples = num_samples if split == "train" else num_samples // 5
            
            for i in range(split_samples):
                # 生成占位符图像 (PPM格式，无需PIL)
                img_name = f"placeholder_{self.voltage_level}_transformer_{split}_{i:04d}"
                img_path = self.data_path / "images" / split / f"{img_name}.ppm"
                
                # 创建简单的PPM图像
                self._create_placeholder_image(img_path, 640, 640)
                generated["images"].append(img_path)
                
                # 生成占位符标注
                label_path = self.data_path / "labels" / split / f"{img_name}.txt"
                self._create_placeholder_label(label_path)
                generated["labels"].append(label_path)
        
        logger.info(f"生成{len(generated['images'])}个占位符样本")
        return generated
    
    def _create_placeholder_image(self, path: Path, width: int, height: int):
        """创建占位符PPM图像"""
        header = f"P6\n{width} {height}\n255\n"
        # 灰色背景
        pixels = bytes([128, 128, 128] * width * height)
        
        with open(path, 'wb') as f:
            f.write(header.encode())
            f.write(pixels)
    
    def _create_placeholder_label(self, path: Path):
        """创建占位符YOLO标注"""
        # 随机选择一个类别，生成示例标注
        class_id = random.randint(0, len(self.classes) - 1)
        x_center = random.uniform(0.2, 0.8)
        y_center = random.uniform(0.2, 0.8)
        width = random.uniform(0.1, 0.3)
        height = random.uniform(0.1, 0.3)
        
        content = f"# 占位符标注 - {self.voltage_level} transformer\n"
        content += f"# 类别: {self.classes[class_id]}\n"
        content += f"# 需要用真实数据替换\n"
        content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        with open(path, 'w') as f:
            f.write(content)
    
    def generate_readme(self) -> Path:
        """生成数据集说明文件"""
        readme_content = f"""# 主变压器训练数据集 - {self.voltage_level}

## 电压等级
{self.voltage_level}

## 设备规格
{json.dumps(self.config.get('equipment_specs', {}), indent=2, ensure_ascii=False)}

## 检测类别 ({len(self.classes)}类)
| ID | 类别名称 | 说明 |
|----|----------|------|
"""
        class_descriptions = {
            "oil_leak": "油泄漏",
            "rust": "锈蚀",
            "surface_damage": "表面破损",
            "foreign_object": "异物",
            "silica_gel_normal": "硅胶颜色正常(蓝色)",
            "silica_gel_abnormal": "硅胶颜色异常(粉红/白色)",
            "oil_level_normal": "油位正常",
            "oil_level_abnormal": "油位异常",
            "bushing_crack": "套管裂纹",
            "porcelain_contamination": "瓷套污损",
            "partial_discharge": "局部放电痕迹",
            "core_ground_current": "铁芯接地电流异常",
            "winding_deformation": "绕组变形",
            "converter_transformer": "换流变压器",
            "cooling_system_abnormal": "冷却系统异常",
            "tap_changer_abnormal": "有载调压开关异常",
            "valve_winding": "阀侧绕组",
            "dc_bias": "直流偏磁",
            "conductor_rust": "导电接头锈蚀",
            "radiator_damage": "散热片损伤",
            "box_transformer": "箱式变压器",
            "terminal_damage": "接线端子损坏",
        }
        
        for i, cls in enumerate(self.classes):
            desc = class_descriptions.get(cls, cls)
            readme_content += f"| {i} | {cls} | {desc} |\n"
        
        readme_content += f"""

## 热成像阈值
- 正常温度: ≤{self.config['thermal_thresholds']['normal']}°C
- 预警温度: {self.config['thermal_thresholds']['warning']}°C
- 报警温度: ≥{self.config['thermal_thresholds']['alarm']}°C

## 数据来源
### 公开数据集
1. **Transformer Thermal Images Dataset** (Mendeley)
   - URL: https://data.mendeley.com/datasets/8mg8mkc7k5/3
   - 255张热成像图像

2. **Substation Equipment Defect Dataset**
   - 包含油泄漏、锈蚀等缺陷

### 需要现场采集的数据
- 套管裂纹高清图像
- 瓷套污损图像
- 局部放电痕迹图像
- 油位刻度不同状态图像
- 硅胶颜色变化图像

## 数据格式
- 图像格式: JPG/PNG, 建议分辨率1920x1080或更高
- 标注格式: YOLO格式 (class_id x_center y_center width height)
- 标注文件: 与图像同名的.txt文件

## 目录结构
```
{self.voltage_level}/transformer/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
├── classes.txt
└── README.md
```

## 训练建议
- 建议每类至少100张图像
- 包含正常和异常状态
- 多角度、多光照条件
- 包含热成像和可见光图像

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = self.data_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        return readme_path
    
    def prepare_dataset(self, generate_placeholders: bool = True) -> Dict[str, Any]:
        """
        准备完整数据集
        """
        result = {
            "voltage_level": self.voltage_level,
            "data_path": str(self.data_path),
            "classes": self.classes,
            "num_classes": len(self.classes),
            "files": {}
        }
        
        # 1. 创建目录
        self.setup_directories()
        
        # 2. 生成配置文件
        result["files"]["data_yaml"] = str(self.generate_data_yaml())
        result["files"]["classes_txt"] = str(self.generate_classes_file())
        result["files"]["readme"] = str(self.generate_readme())
        
        # 3. 生成占位符数据(如果需要)
        if generate_placeholders:
            placeholders = self.generate_placeholder_samples(num_samples=20)
            result["placeholder_count"] = len(placeholders["images"])
        
        logger.info(f"数据集准备完成: {self.voltage_level}/transformer")
        return result


# =============================================================================
# 批量数据准备
# =============================================================================
def prepare_all_transformer_data():
    """准备所有电压等级的主变压器训练数据"""
    results = {}
    
    for voltage_level in TRANSFORMER_CLASSES.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"准备数据: {voltage_level}")
        logger.info(f"{'='*60}")
        
        loader = TransformerDataLoader(voltage_level)
        results[voltage_level] = loader.prepare_dataset()
    
    # 生成汇总报告
    report_path = DATA_PATH / "transformer_data_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n汇总报告: {report_path}")
    return results


# =============================================================================
# 命令行接口
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="主变压器训练数据准备")
    parser.add_argument("--voltage", "-v", type=str, help="电压等级")
    parser.add_argument("--all", action="store_true", help="准备所有电压等级")
    parser.add_argument("--list", action="store_true", help="列出所有电压等级")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n可用电压等级:")
        for level, config in TRANSFORMER_CLASSES.items():
            print(f"  - {level}: {len(config['classes'])}个检测类别")
    elif args.all:
        prepare_all_transformer_data()
    elif args.voltage:
        loader = TransformerDataLoader(args.voltage)
        result = loader.prepare_dataset()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
