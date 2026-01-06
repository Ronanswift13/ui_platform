#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 电容器巡检训练数据加载器
Capacitor Plugin Training Data Loader

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training/checkpoints/capacitor
"""

import os
import json
import yaml
import logging
import random
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
DATA_PATH = BASE_PATH / "data"
CHECKPOINT_PATH = BASE_PATH / "checkpoints" / "capacitor"

# =============================================================================
# 电容器巡检检测类别定义 - 按电压等级
# =============================================================================
CAPACITOR_CLASSES = {
    "UHV_1000kV_AC": {
        "classes": [
            "capacitor_unit",         # 电容器单元
            "capacitor_tilted",       # 电容器倾斜
            "capacitor_fallen",       # 电容器掉落
            "capacitor_missing",      # 电容器缺失
            "fuse_normal",            # 熔丝正常
            "fuse_blown",             # 熔丝熔断
            "silver_contact_normal",  # 银触点正常
            "silver_contact_damage",  # 银触点损坏
            "person",                 # 人员入侵
            "vehicle",                # 车辆入侵
            "foreign_object",         # 异物入侵
        ],
        "specs": {
            "type": "高压并联电容器组",
            "capacity_mvar": "60-120",
        }
    },
    
    "EHV_500kV": {
        "classes": [
            "capacitor_unit",
            "capacitor_tilted",
            "capacitor_fallen",
            "capacitor_missing",
            "fuse_normal",
            "fuse_blown",
            "person",
            "vehicle",
        ],
        "specs": {
            "type": "高压并联电容器组",
            "capacity_mvar": "30-60",
        }
    },
    
    "EHV_330kV": {
        "classes": [
            "capacitor_unit",
            "capacitor_tilted",
            "capacitor_fallen",
            "fuse_blown",
        ],
        "specs": {
            "type": "高压并联电容器组",
            "capacity_mvar": "20-40",
        }
    },
    
    "HV_220kV": {
        "classes": [
            "capacitor_unit",
            "capacitor_tilted",
            "capacitor_fallen",
            "capacitor_missing",
            "person",
            "vehicle",
        ],
        "specs": {
            "model": "CW1-220",
            "type": "高压并联电容器",
        }
    },
    
    "HV_110kV": {
        "classes": [
            "capacitor_unit",
            "capacitor_tilted",
            "capacitor_fallen",
            "person",
            "vehicle",
        ],
        "specs": {
            "model": "TBB10-6600",
            "type": "高压并联电容器",
        }
    },
    
    "MV_35kV": {
        "classes": [
            "capacitor_unit",
            "capacitor_tilted",
            "capacitor_missing",
            "safety_distance_violation",  # 安全距离违规
        ],
        "specs": {
            "type": "中压并联电容器",
        }
    },
    
    "LV_10kV": {
        "classes": [
            "capacitor_normal",       # 电容器正常
            "capacitor_bulge",        # 电容器鼓包
            "capacitor_leak",         # 电容器漏液
            "capacitor_discolor",     # 电容器变色
        ],
        "specs": {
            "type": "低压补偿电容器/智能电容器",
        }
    },
}


class CapacitorDataLoader:
    """电容器巡检训练数据加载器"""
    
    def __init__(self, voltage_level: str):
        self.voltage_level = voltage_level
        self.config = CAPACITOR_CLASSES.get(voltage_level, CAPACITOR_CLASSES["HV_220kV"])
        self.classes = self.config["classes"]
        self.data_path = DATA_PATH / "processed" / voltage_level / "capacitor"
        self.checkpoint_path = CHECKPOINT_PATH / voltage_level
        
    def setup_directories(self):
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
        yaml_content = {
            "path": str(self.data_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": {i: cls for i, cls in enumerate(self.classes)},
            "voltage_level": self.voltage_level,
            "specs": self.config.get("specs", {}),
        }
        yaml_path = self.data_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        return yaml_path
    
    def generate_classes_file(self) -> Path:
        classes_path = self.data_path / "classes.txt"
        with open(classes_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.classes))
        return classes_path
    
    def generate_placeholder_samples(self, num_samples: int = 10) -> Dict[str, List[Path]]:
        generated = {"images": [], "labels": []}
        for split in ["train", "val", "test"]:
            split_samples = num_samples if split == "train" else num_samples // 5
            for i in range(split_samples):
                img_name = f"placeholder_{self.voltage_level}_capacitor_{split}_{i:04d}"
                img_path = self.data_path / "images" / split / f"{img_name}.ppm"
                self._create_placeholder_image(img_path, 640, 640)
                generated["images"].append(img_path)
                label_path = self.data_path / "labels" / split / f"{img_name}.txt"
                self._create_placeholder_label(label_path)
                generated["labels"].append(label_path)
        return generated
    
    def _create_placeholder_image(self, path: Path, width: int, height: int):
        header = f"P6\n{width} {height}\n255\n"
        pixels = bytes([200, 200, 200] * width * height)
        with open(path, 'wb') as f:
            f.write(header.encode())
            f.write(pixels)
    
    def _create_placeholder_label(self, path: Path):
        class_id = random.randint(0, len(self.classes) - 1)
        x_center = random.uniform(0.3, 0.7)
        y_center = random.uniform(0.3, 0.7)
        width = random.uniform(0.1, 0.3)
        height = random.uniform(0.1, 0.3)
        content = f"# 占位符 - {self.voltage_level} capacitor\n"
        content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        with open(path, 'w') as f:
            f.write(content)
    
    def generate_readme(self) -> Path:
        readme_content = f"""# 电容器巡检训练数据集 - {self.voltage_level}

## 设备规格
{json.dumps(self.config.get('specs', {}), indent=2, ensure_ascii=False)}

## 检测类别 ({len(self.classes)}类)
| ID | 类别 | 说明 |
|----|------|------|
"""
        descriptions = {
            "capacitor_unit": "电容器单元",
            "capacitor_tilted": "电容器倾斜",
            "capacitor_fallen": "电容器掉落",
            "capacitor_missing": "电容器缺失",
            "fuse_normal": "熔丝正常",
            "fuse_blown": "熔丝熔断",
            "silver_contact_normal": "银触点正常",
            "silver_contact_damage": "银触点损坏",
            "person": "人员入侵",
            "vehicle": "车辆入侵",
            "foreign_object": "异物入侵",
            "safety_distance_violation": "安全距离违规",
            "capacitor_normal": "电容器正常",
            "capacitor_bulge": "电容器鼓包",
            "capacitor_leak": "电容器漏液",
            "capacitor_discolor": "电容器变色",
        }
        for i, cls in enumerate(self.classes):
            readme_content += f"| {i} | {cls} | {descriptions.get(cls, cls)} |\n"
        
        readme_content += f"""

## 数据采集建议
- 电容器正常/异常状态各100张以上
- 倾斜、掉落需模拟采集
- 人员/车辆入侵使用安全监控数据集预训练

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        readme_path = self.data_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        return readme_path
    
    def prepare_dataset(self, generate_placeholders: bool = True) -> Dict[str, Any]:
        result = {
            "voltage_level": self.voltage_level,
            "data_path": str(self.data_path),
            "classes": self.classes,
            "num_classes": len(self.classes),
            "files": {}
        }
        self.setup_directories()
        result["files"]["data_yaml"] = str(self.generate_data_yaml())
        result["files"]["classes_txt"] = str(self.generate_classes_file())
        result["files"]["readme"] = str(self.generate_readme())
        if generate_placeholders:
            placeholders = self.generate_placeholder_samples(num_samples=15)
            result["placeholder_count"] = len(placeholders["images"])
        logger.info(f"数据集准备完成: {self.voltage_level}/capacitor")
        return result


def prepare_all_capacitor_data():
    results = {}
    for voltage_level in CAPACITOR_CLASSES.keys():
        loader = CapacitorDataLoader(voltage_level)
        results[voltage_level] = loader.prepare_dataset()
    report_path = DATA_PATH / "capacitor_data_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


# =============================================================================
# 表计读数数据加载器
# =============================================================================
METER_CLASSES = {
    "UHV_1000kV_AC": {
        "classes": [
            "sf6_pressure_gauge",     # SF6压力表
            "sf6_density_relay",      # SF6密度继电器
            "oil_temp_gauge",         # 油温表
            "oil_level_gauge",        # 油位计
            "gas_relay",              # 气体继电器
            "digital_display",        # 数字显示屏
            "pointer_gauge",          # 指针式表计
            "dial_0", "dial_1", "dial_2", "dial_3", "dial_4",
            "dial_5", "dial_6", "dial_7", "dial_8", "dial_9",
        ],
        "specs": {
            "sf6_pressure_range_mpa": "0.4-0.6",
        }
    },
    
    "EHV_500kV": {
        "classes": [
            "sf6_pressure_gauge",
            "sf6_density_relay",
            "gas_relay",
            "oil_temp_gauge",
            "oil_level_gauge",
            "dial_0", "dial_1", "dial_2", "dial_3", "dial_4",
            "dial_5", "dial_6", "dial_7", "dial_8", "dial_9",
        ],
        "specs": {
            "sf6_pressure_range_mpa": "0.35-0.55",
        }
    },
    
    "EHV_330kV": {
        "classes": [
            "sf6_pressure_gauge",
            "oil_temp_gauge",
            "oil_level_gauge",
            "dial_0", "dial_1", "dial_2", "dial_3", "dial_4",
            "dial_5", "dial_6", "dial_7", "dial_8", "dial_9",
        ],
        "specs": {}
    },
    
    "HV_220kV": {
        "classes": [
            "sf6_pressure_gauge",
            "oil_temp_gauge",
            "oil_level_gauge",
            "dial_0", "dial_1", "dial_2", "dial_3", "dial_4",
            "dial_5", "dial_6", "dial_7", "dial_8", "dial_9",
        ],
        "specs": {}
    },
    
    "HV_110kV": {
        "classes": [
            "sf6_pressure_gauge",
            "oil_temp_gauge",
            "oil_level_gauge",
            "dial_0", "dial_1", "dial_2", "dial_3", "dial_4",
            "dial_5", "dial_6", "dial_7", "dial_8", "dial_9",
        ],
        "specs": {}
    },
    
    "MV_35kV": {
        "classes": [
            "digital_meter",          # 数字电表
            "analog_meter",           # 模拟表计
            "transformer_monitor",    # 变压器监测仪
            "dial_0", "dial_1", "dial_2", "dial_3", "dial_4",
            "dial_5", "dial_6", "dial_7", "dial_8", "dial_9",
        ],
        "specs": {}
    },
    
    "LV_10kV": {
        "classes": [
            "energy_meter",           # 电能表
            "water_meter",            # 水表(多功能仪表)
            "pointer_meter",          # 指针表
            "digital_meter",          # 数字表
            "seven_segment",          # 七段数码管
            "dial_0", "dial_1", "dial_2", "dial_3", "dial_4",
            "dial_5", "dial_6", "dial_7", "dial_8", "dial_9",
        ],
        "specs": {}
    },
}

# 公开表计读数数据集
PUBLIC_DATASETS_METER = {
    "ufpr_amr": {
        "name": "UFPR-AMR Dataset",
        "url": "https://github.com/raysonlaroca/ufpr-amr-dataset",
        "images": 2000,
        "description": "电表自动抄表数据集",
    },
    "copel_amr": {
        "name": "Copel-AMR Dataset",
        "url": "https://github.com/raysonlaroca/copel-amr-dataset",
        "images": 12500,
        "description": "现场采集电表图像",
    },
    "yuva_eb": {
        "name": "YUVA EB Dataset",
        "url": "ResearchGate论文",
        "description": "七段数码管电表数据集",
    },
}


class MeterDataLoader:
    """表计读数训练数据加载器"""
    
    def __init__(self, voltage_level: str):
        self.voltage_level = voltage_level
        self.config = METER_CLASSES.get(voltage_level, METER_CLASSES["HV_220kV"])
        self.classes = self.config["classes"]
        self.data_path = DATA_PATH / "processed" / voltage_level / "meter"
        self.checkpoint_path = BASE_PATH / "checkpoints" / "meter" / voltage_level
        
    def setup_directories(self):
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
        yaml_content = {
            "path": str(self.data_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": {i: cls for i, cls in enumerate(self.classes)},
            "voltage_level": self.voltage_level,
        }
        yaml_path = self.data_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        return yaml_path
    
    def generate_classes_file(self) -> Path:
        classes_path = self.data_path / "classes.txt"
        with open(classes_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.classes))
        return classes_path
    
    def generate_placeholder_samples(self, num_samples: int = 10) -> Dict[str, List[Path]]:
        generated = {"images": [], "labels": []}
        for split in ["train", "val", "test"]:
            split_samples = num_samples if split == "train" else num_samples // 5
            for i in range(split_samples):
                img_name = f"placeholder_{self.voltage_level}_meter_{split}_{i:04d}"
                img_path = self.data_path / "images" / split / f"{img_name}.ppm"
                self._create_placeholder_image(img_path, 416, 416)
                generated["images"].append(img_path)
                label_path = self.data_path / "labels" / split / f"{img_name}.txt"
                self._create_placeholder_label(label_path)
                generated["labels"].append(label_path)
        return generated
    
    def _create_placeholder_image(self, path: Path, width: int, height: int):
        header = f"P6\n{width} {height}\n255\n"
        pixels = bytes([50, 50, 50] * width * height)  # 深色背景
        with open(path, 'wb') as f:
            f.write(header.encode())
            f.write(pixels)
    
    def _create_placeholder_label(self, path: Path):
        class_id = random.randint(0, len(self.classes) - 1)
        x_center = random.uniform(0.3, 0.7)
        y_center = random.uniform(0.3, 0.7)
        width = random.uniform(0.15, 0.4)
        height = random.uniform(0.1, 0.3)
        content = f"# 占位符 - {self.voltage_level} meter\n"
        content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        with open(path, 'w') as f:
            f.write(content)
    
    def generate_readme(self) -> Path:
        readme_content = f"""# 表计读数训练数据集 - {self.voltage_level}

## 检测类别 ({len(self.classes)}类)
| ID | 类别 | 说明 |
|----|------|------|
"""
        descriptions = {
            "sf6_pressure_gauge": "SF6压力表",
            "sf6_density_relay": "SF6密度继电器",
            "oil_temp_gauge": "油温表",
            "oil_level_gauge": "油位计",
            "gas_relay": "气体继电器",
            "digital_display": "数字显示屏",
            "pointer_gauge": "指针式表计",
            "digital_meter": "数字电表",
            "analog_meter": "模拟表计",
            "transformer_monitor": "变压器监测仪",
            "energy_meter": "电能表",
            "water_meter": "水表",
            "pointer_meter": "指针表",
            "seven_segment": "七段数码管",
        }
        for i, cls in enumerate(self.classes):
            if cls.startswith("dial_"):
                desc = f"数字{cls[-1]}"
            else:
                desc = descriptions.get(cls, cls)
            readme_content += f"| {i} | {cls} | {desc} |\n"
        
        readme_content += f"""

## 公开数据集
| 数据集 | 图像数 | URL |
|--------|--------|-----|
"""
        for name, info in PUBLIC_DATASETS_METER.items():
            readme_content += f"| {info['name']} | {info.get('images', 'N/A')} | {info['url']} |\n"
        
        readme_content += f"""

## OCR模型建议
- 表计检测: YOLOv8
- 数字识别: CRNN/Transformer OCR
- 预训练: MNIST + 七段数码管数据集

## 下载公开数据集
```bash
# UFPR-AMR数据集
git clone https://github.com/raysonlaroca/ufpr-amr-dataset.git

# Copel-AMR数据集
git clone https://github.com/raysonlaroca/copel-amr-dataset.git
```

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        readme_path = self.data_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        return readme_path
    
    def prepare_dataset(self, generate_placeholders: bool = True) -> Dict[str, Any]:
        result = {
            "voltage_level": self.voltage_level,
            "data_path": str(self.data_path),
            "classes": self.classes,
            "num_classes": len(self.classes),
            "files": {}
        }
        self.setup_directories()
        result["files"]["data_yaml"] = str(self.generate_data_yaml())
        result["files"]["classes_txt"] = str(self.generate_classes_file())
        result["files"]["readme"] = str(self.generate_readme())
        if generate_placeholders:
            placeholders = self.generate_placeholder_samples(num_samples=20)
            result["placeholder_count"] = len(placeholders["images"])
        logger.info(f"数据集准备完成: {self.voltage_level}/meter")
        return result


def prepare_all_meter_data():
    results = {}
    for voltage_level in METER_CLASSES.keys():
        loader = MeterDataLoader(voltage_level)
        results[voltage_level] = loader.prepare_dataset()
    report_path = DATA_PATH / "meter_data_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="电容器/表计训练数据准备")
    parser.add_argument("--plugin", "-p", choices=["capacitor", "meter"], required=True)
    parser.add_argument("--voltage", "-v", type=str, help="电压等级")
    parser.add_argument("--all", action="store_true", help="准备所有电压等级")
    parser.add_argument("--list", action="store_true", help="列出所有电压等级")
    
    args = parser.parse_args()
    
    if args.plugin == "capacitor":
        classes_dict = CAPACITOR_CLASSES
        prepare_func = prepare_all_capacitor_data
        loader_class = CapacitorDataLoader
    else:
        classes_dict = METER_CLASSES
        prepare_func = prepare_all_meter_data
        loader_class = MeterDataLoader
    
    if args.list:
        print(f"\n{args.plugin} 可用电压等级:")
        for level, config in classes_dict.items():
            print(f"  - {level}: {len(config['classes'])}个检测类别")
    elif args.all:
        prepare_func()
    elif args.voltage:
        loader = loader_class(args.voltage)
        result = loader.prepare_dataset()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
