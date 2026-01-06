#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 母线巡检训练数据加载器
Busbar Plugin Training Data Loader

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training/checkpoints/busbar
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
CHECKPOINT_PATH = BASE_PATH / "checkpoints" / "busbar"

# =============================================================================
# 母线巡检检测类别定义 - 按电压等级
# =============================================================================
BUSBAR_CLASSES = {
    # 特高压 1000kV - 母线高度约20m，相间距约15m
    "UHV_1000kV_AC": {
        "classes": [
            "insulator_crack",        # 绝缘子破裂
            "insulator_dirty",        # 绝缘子污秽
            "insulator_flashover",    # 绝缘子闪络痕迹
            "fitting_loose",          # 金具松动
            "fitting_rust",           # 金具锈蚀
            "wire_damage",            # 导线损伤
            "foreign_object",         # 异物
            "bird",                   # 鸟类
            "bird_nest",              # 鸟巢
            "pin_missing",            # 销钉缺失
            "spacer_damage",          # 间隔棒损坏
            "corona_discharge",       # 电晕放电痕迹
            "insulator_tilt",         # 绝缘子串倾斜
            "sag_abnormal",           # 弧垂异常
        ],
        "specs": {
            "busbar_height_m": 20,
            "phase_spacing_m": 15,
            "conductor": "8×LGJ-630/45",
            "insulator_string": "40片以上",
            "min_target_size_px": 20,
        }
    },
    
    # 超高压 500kV - 母线高度约15m，相间距约9m
    "EHV_500kV": {
        "classes": [
            "insulator_crack",
            "insulator_dirty",
            "insulator_flashover",
            "fitting_loose",
            "fitting_rust",
            "wire_damage",
            "foreign_object",
            "bird",
            "bird_nest",
            "pin_missing",
            "spacer_damage",
        ],
        "specs": {
            "busbar_height_m": 15,
            "phase_spacing_m": 9,
            "conductor": "4×LGJ-630/45",
            "insulator_string": "28片",
            "min_target_size_px": 15,
        }
    },
    
    # 超高压 330kV - 母线高度约12m，相间距约7m
    "EHV_330kV": {
        "classes": [
            "insulator_crack",
            "insulator_dirty",
            "fitting_loose",
            "fitting_rust",
            "wire_damage",
            "foreign_object",
            "bird",
        ],
        "specs": {
            "busbar_height_m": 12,
            "phase_spacing_m": 7,
            "conductor": "2×LGJ-500/35",
            "insulator_string": "19片",
        }
    },
    
    # 高压 220kV - 母线高度约8m，相间距约4.5m
    "HV_220kV": {
        "classes": [
            "insulator_crack",
            "insulator_dirty",
            "insulator_flashover",
            "fitting_loose",
            "fitting_rust",
            "wire_damage",
            "foreign_object",
            "bird",
            "bird_nest",
        ],
        "specs": {
            "busbar_height_m": 8,
            "phase_spacing_m": 4.5,
            "conductor": "LGJ-400/35",
            "insulator_string": "13片",
            "min_target_size_px": 15,
        }
    },
    
    # 高压 110kV - 母线高度约6m，相间距约3m
    "HV_110kV": {
        "classes": [
            "insulator_crack",
            "insulator_dirty",
            "fitting_loose",
            "fitting_rust",
            "wire_damage",
            "foreign_object",
            "bird",
            "bird_nest",
        ],
        "specs": {
            "busbar_height_m": 6,
            "phase_spacing_m": 3,
            "conductor": "LGJ-240/30",
            "insulator_string": "7片",
        }
    },
    
    # 中压 35kV - 母线高度约4m，相间距约1.5m
    "MV_35kV": {
        "classes": [
            "insulator_crack",
            "insulator_dirty",
            "fitting_loose",
            "wire_damage",
            "foreign_object",
            "wire_entangle",          # 异物缠绕
        ],
        "specs": {
            "busbar_height_m": 4,
            "phase_spacing_m": 1.5,
            "conductor": "LGJ-120/20",
        }
    },
    
    # 低压 10kV - 母线高度约2.5m，相间距约0.3m
    "LV_10kV": {
        "classes": [
            "bolt_loose",             # 螺栓松动
            "wire_damage",
            "foreign_object",
            "insulator_damage",       # 绝缘子损坏
        ],
        "specs": {
            "busbar_height_m": 2.5,
            "phase_spacing_m": 0.3,
        }
    },
}

# 公开数据集配置
PUBLIC_DATASETS_BUSBAR = {
    "cplid": {
        "name": "Chinese Power Line Insulator Dataset",
        "url": "https://github.com/InsulatorData/InsulatorDataSet",
        "images": 848,
        "classes": ["normal_insulator", "defective_insulator"],
        "applicable": True,
    },
    "insulator_defect": {
        "name": "Insulator-Defect Detection Dataset",
        "url": "https://datasetninja.com/insulator-defect-detection",
        "images": 1600,
        "classes": ["insulator", "damaged", "flashover"],
        "applicable": True,
    },
    "mpid": {
        "name": "Merged Public Insulator Dataset",
        "url": "https://github.com/phd-benel/MPID",
        "images": 6000,
        "classes": ["insulator", "damaged", "flashover", "hammer"],
        "applicable": True,
    },
    "idid": {
        "name": "IEEE Insulator Defect Image Dataset",
        "url": "https://ieee-dataport.org/competitions/insulator-defect-detection",
        "images": 1600,
        "classes": ["good", "broken", "flashed", "insulator_string"],
        "applicable": True,
    },
}


class BusbarDataLoader:
    """母线巡检训练数据加载器"""
    
    def __init__(self, voltage_level: str):
        self.voltage_level = voltage_level
        self.config = BUSBAR_CLASSES.get(voltage_level, BUSBAR_CLASSES["HV_220kV"])
        self.classes = self.config["classes"]
        self.data_path = DATA_PATH / "processed" / voltage_level / "busbar"
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
            "voltage_level": self.voltage_level,
            "specs": self.config.get("specs", {}),
        }
        
        yaml_path = self.data_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
            
        return yaml_path
    
    def generate_classes_file(self) -> Path:
        """生成类别文件"""
        classes_path = self.data_path / "classes.txt"
        with open(classes_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.classes))
        return classes_path
    
    def generate_placeholder_samples(self, num_samples: int = 10) -> Dict[str, List[Path]]:
        """生成占位符样本"""
        generated = {"images": [], "labels": []}
        
        for split in ["train", "val", "test"]:
            split_samples = num_samples if split == "train" else num_samples // 5
            
            for i in range(split_samples):
                img_name = f"placeholder_{self.voltage_level}_busbar_{split}_{i:04d}"
                img_path = self.data_path / "images" / split / f"{img_name}.ppm"
                
                self._create_placeholder_image(img_path, 1280, 720)  # 更高分辨率用于小目标
                generated["images"].append(img_path)
                
                label_path = self.data_path / "labels" / split / f"{img_name}.txt"
                self._create_placeholder_label(label_path)
                generated["labels"].append(label_path)
        
        return generated
    
    def _create_placeholder_image(self, path: Path, width: int, height: int):
        """创建占位符PPM图像"""
        header = f"P6\n{width} {height}\n255\n"
        pixels = bytes([135, 206, 235] * width * height)  # 天蓝色背景
        
        with open(path, 'wb') as f:
            f.write(header.encode())
            f.write(pixels)
    
    def _create_placeholder_label(self, path: Path):
        """创建占位符YOLO标注"""
        num_objects = random.randint(1, 3)
        content = f"# 占位符标注 - {self.voltage_level} busbar\n"
        
        for _ in range(num_objects):
            class_id = random.randint(0, len(self.classes) - 1)
            x_center = random.uniform(0.2, 0.8)
            y_center = random.uniform(0.2, 0.8)
            width = random.uniform(0.05, 0.15)  # 小目标
            height = random.uniform(0.05, 0.2)
            content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        with open(path, 'w') as f:
            f.write(content)
    
    def generate_readme(self) -> Path:
        """生成数据集说明文件"""
        specs = self.config.get("specs", {})
        
        readme_content = f"""# 母线巡检训练数据集 - {self.voltage_level}

## 电压等级
{self.voltage_level}

## 设备规格
- 母线高度: {specs.get('busbar_height_m', 'N/A')} m
- 相间距: {specs.get('phase_spacing_m', 'N/A')} m
- 导线规格: {specs.get('conductor', 'N/A')}
- 绝缘子串: {specs.get('insulator_string', 'N/A')}
- 最小目标尺寸: {specs.get('min_target_size_px', 15)} px

## 检测类别 ({len(self.classes)}类)
| ID | 类别名称 | 说明 |
|----|----------|------|
"""
        class_descriptions = {
            "insulator_crack": "绝缘子破裂/自爆",
            "insulator_dirty": "绝缘子污秽",
            "insulator_flashover": "绝缘子闪络痕迹",
            "fitting_loose": "金具松动",
            "fitting_rust": "金具锈蚀",
            "wire_damage": "导线损伤",
            "foreign_object": "异物(塑料袋/风筝等)",
            "bird": "鸟类",
            "bird_nest": "鸟巢",
            "pin_missing": "销钉缺失",
            "spacer_damage": "间隔棒损坏",
            "corona_discharge": "电晕放电痕迹",
            "insulator_tilt": "绝缘子串倾斜",
            "sag_abnormal": "弧垂异常",
            "wire_entangle": "异物缠绕",
            "bolt_loose": "螺栓松动",
            "insulator_damage": "绝缘子损坏",
        }
        
        for i, cls in enumerate(self.classes):
            desc = class_descriptions.get(cls, cls)
            readme_content += f"| {i} | {cls} | {desc} |\n"
        
        readme_content += f"""

## 公开数据集
以下公开数据集可用于预训练:

| 数据集 | 图像数 | URL |
|--------|--------|-----|
"""
        for name, info in PUBLIC_DATASETS_BUSBAR.items():
            if info.get("applicable"):
                readme_content += f"| {info['name']} | {info['images']} | {info['url']} |\n"
        
        readme_content += f"""

## 数据采集建议
1. **绝缘子缺陷**
   - 使用无人机或望远镜头采集
   - 包含破裂、污秽、闪络等各类缺陷
   - 多角度、多距离拍摄

2. **金具状态**
   - 金具松动、锈蚀、销钉缺失
   - 近距离清晰图像

3. **异物检测**
   - 风筝、塑料袋、鸟巢等
   - 各种背景条件下的样本

4. **小目标检测**
   - 建议使用1920×1080或更高分辨率
   - 确保缺陷在图像中至少占15×15像素

## 下载公开数据集
```bash
# CPLID数据集
git clone https://github.com/InsulatorData/InsulatorDataSet.git

# MPID数据集
git clone https://github.com/phd-benel/MPID.git

# Insulator-Defect数据集 (需要从datasetninja下载)
# 访问: https://datasetninja.com/insulator-defect-detection
```

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = self.data_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        return readme_path
    
    def prepare_dataset(self, generate_placeholders: bool = True) -> Dict[str, Any]:
        """准备完整数据集"""
        result = {
            "voltage_level": self.voltage_level,
            "data_path": str(self.data_path),
            "classes": self.classes,
            "num_classes": len(self.classes),
            "specs": self.config.get("specs", {}),
            "files": {}
        }
        
        self.setup_directories()
        result["files"]["data_yaml"] = str(self.generate_data_yaml())
        result["files"]["classes_txt"] = str(self.generate_classes_file())
        result["files"]["readme"] = str(self.generate_readme())
        
        if generate_placeholders:
            placeholders = self.generate_placeholder_samples(num_samples=20)
            result["placeholder_count"] = len(placeholders["images"])
        
        logger.info(f"数据集准备完成: {self.voltage_level}/busbar")
        return result


def prepare_all_busbar_data():
    """准备所有电压等级的母线巡检训练数据"""
    results = {}
    
    for voltage_level in BUSBAR_CLASSES.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"准备数据: {voltage_level}")
        logger.info(f"{'='*60}")
        
        loader = BusbarDataLoader(voltage_level)
        results[voltage_level] = loader.prepare_dataset()
    
    report_path = DATA_PATH / "busbar_data_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="母线巡检训练数据准备")
    parser.add_argument("--voltage", "-v", type=str, help="电压等级")
    parser.add_argument("--all", action="store_true", help="准备所有电压等级")
    parser.add_argument("--list", action="store_true", help="列出所有电压等级")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n可用电压等级:")
        for level, config in BUSBAR_CLASSES.items():
            print(f"  - {level}: {len(config['classes'])}个检测类别")
    elif args.all:
        prepare_all_busbar_data()
    elif args.voltage:
        loader = BusbarDataLoader(args.voltage)
        result = loader.prepare_dataset()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
