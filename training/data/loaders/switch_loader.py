#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 开关间隔训练数据加载器
Switch Plugin Training Data Loader

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training/checkpoints/switch
"""

import os
import json
import yaml
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
DATA_PATH = BASE_PATH / "data"
CHECKPOINT_PATH = BASE_PATH / "checkpoints" / "switch"

# =============================================================================
# 开关间隔检测类别定义 - 按电压等级
# =============================================================================
SWITCH_CLASSES = {
    # 特高压 1000kV
    "UHV_1000kV_AC": {
        "classes": [
            "breaker_open",          # 断路器分位
            "breaker_closed",        # 断路器合位
            "isolator_open",         # 隔离开关分位
            "isolator_closed",       # 隔离开关合位
            "grounding_open",        # 接地开关分位
            "grounding_closed",      # 接地开关合位
            "indicator_red",         # 指示灯红色
            "indicator_green",       # 指示灯绿色
            "gis_position_indicator", # GIS位置指示器
            "sf6_density_relay",     # SF6密度继电器
            "gis_double_break",      # GIS双断口
            "gis_four_break",        # GIS四断口
        ],
        "angle_reference": {
            "breaker": {"open_deg": -65, "closed_deg": 35},
            "isolator": {"open_deg": -75, "closed_deg": 25},
            "grounding": {"open_deg": -85, "closed_deg": 15},
        },
        "equipment_specs": {
            "breaker_type": "GIS SF6断路器",
            "rated_current_ka": "50/63",
            "break_type": "双断口/四断口",
        }
    },
    
    # 特高压 ±800kV直流
    "UHV_800kV_DC": {
        "classes": [
            "converter_valve",       # 换流阀
            "thyristor_normal",      # 晶闸管正常
            "thyristor_fault",       # 晶闸管故障
            "cooling_water_normal",  # 冷却水正常
            "cooling_water_abnormal", # 冷却水异常
            "valve_temperature_normal", # 阀温正常
            "valve_temperature_abnormal", # 阀温异常
            "dc_breaker_open",       # 直流断路器分位
            "dc_breaker_closed",     # 直流断路器合位
            "bypass_switch",         # 旁路开关
        ],
        "angle_reference": {
            "dc_breaker": {"open_deg": -60, "closed_deg": 30},
        },
        "equipment_specs": {
            "type": "晶闸管换流阀",
            "pulse": "12脉动",
            "transmission_mw": 8000,
        }
    },
    
    # 超高压 500kV
    "EHV_500kV": {
        "classes": [
            "breaker_open",
            "breaker_closed", 
            "isolator_open",
            "isolator_closed",
            "grounding_open",
            "grounding_closed",
            "indicator_red",
            "indicator_green",
            "gis_position_indicator",
            "sf6_pressure_normal",
            "sf6_pressure_abnormal",
        ],
        "angle_reference": {
            "breaker": {"open_deg": -60, "closed_deg": 30},
            "isolator": {"open_deg": -70, "closed_deg": 20},
            "grounding": {"open_deg": -80, "closed_deg": 10},
        },
        "equipment_specs": {
            "breaker_type": "GIS/AIS SF6断路器",
            "rated_current_ka": "40/50",
        }
    },
    
    # 超高压 330kV
    "EHV_330kV": {
        "classes": [
            "breaker_open",
            "breaker_closed",
            "isolator_open", 
            "isolator_closed",
            "grounding_open",
            "grounding_closed",
            "indicator_red",
            "indicator_green",
        ],
        "angle_reference": {
            "breaker": {"open_deg": -58, "closed_deg": 32},
            "isolator": {"open_deg": -68, "closed_deg": 22},
            "grounding": {"open_deg": -78, "closed_deg": 12},
        },
        "equipment_specs": {
            "breaker_type": "SF6断路器",
            "rated_current_ka": "31.5/40",
        }
    },
    
    # 高压 220kV
    "HV_220kV": {
        "classes": [
            "breaker_open",
            "breaker_closed",
            "isolator_open",
            "isolator_closed",
            "grounding_open",
            "grounding_closed",
            "indicator_red",
            "indicator_green",
            "mechanism_box",         # 机构箱
        ],
        "angle_reference": {
            "breaker": {"open_deg": -55, "closed_deg": 35},
            "isolator": {"open_deg": -65, "closed_deg": 25},
            "grounding": {"open_deg": -75, "closed_deg": 15},
        },
        "equipment_specs": {
            "breaker_type": "SF6/真空断路器",
            "rated_current_ka": "31.5/40",
        }
    },
    
    # 高压 110kV
    "HV_110kV": {
        "classes": [
            "breaker_open",
            "breaker_closed",
            "isolator_open",
            "isolator_closed",
            "grounding_open",
            "grounding_closed",
            "indicator_red",
            "indicator_green",
        ],
        "angle_reference": {
            "breaker": {"open_deg": -55, "closed_deg": 35},
            "isolator": {"open_deg": -65, "closed_deg": 25},
            "grounding": {"open_deg": -75, "closed_deg": 15},
        },
        "equipment_specs": {
            "breaker_type": "SF6/真空断路器",
            "rated_current_ka": "25/31.5",
        }
    },
    
    # 中压 35kV
    "MV_35kV": {
        "classes": [
            "breaker_open",
            "breaker_closed",
            "isolator_open",
            "isolator_closed",
            "indicator_red",
            "indicator_green",
            "cabinet_door_open",     # 柜门打开
            "cabinet_door_closed",   # 柜门关闭
            "handcart_test",         # 手车试验位
            "handcart_work",         # 手车工作位
        ],
        "angle_reference": {
            "breaker": {"open_deg": -50, "closed_deg": 40},
            "isolator": {"open_deg": -60, "closed_deg": 30},
        },
        "equipment_specs": {
            "breaker_type": "真空断路器/SF6断路器",
            "cabinet_type": "户内开关柜",
        }
    },
    
    # 低压 10kV
    "LV_10kV": {
        "classes": [
            "cabinet_switch_on",     # 开关柜合闸
            "cabinet_switch_off",    # 开关柜分闸
            "air_breaker_on",        # 空气断路器合闸
            "air_breaker_off",       # 空气断路器分闸
            "indicator_light_on",    # 指示灯亮
            "indicator_light_off",   # 指示灯灭
            "label_readable",        # 标签可读
            "label_damaged",         # 标签损坏
            "ring_main_unit",        # 环网柜
        ],
        "angle_reference": {},
        "equipment_specs": {
            "breaker_type": "真空断路器/负荷开关",
            "cabinet_type": "环网柜/充气柜",
        }
    },
}


class SwitchDataLoader:
    """开关间隔训练数据加载器"""
    
    def __init__(self, voltage_level: str):
        self.voltage_level = voltage_level
        self.config = SWITCH_CLASSES.get(voltage_level, SWITCH_CLASSES["HV_220kV"])
        self.classes = self.config["classes"]
        self.data_path = DATA_PATH / "processed" / voltage_level / "switch"
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
            "angle_reference": self.config.get("angle_reference", {}),
            "equipment_specs": self.config.get("equipment_specs", {}),
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
                img_name = f"placeholder_{self.voltage_level}_switch_{split}_{i:04d}"
                img_path = self.data_path / "images" / split / f"{img_name}.ppm"
                
                self._create_placeholder_image(img_path, 640, 640)
                generated["images"].append(img_path)
                
                label_path = self.data_path / "labels" / split / f"{img_name}.txt"
                self._create_placeholder_label(label_path)
                generated["labels"].append(label_path)
        
        return generated
    
    def _create_placeholder_image(self, path: Path, width: int, height: int):
        """创建占位符PPM图像"""
        header = f"P6\n{width} {height}\n255\n"
        pixels = bytes([100, 100, 100] * width * height)
        
        with open(path, 'wb') as f:
            f.write(header.encode())
            f.write(pixels)
    
    def _create_placeholder_label(self, path: Path):
        """创建占位符YOLO标注"""
        class_id = random.randint(0, len(self.classes) - 1)
        x_center = random.uniform(0.3, 0.7)
        y_center = random.uniform(0.3, 0.7)
        width = random.uniform(0.1, 0.25)
        height = random.uniform(0.1, 0.25)
        
        content = f"# 占位符标注 - {self.voltage_level} switch\n"
        content += f"# 类别: {self.classes[class_id]}\n"
        content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        with open(path, 'w') as f:
            f.write(content)
    
    def generate_readme(self) -> Path:
        """生成数据集说明文件"""
        readme_content = f"""# 开关间隔训练数据集 - {self.voltage_level}

## 电压等级
{self.voltage_level}

## 设备规格
{json.dumps(self.config.get('equipment_specs', {}), indent=2, ensure_ascii=False)}

## 角度参考值
"""
        for switch_type, angles in self.config.get("angle_reference", {}).items():
            readme_content += f"- **{switch_type}**: 分位 {angles.get('open_deg', 'N/A')}°, 合位 {angles.get('closed_deg', 'N/A')}°\n"

        readme_content += f"""

## 检测类别 ({len(self.classes)}类)
| ID | 类别名称 | 说明 |
|----|----------|------|
"""
        class_descriptions = {
            "breaker_open": "断路器分位",
            "breaker_closed": "断路器合位",
            "isolator_open": "隔离开关分位",
            "isolator_closed": "隔离开关合位",
            "grounding_open": "接地开关分位",
            "grounding_closed": "接地开关合位",
            "indicator_red": "指示灯红色(分位)",
            "indicator_green": "指示灯绿色(合位)",
            "gis_position_indicator": "GIS位置指示器",
            "sf6_density_relay": "SF6密度继电器",
            "gis_double_break": "GIS双断口断路器",
            "gis_four_break": "GIS四断口断路器",
            "converter_valve": "换流阀",
            "thyristor_normal": "晶闸管正常",
            "thyristor_fault": "晶闸管故障",
            "cooling_water_normal": "冷却水正常",
            "cooling_water_abnormal": "冷却水异常",
            "valve_temperature_normal": "阀温正常",
            "valve_temperature_abnormal": "阀温异常",
            "dc_breaker_open": "直流断路器分位",
            "dc_breaker_closed": "直流断路器合位",
            "bypass_switch": "旁路开关",
            "sf6_pressure_normal": "SF6压力正常",
            "sf6_pressure_abnormal": "SF6压力异常",
            "mechanism_box": "机构箱",
            "cabinet_door_open": "柜门打开",
            "cabinet_door_closed": "柜门关闭",
            "handcart_test": "手车试验位",
            "handcart_work": "手车工作位",
            "cabinet_switch_on": "开关柜合闸",
            "cabinet_switch_off": "开关柜分闸",
            "air_breaker_on": "空气断路器合闸",
            "air_breaker_off": "空气断路器分闸",
            "indicator_light_on": "指示灯亮",
            "indicator_light_off": "指示灯灭",
            "label_readable": "标签可读",
            "label_damaged": "标签损坏",
            "ring_main_unit": "环网柜",
        }
        
        for i, cls in enumerate(self.classes):
            desc = class_descriptions.get(cls, cls)
            readme_content += f"| {i} | {cls} | {desc} |\n"
        
        readme_content += f"""

## 数据采集建议
1. **断路器/隔离开关状态**
   - 分位和合位各拍摄多角度图像
   - 包含机构箱、指示器在同一画面
   - 不同光照条件下采集

2. **指示灯识别**
   - 红色和绿色指示灯
   - 包含亮灭状态
   - 不同背景光照

3. **GIS设备**(如适用)
   - 位置指示器各状态
   - SF6密度继电器
   - 操作机构

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
            "angle_reference": self.config.get("angle_reference", {}),
            "files": {}
        }
        
        self.setup_directories()
        result["files"]["data_yaml"] = str(self.generate_data_yaml())
        result["files"]["classes_txt"] = str(self.generate_classes_file())
        result["files"]["readme"] = str(self.generate_readme())
        
        if generate_placeholders:
            placeholders = self.generate_placeholder_samples(num_samples=20)
            result["placeholder_count"] = len(placeholders["images"])
        
        logger.info(f"数据集准备完成: {self.voltage_level}/switch")
        return result


def prepare_all_switch_data():
    """准备所有电压等级的开关间隔训练数据"""
    results = {}
    
    for voltage_level in SWITCH_CLASSES.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"准备数据: {voltage_level}")
        logger.info(f"{'='*60}")
        
        loader = SwitchDataLoader(voltage_level)
        results[voltage_level] = loader.prepare_dataset()
    
    report_path = DATA_PATH / "switch_data_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="开关间隔训练数据准备")
    parser.add_argument("--voltage", "-v", type=str, help="电压等级")
    parser.add_argument("--all", action="store_true", help="准备所有电压等级")
    parser.add_argument("--list", action="store_true", help="列出所有电压等级")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n可用电压等级:")
        for level, config in SWITCH_CLASSES.items():
            print(f"  - {level}: {len(config['classes'])}个检测类别")
    elif args.all:
        prepare_all_switch_data()
    elif args.voltage:
        loader = SwitchDataLoader(args.voltage)
        result = loader.prepare_dataset()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
