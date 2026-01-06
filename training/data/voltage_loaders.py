#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 各电压等级综合数据加载器
按电压等级分类加载训练数据，支持公开数据集和占位符数据

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training
"""

import os
import cv2
import json
import yaml
import shutil
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 路径配置
# =============================================================================
BASE_TRAINING_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
DATA_PATH = BASE_TRAINING_PATH / "data"
RAW_PATH = DATA_PATH / "raw"
PROCESSED_PATH = DATA_PATH / "processed"
PLACEHOLDER_PATH = DATA_PATH / "placeholder"

# =============================================================================
# 数据类定义
# =============================================================================
@dataclass
class BoundingBox:
    """边界框"""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    class_name: str = ""
    confidence: float = 1.0
    
    def to_yolo(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        x1 = int((self.x_center - self.width/2) * img_width)
        y1 = int((self.y_center - self.height/2) * img_height)
        x2 = int((self.x_center + self.width/2) * img_width)
        y2 = int((self.y_center + self.height/2) * img_height)
        return (x1, y1, x2, y2)

@dataclass
class Sample:
    """训练样本"""
    image_path: Path
    label_path: Optional[Path]
    boxes: List[BoundingBox] = field(default_factory=list)
    voltage_level: str = ""
    plugin: str = ""
    split: str = "train"
    is_placeholder: bool = False

@dataclass
class DatasetStats:
    """数据集统计信息"""
    total_images: int = 0
    total_annotations: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    split_distribution: Dict[str, int] = field(default_factory=dict)
    placeholder_count: int = 0

# =============================================================================
# 基础数据加载器
# =============================================================================
class BaseVoltageDataLoader(ABC):
    """电压等级数据加载器基类"""
    
    def __init__(self, voltage_level: str, plugin: str):
        self.voltage_level = voltage_level
        self.plugin = plugin
        self.classes: List[str] = []
        self.samples: List[Sample] = []
        self.stats = DatasetStats()
        
        # 设置路径
        self.processed_path = PROCESSED_PATH / voltage_level / plugin
        self.placeholder_path = PLACEHOLDER_PATH / voltage_level / plugin
        
    @abstractmethod
    def get_classes(self) -> List[str]:
        """获取检测类别"""
        pass
    
    @abstractmethod
    def get_thermal_thresholds(self) -> Dict[str, float]:
        """获取热成像阈值"""
        pass
    
    def load_samples(self) -> List[Sample]:
        """加载所有样本"""
        self.samples = []
        self.classes = self.get_classes()
        
        # 加载处理后的数据
        for split in ["train", "val", "test"]:
            images_dir = self.processed_path / "images" / split
            labels_dir = self.processed_path / "labels" / split
            
            if images_dir.exists():
                self._load_from_directory(images_dir, labels_dir, split, is_placeholder=False)
        
        # 加载占位符数据
        if self.placeholder_path.exists():
            images_dir = self.placeholder_path / "images"
            labels_dir = self.placeholder_path / "labels"
            if images_dir.exists():
                self._load_from_directory(images_dir, labels_dir, "train", is_placeholder=True)
        
        # 更新统计
        self._update_stats()
        
        return self.samples
    
    def _load_from_directory(
        self,
        images_dir: Path,
        labels_dir: Path,
        split: str,
        is_placeholder: bool
    ):
        """从目录加载样本"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
        
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                # 查找对应的标注文件
                label_path = labels_dir / img_path.with_suffix('.txt').name
                if not label_path.exists():
                    label_path = None
                
                # 解析标注
                boxes = self._parse_labels(label_path) if label_path else []
                
                sample = Sample(
                    image_path=img_path,
                    label_path=label_path,
                    boxes=boxes,
                    voltage_level=self.voltage_level,
                    plugin=self.plugin,
                    split=split,
                    is_placeholder=is_placeholder
                )
                self.samples.append(sample)
    
    def _parse_labels(self, label_path: Path) -> List[BoundingBox]:
        """解析YOLO格式标注"""
        boxes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            class_name = self.classes[class_id] if class_id < len(self.classes) else ""
                            
                            boxes.append(BoundingBox(
                                class_id=class_id,
                                x_center=x_center,
                                y_center=y_center,
                                width=width,
                                height=height,
                                class_name=class_name
                            ))
        except Exception as e:
            logger.warning(f"解析标注失败 {label_path}: {e}")
        return boxes
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats = DatasetStats()
        self.stats.total_images = len(self.samples)
        
        for sample in self.samples:
            self.stats.total_annotations += len(sample.boxes)
            
            # 分割统计
            if sample.split not in self.stats.split_distribution:
                self.stats.split_distribution[sample.split] = 0
            self.stats.split_distribution[sample.split] += 1
            
            # 类别统计
            for box in sample.boxes:
                if box.class_name not in self.stats.class_distribution:
                    self.stats.class_distribution[box.class_name] = 0
                self.stats.class_distribution[box.class_name] += 1
            
            # 占位符统计
            if sample.is_placeholder:
                self.stats.placeholder_count += 1
    
    def get_split(self, split: str) -> List[Sample]:
        """获取指定分割的样本"""
        return [s for s in self.samples if s.split == split]
    
    def generate_batches(
        self,
        split: str = "train",
        batch_size: int = 16,
        shuffle: bool = True
    ) -> Generator[List[Sample], None, None]:
        """生成批次数据"""
        samples = self.get_split(split)
        if shuffle:
            random.shuffle(samples)
        
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]
    
    def export_data_yaml(self, output_path: Path = None) -> Path:
        """导出data.yaml文件"""
        output_path = output_path or (self.processed_path / "data.yaml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        yaml_content = {
            "path": str(self.processed_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: cls for i, cls in enumerate(self.classes)},
            "nc": len(self.classes),
            # 额外信息
            "voltage_level": self.voltage_level,
            "plugin": self.plugin,
            "thermal_thresholds": self.get_thermal_thresholds()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"导出data.yaml: {output_path}")
        return output_path


# =============================================================================
# 特高压数据加载器 (UHV)
# =============================================================================
class UHV1000kVACLoader(BaseVoltageDataLoader):
    """1000kV交流特高压数据加载器"""
    
    def __init__(self, plugin: str):
        super().__init__("UHV_1000kV_AC", plugin)
    
    def get_classes(self) -> List[str]:
        plugin_classes = {
            "transformer": [
                "oil_leak", "rust", "surface_damage", "foreign_object",
                "silica_gel_normal", "silica_gel_abnormal", 
                "oil_level_normal", "oil_level_abnormal",
                "bushing_crack", "porcelain_contamination",
                "partial_discharge", "core_ground_current", "winding_deformation"
            ],
            "switch": [
                "breaker_open", "breaker_closed",
                "isolator_open", "isolator_closed",
                "grounding_open", "grounding_closed",
                "indicator_red", "indicator_green",
                "gis_position", "gis_sf6_density"
            ],
            "busbar": [
                "insulator_crack", "insulator_dirty", "insulator_flashover",
                "fitting_loose", "fitting_rust", "wire_damage",
                "foreign_object", "bird", "pin_missing",
                "spacer_damage", "corona_discharge"
            ],
            "capacitor": [
                "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
                "capacitor_missing", "person", "vehicle",
                "fuse_blown", "silver_contact_damage"
            ],
            "meter": [
                "sf6_pressure_gauge", "sf6_density_relay",
                "oil_temp_gauge", "oil_level_gauge",
                "gas_relay", "digital_display", "pointer_gauge"
            ]
        }
        return plugin_classes.get(self.plugin, [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        return {"normal": 70.0, "warning": 85.0, "alarm": 100.0}


class UHV800kVDCLoader(BaseVoltageDataLoader):
    """±800kV直流特高压数据加载器"""
    
    def __init__(self, plugin: str):
        super().__init__("UHV_800kV_DC", plugin)
    
    def get_classes(self) -> List[str]:
        plugin_classes = {
            "transformer": [
                "converter_transformer", "oil_leak", "bushing_crack",
                "cooling_system", "tap_changer"
            ],
            "switch": [
                "converter_valve", "thyristor_status",
                "cooling_water", "valve_temperature", "dc_breaker"
            ]
        }
        return plugin_classes.get(self.plugin, [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        return {"normal": 70.0, "warning": 85.0, "alarm": 100.0}


# =============================================================================
# 超高压数据加载器 (EHV)
# =============================================================================
class EHV500kVLoader(BaseVoltageDataLoader):
    """500kV超高压数据加载器"""
    
    def __init__(self, plugin: str):
        super().__init__("EHV_500kV", plugin)
    
    def get_classes(self) -> List[str]:
        plugin_classes = {
            "transformer": [
                "oil_leak", "rust", "surface_damage", "foreign_object",
                "silica_gel_normal", "silica_gel_abnormal",
                "oil_level_normal", "oil_level_abnormal",
                "bushing_crack", "porcelain_contamination"
            ],
            "switch": [
                "breaker_open", "breaker_closed",
                "isolator_open", "isolator_closed",
                "grounding_open", "grounding_closed",
                "indicator_red", "indicator_green", "gis_position"
            ],
            "busbar": [
                "insulator_crack", "insulator_dirty", "insulator_flashover",
                "fitting_loose", "fitting_rust", "wire_damage",
                "foreign_object", "bird", "pin_missing", "spacer_damage"
            ],
            "capacitor": [
                "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
                "capacitor_missing", "person", "vehicle", "fuse_blown"
            ],
            "meter": [
                "sf6_pressure_gauge", "sf6_density_relay",
                "gas_relay", "oil_temp_gauge", "oil_level_gauge"
            ]
        }
        return plugin_classes.get(self.plugin, [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        return {"normal": 65.0, "warning": 80.0, "alarm": 95.0}


class EHV330kVLoader(BaseVoltageDataLoader):
    """330kV超高压数据加载器"""
    
    def __init__(self, plugin: str):
        super().__init__("EHV_330kV", plugin)
    
    def get_classes(self) -> List[str]:
        plugin_classes = {
            "transformer": [
                "oil_leak", "rust", "surface_damage",
                "silica_gel_normal", "silica_gel_abnormal",
                "oil_level_normal", "oil_level_abnormal"
            ],
            "switch": [
                "breaker_open", "breaker_closed",
                "isolator_open", "isolator_closed",
                "grounding_open", "grounding_closed",
                "indicator_red", "indicator_green"
            ],
            "busbar": [
                "insulator_crack", "insulator_dirty",
                "fitting_loose", "fitting_rust",
                "wire_damage", "foreign_object", "bird"
            ],
            "capacitor": [
                "capacitor_unit", "capacitor_tilted",
                "capacitor_fallen", "fuse_blown"
            ],
            "meter": [
                "sf6_pressure_gauge", "oil_temp_gauge", "oil_level_gauge"
            ]
        }
        return plugin_classes.get(self.plugin, [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        return {"normal": 63.0, "warning": 78.0, "alarm": 92.0}


# =============================================================================
# 高压数据加载器 (HV)
# =============================================================================
class HV220kVLoader(BaseVoltageDataLoader):
    """220kV高压数据加载器"""
    
    def __init__(self, plugin: str):
        super().__init__("HV_220kV", plugin)
    
    def get_classes(self) -> List[str]:
        plugin_classes = {
            "transformer": [
                "oil_leak", "rust", "surface_damage", "foreign_object",
                "silica_gel_normal", "silica_gel_abnormal",
                "oil_level_normal", "oil_level_abnormal"
            ],
            "switch": [
                "breaker_open", "breaker_closed",
                "isolator_open", "isolator_closed",
                "grounding_open", "grounding_closed",
                "indicator_red", "indicator_green"
            ],
            "busbar": [
                "insulator_crack", "insulator_dirty",
                "fitting_loose", "fitting_rust",
                "wire_damage", "foreign_object", "bird", "bird_nest"
            ],
            "capacitor": [
                "capacitor_unit", "capacitor_tilted",
                "capacitor_fallen", "capacitor_missing",
                "person", "vehicle"
            ],
            "meter": [
                "sf6_pressure_gauge", "oil_temp_gauge", "oil_level_gauge"
            ]
        }
        return plugin_classes.get(self.plugin, [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        return {"normal": 60.0, "warning": 75.0, "alarm": 85.0}


class HV110kVLoader(BaseVoltageDataLoader):
    """110kV高压数据加载器"""
    
    def __init__(self, plugin: str):
        super().__init__("HV_110kV", plugin)
    
    def get_classes(self) -> List[str]:
        plugin_classes = {
            "transformer": [
                "oil_leak", "rust", "surface_damage",
                "silica_gel_normal", "silica_gel_abnormal",
                "oil_level_normal", "oil_level_abnormal"
            ],
            "switch": [
                "breaker_open", "breaker_closed",
                "isolator_open", "isolator_closed",
                "grounding_open", "grounding_closed",
                "indicator_red", "indicator_green"
            ],
            "busbar": [
                "insulator_crack", "insulator_dirty",
                "fitting_loose", "fitting_rust",
                "wire_damage", "foreign_object", "bird", "bird_nest"
            ],
            "capacitor": [
                "capacitor_unit", "capacitor_tilted",
                "capacitor_fallen", "person", "vehicle"
            ],
            "meter": [
                "sf6_pressure_gauge", "oil_temp_gauge", "oil_level_gauge"
            ]
        }
        return plugin_classes.get(self.plugin, [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        return {"normal": 55.0, "warning": 70.0, "alarm": 80.0}


# =============================================================================
# 中压数据加载器 (MV)
# =============================================================================
class MV35kVLoader(BaseVoltageDataLoader):
    """35kV中压数据加载器"""
    
    def __init__(self, plugin: str):
        super().__init__("MV_35kV", plugin)
    
    def get_classes(self) -> List[str]:
        plugin_classes = {
            "transformer": [
                "oil_leak", "rust",
                "silica_gel_normal", "silica_gel_abnormal",
                "oil_level_normal", "oil_level_abnormal",
                "conductor_rust"
            ],
            "switch": [
                "breaker_open", "breaker_closed",
                "isolator_open", "isolator_closed",
                "indicator_red", "indicator_green",
                "cabinet_door"
            ],
            "busbar": [
                "insulator_crack", "insulator_dirty",
                "fitting_loose", "wire_damage", "foreign_object"
            ],
            "capacitor": [
                "capacitor_unit", "capacitor_tilted",
                "capacitor_missing", "safety_distance"
            ],
            "meter": [
                "digital_meter", "analog_meter", "transformer_monitor"
            ]
        }
        return plugin_classes.get(self.plugin, [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        return {"normal": 50.0, "warning": 65.0, "alarm": 75.0}


# =============================================================================
# 低压数据加载器 (LV)
# =============================================================================
class LV10kVLoader(BaseVoltageDataLoader):
    """10kV低压数据加载器"""
    
    def __init__(self, plugin: str):
        super().__init__("LV_10kV", plugin)
    
    def get_classes(self) -> List[str]:
        plugin_classes = {
            "transformer": [
                "rust", "oil_leak", "radiator_damage", "box_transformer"
            ],
            "switch": [
                "cabinet_switch", "air_breaker",
                "indicator_light", "label"
            ],
            "busbar": [
                "bolt_loose", "wire_damage", "foreign_object"
            ],
            "capacitor": [
                "capacitor_bulge", "capacitor_leak"
            ],
            "meter": [
                "energy_meter", "water_meter",
                "pointer_meter", "digital_meter"
            ]
        }
        return plugin_classes.get(self.plugin, [])
    
    def get_thermal_thresholds(self) -> Dict[str, float]:
        return {"normal": 45.0, "warning": 60.0, "alarm": 70.0}


# =============================================================================
# 数据加载器工厂
# =============================================================================
class DataLoaderFactory:
    """数据加载器工厂"""
    
    LOADERS = {
        "UHV_1000kV_AC": UHV1000kVACLoader,
        "UHV_800kV_DC": UHV800kVDCLoader,
        "EHV_500kV": EHV500kVLoader,
        "EHV_330kV": EHV330kVLoader,
        "HV_220kV": HV220kVLoader,
        "HV_110kV": HV110kVLoader,
        "MV_35kV": MV35kVLoader,
        "LV_10kV": LV10kVLoader
    }
    
    @classmethod
    def create(cls, voltage_level: str, plugin: str) -> BaseVoltageDataLoader:
        """创建数据加载器"""
        loader_class = cls.LOADERS.get(voltage_level)
        if loader_class is None:
            raise ValueError(f"不支持的电压等级: {voltage_level}")
        return loader_class(plugin)
    
    @classmethod
    def get_supported_voltage_levels(cls) -> List[str]:
        """获取支持的电压等级"""
        return list(cls.LOADERS.keys())
    
    @classmethod
    def load_all(cls, plugins: List[str] = None) -> Dict[str, Dict[str, BaseVoltageDataLoader]]:
        """加载所有电压等级的数据"""
        if plugins is None:
            plugins = ["transformer", "switch", "busbar", "capacitor", "meter"]
        
        loaders = {}
        for voltage_level in cls.LOADERS.keys():
            loaders[voltage_level] = {}
            for plugin in plugins:
                try:
                    loader = cls.create(voltage_level, plugin)
                    loader.load_samples()
                    loaders[voltage_level][plugin] = loader
                    logger.info(f"加载 {voltage_level}/{plugin}: {loader.stats.total_images} 张图像")
                except Exception as e:
                    logger.warning(f"加载失败 {voltage_level}/{plugin}: {e}")
        
        return loaders


# =============================================================================
# 命令行接口
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="电压等级数据加载器")
    parser.add_argument("--voltage", "-v", type=str, help="电压等级")
    parser.add_argument("--plugin", "-p", type=str, help="插件类型")
    parser.add_argument("--list", "-l", action="store_true", help="列出支持的电压等级")
    parser.add_argument("--export", "-e", action="store_true", help="导出data.yaml")
    parser.add_argument("--stats", "-s", action="store_true", help="显示统计信息")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n支持的电压等级:")
        for level in DataLoaderFactory.get_supported_voltage_levels():
            print(f"  - {level}")
        return
    
    if args.voltage and args.plugin:
        loader = DataLoaderFactory.create(args.voltage, args.plugin)
        loader.load_samples()
        
        if args.stats:
            print(f"\n数据统计 - {args.voltage}/{args.plugin}:")
            print(f"  总图像数: {loader.stats.total_images}")
            print(f"  总标注数: {loader.stats.total_annotations}")
            print(f"  占位符数: {loader.stats.placeholder_count}")
            print(f"  分割分布: {loader.stats.split_distribution}")
            print(f"  类别分布: {loader.stats.class_distribution}")
        
        if args.export:
            yaml_path = loader.export_data_yaml()
            print(f"\n导出data.yaml: {yaml_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
