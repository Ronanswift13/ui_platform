#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 训练数据下载与准备工具
==============================================

功能:
1. 自动下载中国大陆区公开电力数据集
2. 数据格式转换 (VOC -> COCO -> YOLO)
3. 数据集划分和组织
4. 适配 220kV/500kV 不同场景的标签映射

公开数据集来源:
- CPLID: 中国电力线路绝缘子数据集 (GitHub)
- 变电站缺陷检测数据集 (CSDN/知乎汇总)
- 输电线路巡检数据集 (百度飞桨AI Studio)
- 开关状态检测数据集

使用方法:
    # 下载所有数据集
    python prepare_training_data.py --download-all
    
    # 准备特定插件数据
    python prepare_training_data.py --prepare switch --voltage 500kV
    
    # 格式转换
    python prepare_training_data.py --convert voc2coco --input data/raw --output data/coco

作者: 破夜绘明团队
日期: 2025
"""

import os
import sys
import json
import shutil
import logging
import argparse
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
from datetime import datetime

# 可选依赖
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据集信息定义
# =============================================================================
@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    description: str
    url: str
    download_method: str  # direct, github, baidu_pan, manual
    size_mb: int
    format: str  # voc, coco, yolo, classification
    num_images: int
    classes: List[str]
    applicable_plugins: List[str]
    voltage_levels: List[str]
    license: str = "Unknown"
    notes: str = ""


# 中国大陆区可用的公开电力数据集
AVAILABLE_DATASETS = {
    # =========================================================================
    # 绝缘子相关数据集
    # =========================================================================
    "cplid": DatasetInfo(
        name="Chinese Power Line Insulator Dataset (CPLID)",
        description="中国电力线路绝缘子数据集，包含无人机拍摄的正常和缺陷绝缘子图像",
        url="https://github.com/InsulatorData/InsulatorDataSet",
        download_method="github",
        size_mb=200,
        format="voc",
        num_images=848,
        classes=["insulator", "defective_insulator"],
        applicable_plugins=["busbar", "transformer"],
        voltage_levels=["220kV", "500kV"],
        license="Research Use",
        notes="600张正常 + 248张缺陷，VOC2007格式标注"
    ),
    
    "insulator_defect": DatasetInfo(
        name="绝缘子缺陷检测数据集",
        description="包含破损、闪络、掉串等缺陷的绝缘子图像",
        url="https://aistudio.baidu.com/datasetdetail/90042",
        download_method="baidu_pan",
        size_mb=500,
        format="voc",
        num_images=2000,
        classes=["insulator_normal", "insulator_broken", "insulator_flashover", "insulator_missing"],
        applicable_plugins=["busbar"],
        voltage_levels=["220kV", "500kV"],
        license="Research Use"
    ),
    
    # =========================================================================
    # 变电站设备数据集
    # =========================================================================
    "substation_defect_8000": DatasetInfo(
        name="变电站缺陷检测数据集 (8000+)",
        description="变电站各类设备缺陷检测，包含人员安全、设备异常等",
        url="https://blog.csdn.net/qq_58995858/article/details/134962836",
        download_method="manual",
        size_mb=2600,
        format="voc",
        num_images=8307,
        classes=[
            "bjdsyc", "bj_wkps", "yw_nc", "xmbhyc", "kgg_ybh",
            "gbps", "yw_gkxfw", "hxq_gjbs", "bj_bpmh", "jyz_pl",
            "bj_bpps", "sly_dmyw", "wcaqm", "wcgz", "ywzt_yfyc",
            "hxq_gjtps", "xy"
        ],
        applicable_plugins=["transformer", "switch", "busbar", "capacitor"],
        voltage_levels=["220kV", "500kV"],
        notes="含VOC和YOLO两种格式标签，17个类别"
    ),
    
    "substation_real_7500": DatasetInfo(
        name="变电站真实巡检电力设备检测数据集",
        description="真实变电站巡检场景的电力设备检测",
        url="参见CSDN汇总",
        download_method="manual",
        size_mb=3000,
        format="yolo",
        num_images=7500,
        classes=[
            "transformer", "breaker", "isolator", "grounding_switch",
            "busbar", "insulator", "capacitor", "reactor",
            "current_transformer", "voltage_transformer",
            "surge_arrester", "cable", "meter", "indicator", "person"
        ],
        applicable_plugins=["transformer", "switch", "busbar", "capacitor", "meter"],
        voltage_levels=["220kV", "500kV"],
        notes="15个类别，YOLO格式"
    ),
    
    # =========================================================================
    # 开关状态数据集
    # =========================================================================
    "switch_state_600": DatasetInfo(
        name="断路器分合闸指示位检测数据集",
        description="断路器分合闸状态指示检测",
        url="参见CSDN汇总",
        download_method="manual",
        size_mb=150,
        format="yolo",
        num_images=600,
        classes=["breaker_open", "breaker_closed"],
        applicable_plugins=["switch"],
        voltage_levels=["220kV", "500kV"],
        notes="2个类别，txt标签"
    ),
    
    "control_panel_1800": DatasetInfo(
        name="变电站控制柜面板状态检测数据集",
        description="控制柜面板指示灯、按钮状态检测",
        url="参见知乎汇总",
        download_method="manual",
        size_mb=400,
        format="voc",
        num_images=1800,
        classes=["indicator_red", "indicator_green", "button", "switch_on", "switch_off"],
        applicable_plugins=["switch"],
        voltage_levels=["220kV", "500kV"]
    ),
    
    # =========================================================================
    # 表计读数数据集
    # =========================================================================
    "meter_pointer_500": DatasetInfo(
        name="变电站指针式仪表目标检测数据集",
        description="指针式仪表检测与读数",
        url="参见CSDN",
        download_method="manual",
        size_mb=100,
        format="voc",
        num_images=500,
        classes=["meter", "pointer", "scale"],
        applicable_plugins=["meter"],
        voltage_levels=["220kV", "500kV"]
    ),
    
    # =========================================================================
    # 红外热成像数据集
    # =========================================================================
    "thermal_defect_1900": DatasetInfo(
        name="变电站红外过热缺陷检测数据集",
        description="红外热成像过热缺陷检测",
        url="参见CSDN汇总",
        download_method="manual",
        size_mb=500,
        format="voc",
        num_images=1900,
        classes=["overheating_mild", "overheating_moderate", "overheating_severe"],
        applicable_plugins=["transformer", "switch"],
        voltage_levels=["220kV", "500kV"],
        notes="含温度信息"
    ),
    
    "switchgear_thermal_3000": DatasetInfo(
        name="开关柜红外过热检测数据集",
        description="开关柜接头红外过热检测",
        url="参见知乎汇总",
        download_method="manual",
        size_mb=800,
        format="voc",
        num_images=3000,
        classes=["junction_normal", "junction_overheating"],
        applicable_plugins=["switch"],
        voltage_levels=["220kV", "500kV"],
        notes="含温度信息"
    ),
    
    # =========================================================================
    # 电力设备图像分割数据集
    # =========================================================================
    "elek_seg": DatasetInfo(
        name="电力设备图像分割数据集 (elek-seg)",
        description="电力设备语义分割，16个类别",
        url="参见CSDN",
        download_method="manual",
        size_mb=1000,
        format="coco",
        num_images=2000,
        classes=[
            "background", "breaker", "isolator_closed", "isolator_double_closed",
            "current_transformer", "fuse", "glass_insulator", "surge_arrester",
            "muffler", "isolator_open", "isolator_double_open", "porcelain_insulator",
            "potential_transformer", "power_transformer", "recloser", "triple_isolator"
        ],
        applicable_plugins=["transformer", "switch", "busbar"],
        voltage_levels=["220kV", "500kV"]
    ),
}


# =============================================================================
# 标签映射表 - 将各数据集标签映射到统一的标签体系
# =============================================================================
# A组 - 主变巡视
TRANSFORMER_LABEL_MAP = {
    "oil_leak": "oil_leak",
    "OIL_LEAKAGE": "oil_leak",
    "漏油": "oil_leak",
    "rust": "rust_corrosion",
    "rust_corrosion": "rust_corrosion",
    "锈蚀": "rust_corrosion",
    "damage": "surface_damage",
    "破损": "surface_damage",
    "foreign_object": "foreign_object",
    "异物": "foreign_object",
    "silica_blue": "silica_gel_blue",
    "silica_pink": "silica_gel_pink",
    "oil_level": "oil_level_normal",
}

# B组 - 开关间隔
SWITCH_LABEL_MAP = {
    "breaker_open": "breaker_open",
    "breaker_closed": "breaker_closed",
    "isolator_open": "isolator_open",
    "isolator_closed": "isolator_closed",
    "grounding_open": "grounding_open",
    "grounding_closed": "grounding_closed",
    "indicator_red": "indicator_red",
    "indicator_green": "indicator_green",
    "分": "breaker_open",
    "合": "breaker_closed",
}

# C组 - 母线巡视
BUSBAR_LABEL_MAP = {
    "insulator": "insulator_normal",
    "insulator_crack": "insulator_crack",
    "insulator_dirty": "insulator_dirty",
    "defective_insulator": "insulator_damaged",
    "crack": "insulator_crack",
    "fitting_loose": "fitting_loose",
    "fitting_rust": "fitting_rust",
    "foreign_object": "foreign_object",
    "bird": "bird",
    "pin_missing": "pin_missing",
}

# 插件到标签映射的对应关系
PLUGIN_LABEL_MAPS = {
    "transformer": TRANSFORMER_LABEL_MAP,
    "switch": SWITCH_LABEL_MAP,
    "busbar": BUSBAR_LABEL_MAP,
}


# =============================================================================
# 数据格式转换器
# =============================================================================
class DataFormatConverter:
    """数据格式转换器"""
    
    @staticmethod
    def voc_to_coco(voc_dir: str, output_file: str, classes: List[str]) -> Dict:
        """
        VOC 格式转换为 COCO 格式
        
        Args:
            voc_dir: VOC 标注目录 (包含 xml 文件)
            output_file: COCO JSON 输出文件
            classes: 类别列表
        
        Returns:
            COCO 格式字典
        """
        coco = {
            "info": {
                "description": "Converted from VOC format",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 添加类别
        for i, cls in enumerate(classes):
            coco["categories"].append({
                "id": i + 1,
                "name": cls,
                "supercategory": "object"
            })
        
        class_to_id = {cls: i + 1 for i, cls in enumerate(classes)}
        
        voc_path = Path(voc_dir)
        annotation_id = 1
        
        for img_id, xml_file in enumerate(voc_path.glob("*.xml"), 1):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 图像信息
            filename = root.find("filename").text
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            
            coco["images"].append({
                "id": img_id,
                "file_name": filename,
                "width": width,
                "height": height
            })
            
            # 标注信息
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in class_to_id:
                    continue
                
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # COCO 格式: [x, y, width, height]
                coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = (xmax - xmin) * (ymax - ymin)
                
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_to_id[cls_name],
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1
        
        # 保存
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        
        logger.info(f"VOC -> COCO 转换完成: {output_file}")
        logger.info(f"  图像数: {len(coco['images'])}")
        logger.info(f"  标注数: {len(coco['annotations'])}")
        
        return coco
    
    @staticmethod
    def coco_to_yolo(coco_file: str, output_dir: str, classes: List[str]):
        """
        COCO 格式转换为 YOLO 格式
        
        Args:
            coco_file: COCO JSON 文件
            output_dir: YOLO 标注输出目录
            classes: 类别列表
        """
        with open(coco_file, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 建立映射
        category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        image_map = {img["id"]: img for img in coco["images"]}
        
        # 按图像组织标注
        image_annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # 生成 YOLO 格式标注
        for img_id, annotations in image_annotations.items():
            img = image_map[img_id]
            width, height = img["width"], img["height"]
            
            # YOLO 标注文件名
            txt_name = Path(img["file_name"]).stem + ".txt"
            txt_path = output_path / txt_name
            
            lines = []
            for ann in annotations:
                cat_name = category_map[ann["category_id"]]
                if cat_name not in class_to_idx:
                    continue
                
                class_idx = class_to_idx[cat_name]
                
                # COCO bbox [x, y, w, h] -> YOLO [cx, cy, w, h] (归一化)
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                
                lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
            with open(txt_path, 'w') as f:
                f.write("\n".join(lines))
        
        # 保存类别文件
        classes_file = output_path / "classes.txt"
        with open(classes_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(classes))
        
        logger.info(f"COCO -> YOLO 转换完成: {output_dir}")
        logger.info(f"  标注文件数: {len(image_annotations)}")
    
    @staticmethod
    def apply_label_mapping(
        input_dir: str,
        output_dir: str,
        label_map: Dict[str, str],
        format: str = "yolo"
    ):
        """
        应用标签映射，统一标签体系
        
        Args:
            input_dir: 输入标注目录
            output_dir: 输出标注目录
            label_map: 标签映射字典
            format: 标注格式 (yolo, voc)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == "yolo":
            # YOLO 格式: 需要同时修改 classes.txt 和标注文件
            classes_file = input_path / "classes.txt"
            if classes_file.exists():
                with open(classes_file, 'r', encoding='utf-8') as f:
                    old_classes = [line.strip() for line in f]
                
                # 创建新的类别列表
                new_classes = []
                old_to_new_idx = {}
                
                for old_idx, old_cls in enumerate(old_classes):
                    new_cls = label_map.get(old_cls, old_cls)
                    if new_cls not in new_classes:
                        new_classes.append(new_cls)
                    old_to_new_idx[old_idx] = new_classes.index(new_cls)
                
                # 保存新的类别文件
                with open(output_path / "classes.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_classes))
                
                # 转换标注文件
                for txt_file in input_path.glob("*.txt"):
                    if txt_file.name == "classes.txt":
                        continue
                    
                    with open(txt_file, 'r') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_idx = int(parts[0])
                            if old_idx in old_to_new_idx:
                                parts[0] = str(old_to_new_idx[old_idx])
                                new_lines.append(" ".join(parts))
                    
                    with open(output_path / txt_file.name, 'w') as f:
                        f.write("\n".join(new_lines))
        
        elif format == "voc":
            # VOC 格式: 修改 XML 中的类别名
            for xml_file in input_path.glob("*.xml"):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for obj in root.findall("object"):
                    name_elem = obj.find("name")
                    old_name = name_elem.text
                    new_name = label_map.get(old_name, old_name)
                    name_elem.text = new_name
                
                tree.write(output_path / xml_file.name, encoding='utf-8')
        
        logger.info(f"标签映射完成: {input_dir} -> {output_dir}")


# =============================================================================
# 数据集下载器
# =============================================================================
class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, download_dir: str = "data/raw"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def download_github_dataset(self, url: str, name: str) -> Optional[str]:
        """
        从 GitHub 下载数据集
        
        Args:
            url: GitHub 仓库 URL
            name: 数据集名称
        
        Returns:
            下载目录路径
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests 库未安装，无法下载")
            return None
        
        # 构建下载 URL
        if "github.com" in url:
            # 转换为 zip 下载链接
            zip_url = url.rstrip('/') + "/archive/refs/heads/main.zip"
            if "InsulatorData" in url:
                zip_url = url.rstrip('/') + "/archive/refs/heads/master.zip"
        else:
            zip_url = url
        
        output_dir = self.download_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = output_dir / "dataset.zip"
        
        try:
            logger.info(f"开始下载: {zip_url}")
            response = requests.get(zip_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded * 100 // total_size
                        print(f"\r下载进度: {percent}%", end='', flush=True)
            
            print()  # 换行
            
            # 解压
            logger.info(f"解压文件: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(output_dir)
            
            # 删除 zip 文件
            zip_path.unlink()
            
            logger.info(f"数据集下载完成: {output_dir}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return None
    
    def generate_manual_download_guide(self, dataset_name: str) -> str:
        """
        生成手动下载指南
        
        Args:
            dataset_name: 数据集名称
        
        Returns:
            指南文件路径
        """
        if dataset_name not in AVAILABLE_DATASETS:
            logger.error(f"未知数据集: {dataset_name}")
            return ""
        
        dataset = AVAILABLE_DATASETS[dataset_name]
        
        guide_content = f"""# {dataset.name} 下载指南

## 数据集信息
- 描述: {dataset.description}
- 图像数量: {dataset.num_images}
- 大小: {dataset.size_mb} MB
- 格式: {dataset.format}
- 类别: {', '.join(dataset.classes)}

## 下载方式: {dataset.download_method}

### 下载步骤:

"""
        
        if dataset.download_method == "github":
            guide_content += f"""
1. 访问 GitHub 仓库: {dataset.url}
2. 点击 "Code" -> "Download ZIP"
3. 解压到 data/raw/{dataset_name}/ 目录
"""
        elif dataset.download_method == "baidu_pan":
            guide_content += f"""
1. 访问百度飞桨 AI Studio: {dataset.url}
2. 登录账号后下载数据集
3. 解压到 data/raw/{dataset_name}/ 目录
"""
        elif dataset.download_method == "manual":
            guide_content += f"""
1. 访问参考链接获取下载地址: {dataset.url}
2. 部分数据集可能需要付费或申请
3. 下载后解压到 data/raw/{dataset_name}/ 目录

### 参考资源:
- 知乎汇总: https://zhuanlan.zhihu.com/p/484933022
- CSDN汇总: https://blog.csdn.net/DM_zx/article/details/129227962
- 飞桨AI Studio: https://aistudio.baidu.com/
"""
        
        guide_content += f"""

## 数据组织结构
下载解压后，请确保目录结构如下:

data/raw/{dataset_name}/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── labels/
    ├── image_001.{dataset.format if dataset.format != 'yolo' else 'txt'}
    ├── image_002.{dataset.format if dataset.format != 'yolo' else 'txt'}
    └── ...

## 适用插件
- {', '.join(dataset.applicable_plugins)}

## 适用电压等级
- {', '.join(dataset.voltage_levels)}

## 注意事项
{dataset.notes if dataset.notes else '无特殊注意事项'}
"""
        
        # 保存指南
        guide_dir = self.download_dir / "guides"
        guide_dir.mkdir(parents=True, exist_ok=True)
        guide_path = guide_dir / f"{dataset_name}_download_guide.md"
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"下载指南已生成: {guide_path}")
        return str(guide_path)
    
    def list_available_datasets(self) -> List[Dict]:
        """列出所有可用数据集"""
        datasets = []
        for name, info in AVAILABLE_DATASETS.items():
            datasets.append({
                "name": name,
                "description": info.description,
                "num_images": info.num_images,
                "size_mb": info.size_mb,
                "format": info.format,
                "download_method": info.download_method,
                "applicable_plugins": info.applicable_plugins,
            })
        return datasets


# =============================================================================
# 数据集准备器
# =============================================================================
class DatasetPreparer:
    """数据集准备器 - 整合数据用于训练"""
    
    def __init__(self, 
                 raw_dir: str = "data/raw",
                 processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.converter = DataFormatConverter()
    
    def prepare_plugin_dataset(
        self,
        plugin_name: str,
        voltage_level: str = "500kV",
        output_format: str = "yolo",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> str:
        """
        为指定插件准备训练数据集
        
        Args:
            plugin_name: 插件名称 (transformer, switch, busbar, capacitor, meter)
            voltage_level: 电压等级 (220kV, 500kV)
            output_format: 输出格式 (yolo, coco)
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        
        Returns:
            准备好的数据集目录
        """
        logger.info(f"准备 {plugin_name} 插件数据集 ({voltage_level})")
        
        # 输出目录
        output_dir = self.processed_dir / voltage_level / plugin_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 找到适用的数据集
        applicable_datasets = []
        for name, info in AVAILABLE_DATASETS.items():
            if (plugin_name in info.applicable_plugins and 
                voltage_level in info.voltage_levels):
                applicable_datasets.append(name)
        
        logger.info(f"找到 {len(applicable_datasets)} 个适用数据集: {applicable_datasets}")
        
        # TODO: 实际的数据整合逻辑
        # 这里创建一个配置文件记录使用的数据集
        config = {
            "plugin": plugin_name,
            "voltage_level": voltage_level,
            "output_format": output_format,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1 - train_ratio - val_ratio,
            "applicable_datasets": applicable_datasets,
            "label_map": PLUGIN_LABEL_MAPS.get(plugin_name, {}),
            "created_at": datetime.now().isoformat(),
        }
        
        config_path = output_dir / "dataset_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 创建目录结构
        for split in ["train", "val", "test"]:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据集目录已创建: {output_dir}")
        logger.info(f"配置文件: {config_path}")
        logger.info(f"请将处理后的数据放入对应目录")
        
        return str(output_dir)


# =============================================================================
# 主程序
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="训练数据下载与准备工具")
    parser.add_argument("--list", action="store_true", help="列出所有可用数据集")
    parser.add_argument("--download", type=str, help="下载指定数据集")
    parser.add_argument("--download-all-guides", action="store_true", 
                       help="生成所有数据集的下载指南")
    parser.add_argument("--prepare", type=str, 
                       choices=["transformer", "switch", "busbar", "capacitor", "meter"],
                       help="为指定插件准备数据")
    parser.add_argument("--voltage", type=str, default="500kV",
                       choices=["220kV", "500kV"], help="电压等级")
    parser.add_argument("--convert", type=str, choices=["voc2coco", "coco2yolo"],
                       help="数据格式转换")
    parser.add_argument("--input", type=str, help="输入目录")
    parser.add_argument("--output", type=str, help="输出目录/文件")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("破夜绘明激光监测平台 - 训练数据下载与准备工具")
    print("=" * 70)
    
    downloader = DatasetDownloader()
    preparer = DatasetPreparer()
    
    if args.list:
        print("\n【可用数据集列表】\n")
        datasets = downloader.list_available_datasets()
        for i, ds in enumerate(datasets, 1):
            print(f"{i}. {ds['name']}")
            print(f"   描述: {ds['description']}")
            print(f"   图像数: {ds['num_images']} | 大小: {ds['size_mb']}MB | 格式: {ds['format']}")
            print(f"   下载方式: {ds['download_method']}")
            print(f"   适用插件: {', '.join(ds['applicable_plugins'])}")
            print()
    
    elif args.download:
        if args.download in AVAILABLE_DATASETS:
            dataset = AVAILABLE_DATASETS[args.download]
            if dataset.download_method == "github":
                downloader.download_github_dataset(dataset.url, args.download)
            else:
                guide = downloader.generate_manual_download_guide(args.download)
                print(f"\n此数据集需要手动下载，请查看指南: {guide}")
        else:
            print(f"未知数据集: {args.download}")
    
    elif args.download_all_guides:
        print("\n生成所有数据集的下载指南...\n")
        for name in AVAILABLE_DATASETS:
            downloader.generate_manual_download_guide(name)
        print("\n所有下载指南已生成在 data/raw/guides/ 目录")
    
    elif args.prepare:
        preparer.prepare_plugin_dataset(
            args.prepare,
            voltage_level=args.voltage
        )
    
    elif args.convert:
        if not args.input or not args.output:
            print("错误: 格式转换需要指定 --input 和 --output")
            return
        
        converter = DataFormatConverter()
        if args.convert == "voc2coco":
            # 需要提供类别列表，这里使用示例
            classes = ["insulator", "defective_insulator", "bird", "foreign_object"]
            converter.voc_to_coco(args.input, args.output, classes)
        elif args.convert == "coco2yolo":
            classes = ["insulator", "defective_insulator", "bird", "foreign_object"]
            converter.coco_to_yolo(args.input, args.output, classes)
    
    else:
        print("\n使用 --help 查看所有可用选项")
        print("\n快速开始:")
        print("  1. 列出可用数据集: python prepare_training_data.py --list")
        print("  2. 生成下载指南:   python prepare_training_data.py --download-all-guides")
        print("  3. 准备插件数据:   python prepare_training_data.py --prepare switch --voltage 500kV")


if __name__ == "__main__":
    main()
