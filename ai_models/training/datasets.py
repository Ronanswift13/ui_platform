#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变电站巡检数据集模块
===================

支持:
- 公开500kV变电站通用数据集
- 云南保山站专用数据集
- 多种数据增强策略
- 自动数据划分

数据来源:
1. 公开电力设备缺陷数据集
2. 输电线路航拍数据集
3. 变电站表计读数数据集
4. 工业声音异常检测数据集

作者: 破夜绘明团队
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass

import numpy as np

# PyTorch导入
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split, Subset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# OpenCV导入
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# PIL导入
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# 数据增强
# =============================================================================
class BaseTransform:
    """变换基类"""
    def __call__(self, image, target=None):
        raise NotImplementedError


class Compose:
    """组合多个变换"""
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(BaseTransform):
    """调整大小"""
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, image, target=None):
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, self.size)
        elif PIL_AVAILABLE and isinstance(image, Image.Image):
            image = image.resize(self.size)
        return image, target


class RandomHorizontalFlip(BaseTransform):
    """随机水平翻转"""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            if isinstance(image, np.ndarray):
                image = cv2.flip(image, 1)
            elif PIL_AVAILABLE and isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 更新目标框
            if target is not None and 'boxes' in target:
                boxes = target['boxes']
                if isinstance(image, np.ndarray):
                    w = image.shape[1]
                else:
                    w = image.width
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        return image, target


class RandomVerticalFlip(BaseTransform):
    """随机垂直翻转"""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            if isinstance(image, np.ndarray):
                image = cv2.flip(image, 0)
            elif PIL_AVAILABLE and isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image, target


class RandomRotation(BaseTransform):
    """随机旋转"""
    def __init__(self, degrees: float = 15):
        self.degrees = degrees
    
    def __call__(self, image, target=None):
        angle = random.uniform(-self.degrees, self.degrees)
        
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        elif PIL_AVAILABLE and isinstance(image, Image.Image):
            image = image.rotate(angle)
        
        return image, target


class ColorJitter(BaseTransform):
    """颜色抖动"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image, target=None):
        if isinstance(image, np.ndarray):
            # 亮度
            if self.brightness > 0:
                factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
            # 对比度
            if self.contrast > 0:
                factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                mean = np.mean(image)
                image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return image, target


class Normalize(BaseTransform):
    """归一化"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
    
    def __call__(self, image, target=None):
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std
        return image, target


class ToTensor(BaseTransform):
    """转换为Tensor"""
    def __call__(self, image, target=None):
        if isinstance(image, np.ndarray):
            # HWC -> CHW
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image.copy()).float()
        
        return image, target


def get_train_transforms(input_size: Tuple[int, int], 
                         augmentation: bool = True) -> Compose:
    """获取训练变换"""
    transforms = [Resize(input_size)]
    
    if augmentation:
        transforms.extend([
            RandomHorizontalFlip(0.5),
            RandomRotation(10),
            ColorJitter(0.2, 0.2, 0.2, 0.1),
        ])
    
    transforms.extend([
        Normalize(),
        ToTensor(),
    ])
    
    return Compose(transforms)


def get_val_transforms(input_size: Tuple[int, int]) -> Compose:
    """获取验证变换"""
    return Compose([
        Resize(input_size),
        Normalize(),
        ToTensor(),
    ])


# =============================================================================
# 基础数据集类
# =============================================================================
class SubstationDataset(Dataset):
    """
    变电站巡检基础数据集
    
    支持多种标注格式:
    - COCO格式 (目标检测)
    - VOC格式 (目标检测)
    - 分类目录格式 (图像分类)
    - 语义分割格式 (mask图像)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Callable = None,
        annotation_format: str = "coco",  # coco, voc, classification, segmentation
        classes: List[str] = None,
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            split: 数据集划分 (train, val, test)
            transform: 数据变换
            annotation_format: 标注格式
            classes: 类别列表
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.annotation_format = annotation_format
        self.classes = classes or []
        
        # 加载数据索引
        self.samples = self._load_samples()
        
        logger.info(f"数据集加载完成: {len(self.samples)} 样本 ({split})")
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        
        if self.annotation_format == "coco":
            samples = self._load_coco_samples()
        elif self.annotation_format == "voc":
            samples = self._load_voc_samples()
        elif self.annotation_format == "classification":
            samples = self._load_classification_samples()
        elif self.annotation_format == "segmentation":
            samples = self._load_segmentation_samples()
        else:
            # 自动检测格式
            samples = self._auto_detect_and_load()
        
        return samples
    
    def _load_coco_samples(self) -> List[Dict]:
        """加载COCO格式数据"""
        samples = []
        
        ann_file = self.data_dir / "annotations" / f"{self.split}.json"
        img_dir = self.data_dir / "images" / self.split
        
        if not ann_file.exists():
            logger.warning(f"COCO标注文件不存在: {ann_file}")
            return samples
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 构建图像ID到标注的映射
        img_to_anns = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # 构建类别映射
        self.classes = [c['name'] for c in coco_data.get('categories', [])]
        cat_id_to_idx = {c['id']: i for i, c in enumerate(coco_data.get('categories', []))}
        
        for img_info in coco_data.get('images', []):
            img_path = img_dir / img_info['file_name']
            if not img_path.exists():
                continue
            
            anns = img_to_anns.get(img_info['id'], [])
            
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(cat_id_to_idx.get(ann['category_id'], 0))
            
            samples.append({
                'image_path': str(img_path),
                'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4)),
                'labels': np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64),
            })
        
        return samples
    
    def _load_voc_samples(self) -> List[Dict]:
        """加载VOC格式数据"""
        samples = []
        
        img_dir = self.data_dir / "JPEGImages"
        ann_dir = self.data_dir / "Annotations"
        split_file = self.data_dir / "ImageSets" / "Main" / f"{self.split}.txt"
        
        if not split_file.exists():
            # 使用所有图像
            image_ids = [p.stem for p in img_dir.glob("*.jpg")]
        else:
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
        
        for img_id in image_ids:
            img_path = img_dir / f"{img_id}.jpg"
            ann_path = ann_dir / f"{img_id}.xml"
            
            if not img_path.exists():
                continue
            
            boxes, labels = [], []
            if ann_path.exists():
                boxes, labels = self._parse_voc_xml(ann_path)
            
            samples.append({
                'image_path': str(img_path),
                'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4)),
                'labels': np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64),
            })
        
        return samples
    
    def _parse_voc_xml(self, xml_path: Path) -> Tuple[List, List]:
        """解析VOC XML标注"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            boxes, labels = [], []
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in self.classes:
                    self.classes.append(name)
                
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.classes.index(name))
            
            return boxes, labels
            
        except Exception as e:
            logger.warning(f"解析XML失败: {xml_path}, {e}")
            return [], []
    
    def _load_classification_samples(self) -> List[Dict]:
        """加载分类格式数据 (按类别目录组织)"""
        samples = []
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            split_dir = self.data_dir
        
        # 获取类别目录
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        
        for class_idx, class_dir in enumerate(class_dirs):
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append({
                        'image_path': str(img_path),
                        'label': class_idx,
                        'class_name': class_dir.name,
                    })
        
        return samples
    
    def _load_segmentation_samples(self) -> List[Dict]:
        """加载语义分割格式数据"""
        samples = []
        
        img_dir = self.data_dir / "images" / self.split
        mask_dir = self.data_dir / "masks" / self.split
        
        if not img_dir.exists():
            img_dir = self.data_dir / "images"
            mask_dir = self.data_dir / "masks"
        
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                mask_path = mask_dir / f"{img_path.stem}.png"
            
            samples.append({
                'image_path': str(img_path),
                'mask_path': str(mask_path) if mask_path.exists() else None,
            })
        
        return samples
    
    def _auto_detect_and_load(self) -> List[Dict]:
        """自动检测格式并加载"""
        # 检测COCO格式
        if (self.data_dir / "annotations").exists():
            return self._load_coco_samples()
        
        # 检测VOC格式
        if (self.data_dir / "Annotations").exists():
            return self._load_voc_samples()
        
        # 检测分割格式
        if (self.data_dir / "masks").exists():
            return self._load_segmentation_samples()
        
        # 默认分类格式
        return self._load_classification_samples()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        sample = self.samples[idx]
        
        # 加载图像
        image = self._load_image(sample['image_path'])
        
        # 构建目标
        if 'boxes' in sample:
            target = {
                'boxes': sample['boxes'],
                'labels': sample['labels'],
            }
        elif 'mask_path' in sample and sample['mask_path']:
            target = self._load_image(sample['mask_path'], grayscale=True)
        elif 'label' in sample:
            target = sample['label']
        else:
            target = None
        
        # 应用变换
        if self.transform:
            if isinstance(target, dict) or isinstance(target, np.ndarray):
                image, target = self.transform(image, target)
            else:
                image, _ = self.transform(image, None)
        
        # 转换目标格式
        if isinstance(target, dict):
            target = {
                'boxes': torch.from_numpy(target['boxes']).float(),
                'labels': torch.from_numpy(target['labels']).long(),
            }
        elif isinstance(target, np.ndarray):
            target = torch.from_numpy(target).long()
        elif isinstance(target, int):
            target = torch.tensor(target).long()
        
        return image, target
    
    def _load_image(self, path: str, grayscale: bool = False) -> np.ndarray:
        """加载图像"""
        if CV2_AVAILABLE:
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            image = cv2.imread(path, flag)
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {path}")
            if not grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        elif PIL_AVAILABLE:
            mode = 'L' if grayscale else 'RGB'
            image = Image.open(path).convert(mode)
            return np.array(image)
        else:
            raise ImportError("需要安装opencv-python或Pillow")


# =============================================================================
# 各插件专用数据集
# =============================================================================
class TransformerDataset(SubstationDataset):
    """
    主变巡视数据集 (A组)
    
    包含:
    - 油泄漏检测
    - 锈蚀检测
    - 破损检测
    - 异物检测
    - 硅胶颜色分类
    - 热成像异常检测
    """
    
    DEFAULT_CLASSES = [
        "oil_leak",       # 油泄漏
        "rust",           # 锈蚀
        "damage",         # 破损
        "foreign_object", # 异物
        "crack",          # 裂纹
        "discoloration",  # 变色
    ]
    
    def __init__(self, data_dir: str, split: str = "train", 
                 transform: Callable = None, task: str = "detection"):
        """
        Args:
            task: detection, silica_classification, thermal_classification
        """
        self.task = task
        
        if task == "silica_classification":
            annotation_format = "classification"
            classes = ["blue", "pink", "white", "unknown"]
        elif task == "thermal_classification":
            annotation_format = "classification"
            classes = ["normal", "warning", "alarm"]
        else:
            annotation_format = "coco"
            classes = self.DEFAULT_CLASSES
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            annotation_format=annotation_format,
            classes=classes
        )


class SwitchDataset(SubstationDataset):
    """
    开关间隔数据集 (B组)
    
    包含:
    - 断路器分合状态
    - 隔离开关状态
    - 接地开关状态
    - 指示灯状态
    - OCR文字识别
    """
    
    DEFAULT_CLASSES = [
        "breaker_open",     # 断路器分
        "breaker_closed",   # 断路器合
        "isolator_open",    # 隔离开关分
        "isolator_closed",  # 隔离开关合
        "grounding_open",   # 接地开关分
        "grounding_closed", # 接地开关合
        "indicator_red",    # 红色指示灯
        "indicator_green",  # 绿色指示灯
    ]
    
    def __init__(self, data_dir: str, split: str = "train",
                 transform: Callable = None, task: str = "detection"):
        """
        Args:
            task: detection, ocr
        """
        self.task = task
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            annotation_format="coco" if task == "detection" else "ocr",
            classes=self.DEFAULT_CLASSES if task == "detection" else None
        )


class BusbarDataset(SubstationDataset):
    """
    母线巡视数据集 (C组)
    
    专门针对远距小目标检测优化
    
    包含:
    - 绝缘子裂纹/污损
    - 金具松动/锈蚀
    - 导线破损
    - 异物悬挂
    """
    
    DEFAULT_CLASSES = [
        "insulator_crack",   # 绝缘子裂纹
        "insulator_dirty",   # 绝缘子污损
        "fitting_loose",     # 金具松动
        "fitting_rust",      # 金具锈蚀
        "wire_damage",       # 导线破损
        "foreign_object",    # 异物悬挂
        "bird",              # 鸟类(干扰)
        "insect",            # 飞虫(干扰)
    ]
    
    def __init__(self, data_dir: str, split: str = "train",
                 transform: Callable = None, use_slicing: bool = True):
        """
        Args:
            use_slicing: 是否使用切片增强(4K大图)
        """
        self.use_slicing = use_slicing
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            annotation_format="coco",
            classes=self.DEFAULT_CLASSES
        )
    
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        
        # 4K切片处理
        if self.use_slicing and image.shape[-1] > 2000:
            image, target = self._apply_slicing(image, target)
        
        return image, target
    
    def _apply_slicing(self, image, target, tile_size=1280, overlap=128):
        """切片处理大图"""
        # 简化实现,实际应该返回多个切片
        return image, target


class CapacitorDataset(SubstationDataset):
    """
    电容器巡视数据集 (D组)
    
    包含:
    - 电容器单元检测
    - 倾斜检测
    - 倒塌检测
    - 缺失检测
    - 入侵检测
    """
    
    DEFAULT_CLASSES = [
        "capacitor_unit",    # 电容器单元
        "capacitor_tilted",  # 倾斜电容器
        "capacitor_fallen",  # 倒塌电容器
        "capacitor_missing", # 缺失位置
        "person",            # 人员(入侵)
        "vehicle",           # 车辆(入侵)
        "animal",            # 动物(入侵)
    ]
    
    def __init__(self, data_dir: str, split: str = "train",
                 transform: Callable = None, task: str = "detection"):
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            annotation_format="coco",
            classes=self.DEFAULT_CLASSES
        )


class MeterDataset(SubstationDataset):
    """
    表计读数数据集 (E组)
    
    包含:
    - 表计关键点检测
    - 数字OCR识别
    - 表计类型分类
    """
    
    METER_TYPES = [
        "pressure_gauge",  # 压力表
        "temperature",     # 温度表
        "oil_level",       # 油位表
        "sf6_pressure",    # SF6压力表
        "digital",         # 数字表
    ]
    
    def __init__(self, data_dir: str, split: str = "train",
                 transform: Callable = None, task: str = "keypoint"):
        """
        Args:
            task: keypoint, ocr, classification
        """
        self.task = task
        
        if task == "keypoint":
            annotation_format = "keypoint"
        elif task == "ocr":
            annotation_format = "ocr"
        else:
            annotation_format = "classification"
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            annotation_format=annotation_format,
            classes=self.METER_TYPES if task == "classification" else None
        )


# =============================================================================
# 模拟数据集 (用于预训练和测试)
# =============================================================================
class SimulatedDataset(Dataset):
    """
    模拟数据集
    
    用于在没有真实数据时进行代码验证和预训练
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        input_size: Tuple[int, int] = (640, 640),
        num_classes: int = 10,
        task: str = "classification",  # classification, detection, segmentation
        transform: Callable = None,
    ):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        self.task = task
        self.transform = transform
        
        # 生成模拟数据索引
        self.samples = list(range(num_samples))
        
        logger.info(f"模拟数据集创建: {num_samples} 样本, {task}任务")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # 生成随机图像
        np.random.seed(idx)
        image = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
        
        if self.task == "classification":
            label = idx % self.num_classes
            target = label
        elif self.task == "detection":
            # 生成随机框
            num_boxes = np.random.randint(1, 5)
            boxes = []
            labels = []
            for _ in range(num_boxes):
                x1 = np.random.randint(0, self.input_size[0] - 50)
                y1 = np.random.randint(0, self.input_size[1] - 50)
                x2 = x1 + np.random.randint(30, 100)
                y2 = y1 + np.random.randint(30, 100)
                boxes.append([x1, y1, x2, y2])
                labels.append(np.random.randint(0, self.num_classes))
            
            target = {
                'boxes': np.array(boxes, dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64),
            }
        elif self.task == "segmentation":
            target = np.random.randint(0, self.num_classes, self.input_size, dtype=np.int64)
        else:
            target = idx % self.num_classes
        
        # 应用变换
        if self.transform:
            if isinstance(target, dict) or isinstance(target, np.ndarray):
                image, target = self.transform(image, target)
            else:
                image, _ = self.transform(image, None)
        
        # 转换目标
        if isinstance(target, dict):
            target = {
                'boxes': torch.from_numpy(target['boxes']).float(),
                'labels': torch.from_numpy(target['labels']).long(),
            }
        elif isinstance(target, np.ndarray):
            target = torch.from_numpy(target).long()
        elif isinstance(target, int):
            target = torch.tensor(target).long()
        
        return image, target


# =============================================================================
# 数据加载器工厂函数
# =============================================================================
def create_dataloader(
    plugin_name: str,
    model_type: str,
    data_dir: str,
    batch_size: int = 16,
    input_size: Tuple[int, int] = (640, 640),
    num_workers: int = 4,
    val_split: float = 0.2,
    use_simulated: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        plugin_name: 插件名称
        model_type: 模型类型
        data_dir: 数据目录
        batch_size: 批大小
        input_size: 输入尺寸
        num_workers: 工作进程数
        val_split: 验证集比例
        use_simulated: 是否使用模拟数据
    
    Returns:
        (train_loader, val_loader)
    """
    # 获取变换
    train_transform = get_train_transforms(input_size, augmentation=True)
    val_transform = get_val_transforms(input_size)
    
    # 根据插件选择数据集类
    if use_simulated:
        # 使用模拟数据
        task = "detection" if model_type == "detection" else "classification"
        train_dataset = SimulatedDataset(
            num_samples=1000,
            input_size=input_size,
            num_classes=10,
            task=task,
            transform=train_transform
        )
        val_dataset = SimulatedDataset(
            num_samples=200,
            input_size=input_size,
            num_classes=10,
            task=task,
            transform=val_transform
        )
    else:
        # 真实数据
        dataset_class = {
            "transformer": TransformerDataset,
            "switch": SwitchDataset,
            "busbar": BusbarDataset,
            "capacitor": CapacitorDataset,
            "meter": MeterDataset,
        }.get(plugin_name, SubstationDataset)
        
        try:
            # 尝试加载训练集和验证集
            train_dataset = dataset_class(
                data_dir=data_dir,
                split="train",
                transform=train_transform,
            )
            val_dataset = dataset_class(
                data_dir=data_dir,
                split="val",
                transform=val_transform,
            )
        except Exception as e:
            logger.warning(f"无法加载真实数据: {e}, 使用模拟数据")
            return create_dataloader(
                plugin_name, model_type, data_dir, batch_size, 
                input_size, num_workers, val_split, use_simulated=True
            )
    
    # 如果数据集太小,自动划分
    if len(train_dataset) < 10:
        logger.warning("数据集太小,使用模拟数据")
        return create_dataloader(
            plugin_name, model_type, data_dir, batch_size,
            input_size, num_workers, val_split, use_simulated=True
        )
    
    # 检测设备以决定是否使用pin_memory
    use_pin_memory = torch.cuda.is_available()  # 仅CUDA支持pin_memory，MPS不支持

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
        collate_fn=detection_collate_fn if model_type == "detection" else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        collate_fn=detection_collate_fn if model_type == "detection" else None
    )
    
    return train_loader, val_loader


def detection_collate_fn(batch):
    """目标检测数据集的collate函数"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets


# =============================================================================
# 公开数据集下载工具
# =============================================================================
class DatasetDownloader:
    """
    公开数据集下载器
    
    支持下载和预处理常用的电力设备数据集
    """
    
    # 可用数据集列表
    AVAILABLE_DATASETS = {
        "power_equipment_defect": {
            "url": "https://example.com/power_defect.zip",  # 示例URL
            "description": "电力设备缺陷检测数据集",
            "size": "2GB",
            "format": "coco",
        },
        "transmission_line": {
            "url": "https://example.com/transmission.zip",
            "description": "输电线路航拍缺陷数据集",
            "size": "5GB",
            "format": "voc",
        },
        "meter_reading": {
            "url": "https://example.com/meter.zip",
            "description": "工业表计读数数据集",
            "size": "1GB",
            "format": "classification",
        },
        "industrial_sound": {
            "url": "https://example.com/sound.zip",
            "description": "工业声音异常检测数据集",
            "size": "500MB",
            "format": "audio",
        },
    }
    
    @classmethod
    def list_datasets(cls) -> Dict:
        """列出可用数据集"""
        return cls.AVAILABLE_DATASETS
    
    @classmethod
    def download(cls, dataset_name: str, output_dir: str) -> str:
        """
        下载数据集
        
        Args:
            dataset_name: 数据集名称
            output_dir: 输出目录
        
        Returns:
            数据集路径
        """
        if dataset_name not in cls.AVAILABLE_DATASETS:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        info = cls.AVAILABLE_DATASETS[dataset_name]
        logger.info(f"下载数据集: {dataset_name}")
        logger.info(f"描述: {info['description']}")
        logger.info(f"大小: {info['size']}")
        
        # 实际下载逻辑需要根据具体数据集实现
        output_path = Path(output_dir) / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据集将保存到: {output_path}")
        logger.warning("注意: 实际下载需要配置真实的数据集URL")
        
        return str(output_path)
