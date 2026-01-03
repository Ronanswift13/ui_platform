#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 数据增强模块
提供针对变电站设备检测的专用数据增强方法

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training
"""

import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 路径配置
# =============================================================================
BASE_TRAINING_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
DATA_PATH = BASE_TRAINING_PATH / "data"
AUGMENTED_PATH = DATA_PATH / "augmented"


# =============================================================================
# 基础增强类
# =============================================================================
class BaseAugmentation:
    """基础增强类"""
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def __call__(self, image: np.ndarray, bboxes: List[List[float]] = None) -> Tuple[np.ndarray, List[List[float]]]:
        if random.random() < self.probability:
            return self.apply(image, bboxes)
        return image, bboxes
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        raise NotImplementedError


# =============================================================================
# 颜色增强
# =============================================================================
class ColorJitter(BaseAugmentation):
    """颜色抖动增强"""
    
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, 
                 saturation: float = 0.2, hue: float = 0.1, probability: float = 0.5):
        super().__init__(probability)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        # 亮度调整
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # 对比度调整
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # HSV调整
        if self.saturation > 0 or self.hue > 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            if self.saturation > 0:
                hsv[:, :, 1] *= 1 + random.uniform(-self.saturation, self.saturation)
            
            if self.hue > 0:
                hsv[:, :, 0] += random.uniform(-self.hue, self.hue) * 180
            
            hsv = np.clip(hsv, 0, 255)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image, bboxes


class GaussianNoise(BaseAugmentation):
    """高斯噪声"""
    
    def __init__(self, mean: float = 0, std: float = 25, probability: float = 0.3):
        super().__init__(probability)
        self.mean = mean
        self.std = std
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, bboxes


class GaussianBlur(BaseAugmentation):
    """高斯模糊"""
    
    def __init__(self, kernel_size: Tuple[int, int] = (5, 5), probability: float = 0.3):
        super().__init__(probability)
        self.kernel_size = kernel_size
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        image = cv2.GaussianBlur(image, self.kernel_size, 0)
        return image, bboxes


# =============================================================================
# 几何增强
# =============================================================================
class RandomFlip(BaseAugmentation):
    """随机翻转"""
    
    def __init__(self, horizontal: bool = True, vertical: bool = False, probability: float = 0.5):
        super().__init__(probability)
        self.horizontal = horizontal
        self.vertical = vertical
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        h, w = image.shape[:2]
        
        flip_h = self.horizontal and random.random() < 0.5
        flip_v = self.vertical and random.random() < 0.5
        
        if flip_h:
            image = cv2.flip(image, 1)
            if bboxes:
                bboxes = [[1 - x - bw, y, bw, bh] for x, y, bw, bh in bboxes]
        
        if flip_v:
            image = cv2.flip(image, 0)
            if bboxes:
                bboxes = [[x, 1 - y - bh, bw, bh] for x, y, bw, bh in bboxes]
        
        return image, bboxes


class RandomRotate(BaseAugmentation):
    """随机旋转"""
    
    def __init__(self, max_angle: float = 15, probability: float = 0.3):
        super().__init__(probability)
        self.max_angle = max_angle
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        h, w = image.shape[:2]
        angle = random.uniform(-self.max_angle, self.max_angle)
        
        # 旋转图像
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # 旋转边界框 (简化处理，保持AABB)
        if bboxes:
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            new_bboxes = []
            for bbox in bboxes:
                x, y, bw, bh = bbox
                # 转换为绝对坐标
                cx, cy = (x + bw/2) * w, (y + bh/2) * h
                # 旋转中心点
                new_cx = cos_a * (cx - w/2) - sin_a * (cy - h/2) + w/2
                new_cy = sin_a * (cx - w/2) + cos_a * (cy - h/2) + h/2
                # 转回相对坐标
                new_x = (new_cx - bw*w/2) / w
                new_y = (new_cy - bh*h/2) / h
                
                # 确保边界框在图像内
                new_x = np.clip(new_x, 0, 1 - bw)
                new_y = np.clip(new_y, 0, 1 - bh)
                new_bboxes.append([new_x, new_y, bw, bh])
            
            bboxes = new_bboxes
        
        return image, bboxes


class RandomScale(BaseAugmentation):
    """随机缩放"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), probability: float = 0.3):
        super().__init__(probability)
        self.scale_range = scale_range
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        h, w = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        
        # 如果缩小，则填充；如果放大，则裁剪中心
        if scale < 1:
            # 填充
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image
            image = padded
            
            # 调整边界框
            if bboxes:
                bboxes = [
                    [(x * scale + pad_w/w), (y * scale + pad_h/h), bw * scale, bh * scale]
                    for x, y, bw, bh in bboxes
                ]
        else:
            # 裁剪中心
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = image[start_h:start_h+h, start_w:start_w+w]
            
            # 调整边界框
            if bboxes:
                bboxes = [
                    [(x * scale - start_w/w), (y * scale - start_h/h), bw * scale, bh * scale]
                    for x, y, bw, bh in bboxes
                ]
        
        return image, bboxes


class RandomCrop(BaseAugmentation):
    """随机裁剪"""
    
    def __init__(self, crop_ratio: Tuple[float, float] = (0.8, 1.0), probability: float = 0.3):
        super().__init__(probability)
        self.crop_ratio = crop_ratio
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        h, w = image.shape[:2]
        
        crop_h = int(h * random.uniform(*self.crop_ratio))
        crop_w = int(w * random.uniform(*self.crop_ratio))
        
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        
        image = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        image = cv2.resize(image, (w, h))
        
        # 调整边界框
        if bboxes:
            new_bboxes = []
            for bbox in bboxes:
                x, y, bw, bh = bbox
                # 转换到裁剪后的坐标
                new_x = (x * w - start_w) / crop_w
                new_y = (y * h - start_h) / crop_h
                new_bw = bw * w / crop_w
                new_bh = bh * h / crop_h
                
                # 检查边界框是否在裁剪区域内
                if 0 <= new_x < 1 and 0 <= new_y < 1 and new_x + new_bw <= 1 and new_y + new_bh <= 1:
                    new_bboxes.append([new_x, new_y, new_bw, new_bh])
            
            bboxes = new_bboxes
        
        return image, bboxes


# =============================================================================
# 变电站专用增强
# =============================================================================
class WeatherSimulation(BaseAugmentation):
    """天气模拟增强 (适用于户外变电站)"""
    
    def __init__(self, weather_types: List[str] = None, probability: float = 0.3):
        super().__init__(probability)
        self.weather_types = weather_types or ["rain", "fog", "snow", "sun_glare"]
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        weather = random.choice(self.weather_types)
        
        if weather == "rain":
            image = self._add_rain(image)
        elif weather == "fog":
            image = self._add_fog(image)
        elif weather == "snow":
            image = self._add_snow(image)
        elif weather == "sun_glare":
            image = self._add_sun_glare(image)
        
        return image, bboxes
    
    def _add_rain(self, image: np.ndarray) -> np.ndarray:
        """添加雨滴效果"""
        h, w = image.shape[:2]
        rain = np.zeros((h, w), dtype=np.uint8)
        
        # 生成雨滴
        for _ in range(random.randint(100, 300)):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(10, 30)
            cv2.line(rain, (x, y), (x + 1, y + length), 200, 1)
        
        # 模糊雨滴
        rain = cv2.GaussianBlur(rain, (3, 3), 0)
        
        # 叠加到图像
        rain_rgb = cv2.cvtColor(rain, cv2.COLOR_GRAY2BGR)
        image = cv2.addWeighted(image, 0.85, rain_rgb, 0.15, 0)
        
        return image
    
    def _add_fog(self, image: np.ndarray) -> np.ndarray:
        """添加雾效果"""
        h, w = image.shape[:2]
        fog = np.ones((h, w, 3), dtype=np.uint8) * 200
        
        # 生成渐变雾
        fog_density = random.uniform(0.3, 0.6)
        image = cv2.addWeighted(image, 1 - fog_density, fog, fog_density, 0)
        
        return image
    
    def _add_snow(self, image: np.ndarray) -> np.ndarray:
        """添加雪花效果"""
        h, w = image.shape[:2]
        snow = np.zeros((h, w), dtype=np.uint8)
        
        # 生成雪花
        for _ in range(random.randint(500, 1000)):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            radius = random.randint(1, 3)
            cv2.circle(snow, (x, y), radius, 255, -1)
        
        # 模糊雪花
        snow = cv2.GaussianBlur(snow, (5, 5), 0)
        
        # 叠加到图像
        snow_rgb = cv2.cvtColor(snow, cv2.COLOR_GRAY2BGR)
        image = cv2.addWeighted(image, 0.9, snow_rgb, 0.1, 0)
        
        return image
    
    def _add_sun_glare(self, image: np.ndarray) -> np.ndarray:
        """添加阳光眩光效果"""
        h, w = image.shape[:2]
        
        # 随机光源位置
        center_x = random.randint(0, w)
        center_y = random.randint(0, h // 3)  # 光源在上方
        
        # 创建径向渐变
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(w**2 + h**2)
        
        glare = (1 - dist / max_dist) * 0.5
        glare = np.clip(glare, 0, 1)
        glare = np.stack([glare * 255] * 3, axis=-1).astype(np.uint8)
        
        image = cv2.addWeighted(image, 0.7, glare, 0.3, 0)
        
        return image


class InfraredSimulation(BaseAugmentation):
    """红外图像模拟 (用于热成像训练)"""
    
    def __init__(self, probability: float = 0.3):
        super().__init__(probability)
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用伪彩色映射
        colormap = random.choice([
            cv2.COLORMAP_JET,
            cv2.COLORMAP_HOT,
            cv2.COLORMAP_INFERNO,
            cv2.COLORMAP_PLASMA
        ])
        image = cv2.applyColorMap(gray, colormap)
        
        return image, bboxes


class LightingVariation(BaseAugmentation):
    """光照变化模拟 (模拟白天/夜间/阴影)"""
    
    def __init__(self, probability: float = 0.4):
        super().__init__(probability)
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        lighting_type = random.choice(["day", "night", "shadow", "overexpose"])
        
        if lighting_type == "day":
            # 正常日光
            factor = random.uniform(1.0, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
        elif lighting_type == "night":
            # 夜间/低光照
            factor = random.uniform(0.3, 0.6)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
            # 添加噪声
            noise = np.random.normal(0, 15, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
        elif lighting_type == "shadow":
            # 添加随机阴影
            h, w = image.shape[:2]
            shadow_mask = np.ones((h, w), dtype=np.float32)
            
            # 随机多边形阴影
            pts = np.array([
                [random.randint(0, w), random.randint(0, h)]
                for _ in range(random.randint(3, 6))
            ])
            cv2.fillPoly(shadow_mask, [pts], random.uniform(0.3, 0.7))
            
            image = (image * shadow_mask[:, :, np.newaxis]).astype(np.uint8)
            
        elif lighting_type == "overexpose":
            # 过曝
            factor = random.uniform(1.5, 2.0)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image, bboxes


# =============================================================================
# 增强组合器
# =============================================================================
class AugmentationCompose:
    """增强组合器"""
    
    def __init__(self, augmentations: List[BaseAugmentation]):
        self.augmentations = augmentations
    
    def __call__(self, image: np.ndarray, bboxes: List[List[float]] = None) -> Tuple[np.ndarray, List[List[float]]]:
        for aug in self.augmentations:
            image, bboxes = aug(image, bboxes)
        return image, bboxes


def get_default_augmentation(voltage_level: str, plugin: str) -> AugmentationCompose:
    """
    获取默认的数据增强配置
    根据电压等级和插件类型返回合适的增强组合
    """
    # 基础增强
    base_augs = [
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, probability=0.5),
        RandomFlip(horizontal=True, vertical=False, probability=0.5),
    ]
    
    # 根据插件类型添加特定增强
    if plugin == "transformer":
        # 变压器巡检 - 添加热成像模拟
        augs = base_augs + [
            GaussianNoise(std=15, probability=0.2),
            LightingVariation(probability=0.4),
            InfraredSimulation(probability=0.2),
        ]
        
    elif plugin == "switch":
        # 开关间隔 - 关注指示灯识别
        augs = base_augs + [
            GaussianBlur(kernel_size=(3, 3), probability=0.2),
            LightingVariation(probability=0.3),
        ]
        
    elif plugin == "busbar":
        # 母线巡检 - 户外场景，添加天气模拟
        augs = base_augs + [
            RandomRotate(max_angle=10, probability=0.3),
            WeatherSimulation(probability=0.3),
            RandomScale(scale_range=(0.8, 1.2), probability=0.3),
        ]
        
    elif plugin == "capacitor":
        # 电容器巡检
        augs = base_augs + [
            RandomCrop(crop_ratio=(0.85, 1.0), probability=0.3),
            LightingVariation(probability=0.3),
        ]
        
    elif plugin == "meter":
        # 表计读数 - 关注数字清晰度
        augs = base_augs + [
            GaussianNoise(std=10, probability=0.2),
            RandomRotate(max_angle=5, probability=0.3),  # 小角度旋转
        ]
    else:
        augs = base_augs
    
    return AugmentationCompose(augs)


# =============================================================================
# 数据增强执行器
# =============================================================================
class DataAugmentor:
    """
    数据增强执行器
    批量处理训练数据增强
    """
    
    def __init__(self, voltage_level: str, plugin: str):
        self.voltage_level = voltage_level
        self.plugin = plugin
        self.augmentation = get_default_augmentation(voltage_level, plugin)
        
        # 设置路径
        self.source_path = DATA_PATH / "processed" / voltage_level / plugin
        self.output_path = AUGMENTED_PATH / voltage_level / plugin
        
    def augment_dataset(
        self,
        augment_factor: int = 3,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        增强整个数据集
        
        Args:
            augment_factor: 每张原图生成的增强图数量
            max_workers: 并行处理的工作线程数
        """
        # 创建输出目录
        for split in ["train", "val"]:  # 只增强训练集和验证集
            (self.output_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        stats = {"train": 0, "val": 0}
        
        for split in ["train", "val"]:
            images_dir = self.source_path / "images" / split
            labels_dir = self.source_path / "labels" / split
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + \
                         list(images_dir.glob("*.png")) + \
                         list(images_dir.glob("*.jpeg"))
            
            for img_path in image_files:
                # 复制原图
                shutil.copy2(img_path, self.output_path / "images" / split / img_path.name)
                
                # 复制原标注
                label_path = labels_dir / img_path.with_suffix('.txt').name
                if label_path.exists():
                    shutil.copy2(label_path, self.output_path / "labels" / split / label_path.name)
                
                # 生成增强图像
                for i in range(augment_factor):
                    self._augment_single(img_path, label_path, split, i)
                    stats[split] += 1
        
        # 生成data.yaml
        self._generate_data_yaml()
        
        logger.info(f"数据增强完成: {self.voltage_level}/{self.plugin}")
        logger.info(f"  训练集增强: {stats['train']} 张")
        logger.info(f"  验证集增强: {stats['val']} 张")
        
        return stats
    
    def _augment_single(
        self,
        img_path: Path,
        label_path: Path,
        split: str,
        aug_idx: int
    ):
        """增强单张图像"""
        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                return
            
            # 读取标注 (YOLO格式)
            bboxes = []
            class_ids = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_ids.append(int(parts[0]))
                            bboxes.append([float(x) for x in parts[1:5]])
            
            # 应用增强
            aug_image, aug_bboxes = self.augmentation(image, bboxes)
            
            # 保存增强后的图像
            aug_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
            aug_img_path = self.output_path / "images" / split / aug_name
            cv2.imwrite(str(aug_img_path), aug_image)
            
            # 保存增强后的标注
            aug_label_path = self.output_path / "labels" / split / f"{img_path.stem}_aug{aug_idx}.txt"
            with open(aug_label_path, 'w') as f:
                for cls_id, bbox in zip(class_ids, aug_bboxes):
                    f.write(f"{cls_id} {' '.join(f'{x:.6f}' for x in bbox)}\n")
                    
        except Exception as e:
            logger.error(f"增强失败 {img_path}: {e}")
    
    def _generate_data_yaml(self):
        """生成增强数据集的data.yaml"""
        import yaml
        
        # 读取原始classes
        classes_file = self.source_path / "classes.txt"
        classes = []
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
        
        yaml_content = {
            "path": str(self.output_path),
            "train": "images/train",
            "val": "images/val",
            "test": "../processed/{}/{}/images/test".format(self.voltage_level, self.plugin),
            "names": {i: cls for i, cls in enumerate(classes)},
            "nc": len(classes)
        }
        
        yaml_path = self.output_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)


# =============================================================================
# 命令行接口
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="数据增强工具")
    parser.add_argument("--voltage", "-v", type=str, required=True, help="电压等级")
    parser.add_argument("--plugin", "-p", type=str, required=True, help="插件类型")
    parser.add_argument("--factor", "-f", type=int, default=3, help="增强倍数")
    
    args = parser.parse_args()
    
    augmentor = DataAugmentor(args.voltage, args.plugin)
    stats = augmentor.augment_dataset(augment_factor=args.factor)
    
    print(f"增强完成: {stats}")


if __name__ == "__main__":
    main()
