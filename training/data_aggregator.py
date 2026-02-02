#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据聚合器 (Training Data Aggregator V2.0)
================================================

本模块是 train_main.py 的数据加载增强插件:
  - 接收 voltage 和 plugin 参数
  - 自动遍历 training/data/raw/{voltage}/{plugin}/ 下所有有效子目录
  - 虚拟合并: 不物理移动文件, 生成临时 dataset_config.yaml
  - 将聚合后的 YAML 配置喂给 YOLO 训练引擎

使用方式 (在 train_main.py 中):
    from training.data_aggregator import TrainingDataAggregator
    
    agg = TrainingDataAggregator()
    yaml_path = agg.prepare_for_training(
        voltage="HV_220kV", 
        plugin="transformer"
    )
    # yaml_path 可直接传给 YOLO model.train(data=yaml_path)

命令行使用:
    python training/data_aggregator.py --voltage HV_220kV --plugin transformer
    python training/data_aggregator.py --voltage HV_220kV --plugin transformer --split

作者: 破夜绘明团队
日期: 2026-02-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 路径设置
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# 确保可以导入 platform_core
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML 未安装, 将使用 JSON 格式")

# 图像扩展名
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm', '.webp'}
LABEL_EXTS = {'.txt'}

# 训练数据根路径
DATA_ROOT = SCRIPT_DIR / "data"
RAW_PATH = DATA_ROOT / "raw"
PROCESSED_PATH = DATA_ROOT / "processed"


# =============================================================================
# 检测类别定义 (与 data_manager.py 保持一致)
# =============================================================================

PLUGIN_CLASSES = {
    "transformer": {
        "default": [
            "oil_leak", "rust", "surface_damage", "foreign_object",
            "silica_gel_normal", "silica_gel_abnormal",
            "oil_level_normal", "oil_level_abnormal",
            "bushing_crack", "porcelain_contamination",
            "thermal_normal", "thermal_warning", "thermal_danger",
        ],
        "UHV": [
            "oil_leak", "rust", "surface_damage", "foreign_object",
            "silica_gel_normal", "silica_gel_abnormal",
            "oil_level_normal", "oil_level_abnormal",
            "bushing_crack", "porcelain_contamination",
            "partial_discharge", "core_ground_current", "winding_deformation",
            "thermal_normal", "thermal_warning", "thermal_danger",
        ],
    },
    "switch": {
        "default": [
            "breaker_open", "breaker_closed", "breaker_intermediate",
            "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed",
            "indicator_red", "indicator_green",
        ],
    },
    "busbar": {
        "default": [
            "insulator_crack", "insulator_dirty", "insulator_flashover",
            "fitting_loose", "fitting_rust", "wire_damage",
            "foreign_object", "bird", "pin_missing",
        ],
    },
    "capacitor": {
        "default": [
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle", "fuse_blown",
        ],
    },
    "meter": {
        "default": [
            "sf6_pressure_gauge", "oil_temp_gauge", "oil_level_gauge",
            "digital_display", "pointer_gauge", "led_indicator",
        ],
    },
    "thermal": {
        "default": [
            "thermal_normal", "thermal_warning", "thermal_danger",
            "hotspot", "cold_spot",
        ],
    },
}


def get_classes(plugin: str, voltage: str) -> List[str]:
    """获取指定 插件+电压 的检测类别"""
    plugin_map = PLUGIN_CLASSES.get(plugin, {})
    category = voltage.split("_")[0]  # UHV, EHV, HV, MV, LV
    return plugin_map.get(category, plugin_map.get("default", ["class_0"]))


# =============================================================================
# 核心聚合器
# =============================================================================

class TrainingDataAggregator:
    """
    训练数据聚合器
    
    替代 train_main.py 中手动指定数据路径的方式,
    自动扫描并聚合所有有效数据集。
    """

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or DATA_ROOT
        self.raw_path = self.data_root / "raw"
        self.processed_path = self.data_root / "processed"

    def discover_datasets(
        self, voltage: str, plugin: str
    ) -> List[Dict[str, Any]]:
        """
        发现指定 电压+插件 下的所有数据集子目录
        
        Returns:
            [{"path": Path, "id": str, "images": int, "labels": int, 
              "status": str, "meta": dict}, ...]
        """
        base = self.raw_path / voltage / plugin
        datasets = []

        if not base.exists():
            logger.warning(f"数据目录不存在: {base}")
            return datasets

        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue

            meta = {}
            meta_file = entry / "meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            # 统计图片和标签
            images = self._count_files(entry, IMAGE_EXTS)
            labels = self._count_files(entry, LABEL_EXTS)
            status = meta.get("status", "unknown")

            datasets.append({
                "path": entry,
                "id": entry.name,
                "images": images,
                "labels": labels,
                "status": status,
                "meta": meta,
            })

        return datasets

    def get_valid_datasets(
        self, voltage: str, plugin: str
    ) -> List[Dict[str, Any]]:
        """
        仅返回可用的数据集 (status 为 verified/warning/unknown)
        
        对于没有 meta.json 的目录 (手动拷贝的), 只要有图片就认为可用
        """
        all_ds = self.discover_datasets(voltage, plugin)
        valid = []
        for ds in all_ds:
            status = ds["status"]
            if status in ("verified", "warning", "unknown") and ds["images"] > 0:
                valid.append(ds)
            elif status == "error":
                logger.warning(f"跳过错误数据集: {ds['id']}")
        return valid

    def prepare_for_training(
        self,
        voltage: str,
        plugin: str,
        output_yaml: Optional[str] = None,
    ) -> Optional[str]:
        """
        核心方法: 准备训练数据
        
        1. 扫描 raw/{voltage}/{plugin}/ 下所有有效子目录
        2. 收集所有图片和标签路径
        3. 生成 YOLO 训练用 YAML 配置
        
        Args:
            voltage: 电压等级 (如 HV_220kV)
            plugin: 插件类型 (如 transformer)
            output_yaml: 输出 YAML 路径 (默认 processed/{v}/{p}/aggregated_data.yaml)
            
        Returns:
            生成的 YAML 文件路径 (可直接传给 model.train(data=...))
        """
        valid_datasets = self.get_valid_datasets(voltage, plugin)

        if not valid_datasets:
            logger.error(f"未找到有效数据集: {voltage}/{plugin}")
            return None

        logger.info(f"发现 {len(valid_datasets)} 个有效数据集:")
        total_images = 0
        total_labels = 0

        # 收集所有图片目录
        all_image_dirs = []
        all_label_dirs = []

        for ds in valid_datasets:
            ds_path = ds["path"]
            img_count = ds["images"]
            lbl_count = ds["labels"]
            total_images += img_count
            total_labels += lbl_count

            logger.info(f"  - {ds['id']}: {img_count} images, {lbl_count} labels")

            # 确定图片和标签目录
            images_subdir = ds_path / "images"
            labels_subdir = ds_path / "labels"

            if images_subdir.exists() and self._count_files(images_subdir, IMAGE_EXTS) > 0:
                all_image_dirs.append(str(images_subdir))
            else:
                # 图片散落在根目录
                all_image_dirs.append(str(ds_path))

            if labels_subdir.exists() and self._count_files(labels_subdir, LABEL_EXTS) > 0:
                all_label_dirs.append(str(labels_subdir))
            else:
                all_label_dirs.append(str(ds_path))

        # 获取类别
        classes = get_classes(plugin, voltage)
        names = {i: name for i, name in enumerate(classes)}

        # 输出路径
        if output_yaml is None:
            out_dir = self.processed_path / voltage / plugin
            out_dir.mkdir(parents=True, exist_ok=True)
            output_yaml = str(out_dir / "aggregated_data.yaml")

        # 构建 YAML 配置
        # YOLO 支持 train 字段为路径列表
        yaml_config = {
            "path": str(self.raw_path / voltage / plugin),
            "train": all_image_dirs,
            "val": all_image_dirs,
            "nc": len(classes),
            "names": names,
        }

        if YAML_AVAILABLE:
            with open(output_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)
        else:
            # Fallback: JSON 格式 (YOLO 不直接支持, 但可以程序中读取)
            json_path = output_yaml.replace('.yaml', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(yaml_config, f, ensure_ascii=False, indent=2)
            output_yaml = json_path

        logger.info(f"聚合 YAML 已生成: {output_yaml}")
        logger.info(f"总计: {total_images} 张图像, {total_labels} 个标签, "
                    f"{len(valid_datasets)} 个数据集")

        return output_yaml

    def prepare_split_for_training(
        self,
        voltage: str,
        plugin: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        output_yaml: Optional[str] = None,
    ) -> Optional[str]:
        """
        带 train/val/test 拆分的数据准备 (需要物理拷贝文件)
        
        步骤:
        1. 收集所有有效数据集的图片-标签对
        2. 随机打乱后按比例拆分
        3. 拷贝到 processed/{voltage}/{plugin}/images/{train,val,test}
        4. 生成标准 YOLO data.yaml
        """
        valid_datasets = self.get_valid_datasets(voltage, plugin)
        if not valid_datasets:
            logger.error(f"未找到有效数据集: {voltage}/{plugin}")
            return None

        # 收集所有图片-标签对
        all_pairs = []  # [(img_path, lbl_path or None), ...]

        for ds in valid_datasets:
            ds_path = ds["path"]
            images = self._find_images(ds_path)
            labels = self._find_labels(ds_path)
            label_map = {l.stem: l for l in labels}

            for img in images:
                lbl = label_map.get(img.stem)
                all_pairs.append((img, lbl))

        if not all_pairs:
            logger.error("未找到任何图片文件")
            return None

        random.shuffle(all_pairs)
        n = len(all_pairs)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": all_pairs[:train_end],
            "val": all_pairs[train_end:val_end],
            "test": all_pairs[val_end:],
        }

        # 输出目录
        out_base = self.processed_path / voltage / plugin
        for split_name in ["train", "val", "test"]:
            img_dir = out_base / "images" / split_name
            lbl_dir = out_base / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            # 清空旧数据
            for f in img_dir.iterdir():
                f.unlink()
            for f in lbl_dir.iterdir():
                f.unlink()

        # 拷贝文件
        stats = {}
        for split_name, pairs in splits.items():
            img_dest = out_base / "images" / split_name
            lbl_dest = out_base / "labels" / split_name

            for img_path, lbl_path in pairs:
                shutil.copy2(img_path, img_dest / img_path.name)
                if lbl_path and lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_dest / lbl_path.name)

            stats[split_name] = len(pairs)

        # 生成 data.yaml
        classes = get_classes(plugin, voltage)
        yaml_config = {
            "path": str(out_base),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(classes),
            "names": {i: name for i, name in enumerate(classes)},
        }

        if output_yaml is None:
            output_yaml = str(out_base / "data.yaml")

        if YAML_AVAILABLE:
            with open(output_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)
        else:
            json_path = output_yaml.replace('.yaml', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(yaml_config, f, ensure_ascii=False, indent=2)
            output_yaml = json_path

        logger.info(f"训练拆分完成: train={stats['train']}, "
                    f"val={stats['val']}, test={stats['test']}")
        logger.info(f"data.yaml: {output_yaml}")

        return output_yaml

    def get_summary(self, voltage: str, plugin: str) -> Dict[str, Any]:
        """获取数据摘要"""
        datasets = self.discover_datasets(voltage, plugin)
        valid = [d for d in datasets if d["status"] in ("verified", "warning", "unknown") and d["images"] > 0]
        total_images = sum(d["images"] for d in valid)
        total_labels = sum(d["labels"] for d in valid)

        return {
            "voltage": voltage,
            "plugin": plugin,
            "total_datasets": len(datasets),
            "valid_datasets": len(valid),
            "total_images": total_images,
            "total_labels": total_labels,
            "datasets": [{
                "id": d["id"],
                "images": d["images"],
                "labels": d["labels"],
                "status": d["status"],
            } for d in datasets],
        }

    # ---------- 内部工具方法 ----------

    @staticmethod
    def _count_files(directory: Path, extensions: set) -> int:
        """递归统计指定扩展名的文件数量"""
        count = 0
        for ext in extensions:
            count += len(list(directory.rglob(f"*{ext}")))
            count += len(list(directory.rglob(f"*{ext.upper()}")))
        return count

    @staticmethod
    def _find_images(directory: Path) -> List[Path]:
        """递归查找所有图片"""
        images = []
        for ext in IMAGE_EXTS:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
        return sorted(set(images))

    @staticmethod
    def _find_labels(directory: Path) -> List[Path]:
        """递归查找所有 YOLO 格式标签 (.txt)"""
        labels = list(directory.rglob("*.txt"))
        skip = {'classes.txt', 'meta.json', 'dataset_config.json'}
        return sorted(l for l in labels if l.name not in skip)


# =============================================================================
# train_main.py 集成接口
# =============================================================================

def prepare_training_data(
    voltage: str,
    plugin: str,
    split: bool = False,
    train_ratio: float = 0.8,
) -> Optional[str]:
    """
    供 train_main.py 直接调用的便捷函数
    
    用法:
        from training.data_aggregator import prepare_training_data
        
        data_yaml = prepare_training_data("HV_220kV", "transformer")
        if data_yaml:
            model.train(data=data_yaml, epochs=100)
    
    Args:
        voltage: 电压等级
        plugin: 插件类型
        split: 是否进行 train/val/test 物理拆分
        train_ratio: 训练集比例 (仅 split=True 时有效)
        
    Returns:
        YAML 配置文件路径, 或 None
    """
    agg = TrainingDataAggregator()

    if split:
        return agg.prepare_split_for_training(
            voltage, plugin, train_ratio=train_ratio
        )
    else:
        return agg.prepare_for_training(voltage, plugin)


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="训练数据聚合器 - 动态合并多数据集"
    )
    parser.add_argument("--voltage", "-v", required=True,
                       help="电压等级 (如 HV_220kV)")
    parser.add_argument("--plugin", "-p", required=True,
                       help="插件类型 (如 transformer)")
    parser.add_argument("--split", action="store_true",
                       help="进行 train/val/test 物理拆分")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="训练集比例 (默认 0.8)")
    parser.add_argument("--summary", action="store_true",
                       help="仅显示数据摘要, 不生成配置")
    parser.add_argument("--output", "-o", default=None,
                       help="输出 YAML 路径")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
    )

    agg = TrainingDataAggregator()

    if args.summary:
        summary = agg.get_summary(args.voltage, args.plugin)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.split:
        yaml_path = agg.prepare_split_for_training(
            args.voltage, args.plugin,
            train_ratio=args.train_ratio,
            output_yaml=args.output,
        )
    else:
        yaml_path = agg.prepare_for_training(
            args.voltage, args.plugin,
            output_yaml=args.output,
        )

    if yaml_path:
        print(f"\n✅ YAML 配置已生成: {yaml_path}")
        print(f"用法: model.train(data='{yaml_path}')")
    else:
        print("\n❌ 未找到有效数据集")
        sys.exit(1)


if __name__ == "__main__":
    main()
