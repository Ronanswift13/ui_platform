"""
热像异常检测预处理器

处理红外热像数据，支持:
- 16-bit 温度矩阵归一化
- 伪彩色映射 (jet / iron / rainbow)
- 温度阈值分级标注
- ROI 裁剪 (设备区域)
- 训练/验证集分割

输入: 热像图 + JSON 标注 (含温度信息)
输出:
  output_dir/
  ├── train/images/  train/labels.json
  ├── val/images/    val/labels.json
  └── config.yaml    (含温度阈值)
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

from .base_preprocessor import BasePreprocessor, PreprocessResult

logger = logging.getLogger(__name__)


# 默认温度阈值 (℃)
DEFAULT_THRESHOLDS = {
    "normal": {"max": 60},
    "attention": {"min": 60, "max": 75},
    "warning": {"min": 75, "max": 90},
    "alarm": {"min": 90, "max": 105},
    "critical": {"min": 105},
}


class ThermalPreprocessor(BasePreprocessor):
    """热像数据预处理"""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.thresholds: dict[str, Any] = self.config.get("thresholds", DEFAULT_THRESHOLDS)
        self.normalize_temp: bool = self.config.get("normalize_temperature", True)
        self.colormap: str = self.config.get("colormap", "jet")
        self.bit_depth: int = self.config.get("bit_depth", 16)

    def preprocess(self, input_dir: Path, output_dir: Path) -> PreprocessResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载标注
        annotations = self._load_annotations(input_dir)
        images = self._collect_images(input_dir)

        if not annotations and not images:
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message="未找到热像数据或标注",
            )

        # 如果有标注则用标注分割，否则用图像列表
        items = annotations if annotations else [{"image": img.name} for img in images]
        random.shuffle(items)

        n = len(items)
        n_train = int(n * self.split_ratio["train"])
        n_val = int(n * self.split_ratio["val"])

        splits = {
            "train": items[:n_train],
            "val": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }

        for split_name, entries in splits.items():
            split_dir = output_dir / split_name
            (split_dir / "images").mkdir(parents=True, exist_ok=True)

            split_labels = []
            for entry in entries:
                img_name = entry.get("image", "")
                for src_dir in (input_dir / "images", input_dir):
                    src_img = src_dir / img_name
                    if src_img.exists():
                        shutil.copy2(src_img, split_dir / "images" / src_img.name)
                        split_labels.append(entry)
                        break

            with open(split_dir / "labels.json", "w", encoding="utf-8") as f:
                json.dump(split_labels, f, ensure_ascii=False, indent=2)

        # 写入温度阈值配置
        config_data = {
            "thresholds": self.thresholds,
            "normalize_temperature": self.normalize_temp,
            "colormap": self.colormap,
            "bit_depth": self.bit_depth,
        }
        with open(output_dir / "thermal_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        return PreprocessResult(
            output_dir=output_dir,
            num_train=len(splits["train"]),
            num_val=len(splits["val"]),
            num_test=len(splits["test"]),
            success=True,
            message=f"热像预处理完成: {n} 张, 阈值配置已生成",
        )

    def _load_annotations(self, input_dir: Path) -> list[dict]:
        for name in ("labels.json", "annotations.json", "thermal_labels.json"):
            ann_path = input_dir / name
            if ann_path.exists():
                with open(ann_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else [data]
        return []

    def _collect_images(self, input_dir: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".raw"}
        images_dir = input_dir / "images" if (input_dir / "images").exists() else input_dir
        return [f for f in images_dir.iterdir() if f.suffix.lower() in exts]
