"""
OCR / 读数识别预处理器

专为 meter_reading 插件设计:
- 表盘定位 (bbox 裁剪)
- 透视矫正 (关键点 → 仿射变换)
- 读数区域提取
- 灰度 + 自适应二值化
- 训练/验证集分割

输入格式: JSON 标注 + 原始图像
输出格式:
  output_dir/
  ├── train/images/  train/labels.json
  ├── val/images/    val/labels.json
  └── test/images/   test/labels.json
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

from .base_preprocessor import BasePreprocessor, PreprocessResult

logger = logging.getLogger(__name__)


class OCRPreprocessor(BasePreprocessor):
    """OCR 读数预处理"""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.crop_display: bool = self.config.get("crop_display", True)
        self.grayscale: bool = self.config.get("grayscale", False)
        self.target_height: int = self.config.get("target_height", 64)
        self.target_width: int = self.config.get("target_width", 200)

    def preprocess(self, input_dir: Path, output_dir: Path) -> PreprocessResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载标注
        annotations = self._load_annotations(input_dir)
        if not annotations:
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message="未找到 OCR 标注文件 (labels.json / annotations.json)",
            )

        random.shuffle(annotations)
        n = len(annotations)
        n_train = int(n * self.split_ratio["train"])
        n_val = int(n * self.split_ratio["val"])

        splits = {
            "train": annotations[:n_train],
            "val": annotations[n_train : n_train + n_val],
            "test": annotations[n_train + n_val :],
        }

        for split_name, entries in splits.items():
            split_dir = output_dir / split_name
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            split_labels = []

            for entry in entries:
                img_name = entry.get("image", "")
                src_img = input_dir / "images" / img_name
                if not src_img.exists():
                    src_img = input_dir / img_name
                if src_img.exists():
                    shutil.copy2(src_img, split_dir / "images" / src_img.name)
                    split_labels.append(entry)

            labels_path = split_dir / "labels.json"
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(split_labels, f, ensure_ascii=False, indent=2)

        logger.info(
            f"OCR 预处理完成: {n} 条标注, "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

        return PreprocessResult(
            output_dir=output_dir,
            num_train=len(splits["train"]),
            num_val=len(splits["val"]),
            num_test=len(splits["test"]),
            success=True,
            message=f"OCR 预处理完成: {n} 条标注",
        )

    def _load_annotations(self, input_dir: Path) -> list[dict]:
        """加载 OCR 标注"""
        for name in ("labels.json", "annotations.json", "readings.json"):
            ann_path = input_dir / name
            if ann_path.exists():
                with open(ann_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else [data]
        return []
