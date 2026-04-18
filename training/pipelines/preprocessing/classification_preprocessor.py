"""
图像分类预处理器

输入: 按文件夹组织的图像 (folder_name = class_name) 或 CSV 标注
输出:
  output_dir/
  ├── train/{class_name}/*.jpg
  ├── val/{class_name}/*.jpg
  └── test/{class_name}/*.jpg
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

from .base_preprocessor import BasePreprocessor, PreprocessResult

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class ClassificationPreprocessor(BasePreprocessor):
    """图像分类数据预处理"""

    def preprocess(self, input_dir: Path, output_dir: Path) -> PreprocessResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有类别目录
        class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        if not class_dirs:
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message="未找到类别子目录",
            )

        class_names = [d.name for d in class_dirs]
        class_dist: dict[str, int] = {}
        total = {"train": 0, "val": 0, "test": 0}

        for cls_dir in class_dirs:
            images = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            random.shuffle(images)
            class_dist[cls_dir.name] = len(images)

            n = len(images)
            n_train = int(n * self.split_ratio["train"])
            n_val = int(n * self.split_ratio["val"])

            splits = {
                "train": images[:n_train],
                "val": images[n_train : n_train + n_val],
                "test": images[n_train + n_val :],
            }

            for split_name, files in splits.items():
                dst_dir = output_dir / split_name / cls_dir.name
                dst_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy2(f, dst_dir / f.name)
                total[split_name] += len(files)

        logger.info(f"分类预处理完成: {len(class_names)} 类, {sum(class_dist.values())} 张")

        return PreprocessResult(
            output_dir=output_dir,
            num_train=total["train"],
            num_val=total["val"],
            num_test=total["test"],
            class_distribution=class_dist,
            success=True,
            message=f"分类预处理完成: {len(class_names)} 类",
        )
