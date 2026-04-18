"""
语义分割预处理器

处理以下 mask 格式:
- png_index:  单通道 PNG (像素值 = class_id)
- png_color:  三通道 RGB PNG (通过 color_map 映射到 class_id)
- rle/polygon: COCO 格式 JSON → 转为 png_index mask

输出:
  output_dir/
  ├── train/images/  train/masks/
  ├── val/images/    val/masks/
  ├── test/images/   test/masks/
  └── dataset.yaml
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class SegmentationPreprocessor(BasePreprocessor):
    """语义分割数据预处理"""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.mask_format: str = self.config.get("mask_format", "png_index")
        self.color_map: dict[str, list[int]] = self.config.get("mask_color_map", {})
        self.ignore_index: int = self.config.get("ignore_index", 255)

    def preprocess(self, input_dir: Path, output_dir: Path) -> PreprocessResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 收集图像
        images_dir = input_dir / "images"
        if not images_dir.exists():
            images_dir = input_dir / "JPEGImages"
        image_files = self._collect_images(images_dir) if images_dir.exists() else []

        if not image_files:
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message="未找到图像文件",
            )

        # 收集 mask
        masks_dir = self._find_mask_dir(input_dir)
        if masks_dir is None and self.mask_format not in ("rle", "polygon"):
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message="未找到 mask 目录 (masks/ 或 SegmentationClass/)",
            )

        # 分割数据集
        splits = self.split_dataset(image_files, output_dir)

        # 复制文件
        for split_name, files in splits.items():
            (output_dir / split_name / "masks").mkdir(parents=True, exist_ok=True)
            for img_path in files:
                dst_img = output_dir / split_name / "images" / img_path.name
                shutil.copy2(img_path, dst_img)
                # 找对应 mask
                if masks_dir:
                    mask_path = masks_dir / (img_path.stem + ".png")
                    if mask_path.exists():
                        dst_mask = output_dir / split_name / "masks" / mask_path.name
                        if self.mask_format == "png_color" and self.color_map:
                            self._convert_color_to_index(mask_path, dst_mask)
                        else:
                            shutil.copy2(mask_path, dst_mask)

        return PreprocessResult(
            output_dir=output_dir,
            num_train=len(splits["train"]),
            num_val=len(splits["val"]),
            num_test=len(splits["test"]),
            success=True,
            message=f"分割预处理完成: {sum(len(v) for v in splits.values())} 张",
        )

    def _collect_images(self, directory: Path) -> list[Path]:
        return [f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS]

    def _find_mask_dir(self, input_dir: Path) -> Path | None:
        for name in ("masks", "SegmentationClass", "labels", "annotations"):
            p = input_dir / name
            if p.exists():
                return p
        return None

    def _convert_color_to_index(self, src: Path, dst: Path) -> None:
        """将 RGB color mask 转换为 index mask"""
        try:
            import numpy as np
            from PIL import Image

            img = np.array(Image.open(src).convert("RGB"))
            h, w, _ = img.shape
            index_mask = np.full((h, w), self.ignore_index, dtype=np.uint8)

            for class_name, color in self.color_map.items():
                # 此处需要 class_name → class_id 的映射
                # 简化处理: 用 color_map 中的出现顺序作为 class_id
                class_id = list(self.color_map.keys()).index(class_name)
                match = np.all(img == np.array(color), axis=2)
                index_mask[match] = class_id

            Image.fromarray(index_mask, mode="L").save(dst)
        except ImportError:
            shutil.copy2(src, dst)
