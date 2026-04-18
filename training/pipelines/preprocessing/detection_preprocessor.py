"""
目标检测预处理器

处理 YOLO/COCO/VOC 格式数据，统一输出为 YOLO 格式:
  output_dir/
  ├── train/images/  train/labels/
  ├── val/images/    val/labels/
  ├── test/images/   test/labels/
  └── dataset.yaml
"""

from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .base_preprocessor import BasePreprocessor, PreprocessResult

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DetectionPreprocessor(BasePreprocessor):
    """目标检测数据预处理"""

    def preprocess(self, input_dir: Path, output_dir: Path) -> PreprocessResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 检测输入格式
        source_format = self._detect_format(input_dir)
        logger.info(f"检测到数据格式: {source_format}")

        if source_format == "yolo":
            image_files = self._collect_images(input_dir / "images")
        elif source_format == "coco":
            image_files = self._convert_coco_to_yolo(input_dir, output_dir)
        elif source_format == "voc":
            image_files = self._convert_voc_to_yolo(input_dir, output_dir)
        else:
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message=f"不支持的数据格式: {source_format}",
            )

        # 分割数据集
        splits = self.split_dataset(image_files, output_dir)

        # 复制文件到 split 目录
        for split_name, files in splits.items():
            for img_path in files:
                dst_img = output_dir / split_name / "images" / img_path.name
                shutil.copy2(img_path, dst_img)
                # 复制对应标签
                label_path = self._find_label(img_path, input_dir)
                if label_path and label_path.exists():
                    dst_label = output_dir / split_name / "labels" / label_path.name
                    shutil.copy2(label_path, dst_label)

        result = PreprocessResult(
            output_dir=output_dir,
            num_train=len(splits["train"]),
            num_val=len(splits["val"]),
            num_test=len(splits["test"]),
            success=True,
            message=f"预处理完成: {sum(len(v) for v in splits.values())} 张图像",
        )
        return result

    def _detect_format(self, input_dir: Path) -> str:
        """自动检测数据格式"""
        if (input_dir / "labels").exists():
            return "yolo"
        for ann_file in ("annotations.json", "instances.json"):
            if (input_dir / ann_file).exists():
                return "coco"
        if (input_dir / "Annotations").exists():
            return "voc"
        if (input_dir / "images").exists():
            return "yolo"  # 默认
        return "unknown"

    def _collect_images(self, images_dir: Path) -> list[Path]:
        if not images_dir.exists():
            return []
        return [f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]

    def _find_label(self, img_path: Path, input_dir: Path) -> Path | None:
        """查找对应的 YOLO 标签文件"""
        label_name = img_path.stem + ".txt"
        for labels_dir in (input_dir / "labels", img_path.parent.parent / "labels"):
            candidate = labels_dir / label_name
            if candidate.exists():
                return candidate
        return None

    def _convert_coco_to_yolo(self, input_dir: Path, output_dir: Path) -> list[Path]:
        """COCO JSON → YOLO txt 格式转换"""
        ann_file = None
        for name in ("annotations.json", "instances.json"):
            if (input_dir / name).exists():
                ann_file = input_dir / name
                break
        if not ann_file:
            return []

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # 构建映射
        img_map = {img["id"]: img for img in coco["images"]}
        cat_map = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

        temp_labels = output_dir / "_temp_labels"
        temp_labels.mkdir(parents=True, exist_ok=True)

        for ann in coco["annotations"]:
            img_info = img_map.get(ann["image_id"])
            if not img_info:
                continue
            cat_idx = cat_map.get(ann["category_id"])
            if cat_idx is None:
                continue
            x, y, w, h = ann["bbox"]
            iw, ih = img_info["width"], img_info["height"]
            cx, cy = (x + w / 2) / iw, (y + h / 2) / ih
            nw, nh = w / iw, h / ih

            label_file = temp_labels / (Path(img_info["file_name"]).stem + ".txt")
            with open(label_file, "a") as lf:
                lf.write(f"{cat_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        images_dir = input_dir / "images"
        return self._collect_images(images_dir)

    def _convert_voc_to_yolo(self, input_dir: Path, output_dir: Path) -> list[Path]:
        """VOC XML → YOLO txt 格式转换"""
        ann_dir = input_dir / "Annotations"
        images_dir = input_dir / "JPEGImages"
        if not images_dir.exists():
            images_dir = input_dir / "images"

        if not ann_dir.exists():
            return []

        # 收集所有类别
        all_classes: list[str] = []
        for xml_file in ann_dir.glob("*.xml"):
            tree = ET.parse(xml_file)
            for obj in tree.findall(".//object"):
                name_elem = obj.find("name")
                if name_elem is not None and name_elem.text not in all_classes:
                    all_classes.append(name_elem.text)

        temp_labels = output_dir / "_temp_labels"
        temp_labels.mkdir(parents=True, exist_ok=True)

        for xml_file in ann_dir.glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find("size")
            if size is None:
                continue
            iw = int(size.findtext("width", "1"))
            ih = int(size.findtext("height", "1"))

            label_file = temp_labels / (xml_file.stem + ".txt")
            with open(label_file, "w") as lf:
                for obj in root.findall("object"):
                    name_elem = obj.find("name")
                    if name_elem is None:
                        continue
                    cls_idx = all_classes.index(name_elem.text)
                    bbox = obj.find("bndbox")
                    if bbox is None:
                        continue
                    xmin = float(bbox.findtext("xmin", "0"))
                    ymin = float(bbox.findtext("ymin", "0"))
                    xmax = float(bbox.findtext("xmax", "0"))
                    ymax = float(bbox.findtext("ymax", "0"))
                    cx = (xmin + xmax) / 2 / iw
                    cy = (ymin + ymax) / 2 / ih
                    w = (xmax - xmin) / iw
                    h = (ymax - ymin) / ih
                    lf.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        return self._collect_images(images_dir)
