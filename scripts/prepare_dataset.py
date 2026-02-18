#!/usr/bin/env python3
"""
数据集准备工具 — prepare_dataset.py
====================================
功能:
  1. 从 COCO 数据集提取动物类别子集
  2. 转换标注格式 (COCO JSON -> YOLO TXT)
  3. 数据集划分与校验
  4. 数据增强预览
  5. 数据集统计报告

使用方法:
  # 从 COCO 提取动物子集
  python prepare_dataset.py --mode extract-coco \
      --coco-images /path/to/coco/train2017 \
      --coco-ann /path/to/coco/annotations/instances_train2017.json \
      --output datasets/animal_intrusion

  # 校验已有数据集
  python prepare_dataset.py --mode validate \
      --dataset datasets/animal_intrusion

  # 生成统计报告
  python prepare_dataset.py --mode stats \
      --dataset datasets/animal_intrusion

  # 从自采数据创建数据集 (需先用 LabelImg/CVAT 标注)
  python prepare_dataset.py --mode import-labeled \
      --source /path/to/labeled_data \
      --output datasets/animal_intrusion \
      --format yolo

硬约束: 不生成任何合成/随机数据, 所有数据必须来源可追溯
"""

import argparse
import json
import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_dataset")

# 目标类别
TARGET_CLASSES = ["rat", "cat", "snake", "bird", "dog", "poultry", "other_wildlife"]

# COCO 类别 -> 目标类别映射
COCO_TO_TARGET = {
    "bird": 3,       # -> bird
    "cat": 1,        # -> cat
    "dog": 4,        # -> dog
    "horse": 6,      # -> other_wildlife
    "sheep": 6,      # -> other_wildlife
    "cow": 6,        # -> other_wildlife
    "elephant": 6,   # -> other_wildlife
    "bear": 6,       # -> other_wildlife
    "zebra": 6,      # -> other_wildlife
    "giraffe": 6,    # -> other_wildlife
}


def setup_directory_structure(output_dir: str) -> Dict[str, str]:
    """创建标准 YOLO 数据集目录结构."""
    dirs = {
        "train_images": os.path.join(output_dir, "images", "train"),
        "val_images": os.path.join(output_dir, "images", "val"),
        "test_images": os.path.join(output_dir, "images", "test"),
        "train_labels": os.path.join(output_dir, "labels", "train"),
        "val_labels": os.path.join(output_dir, "labels", "val"),
        "test_labels": os.path.join(output_dir, "labels", "test"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def extract_coco_animals(coco_images_dir: str, coco_ann_path: str,
                         output_dir: str, max_per_class: int = 2000,
                         split_ratio: Tuple[float, ...] = (0.8, 0.1, 0.1)) -> None:
    """
    从 COCO 数据集提取动物类别, 转换为 YOLO 格式.

    参数:
      coco_images_dir: COCO 图片目录 (train2017/)
      coco_ann_path: COCO 标注文件 (instances_train2017.json)
      output_dir: 输出目录
      max_per_class: 每类最多提取图片数
      split_ratio: 训练/验证/测试 划分比例
    """
    logger.info("加载 COCO 标注: %s", coco_ann_path)
    with open(coco_ann_path, "r") as f:
        coco = json.load(f)

    # 建立类别 ID -> 名称映射
    coco_cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    # 筛选动物类别
    target_coco_ids = {}
    for cid, cname in coco_cat_map.items():
        if cname in COCO_TO_TARGET:
            target_coco_ids[cid] = COCO_TO_TARGET[cname]

    if not target_coco_ids:
        logger.error("COCO 数据中未找到动物类别")
        return

    logger.info("COCO 动物类别映射: %s",
                {coco_cat_map[k]: TARGET_CLASSES[v] for k, v in target_coco_ids.items()})

    # 建立图片 ID -> 文件名映射
    img_map = {img["id"]: img for img in coco["images"]}

    # 按类别收集标注
    class_annotations = defaultdict(list)  # target_class_id -> [(img_id, ann)]
    for ann in coco["annotations"]:
        if ann["category_id"] in target_coco_ids:
            target_cls = target_coco_ids[ann["category_id"]]
            class_annotations[target_cls].append((ann["image_id"], ann))

    # 收集需要的图片 ID 及其标注
    # 每类限制 max_per_class 张
    selected_images = {}  # img_id -> list of (target_cls, bbox_normalized)
    class_counts = Counter()

    for target_cls, anns in class_annotations.items():
        random.seed(42)
        random.shuffle(anns)
        count = 0
        for img_id, ann in anns:
            if count >= max_per_class:
                break
            if img_id not in img_map:
                continue

            img_info = img_map[img_id]
            w, h = img_info["width"], img_info["height"]
            if w <= 0 or h <= 0:
                continue

            # COCO bbox: [x, y, width, height] (绝对像素)
            # YOLO bbox: [x_center, y_center, width, height] (归一化)
            bx, by, bw, bh = ann["bbox"]
            x_center = (bx + bw / 2) / w
            y_center = (by + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h

            # 边界检查
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 < bw_norm <= 1 and 0 < bh_norm <= 1):
                continue

            if img_id not in selected_images:
                selected_images[img_id] = []
            selected_images[img_id].append(
                (target_cls, x_center, y_center, bw_norm, bh_norm)
            )
            count += 1
            class_counts[target_cls] += 1

    logger.info("提取图片总数: %d", len(selected_images))
    for cls_id, cnt in sorted(class_counts.items()):
        logger.info("  %s: %d 个标注", TARGET_CLASSES[cls_id], cnt)

    # 划分数据集
    img_ids = sorted(selected_images.keys())
    random.seed(42)
    random.shuffle(img_ids)

    n_total = len(img_ids)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    splits = {
        "train": img_ids[:n_train],
        "val": img_ids[n_train:n_train + n_val],
        "test": img_ids[n_train + n_val:],
    }

    # 创建目录结构
    dirs = setup_directory_structure(output_dir)

    # 复制图片和生成标注
    provenance_log = []
    for split_name, split_ids in splits.items():
        img_dir = dirs[f"{split_name}_images"]
        lbl_dir = dirs[f"{split_name}_labels"]

        for img_id in split_ids:
            img_info = img_map[img_id]
            src_path = os.path.join(coco_images_dir, img_info["file_name"])

            if not os.path.isfile(src_path):
                continue

            # 复制图片
            dst_img = os.path.join(img_dir, img_info["file_name"])
            shutil.copy2(src_path, dst_img)

            # 生成 YOLO 标注文件
            label_name = os.path.splitext(img_info["file_name"])[0] + ".txt"
            dst_lbl = os.path.join(lbl_dir, label_name)
            with open(dst_lbl, "w") as f:
                for cls_id, xc, yc, bw, bh in selected_images[img_id]:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            provenance_log.append({
                "file": img_info["file_name"],
                "split": split_name,
                "source": "COCO",
                "coco_image_id": img_id,
                "annotations": len(selected_images[img_id]),
            })

        logger.info("%s 集: %d 张图片", split_name, len(split_ids))

    # 保存数据来源审计日志
    provenance_path = os.path.join(output_dir, "data_provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "COCO",
            "coco_annotations": coco_ann_path,
            "class_mapping": {coco_cat_map[k]: TARGET_CLASSES[v]
                              for k, v in target_coco_ids.items()},
            "split_ratio": list(split_ratio),
            "total_images": len(provenance_log),
            "class_distribution": {TARGET_CLASSES[k]: v
                                   for k, v in sorted(class_counts.items())},
            "entries": provenance_log,
        }, f, indent=2, ensure_ascii=False)

    logger.info("数据来源审计日志: %s", provenance_path)


def import_labeled_data(source_dir: str, output_dir: str,
                        fmt: str = "yolo",
                        split_ratio: Tuple[float, ...] = (0.8, 0.1, 0.1)) -> None:
    """
    导入已标注的自采数据.

    支持格式:
      - yolo: 图片和同名 .txt 标注在同一目录或 images/labels 子目录
      - coco: COCO JSON 标注
    """
    logger.info("导入标注数据: %s (格式: %s)", source_dir, fmt)

    if fmt == "yolo":
        _import_yolo_format(source_dir, output_dir, split_ratio)
    elif fmt == "coco":
        # 查找 annotation json
        ann_files = list(Path(source_dir).glob("*.json"))
        if not ann_files:
            logger.error("未找到 COCO JSON 标注文件")
            return
        img_dir = os.path.join(source_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = source_dir
        extract_coco_animals(img_dir, str(ann_files[0]), output_dir,
                             max_per_class=99999, split_ratio=split_ratio)
    else:
        logger.error("不支持的格式: %s", fmt)


def _import_yolo_format(source_dir: str, output_dir: str,
                        split_ratio: Tuple[float, ...]) -> None:
    """导入 YOLO 格式标注数据."""
    # 查找图片
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []

    # 尝试 images/ 子目录
    img_dir = os.path.join(source_dir, "images")
    lbl_dir = os.path.join(source_dir, "labels")
    if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
        for f in os.listdir(img_dir):
            if os.path.splitext(f)[1].lower() in img_exts:
                lbl_file = os.path.splitext(f)[0] + ".txt"
                if os.path.isfile(os.path.join(lbl_dir, lbl_file)):
                    images.append((
                        os.path.join(img_dir, f),
                        os.path.join(lbl_dir, lbl_file)
                    ))
    else:
        # 同一目录
        for f in os.listdir(source_dir):
            if os.path.splitext(f)[1].lower() in img_exts:
                lbl_file = os.path.splitext(f)[0] + ".txt"
                lbl_path = os.path.join(source_dir, lbl_file)
                if os.path.isfile(lbl_path):
                    images.append((
                        os.path.join(source_dir, f),
                        lbl_path
                    ))

    if not images:
        logger.error("未找到配对的图片+标注文件")
        return

    logger.info("找到 %d 个配对文件", len(images))

    # 校验标注格式
    nc = len(TARGET_CLASSES)
    valid_images = []
    for img_path, lbl_path in images:
        valid = True
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                if len(parts) != 5:
                    logger.warning("标注格式错误 %s: %s", lbl_path, line.strip())
                    valid = False
                    break
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= nc:
                    logger.warning("类别ID超出范围 %s: %d", lbl_path, cls_id)
                    valid = False
                    break
        if valid:
            valid_images.append((img_path, lbl_path))

    logger.info("有效数据: %d / %d", len(valid_images), len(images))

    # 随机划分
    random.seed(42)
    random.shuffle(valid_images)
    n = len(valid_images)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    splits = {
        "train": valid_images[:n_train],
        "val": valid_images[n_train:n_train + n_val],
        "test": valid_images[n_train + n_val:],
    }

    dirs = setup_directory_structure(output_dir)

    for split_name, split_data in splits.items():
        for img_path, lbl_path in split_data:
            fname = os.path.basename(img_path)
            lname = os.path.splitext(fname)[0] + ".txt"
            shutil.copy2(img_path, os.path.join(dirs[f"{split_name}_images"], fname))
            shutil.copy2(lbl_path, os.path.join(dirs[f"{split_name}_labels"], lname))
        logger.info("%s 集: %d 张", split_name, len(split_data))


def validate_dataset(dataset_dir: str) -> bool:
    """全面校验数据集完整性和质量."""
    logger.info("校验数据集: %s", dataset_dir)
    errors = []
    warnings = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.isdir(img_dir):
            if split == "test":
                warnings.append(f"测试集目录不存在: {img_dir} (可选)")
            else:
                errors.append(f"{split} 图片目录不存在: {img_dir}")
            continue

        if not os.path.isdir(lbl_dir):
            errors.append(f"{split} 标注目录不存在: {lbl_dir}")
            continue

        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_stems = set()
        for f in os.listdir(img_dir):
            if os.path.splitext(f)[1].lower() in img_exts:
                img_stems.add(os.path.splitext(f)[0])

        lbl_stems = set()
        for f in os.listdir(lbl_dir):
            if f.endswith(".txt"):
                lbl_stems.add(os.path.splitext(f)[0])

        # 配对检查
        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems

        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} 张图片缺标注")
        if orphan_labels:
            warnings.append(f"{split}: {len(orphan_labels)} 个孤立标注文件")

        # 标注内容检查
        nc = len(TARGET_CLASSES)
        class_counts = Counter()
        bbox_count = 0
        for stem in lbl_stems & img_stems:
            lpath = os.path.join(lbl_dir, stem + ".txt")
            with open(lpath, "r") as f:
                for line_no, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        errors.append(f"{lpath}:{line_no} 字段数={len(parts)}")
                        continue
                    try:
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                    except ValueError:
                        errors.append(f"{lpath}:{line_no} 数值解析失败")
                        continue

                    if cls_id < 0 or cls_id >= nc:
                        errors.append(f"{lpath}:{line_no} 类别{cls_id}超限")
                    for i, v in enumerate(coords):
                        if v < 0 or v > 1:
                            errors.append(f"{lpath}:{line_no} 坐标[{i}]={v}超限")

                    class_counts[cls_id] += 1
                    bbox_count += 1

        logger.info("%s 集: %d 图片, %d 标注框", split, len(img_stems), bbox_count)
        for cls_id in range(nc):
            cnt = class_counts.get(cls_id, 0)
            status = "⚠" if cnt < 100 else "✓"
            logger.info("  %s %s: %d", status, TARGET_CLASSES[cls_id], cnt)
            if cnt < 50:
                warnings.append(f"{split}/{TARGET_CLASSES[cls_id]} 仅 {cnt} 个样本, 建议 ≥500")

    # 输出结果
    if errors:
        logger.error("=" * 40)
        logger.error("校验失败! %d 个错误:", len(errors))
        for e in errors[:20]:
            logger.error("  ✗ %s", e)
        if len(errors) > 20:
            logger.error("  ... 还有 %d 个错误", len(errors) - 20)
        return False

    if warnings:
        logger.warning("%d 个警告:", len(warnings))
        for w in warnings:
            logger.warning("  ⚠ %s", w)

    logger.info("✓ 数据集校验通过")
    return True


def generate_stats(dataset_dir: str) -> None:
    """生成数据集统计报告."""
    logger.info("生成统计报告: %s", dataset_dir)

    stats = {
        "dataset_dir": dataset_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "class_names": TARGET_CLASSES,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.isdir(img_dir):
            continue

        class_counts = Counter()
        total_boxes = 0
        boxes_per_image = []
        bbox_sizes = []

        lbl_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")] \
            if os.path.isdir(lbl_dir) else []

        for lf in lbl_files:
            n_boxes = 0
            with open(os.path.join(lbl_dir, lf), "r") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        class_counts[cls_id] += 1
                        bbox_sizes.append(w * h)
                        n_boxes += 1
            boxes_per_image.append(n_boxes)
            total_boxes += n_boxes

        n_images = sum(1 for f in os.listdir(img_dir)
                       if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"})

        split_stats = {
            "num_images": n_images,
            "num_annotations": total_boxes,
            "avg_boxes_per_image": round(sum(boxes_per_image) / max(len(boxes_per_image), 1), 2),
            "class_distribution": {TARGET_CLASSES[k]: v for k, v in sorted(class_counts.items())},
        }

        if bbox_sizes:
            bbox_sizes.sort()
            split_stats["bbox_area_stats"] = {
                "min": round(min(bbox_sizes), 6),
                "max": round(max(bbox_sizes), 6),
                "median": round(bbox_sizes[len(bbox_sizes)//2], 6),
                "mean": round(sum(bbox_sizes)/len(bbox_sizes), 6),
            }
            # 小目标占比 (面积 < 0.01, 即 < 1% 图像面积)
            small_count = sum(1 for s in bbox_sizes if s < 0.01)
            split_stats["small_object_ratio"] = round(small_count / len(bbox_sizes), 4)

        stats["splits"][split] = split_stats

    report_path = os.path.join(dataset_dir, "dataset_stats.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("统计报告已保存: %s", report_path)

    # 打印摘要
    for split, ss in stats["splits"].items():
        logger.info("--- %s ---", split)
        logger.info("  图片: %d, 标注: %d, 平均框/图: %.1f",
                     ss["num_images"], ss["num_annotations"], ss["avg_boxes_per_image"])
        for cls_name, cnt in ss["class_distribution"].items():
            logger.info("    %s: %d", cls_name, cnt)


def main():
    parser = argparse.ArgumentParser(description="动物检测数据集准备工具")
    parser.add_argument("--mode", required=True,
                        choices=["extract-coco", "import-labeled", "validate", "stats"])
    parser.add_argument("--coco-images", help="COCO 图片目录")
    parser.add_argument("--coco-ann", help="COCO 标注 JSON")
    parser.add_argument("--source", help="自采数据源目录")
    parser.add_argument("--output", "--dataset", dest="output",
                        default="datasets/animal_intrusion")
    parser.add_argument("--format", default="yolo", choices=["yolo", "coco"])
    parser.add_argument("--max-per-class", type=int, default=2000)
    args = parser.parse_args()

    if args.mode == "extract-coco":
        if not args.coco_images or not args.coco_ann:
            parser.error("extract-coco 需要 --coco-images 和 --coco-ann")
        extract_coco_animals(args.coco_images, args.coco_ann, args.output,
                             args.max_per_class)
    elif args.mode == "import-labeled":
        if not args.source:
            parser.error("import-labeled 需要 --source")
        import_labeled_data(args.source, args.output, args.format)
    elif args.mode == "validate":
        validate_dataset(args.output)
    elif args.mode == "stats":
        generate_stats(args.output)


if __name__ == "__main__":
    main()
