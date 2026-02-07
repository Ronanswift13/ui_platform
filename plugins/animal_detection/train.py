#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - 可复现训练脚本
====================================

使用 Ultralytics YOLOv8 训练动物检测模型:
- 支持迁移学习 (从COCO预训练权重)
- 多尺度训练
- 模型版本化管理
- 指标记录与评估
- ONNX导出

用法:
    python train.py --data configs/animal_data.yaml --model yolov8n.pt --epochs 100
    python train.py --export-only --weights runs/best.pt

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 数据集校验
# =============================================================================

def validate_dataset(data_yaml: str) -> Dict[str, Any]:
    """
    校验数据集完整性

    检查:
    - YAML配置文件格式
    - 训练/验证/测试集目录存在
    - 图像-标签文件一一对应
    - 标签格式正确 (YOLO格式: class cx cy w h)
    - 类别分布统计

    Returns:
        校验报告字典
    """
    import yaml

    report = {
        "valid": True,
        "data_yaml": data_yaml,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # 读取YAML
    data_path = Path(data_yaml)
    if not data_path.exists():
        report["valid"] = False
        report["errors"].append(f"数据集配置文件不存在: {data_yaml}")
        return report

    with open(data_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # 检查必要字段
    required_fields = ["train", "val", "nc", "names"]
    for field in required_fields:
        if field not in data_cfg:
            report["valid"] = False
            report["errors"].append(f"缺少必要字段: {field}")

    if not report["valid"]:
        return report

    base_dir = data_path.parent
    nc = data_cfg["nc"]
    names = data_cfg["names"]

    if len(names) != nc:
        report["valid"] = False
        report["errors"].append(f"类别数不匹配: nc={nc}, names长度={len(names)}")

    # 检查各数据集分割
    for split in ["train", "val", "test"]:
        split_path = data_cfg.get(split)
        if split_path is None:
            if split != "test":
                report["errors"].append(f"缺少{split}集路径")
                report["valid"] = False
            continue

        img_dir = base_dir / split_path
        if not img_dir.exists():
            report["errors"].append(f"{split}集目录不存在: {img_dir}")
            report["valid"] = False
            continue

        # 查找图像和标签
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]

        # 标签目录 (images -> labels)
        label_dir = Path(str(img_dir).replace("/images", "/labels"))
        labels = []
        missing_labels = []
        class_counts = {}

        for img in images:
            label_path = label_dir / (img.stem + ".txt")
            if label_path.exists():
                labels.append(label_path)
                # 解析标签
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            if cls_id not in class_counts:
                                class_counts[cls_id] = 0
                            class_counts[cls_id] += 1
                            if cls_id >= nc:
                                report["warnings"].append(
                                    f"标签中类别ID {cls_id} >= nc={nc}: {label_path}"
                                )
            else:
                missing_labels.append(img.name)

        report["stats"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(missing_labels),
            "class_distribution": {
                names[k] if k < len(names) else f"class_{k}": v
                for k, v in sorted(class_counts.items())
            },
        }

        if missing_labels:
            report["warnings"].append(
                f"{split}集有 {len(missing_labels)} 张图片缺少标签"
            )

    logger.info(f"数据集校验完成: {'通过' if report['valid'] else '失败'}")
    for err in report["errors"]:
        logger.error(f"  ❌ {err}")
    for warn in report["warnings"]:
        logger.warning(f"  ⚠️ {warn}")
    for split, stats in report["stats"].items():
        logger.info(f"  {split}: {stats['images']}张图片, {stats['labels']}个标签")
        for cls_name, cnt in stats["class_distribution"].items():
            logger.info(f"    - {cls_name}: {cnt}")

    return report


# =============================================================================
# 训练
# =============================================================================

def train(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/animal_detection",
    name: str = "train",
    multi_scale: bool = True,
    resume: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    执行YOLOv8训练

    Args:
        data_yaml: 数据集配置文件路径
        model: 预训练模型路径 (yolov8n.pt / yolov8s.pt)
        epochs: 训练轮次
        imgsz: 输入图像尺寸
        batch: 批量大小
        device: GPU设备号 ("0", "0,1", "cpu")
        project: 训练输出目录
        name: 实验名称
        multi_scale: 是否启用多尺度训练
        resume: 是否恢复训练
        seed: 随机种子 (可复现)

    Returns:
        训练结果摘要
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics未安装，请运行: pip install ultralytics")
        return {"success": False, "error": "ultralytics not installed"}

    # 数据集校验
    logger.info("=" * 60)
    logger.info("Step 1: 数据集校验")
    logger.info("=" * 60)
    validation = validate_dataset(data_yaml)
    if not validation["valid"]:
        return {"success": False, "error": "dataset_validation_failed", "validation": validation}

    # 加载模型
    logger.info("=" * 60)
    logger.info(f"Step 2: 加载模型 {model}")
    logger.info("=" * 60)
    yolo_model = YOLO(model)

    # 训练
    logger.info("=" * 60)
    logger.info(f"Step 3: 开始训练 | epochs={epochs} imgsz={imgsz} batch={batch}")
    logger.info("=" * 60)

    train_start = time.time()

    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        multi_scale=multi_scale,
        resume=resume,
        seed=seed,
        # 增强策略 (对小目标友好)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        flipud=0.5,
        fliplr=0.5,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        # 保存策略
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    train_time = time.time() - train_start

    # 收集结果
    train_dir = Path(project) / name
    best_weight = train_dir / "weights" / "best.pt"
    last_weight = train_dir / "weights" / "last.pt"

    result_summary = {
        "success": True,
        "train_time_s": round(train_time, 1),
        "epochs": epochs,
        "model_base": model,
        "data_yaml": data_yaml,
        "imgsz": imgsz,
        "best_weight": str(best_weight) if best_weight.exists() else None,
        "last_weight": str(last_weight) if last_weight.exists() else None,
        "train_dir": str(train_dir),
        "seed": seed,
        "dataset_validation": validation["stats"],
        "timestamp": datetime.now().isoformat(),
    }

    # 提取指标
    if hasattr(results, "results_dict"):
        result_summary["metrics"] = results.results_dict

    # 保存训练摘要
    summary_path = train_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"训练完成! 耗时: {train_time/60:.1f}分钟")
    logger.info(f"最佳权重: {best_weight}")
    logger.info(f"训练摘要: {summary_path}")
    logger.info("=" * 60)

    return result_summary


# =============================================================================
# 模型导出
# =============================================================================

def export_model(
    weights: str,
    format: str = "onnx",
    imgsz: int = 640,
    output_dir: str = "models",
) -> Dict[str, Any]:
    """
    导出模型为ONNX或其他格式

    Args:
        weights: PyTorch权重路径 (.pt)
        format: 导出格式 (onnx / torchscript / tflite / engine)
        imgsz: 输入尺寸
        output_dir: 输出目录

    Returns:
        导出结果
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"success": False, "error": "ultralytics not installed"}

    weights_path = Path(weights)
    if not weights_path.exists():
        return {"success": False, "error": f"权重文件不存在: {weights}"}

    logger.info(f"导出模型: {weights} -> {format}")

    model = YOLO(weights)
    export_path = model.export(format=format, imgsz=imgsz)

    # 复制到输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_file = Path(export_path)
    dest_file = output_dir / f"animal_{export_file.name}"
    shutil.copy2(export_file, dest_file)

    # 计算哈希
    sha256 = hashlib.sha256()
    with open(dest_file, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    result = {
        "success": True,
        "source_weights": str(weights),
        "export_format": format,
        "export_path": str(dest_file),
        "file_size_mb": round(dest_file.stat().st_size / 1024 / 1024, 2),
        "sha256": sha256.hexdigest(),
        "imgsz": imgsz,
        "timestamp": datetime.now().isoformat(),
    }

    # 保存导出清单
    manifest = output_dir / "model_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"导出完成: {dest_file} ({result['file_size_mb']}MB)")
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="室内动物检测模型训练与导出")
    subparsers = parser.add_subparsers(dest="command", help="操作命令")

    # 训练
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data", required=True, help="数据集YAML路径")
    train_parser.add_argument("--model", default="yolov8n.pt", help="预训练模型")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--device", default="0")
    train_parser.add_argument("--project", default="runs/animal_detection")
    train_parser.add_argument("--name", default="train")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--resume", action="store_true")

    # 校验
    val_parser = subparsers.add_parser("validate", help="校验数据集")
    val_parser.add_argument("--data", required=True, help="数据集YAML路径")

    # 导出
    export_parser = subparsers.add_parser("export", help="导出模型")
    export_parser.add_argument("--weights", required=True, help="权重文件路径")
    export_parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "engine"])
    export_parser.add_argument("--imgsz", type=int, default=640)
    export_parser.add_argument("--output", default="models")

    args = parser.parse_args()

    if args.command == "train":
        result = train(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            seed=args.seed,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.command == "validate":
        result = validate_dataset(args.data)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "export":
        result = export_model(
            weights=args.weights,
            format=args.format,
            imgsz=args.imgsz,
            output_dir=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
