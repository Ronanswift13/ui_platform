#!/usr/bin/env python3
"""
bird_monitoring 独立训练/推理入口

示例:
  python plugins/bird_monitoring/main.py train --data /path/to/data.yaml --dry-run
  python plugins/bird_monitoring/main.py infer --input /path/to/test.jpg
  python plugins/bird_monitoring/main.py --config plugins/bird_monitoring/configs/standalone.yaml infer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from darkbreaker_sdk.interfaces import PluginContext, PluginManifest
from darkbreaker_sdk.schemas import BoundingBox, ROI, ROIType
from plugins.bird_monitoring.plugin import BirdMonitoringPlugin


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_file(config_path: Path | None) -> Dict[str, Any]:
    """读取 YAML/JSON 配置文件"""
    if config_path is None:
        return {}

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"配置文件根节点必须是对象: {config_path}")

    return data


def normalize_input_files(raw_value: Any) -> List[str]:
    """
    归一化推理输入:
    - str/Path -> [str]
    - list/tuple/set[str|Path] -> list[str]
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, (str, Path)):
        return [str(raw_value)]
    if isinstance(raw_value, (list, tuple, set)):
        normalized: List[str] = []
        for item in raw_value:
            if item is None:
                continue
            if isinstance(item, (str, Path)):
                normalized.append(str(item))
            else:
                raise TypeError(f"不支持的输入项类型: {type(item)}")
        return normalized
    raise TypeError(f"不支持的输入类型: {type(raw_value)}")


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    """将相对路径解析到 base_dir 下"""
    path_obj = Path(path_value).expanduser()
    if not path_obj.is_absolute():
        path_obj = base_dir / path_obj
    return path_obj


def parse_roi(roi_text: str | None) -> BoundingBox:
    """
    ROI 字符串格式: x,y,width,height (归一化 0~1)
    未提供则默认整图
    """
    if not roi_text:
        return BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)

    parts = [p.strip() for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI 格式错误，需为 x,y,width,height")

    x, y, w, h = [float(v) for v in parts]
    return BoundingBox(x=x, y=y, width=w, height=h)


def load_image(image_path: Path) -> np.ndarray:
    """加载图像，优先 cv2，回退到 Pillow"""
    if not image_path.exists():
        raise FileNotFoundError(f"推理素材不存在: {image_path}")

    if cv2 is not None:
        image = cv2.imread(str(image_path))
        if image is not None:
            return image

    if Image is None:
        raise RuntimeError(
            "无法读取图像: OpenCV 读取失败且 Pillow 未安装"
        )

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)

    # RGB -> BGR
    return arr[:, :, ::-1].copy()


def load_manifest(plugin_dir: Path) -> PluginManifest:
    """加载 bird_monitoring 的 manifest"""
    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 缺失: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    return PluginManifest.from_dict(manifest_data)


def create_plugin(
    plugin_dir: Path,
    plugin_config: Dict[str, Any] | None = None,
) -> BirdMonitoringPlugin:
    """构建并初始化插件"""
    manifest = load_manifest(plugin_dir)
    plugin = BirdMonitoringPlugin(manifest, plugin_dir)
    init_ok = plugin.init(plugin_config or {})
    if not init_ok:
        raise RuntimeError("bird_monitoring 插件初始化失败")
    return plugin


def _labels_dir_from_images_dir(images_dir: Path) -> Path:
    """根据 images 目录推断对应 labels 目录"""
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def validate_yolo_dataset(data_yaml: Path) -> Dict[str, Any]:
    """轻量校验 YOLO 数据集结构"""
    report: Dict[str, Any] = {
        "valid": True,
        "data_yaml": str(data_yaml),
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    if not data_yaml.exists():
        report["valid"] = False
        report["errors"].append(f"数据集配置不存在: {data_yaml}")
        return report

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        report["valid"] = False
        report["errors"].append("data.yaml 根节点必须是对象")
        return report

    for field in ("train", "val", "nc", "names"):
        if field not in cfg:
            report["valid"] = False
            report["errors"].append(f"缺少字段: {field}")

    if not report["valid"]:
        return report

    base_dir = data_yaml.parent
    if cfg.get("path"):
        base_dir = resolve_path(str(cfg["path"]), data_yaml.parent)

    names = cfg.get("names", [])
    nc = int(cfg.get("nc", 0))
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items(), key=lambda x: int(x[0]))]
    if len(names) != nc:
        report["valid"] = False
        report["errors"].append(f"类别数量不一致: nc={nc}, names={len(names)}")

    for split in ("train", "val", "test"):
        split_value = cfg.get(split)
        if split_value is None:
            if split in ("train", "val"):
                report["valid"] = False
                report["errors"].append(f"缺少 {split} 路径")
            continue

        images_dir = resolve_path(str(split_value), base_dir)
        if not images_dir.exists():
            report["valid"] = False
            report["errors"].append(f"{split} 图像目录不存在: {images_dir}")
            continue

        image_files = [
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        labels_dir = _labels_dir_from_images_dir(images_dir)
        missing_labels = 0
        label_files = 0

        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                label_files += 1
            else:
                missing_labels += 1

        report["stats"][split] = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "images": len(image_files),
            "labels": label_files,
            "missing_labels": missing_labels,
        }
        if missing_labels > 0:
            report["warnings"].append(
                f"{split} 有 {missing_labels} 张图像缺少标签"
            )

    return report


def train_once(
    *,
    data_yaml: Path,
    output_dir: Path,
    model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    run_name: str = "bird_train",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    训练入口:
    - dry_run=True: 只做数据集校验并落盘摘要
    - dry_run=False: 尝试调用 ultralytics，缺失时自动退化为 dry_run
    """
    data_yaml = data_yaml.resolve()
    output_dir = output_dir.resolve()
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_yolo_dataset(data_yaml)

    summary: Dict[str, Any] = {
        "success": validation["valid"],
        "timestamp": datetime.now().isoformat(),
        "backend": "dry_run",
        "executed": False,
        "data_yaml": str(data_yaml),
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "run_name": run_name,
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "dry_run": dry_run,
        "validation": validation,
        "error": None,
    }

    if not validation["valid"]:
        summary["error"] = "dataset_validation_failed"
        summary_path = run_dir / "training_summary.json"
        summary["summary_path"] = str(summary_path)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    if not dry_run:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            summary["backend"] = "dry_run_fallback"
            summary["error"] = "ultralytics_not_installed"
        else:
            summary["backend"] = "ultralytics"
            train_start = time.time()
            try:
                yolo_model = YOLO(model)
                results = yolo_model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    project=str(output_dir),
                    name=run_name,
                    save=True,
                    verbose=True,
                )
                summary["executed"] = True
                summary["train_time_s"] = round(time.time() - train_start, 2)

                best_weight = run_dir / "weights" / "best.pt"
                last_weight = run_dir / "weights" / "last.pt"
                summary["best_weight"] = str(best_weight) if best_weight.exists() else None
                summary["last_weight"] = str(last_weight) if last_weight.exists() else None
                if hasattr(results, "results_dict"):
                    summary["metrics"] = getattr(results, "results_dict")
            except Exception as exc:
                summary["success"] = False
                summary["error"] = f"train_failed: {exc}"

    summary_path = run_dir / "training_summary.json"
    summary["summary_path"] = str(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def infer_once(
    *,
    image_paths: List[Path],
    output_json: Path,
    plugin_config: Dict[str, Any] | None = None,
    roi_bbox: BoundingBox | None = None,
    context_overrides: Dict[str, Any] | None = None,
    plugin_dir: Path | None = None,
) -> Dict[str, Any]:
    """单次推理，可输入一张或多张图像"""
    if not image_paths:
        raise ValueError("至少需要一个输入文件")

    plugin_dir = plugin_dir or Path(__file__).resolve().parent
    output_json = output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    roi_bbox = roi_bbox or BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
    context_overrides = context_overrides or {}
    context_metadata = context_overrides.get("metadata", {})
    if not isinstance(context_metadata, dict):
        context_metadata = {}

    context = PluginContext(
        task_id=str(context_overrides.get("task_id", "bird-standalone-task")),
        site_id=str(context_overrides.get("site_id", "site-local")),
        device_id=str(context_overrides.get("device_id", "camera-local")),
        component_id=str(context_overrides.get("component_id", "power-line-1")),
        metadata=context_metadata,
    )

    plugin = create_plugin(plugin_dir, plugin_config=plugin_config or {})
    items: List[Dict[str, Any]] = []

    try:
        for idx, image_path in enumerate(image_paths):
            abs_image_path = image_path.resolve()
            frame = load_image(abs_image_path)
            roi = ROI(
                id=f"roi-{idx}",
                name=f"bird_roi_{idx}",
                component_id=context.component_id,
                roi_type=ROIType.INTRUSION,
                bbox=roi_bbox,
                recognition_types=["bird_detection", "risk_assessment"],
                metadata={"source_image": str(abs_image_path)},
            )

            results = plugin.infer(frame=frame, rois=[roi], context=context)
            alarms = plugin.postprocess(results=results, rules=[])

            items.append({
                "input": str(abs_image_path),
                "result_count": len(results),
                "alarm_count": len(alarms),
                "results": [r.model_dump(mode="json") for r in results],
                "alarms": [a.model_dump(mode="json") for a in alarms],
            })
    finally:
        plugin.cleanup()

    summary: Dict[str, Any] = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "plugin_id": "bird_monitoring",
        "total_images": len(image_paths),
        "items": items,
        "output_json": str(output_json),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def _pick_first(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    """按顺序从字典中取第一个存在且非空值"""
    for key in keys:
        if key in mapping and mapping[key] not in (None, "", []):
            return mapping[key]
    return None


def _collect_infer_inputs(
    args: argparse.Namespace,
    config: Dict[str, Any],
    config_base_dir: Path,
) -> Tuple[List[Path], str]:
    """收集推理输入文件，支持单数和复数输入"""
    cli_inputs: List[str] = []
    if args.input:
        cli_inputs.append(args.input)
    if args.inputs:
        cli_inputs.extend(args.inputs)

    if cli_inputs:
        return [resolve_path(v, Path.cwd()) for v in normalize_input_files(cli_inputs)], "cli"

    infer_cfg = config.get("infer", {})
    if not isinstance(infer_cfg, dict):
        infer_cfg = {}

    raw_inputs = _pick_first(
        infer_cfg,
        ("inputs", "input", "images", "image", "materials", "material", "filename", "file"),
    )
    if raw_inputs is None:
        raw_inputs = _pick_first(
            config,
            ("inputs", "input", "images", "image", "materials", "material", "filename", "file"),
        )

    normalized = normalize_input_files(raw_inputs)
    return [resolve_path(v, config_base_dir) for v in normalized], "config"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="bird_monitoring 独立训练/推理工具")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML/JSON 配置文件路径",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="执行训练")
    train_parser.add_argument("--data", type=str, help="YOLO data.yaml 路径")
    train_parser.add_argument("--model", type=str, default=None, help="预训练模型或权重")
    train_parser.add_argument("--epochs", type=int, default=None, help="训练轮次")
    train_parser.add_argument("--imgsz", type=int, default=None, help="图像尺寸")
    train_parser.add_argument("--batch", type=int, default=None, help="batch size")
    train_parser.add_argument("--device", type=str, default=None, help="训练设备，如 cpu/0")
    train_parser.add_argument("--output", type=str, default=None, help="训练输出目录")
    train_parser.add_argument("--name", type=str, default=None, help="训练任务名")
    train_parser.add_argument("--dry-run", action="store_true", help="仅校验数据并写摘要")

    infer_parser = subparsers.add_parser("infer", help="执行推理")
    infer_parser.add_argument("--input", type=str, default=None, help="单个输入图像路径")
    infer_parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="多个输入图像路径",
    )
    infer_parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI: x,y,width,height (归一化)",
    )
    infer_parser.add_argument("--output", type=str, default=None, help="推理结果JSON输出路径")
    infer_parser.add_argument("--model-path", type=str, default=None, help="覆盖模型路径")
    infer_parser.add_argument("--task-id", type=str, default=None, help="任务ID")
    infer_parser.add_argument("--site-id", type=str, default=None, help="站点ID")
    infer_parser.add_argument("--device-id", type=str, default=None, help="设备ID")
    infer_parser.add_argument("--component-id", type=str, default=None, help="部件ID")
    infer_parser.add_argument(
        "--phase-spacing-m",
        type=float,
        default=None,
        help="相间距(米)，用于风险评估",
    )

    return parser


def run_cli(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).expanduser() if args.config else None
    config = load_config_file(config_path)
    config_base_dir = config_path.parent if config_path else Path.cwd()

    if args.command == "train":
        train_cfg = config.get("train", {})
        if not isinstance(train_cfg, dict):
            train_cfg = {}

        data_raw = args.data if args.data else train_cfg.get("data")
        if not data_raw:
            raise ValueError("训练模式需要 --data 或配置 train.data")
        data_base = Path.cwd() if args.data else config_base_dir
        data_yaml = resolve_path(data_raw, data_base)

        output_raw = args.output if args.output else train_cfg.get("output", "plugins/bird_monitoring/runs")
        output_base = Path.cwd() if args.output else config_base_dir
        output_dir = resolve_path(output_raw, output_base)

        model = args.model if args.model else train_cfg.get("model", "yolov8n.pt")
        epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 50))
        imgsz = int(args.imgsz if args.imgsz is not None else train_cfg.get("imgsz", 640))
        batch = int(args.batch if args.batch is not None else train_cfg.get("batch", 16))
        device = str(args.device if args.device is not None else train_cfg.get("device", "cpu"))
        run_name = str(args.name if args.name else train_cfg.get("name", "bird_train"))
        dry_run = bool(args.dry_run or train_cfg.get("dry_run", False))

        return train_once(
            data_yaml=data_yaml,
            output_dir=output_dir,
            model=model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            run_name=run_name,
            dry_run=dry_run,
        )

    if args.command == "infer":
        infer_cfg = config.get("infer", {})
        if not isinstance(infer_cfg, dict):
            infer_cfg = {}

        image_paths, input_source = _collect_infer_inputs(args, config, config_base_dir)
        if not image_paths:
            raise ValueError(
                "推理模式需要 --input/--inputs 或配置 infer.input(s)"
            )

        output_raw = args.output if args.output else infer_cfg.get(
            "output",
            "plugins/bird_monitoring/output/inference_summary.json",
        )
        output_base = Path.cwd() if args.output else config_base_dir
        output_json = resolve_path(output_raw, output_base)

        roi_text = args.roi if args.roi else infer_cfg.get("roi")
        roi_bbox = parse_roi(roi_text)

        plugin_cfg = config.get("plugin", {})
        if not isinstance(plugin_cfg, dict):
            plugin_cfg = {}
        plugin_cfg = dict(plugin_cfg)

        if args.model_path:
            model_path = str(resolve_path(args.model_path, Path.cwd()))
            plugin_cfg = deep_merge(plugin_cfg, {"model": {"path": model_path}})

        context_cfg = config.get("context", {})
        if not isinstance(context_cfg, dict):
            context_cfg = {}
        context_overrides = dict(context_cfg)

        if args.task_id:
            context_overrides["task_id"] = args.task_id
        if args.site_id:
            context_overrides["site_id"] = args.site_id
        if args.device_id:
            context_overrides["device_id"] = args.device_id
        if args.component_id:
            context_overrides["component_id"] = args.component_id

        if args.phase_spacing_m is not None:
            metadata = context_overrides.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["phase_spacing_m"] = args.phase_spacing_m
            context_overrides["metadata"] = metadata

        summary = infer_once(
            image_paths=image_paths,
            output_json=output_json,
            plugin_config=plugin_cfg,
            roi_bbox=roi_bbox,
            context_overrides=context_overrides,
            plugin_dir=Path(__file__).resolve().parent,
        )
        summary["input_source"] = input_source
        return summary

    raise ValueError(f"未知命令: {args.command}")


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_cli(args)
    except Exception as exc:
        print(f"[bird_monitoring.main] 失败: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
