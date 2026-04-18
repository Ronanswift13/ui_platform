"""
标签格式 Schema 定义

每种 task_type 对应不同的标签格式要求，
本模块定义各类型的标签校验规则。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.multimodal_fusion import validate_multimodal_sample_record


class LabelSchema(ABC):
    """标签校验基类"""

    @abstractmethod
    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        """校验单个标签文件，返回错误列表"""
        ...

    @abstractmethod
    def get_format_description(self) -> str:
        """返回格式说明"""
        ...


class DetectionBBoxSchema(LabelSchema):
    """
    YOLO 格式目标检测标签:
    每行: <class_id> <x_center> <y_center> <width> <height>
    所有值归一化到 [0, 1]
    """

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        errors: list[str] = []
        if not path.exists():
            return [f"标签文件不存在: {path}"]

        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"行 {i}: 期望 5 个字段，实际 {len(parts)}")
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(f"行 {i}: 数值解析失败")
                    continue
                if cls_id < 0 or cls_id >= num_classes:
                    errors.append(f"行 {i}: class_id={cls_id} 超出范围 [0, {num_classes - 1}]")
                for j, v in enumerate(coords):
                    if v < 0.0 or v > 1.0:
                        errors.append(f"行 {i}: 坐标[{j}]={v} 不在 [0, 1] 范围")
        return errors

    def get_format_description(self) -> str:
        return "YOLO bbox: <class_id> <x_center> <y_center> <width> <height> (归一化)"


class ClassificationSchema(LabelSchema):
    """
    图像分类标签:
    按文件夹组织 (每个文件夹名 = 类别名)，或 CSV 文件 (image_path, label)
    """

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        errors: list[str] = []
        if path.is_dir():
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if len(subdirs) != num_classes:
                errors.append(f"文件夹数 {len(subdirs)} ≠ 期望类别数 {num_classes}")
        elif path.suffix == ".csv":
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        errors.append(f"行 {i}: CSV 格式错误，至少需要 image_path, label")
        return errors

    def get_format_description(self) -> str:
        return "分类标签: 文件夹组织 (folder_name = class) 或 CSV (image_path, label)"


class OCRReadingSchema(LabelSchema):
    """
    OCR / 读数识别标签:
    JSON 格式: {"image": "xxx.jpg", "reading": "12.34", "meter_type": "sf6_pressure",
                "keypoints": [[x1,y1], ...], "display_bbox": [x,y,w,h]}
    """

    REQUIRED_FIELDS = {"image", "reading"}
    OPTIONAL_FIELDS = {"meter_type", "keypoints", "display_bbox", "unit", "scale_range"}

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        import json as json_mod

        errors: list[str] = []
        if not path.exists():
            return [f"标签文件不存在: {path}"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json_mod.load(f)
        except json_mod.JSONDecodeError as e:
            return [f"JSON 解析失败: {e}"]

        entries = data if isinstance(data, list) else [data]
        for i, entry in enumerate(entries):
            for field in self.REQUIRED_FIELDS:
                if field not in entry:
                    errors.append(f"条目 {i}: 缺少必填字段 '{field}'")
            reading = entry.get("reading", "")
            if reading and not re.match(r"^-?\d+\.?\d*$", str(reading)):
                errors.append(f"条目 {i}: reading '{reading}' 不是合法数值")
        return errors

    def get_format_description(self) -> str:
        return 'OCR 标签: JSON {"image", "reading", "meter_type"?, "keypoints"?, "display_bbox"?}'


class ThermalAnomalySchema(LabelSchema):
    """
    热像异常检测标签:
    JSON 格式: {"image": "xxx.jpg", "max_temp": 85.2, "anomaly_level": "warning",
                "hotspots": [{"bbox": [x,y,w,h], "temp": 92.1}], "equipment_type": "transformer"}
    """

    VALID_LEVELS = {"normal", "attention", "warning", "alarm", "critical"}

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        import json as json_mod

        errors: list[str] = []
        if not path.exists():
            return [f"标签文件不存在: {path}"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json_mod.load(f)
        except json_mod.JSONDecodeError as e:
            return [f"JSON 解析失败: {e}"]

        entries = data if isinstance(data, list) else [data]
        for i, entry in enumerate(entries):
            if "image" not in entry:
                errors.append(f"条目 {i}: 缺少 'image' 字段")
            level = entry.get("anomaly_level", "")
            if level and level not in self.VALID_LEVELS:
                errors.append(f"条目 {i}: anomaly_level '{level}' 不合法，合法值: {self.VALID_LEVELS}")
            if "max_temp" in entry:
                try:
                    float(entry["max_temp"])
                except (TypeError, ValueError):
                    errors.append(f"条目 {i}: max_temp 不是合法数值")
        return errors

    def get_format_description(self) -> str:
        return '热像标签: JSON {"image", "max_temp", "anomaly_level", "hotspots"?, "equipment_type"?}'


class SegmentationMaskSchema(LabelSchema):
    """
    语义/实例分割标签，支持四种格式:

    1. png_index: 单通道 PNG，像素值 = class_id (0=背景, 255=忽略)
    2. png_color: 三通道 PNG，通过 color_map 映射到 class_id
    3. rle: COCO 风格 RLE 编码 JSON
    4. polygon: COCO 风格多边形 JSON
    """

    def __init__(self, mask_format: str = "png_index", ignore_index: int = 255):
        self.mask_format = mask_format
        self.ignore_index = ignore_index

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        errors: list[str] = []
        if not path.exists():
            return [f"标签文件不存在: {path}"]

        if self.mask_format in ("png_index", "png_color"):
            if path.suffix.lower() != ".png":
                errors.append(f"mask 文件应为 PNG 格式，实际: {path.suffix}")
            else:
                try:
                    from PIL import Image
                    img = Image.open(path)
                    if self.mask_format == "png_index":
                        if img.mode not in ("L", "P"):
                            errors.append(
                                f"png_index mask 应为单通道 (L/P)，实际: {img.mode}"
                            )
                    elif self.mask_format == "png_color":
                        if img.mode != "RGB":
                            errors.append(
                                f"png_color mask 应为 RGB，实际: {img.mode}"
                            )
                except ImportError:
                    pass  # Pillow 不可用时跳过像素级校验
                except Exception as e:
                    errors.append(f"mask 文件读取失败: {e}")

        elif self.mask_format in ("rle", "polygon"):
            import json as json_mod
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json_mod.load(f)
            except json_mod.JSONDecodeError as e:
                return [f"JSON 解析失败: {e}"]

            anns = data.get("annotations", data if isinstance(data, list) else [data])
            for i, ann in enumerate(anns):
                if self.mask_format == "rle":
                    seg = ann.get("segmentation", {})
                    if not isinstance(seg, dict) or "counts" not in seg:
                        errors.append(f"条目 {i}: RLE 需要 segmentation.counts 字段")
                else:
                    seg = ann.get("segmentation", [])
                    if not isinstance(seg, list) or not seg:
                        errors.append(f"条目 {i}: polygon 需要 segmentation 为坐标列表")
                cat = ann.get("category_id")
                if cat is not None and (cat < 0 or cat >= num_classes):
                    errors.append(
                        f"条目 {i}: category_id={cat} 超出 [0, {num_classes - 1}]"
                    )
        return errors

    def get_format_description(self) -> str:
        descs = {
            "png_index": "分割 mask: 单通道 PNG (像素值=class_id, 255=忽略)",
            "png_color": "分割 mask: RGB PNG (通过 color_map 映射 class_id)",
            "rle": "分割标注: COCO RLE JSON (segmentation.counts)",
            "polygon": "分割标注: COCO polygon JSON (segmentation=[坐标列表])",
        }
        return descs.get(self.mask_format, f"分割标注: {self.mask_format}")


class HyperspectralSchema(LabelSchema):
    """
    高光谱标签:
    JSON 格式: {"cube_file": "xxx.hdr", "ground_truth": "xxx_gt.hdr",
                "classes": [...], "pca_applied": false, "num_bands": 224}
    或逐像素标注的 ground truth 图 (与 cube 同尺寸，像素值 = class_id)
    """

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        import json as json_mod

        errors: list[str] = []
        if not path.exists():
            return [f"标签文件不存在: {path}"]

        if path.suffix == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json_mod.load(f)
            except json_mod.JSONDecodeError as e:
                return [f"JSON 解析失败: {e}"]

            if "cube_file" not in data:
                errors.append("缺少 'cube_file' 字段")
            if "ground_truth" not in data and "classes" not in data:
                errors.append("需要 'ground_truth' 或 'classes' 字段之一")
        elif path.suffix in (".hdr", ".dat", ".raw"):
            pass  # ENVI 格式，需要专用库校验
        else:
            errors.append(f"不支持的高光谱标签格式: {path.suffix}")
        return errors

    def get_format_description(self) -> str:
        return '高光谱标签: JSON {"cube_file", "ground_truth"} 或 ENVI .hdr ground truth'


class TemporalSeriesSchema(LabelSchema):
    """
    通用时序标签 Schema:
    - JSON / JSONL: 样本级标签或回归目标
    - CSV: 需包含 sample_id / timestamp / target 之一
    - Parquet / NPZ: 仅做文件级存在性与扩展名检查
    """

    REQUIRED_FIELDS = {"sample_id", "sequence_id", "timestamp", "label", "target"}

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        import json as json_mod

        errors: list[str] = []
        if not path.exists():
            return [f"标签文件不存在: {path}"]

        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json_mod.load(f)
            except json_mod.JSONDecodeError as e:
                return [f"JSON 解析失败: {e}"]

            entries = data if isinstance(data, list) else [data]
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    errors.append(f"条目 {i}: 必须是对象")
                    continue
                if not (set(entry.keys()) & self.REQUIRED_FIELDS):
                    errors.append(
                        f"条目 {i}: 缺少关键字段，至少需要 {sorted(self.REQUIRED_FIELDS)} 之一"
                    )
        elif suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json_mod.loads(line)
                    except json_mod.JSONDecodeError as e:
                        errors.append(f"行 {i}: JSONL 解析失败: {e}")
                        continue
                    if not isinstance(entry, dict):
                        errors.append(f"行 {i}: 必须是对象")
                        continue
                    if not (set(entry.keys()) & self.REQUIRED_FIELDS):
                        errors.append(
                            f"行 {i}: 缺少关键字段，至少需要 {sorted(self.REQUIRED_FIELDS)} 之一"
                        )
        elif suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            header_set = {item.strip() for item in header if item.strip()}
            if not (header_set & self.REQUIRED_FIELDS):
                errors.append(
                    "CSV 标签缺少关键列，至少需要 "
                    f"{sorted(self.REQUIRED_FIELDS)} 之一"
                )
        elif suffix not in {".parquet", ".npz"}:
            errors.append(f"不支持的时序标签格式: {path.suffix}")
        return errors

    def get_format_description(self) -> str:
        return "时序标签: JSON / JSONL / CSV / Parquet / NPZ"


class ActionEventSequenceSchema(LabelSchema):
    """
    动作事件序列标签:
    每条记录至少包含 timestamp / event_type / sequence_id 之一。
    """

    REQUIRED_FIELDS = {"timestamp", "event_type", "sequence_id", "label"}

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        import json as json_mod

        errors: list[str] = []
        if not path.exists():
            return [f"标签文件不存在: {path}"]

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json_mod.loads(line)
                    except json_mod.JSONDecodeError as e:
                        errors.append(f"行 {i}: JSONL 解析失败: {e}")
                        continue
                    if not isinstance(payload, dict):
                        errors.append(f"行 {i}: 必须是对象")
                        continue
                    if not (set(payload.keys()) & self.REQUIRED_FIELDS):
                        errors.append(
                            f"行 {i}: 缺少关键字段，至少需要 {sorted(self.REQUIRED_FIELDS)} 之一"
                        )
        elif suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                header = {item.strip() for item in f.readline().strip().split(",") if item.strip()}
            if not {"timestamp", "event_type"} <= header:
                errors.append("CSV 事件标签至少需要 timestamp,event_type 两列")
        elif suffix == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json_mod.load(f)
            except json_mod.JSONDecodeError as e:
                return [f"JSON 解析失败: {e}"]
            entries = data if isinstance(data, list) else [data]
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    errors.append(f"条目 {i}: 必须是对象")
                    continue
                if not (set(entry.keys()) & self.REQUIRED_FIELDS):
                    errors.append(
                        f"条目 {i}: 缺少关键字段，至少需要 {sorted(self.REQUIRED_FIELDS)} 之一"
                    )
        else:
            errors.append(f"不支持的动作事件标签格式: {path.suffix}")
        return errors

    def get_format_description(self) -> str:
        return "动作事件序列标签: JSON / JSONL / CSV"


class MultimodalSampleSchema(LabelSchema):
    """
    多模态样本索引:
    JSONL 每行一个样本，字段至少包含:
    sample_id / device_id / timestamp / available_modalities / modality_paths / label / diagnosis_target
    """

    def validate_label_file(self, path: Path, num_classes: int) -> list[str]:
        import json as json_mod

        errors: list[str] = []
        if not path.exists():
            return [f"标签文件不存在: {path}"]

        if path.suffix.lower() != ".jsonl":
            return [f"多模态样本索引必须是 .jsonl，实际: {path.suffix}"]

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json_mod.loads(line)
                except json_mod.JSONDecodeError as e:
                    errors.append(f"行 {i}: JSONL 解析失败: {e}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"行 {i}: 必须是对象")
                    continue
                for error in validate_multimodal_sample_record(record):
                    errors.append(f"行 {i}: {error}")
        return errors

    def get_format_description(self) -> str:
        return "多模态样本索引: JSONL (sample_id, device_id, timestamp, available_modalities, modality_paths, label, diagnosis_target)"


# ---- 工厂函数 ----

_SCHEMA_MAP: dict[str, type[LabelSchema]] = {
    "detection": DetectionBBoxSchema,
    "detection_bbox": DetectionBBoxSchema,
    "segmentation": SegmentationMaskSchema,
    "classification": ClassificationSchema,
    "single_label_cls": ClassificationSchema,
    "ocr": OCRReadingSchema,
    "ocr_reading": OCRReadingSchema,
    "thermal_anomaly": ThermalAnomalySchema,
    "binary_cls": ClassificationSchema,
    "hyperspectral_classification": HyperspectralSchema,
    "hyperspectral_anomaly": HyperspectralSchema,
    "hyperspectral_defect": HyperspectralSchema,
    "acoustic_time_frequency_anomaly": TemporalSeriesSchema,
    "multivariate_sensor_anomaly": TemporalSeriesSchema,
    "equipment_health_prediction": TemporalSeriesSchema,
    "health_index_calibration": TemporalSeriesSchema,
    "action_event_sequence_recognition": ActionEventSequenceSchema,
    "multimodal_feature_fusion": MultimodalSampleSchema,
    "multimodal_decision_fusion": MultimodalSampleSchema,
}


def get_schema_for_task_type(task_type: str, **kwargs: Any) -> LabelSchema:
    """
    根据 task_type 获取对应的标签 Schema 实例。

    对 segmentation 类型，可传入 mask_format / ignore_index 等参数。
    """
    schema_cls = _SCHEMA_MAP.get(task_type)
    if schema_cls is None:
        raise ValueError(
            f"未知 task_type: '{task_type}'，合法值: {sorted(_SCHEMA_MAP.keys())}"
        )
    if schema_cls is SegmentationMaskSchema:
        return schema_cls(
            mask_format=kwargs.get("mask_format", "png_index"),
            ignore_index=kwargs.get("ignore_index", 255),
        )
    return schema_cls()
