"""
数据集 manifest.json 定义与校验

每个上传到训练系统的数据包必须包含 manifest.json，
本模块定义其结构并提供校验函数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.temporal_anomaly import (
    TEMPORAL_MODALITIES,
    TEMPORAL_TASK_TYPES,
    is_temporal_task,
    validate_temporal_contract,
)
from training.multimodal_fusion import (
    MULTIMODAL_TASK_TYPES,
    is_multimodal_task,
    validate_multimodal_contract,
)


@dataclass
class DatasetManifest:
    """数据集 manifest 结构"""

    # ── 必填字段 ──
    name: str  # 数据集名称 (全局唯一标识)
    plugin_id: str  # 目标插件 ID (如 busbar_inspection，禁止旧简写)
    task_type: str  # 训练任务类型
    version: str  # 数据集版本 (semver)

    # ── 数据描述 ──
    num_images: int = 0
    num_labels: int = 0
    num_samples: int = 0
    classes: list[str] = field(default_factory=list)
    # class_aliases: 允许多个标注源使用不同名称映射到同一 class
    # 例: {"insulator_broken": "insulator_damaged", "cracked": "insulator_damaged"}
    class_aliases: dict[str, str] = field(default_factory=dict)

    # split: 两种模式
    #   1. 比例模式 (pre_split=false): {"train": 0.8, "val": 0.15, "test": 0.05}
    #   2. 预分割模式 (pre_split=true): 数据包自带 train/ val/ test/ 子目录
    split: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.15, "test": 0.05}
    )
    pre_split: bool = False  # True = 上传者已自行分割

    # ── 格式与模态 ──
    format: str = "yolo"  # yolo / coco / voc / image_folder / coco_seg / custom
    image_format: str = "jpg"  # jpg / png / bmp / tif / raw
    label_format: str = "txt"  # txt (yolo) / json (coco) / xml (voc) / png (seg mask)
    input_modality: str = "rgb"  # rgb / thermal / hyperspectral / rgb+thermal
    voltage_level: str = ""  # 可选，适用电压等级

    # ── 元信息 ──
    description: str = ""
    author: str = ""
    created_at: str = ""
    source: str = ""  # 数据来源 (如 "现场采集" / "公开数据集")
    license: str = ""  # 数据授权

    # ── segmentation 专用 ──
    mask_format: str = ""  # png_index / png_color / rle / polygon
    # mask_color_map: class_name → [R, G, B]，仅 png_color 时必填
    mask_color_map: dict[str, list[int]] = field(default_factory=dict)
    ignore_index: int = 255  # 分割 mask 中的忽略标签

    # ── OCR / reading 专用 ──
    charset: str = ""  # 合法字符集，如 "0123456789.-"
    max_reading_length: int = 10

    # ── thermal 专用 ──
    temperature_unit: str = "celsius"  # celsius / fahrenheit / kelvin
    thermal_bit_depth: int = 16  # 原始热像位深
    temperature_range: list[float] = field(default_factory=list)  # [min_℃, max_℃]

    # ── hyperspectral 专用 ──
    spectral_bands: int | None = None
    wavelength_range: list[float] | None = None  # [start_nm, end_nm]
    spectral_resolution: float | None = None  # nm per band

    # ── temporal anomaly 专用 ──
    sample_rate_hz: int | None = None
    window_size: int | float | None = None
    stride: int | float | None = None
    sequence_length: int | None = None
    prediction_horizon: int | float | None = None
    timestamp_field: str = ""
    timestamp_unit: str = "unix_ms"
    entity_id_field: str = ""
    sequence_id_field: str = ""
    feature_views: list[str] = field(default_factory=list)
    sensor_columns: list[str] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    temporal_schema: dict[str, Any] = field(default_factory=dict)
    target_schema: dict[str, Any] = field(default_factory=dict)

    # ── multimodal fusion 专用 ──
    supported_modalities: list[str] = field(default_factory=list)
    required_modalities: list[str] = field(default_factory=list)
    missing_modality_policy: str = "mask_and_gate"
    fusion_strategy: str = "hybrid"
    alignment_tolerance_ms: int = 5000
    multimodal_schema: dict[str, Any] = field(default_factory=dict)
    diagnosis_target_schema: dict[str, Any] = field(default_factory=dict)
    diagnostic_rule_pack: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> DatasetManifest:
        """从 manifest.json 文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """转为字典"""
        from dataclasses import asdict

        return asdict(self)

    def save(self, path: str | Path) -> None:
        """保存到 JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def all_class_names(self) -> list[str]:
        """返回所有合法类名 (含 aliases 的目标名)"""
        names = list(self.classes)
        for alias_target in self.class_aliases.values():
            if alias_target not in names:
                names.append(alias_target)
        return names


# ---- 合法值定义 ----

VISUAL_TASK_TYPES = {
    "detection",
    "segmentation",
    "classification",
    "ocr",
    "thermal_anomaly",
    "hyperspectral_classification",
    "hyperspectral_anomaly",
}
VALID_TASK_TYPES = VISUAL_TASK_TYPES | set(TEMPORAL_TASK_TYPES) | set(MULTIMODAL_TASK_TYPES)

VALID_FORMATS = {
    "yolo",
    "coco",
    "coco_seg",
    "voc",
    "image_folder",
    "custom",
    "wav_dir",
    "flac_dir",
    "npy_waveform",
    "mel_npz",
    "dual_audio_views",
    "csv_series",
    "parquet_series",
    "jsonl_series",
    "npz_series",
    "jsonl_events",
    "csv_events",
    "parquet_events",
    "multimodal_aligned",
    "multimodal_sparse",
    "jsonl_multimodal",
}
VALID_MODALITIES = {"rgb", "thermal", "hyperspectral", "rgb+thermal"} | set(
    TEMPORAL_MODALITIES
) | {"multimodal"}
VALID_MASK_FORMATS = {"png_index", "png_color", "rle", "polygon", ""}

VALID_PLUGIN_IDS = {
    "busbar_inspection",
    "capacitor_inspection",
    "switch_inspection",
    "transformer_inspection",
    "animal_detection",
    "bird_monitoring",
    "fire_detection",
    "meter_reading",
    "temperature_monitoring",
    "thermal",
    "hyperspectral_detection",
    "acoustic_monitoring",
    "gas_detection",
    "device_monitoring",
    "action_event_monitoring",
    "multimodal_fusion",
}

# plugin_id → 该插件允许的 task_type 集合
PLUGIN_TASK_MATRIX: dict[str, set[str]] = {
    "busbar_inspection": {"detection", "segmentation"},
    "capacitor_inspection": {"detection", "classification", "segmentation"},
    "switch_inspection": {"detection", "classification"},
    "transformer_inspection": {"detection", "classification", "segmentation"},
    "animal_detection": {"detection", "classification"},
    "bird_monitoring": {"detection", "classification"},
    "fire_detection": {"detection", "classification", "segmentation"},
    "meter_reading": {"ocr", "detection"},
    "temperature_monitoring": {"thermal_anomaly", "classification"},
    "thermal": {"thermal_anomaly"},
    "hyperspectral_detection": {"hyperspectral_classification", "hyperspectral_anomaly"},
    "acoustic_monitoring": {"acoustic_time_frequency_anomaly"},
    "gas_detection": {"multivariate_sensor_anomaly", "equipment_health_prediction"},
    "device_monitoring": {
        "multivariate_sensor_anomaly",
        "equipment_health_prediction",
        "health_index_calibration",
    },
    "action_event_monitoring": {"action_event_sequence_recognition"},
    "multimodal_fusion": {
        "multimodal_feature_fusion",
        "multimodal_decision_fusion",
    },
}

# 旧简写 → 新 plugin_id
LEGACY_ALIAS_MAP = {
    "busbar": "busbar_inspection",
    "capacitor": "capacitor_inspection",
    "switch": "switch_inspection",
    "transformer": "transformer_inspection",
    "meter": "meter_reading",
}


def validate_manifest(manifest: DatasetManifest) -> list[str]:
    """
    校验 manifest 合法性，返回错误列表 (空列表 = 通过)。

    校验项:
    1. 必填字段非空
    2. plugin_id 使用规范名称 (拦截旧简写)
    3. task_type 合法
    4. plugin_id × task_type 组合合法
    5. format 合法
    6. input_modality 合法
    7. split 比例 / pre_split 一致性
    8. segmentation 专用校验
    9. OCR 专用校验
    10. thermal 专用校验
    11. 高光谱专用校验
    12. temporal anomaly 专用校验
    13. multimodal fusion 专用校验
    14. classes / target_schema 合法
    15. class_aliases 目标必须在 classes 中
    """
    errors: list[str] = []

    # 1. 必填字段
    for field_name in ("name", "plugin_id", "task_type", "version"):
        if not getattr(manifest, field_name, ""):
            errors.append(f"必填字段 '{field_name}' 为空")

    # 2. plugin_id — 拦截旧简写
    if manifest.plugin_id in LEGACY_ALIAS_MAP:
        correct = LEGACY_ALIAS_MAP[manifest.plugin_id]
        errors.append(
            f"plugin_id '{manifest.plugin_id}' 是旧简写，请改用 '{correct}'"
        )
    elif manifest.plugin_id and manifest.plugin_id not in VALID_PLUGIN_IDS:
        errors.append(
            f"未知 plugin_id: '{manifest.plugin_id}'，"
            f"合法值: {sorted(VALID_PLUGIN_IDS)}"
        )

    # 3. task_type
    if manifest.task_type and manifest.task_type not in VALID_TASK_TYPES:
        errors.append(
            f"未知 task_type: '{manifest.task_type}'，"
            f"合法值: {sorted(VALID_TASK_TYPES)}"
        )

    # 4. plugin_id × task_type 组合
    if (
        manifest.plugin_id in VALID_PLUGIN_IDS
        and manifest.task_type in VALID_TASK_TYPES
    ):
        allowed = PLUGIN_TASK_MATRIX.get(manifest.plugin_id, set())
        if manifest.task_type not in allowed:
            errors.append(
                f"插件 '{manifest.plugin_id}' 不支持 task_type "
                f"'{manifest.task_type}'，允许: {sorted(allowed)}"
            )

    # 5. format
    if manifest.format not in VALID_FORMATS:
        errors.append(
            f"未知 format: '{manifest.format}'，合法值: {sorted(VALID_FORMATS)}"
        )

    # 6. modality
    if manifest.input_modality not in VALID_MODALITIES:
        errors.append(f"未知 input_modality: '{manifest.input_modality}'")

    # 7. split 一致性
    if not manifest.pre_split:
        total = sum(manifest.split.values())
        if abs(total - 1.0) > 0.01:
            errors.append(f"split 比例之和 {total:.2f} ≠ 1.0")
        for key in manifest.split:
            if key not in ("train", "val", "test"):
                errors.append(f"split 含非法键: '{key}'，仅允许 train/val/test")

    # 8. segmentation 专用
    if manifest.task_type == "segmentation":
        if not manifest.mask_format:
            errors.append("segmentation 任务必须指定 mask_format")
        elif manifest.mask_format not in VALID_MASK_FORMATS:
            errors.append(
                f"未知 mask_format: '{manifest.mask_format}'，"
                f"合法值: {sorted(VALID_MASK_FORMATS - {''})}"
            )
        if manifest.mask_format == "png_color" and not manifest.mask_color_map:
            errors.append("mask_format=png_color 时必须提供 mask_color_map")

    # 9. OCR 专用
    if manifest.task_type == "ocr":
        if not manifest.charset:
            errors.append("OCR 任务必须指定 charset")
        if manifest.input_modality != "rgb":
            errors.append("OCR 任务 input_modality 应为 'rgb'")

    # 10. thermal 专用
    if manifest.task_type == "thermal_anomaly":
        if manifest.input_modality not in ("thermal", "rgb+thermal"):
            errors.append(
                "thermal_anomaly 任务 input_modality 应为 'thermal' 或 'rgb+thermal'"
            )

    # 11. 高光谱专用
    if manifest.task_type in ("hyperspectral_classification", "hyperspectral_anomaly"):
        if not manifest.spectral_bands:
            errors.append("高光谱任务必须指定 spectral_bands")
        if manifest.input_modality != "hyperspectral":
            errors.append("高光谱任务 input_modality 必须为 'hyperspectral'")

    # 12. temporal anomaly 专用
    if is_temporal_task(manifest.task_type):
        errors.extend(
            validate_temporal_contract(
                plugin_id=manifest.plugin_id,
                task_type=manifest.task_type,
                input_modality=manifest.input_modality,
                temporal_schema=manifest.temporal_schema,
                target_schema=manifest.target_schema,
                feature_views=manifest.feature_views,
                sensor_columns=manifest.sensor_columns,
                event_types=manifest.event_types,
                timestamp_field=manifest.timestamp_field,
            )
        )
        if manifest.num_samples < 0:
            errors.append("num_samples 不能为负数")
        if manifest.task_type == "acoustic_time_frequency_anomaly" and not manifest.sample_rate_hz:
            errors.append("声学任务必须指定 sample_rate_hz")
        if manifest.task_type in {
            "multivariate_sensor_anomaly",
            "equipment_health_prediction",
            "health_index_calibration",
        } and not manifest.sequence_length:
            errors.append("多变量时序任务必须指定 sequence_length")
        if manifest.task_type == "equipment_health_prediction" and not manifest.prediction_horizon:
            errors.append("equipment_health_prediction 必须指定 prediction_horizon")

    # 13. multimodal fusion 专用
    if is_multimodal_task(manifest.task_type):
        errors.extend(
            validate_multimodal_contract(
                task_type=manifest.task_type,
                input_modality=manifest.input_modality,
                fusion_strategy=manifest.fusion_strategy,
                supported_modalities=manifest.supported_modalities,
                required_modalities=manifest.required_modalities,
                missing_modality_policy=manifest.missing_modality_policy,
                multimodal_schema=manifest.multimodal_schema,
                diagnosis_target_schema=manifest.diagnosis_target_schema,
                diagnostic_rule_pack=manifest.diagnostic_rule_pack,
            )
        )

    # 14. classes / target_schema
    if is_temporal_task(manifest.task_type):
        if manifest.task_type == "action_event_sequence_recognition":
            if not manifest.classes and not manifest.event_types:
                errors.append(
                    "action_event_sequence_recognition 至少需要 classes 或 event_types 之一"
                )
        elif manifest.task_type != "multivariate_sensor_anomaly":
            has_targets = bool((manifest.target_schema or {}).get("outputs"))
            if not manifest.classes and not has_targets:
                errors.append(
                    f"{manifest.task_type} 需要至少一种监督定义: classes 或 target_schema.outputs"
                )
    elif is_multimodal_task(manifest.task_type):
        has_targets = bool((manifest.diagnosis_target_schema or {}).get("outputs"))
        if not manifest.classes and not has_targets:
            errors.append(
                f"{manifest.task_type} 需要至少一种监督定义: classes 或 diagnosis_target_schema.outputs"
            )
    elif not manifest.classes:
        errors.append("classes 列表为空，必须至少包含一个类别")

    # 15. class_aliases 目标合法性
    for alias, target in manifest.class_aliases.items():
        if target not in manifest.classes:
            errors.append(
                f"class_aliases 中 '{alias}' → '{target}'，"
                f"但 '{target}' 不在 classes 列表中"
            )

    return errors
