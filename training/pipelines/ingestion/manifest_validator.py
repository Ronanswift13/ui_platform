"""
数据包 manifest.json 校验器

入口校验流程:
1. 检查数据包根目录是否存在 manifest.json
2. 解析并校验 manifest 结构
3. 根据 task_type 校验标签文件格式 (抽样)
4. 校验图像文件可读性 (抽样)
5. 输出校验报告
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.multimodal_fusion import is_multimodal_task
from training.schemas.dataset_manifest import DatasetManifest, validate_manifest
from training.schemas.label_schemas import get_schema_for_task_type
from training.temporal_anomaly import is_temporal_task

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """校验报告"""

    valid: bool = True
    manifest_errors: list[str] = field(default_factory=list)
    label_errors: list[str] = field(default_factory=list)
    image_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "✅ 通过" if self.valid else "❌ 失败"
        lines = [f"校验结果: {status}"]
        if self.manifest_errors:
            lines.append(f"  Manifest 错误 ({len(self.manifest_errors)}):")
            for e in self.manifest_errors:
                lines.append(f"    - {e}")
        if self.label_errors:
            lines.append(f"  标签错误 ({len(self.label_errors)}):")
            for e in self.label_errors[:10]:
                lines.append(f"    - {e}")
            if len(self.label_errors) > 10:
                lines.append(f"    ... 及其他 {len(self.label_errors) - 10} 个错误")
        if self.image_errors:
            lines.append(f"  图像错误 ({len(self.image_errors)}):")
            for e in self.image_errors[:5]:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  警告 ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.stats:
            lines.append(f"  统计: {json.dumps(self.stats, ensure_ascii=False)}")
        return "\n".join(lines)


class ManifestValidator:
    """数据包校验器"""

    def __init__(self, sample_size: int = 20):
        self.sample_size = sample_size

    def validate(self, dataset_dir: str | Path) -> ValidationReport:
        """
        校验数据包目录

        Args:
            dataset_dir: 数据包根目录 (必须包含 manifest.json)

        Returns:
            ValidationReport
        """
        dataset_dir = Path(dataset_dir)
        report = ValidationReport()

        # 1. 检查 manifest.json 存在
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            report.valid = False
            report.manifest_errors.append(f"manifest.json 不存在: {manifest_path}")
            return report

        # 2. 解析 manifest
        try:
            manifest = DatasetManifest.from_json(manifest_path)
        except Exception as e:
            report.valid = False
            report.manifest_errors.append(f"manifest.json 解析失败: {e}")
            return report

        # 3. 校验 manifest 结构
        manifest_errs = validate_manifest(manifest)
        if manifest_errs:
            report.valid = False
            report.manifest_errors = manifest_errs
            return report

        # 4. 校验标签文件 (抽样)
        label_schema = get_schema_for_task_type(manifest.task_type)
        num_classes = len(manifest.classes)
        label_dir: Path | None = None
        label_files: list[Path] = []
        if is_multimodal_task(manifest.task_type):
            samples_index = dataset_dir / "samples.jsonl"
            if samples_index.exists():
                label_files = [samples_index]
            else:
                label_dir = self._find_label_dir(dataset_dir, manifest)
                if label_dir and label_dir.exists():
                    label_files = list(label_dir.glob("*"))
        else:
            label_dir = self._find_label_dir(dataset_dir, manifest)
            if label_dir and label_dir.exists():
                label_files = list(label_dir.glob("*"))

        if label_files:
            sample = random.sample(label_files, min(self.sample_size, len(label_files)))
            for lf in sample:
                errs = label_schema.validate_label_file(lf, num_classes)
                report.label_errors.extend(errs)

        # 5. 统计
        if is_multimodal_task(manifest.task_type) and (dataset_dir / "samples.jsonl").exists():
            with open(dataset_dir / "samples.jsonl", "r", encoding="utf-8") as f:
                sample_count = sum(1 for line in f if line.strip())
        else:
            data_dir = self._find_primary_data_dir(dataset_dir, manifest)
            sample_count = len(list(data_dir.glob("*"))) if data_dir and data_dir.exists() else 0
        label_count = len(list(label_dir.glob("*"))) if label_dir and label_dir.exists() else 0

        report.stats = {
            "plugin_id": manifest.plugin_id,
            "task_type": manifest.task_type,
            "declared_images": manifest.num_images,
            "declared_samples": manifest.num_samples,
            "actual_samples": sample_count,
            "actual_labels": label_count,
            "classes": manifest.classes,
        }

        if (
            not is_temporal_task(manifest.task_type)
            and not is_multimodal_task(manifest.task_type)
            and manifest.num_images > 0
            and abs(sample_count - manifest.num_images) > 5
        ):
            report.warnings.append(
                f"声明图像数 {manifest.num_images} 与实际 {sample_count} 不符"
            )
        if (
            (is_temporal_task(manifest.task_type) or is_multimodal_task(manifest.task_type))
            and manifest.num_samples > 0
            and abs(sample_count - manifest.num_samples) > 5
        ):
            report.warnings.append(
                f"声明样本数 {manifest.num_samples} 与实际 {sample_count} 不符"
            )

        if report.label_errors:
            report.valid = False

        return report

    def _find_image_dir(self, root: Path, manifest: DatasetManifest) -> Path | None:
        """查找图像目录 (按常见约定)"""
        for candidate in ("images", "train/images", "data/images", "imgs"):
            p = root / candidate
            if p.exists():
                return p
        return root

    def _find_primary_data_dir(self, root: Path, manifest: DatasetManifest) -> Path | None:
        """按 task_type 查找主数据目录"""

        if is_temporal_task(manifest.task_type):
            candidates = {
                "acoustic_time_frequency_anomaly": ("waveforms", "spectrograms", "audio"),
                "multivariate_sensor_anomaly": ("timeseries", "series", "signals"),
                "equipment_health_prediction": ("timeseries", "series", "signals"),
                "action_event_sequence_recognition": ("events", "sequences", "action_events"),
            }.get(manifest.task_type, ())
            for candidate in candidates:
                p = root / candidate
                if p.exists():
                    return p
            return root
        if is_multimodal_task(manifest.task_type):
            for candidate in (
                "visual",
                "thermal",
                "acoustic",
                "ultrasonic",
                "gas",
                "hyperspectral",
                "vibration",
            ):
                p = root / candidate
                if p.exists():
                    return p
            return root
        return self._find_image_dir(root, manifest)

    def _find_label_dir(self, root: Path, manifest: DatasetManifest) -> Path | None:
        """查找标签目录"""
        candidates = ["labels", "train/labels", "data/labels", "annotations"]
        if is_temporal_task(manifest.task_type):
            candidates = ["labels", "targets", "annotations", "metadata"]
        if is_multimodal_task(manifest.task_type):
            candidates = ["labels", "metadata"]
        for candidate in candidates:
            p = root / candidate
            if p.exists():
                return p
        return None
