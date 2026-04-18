"""
时序异常预处理器

统一处理:
- 原始波形
- 频谱图 / 梅尔谱
- 多变量数值时序
- 带时间戳事件序列
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

from training.schemas.dataset_manifest import DatasetManifest
from training.temporal_anomaly import get_temporal_task_profile

from .base_preprocessor import BasePreprocessor, PreprocessResult

logger = logging.getLogger(__name__)


class TemporalPreprocessor(BasePreprocessor):
    """时序数据预处理"""

    SOURCE_DIR_CANDIDATES = {
        "acoustic_time_frequency_anomaly": ("waveforms", "spectrograms", "labels", "metadata"),
        "multivariate_sensor_anomaly": ("timeseries", "labels", "metadata"),
        "equipment_health_prediction": ("timeseries", "labels", "metadata"),
        "health_index_calibration": ("timeseries", "labels", "metadata"),
        "action_event_sequence_recognition": ("events", "labels", "metadata"),
    }

    def preprocess(self, input_dir: Path, output_dir: Path) -> PreprocessResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = input_dir / "manifest.json"
        if not manifest_path.exists():
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message=f"manifest.json 不存在: {manifest_path}",
            )

        manifest = DatasetManifest.from_json(manifest_path)
        profile = get_temporal_task_profile(manifest.task_type)
        if profile is None:
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message=f"未知时序 task_type: {manifest.task_type}",
            )

        if manifest.pre_split and all((input_dir / split).exists() for split in ("train", "val", "test")):
            return self._preprocess_pre_split(input_dir, output_dir, manifest)

        primary_files = self._collect_primary_files(input_dir, manifest.task_type)
        if not primary_files:
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message=f"未找到 {manifest.task_type} 的主数据文件",
            )

        rng = random.Random(self.config.get("random_seed", 42))
        rng.shuffle(primary_files)

        split_ratio = manifest.split or self.split_ratio
        n = len(primary_files)
        n_train = int(n * split_ratio.get("train", 0.8))
        n_val = int(n * split_ratio.get("val", 0.15))
        split_map = {
            "train": primary_files[:n_train],
            "val": primary_files[n_train : n_train + n_val],
            "test": primary_files[n_train + n_val :],
        }

        source_dirs = self.SOURCE_DIR_CANDIDATES.get(manifest.task_type, ())
        for split_name, files in split_map.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            stems = {path.stem for path in files}

            for dir_name in source_dirs:
                src_dir = input_dir / dir_name
                if not src_dir.exists():
                    continue
                dst_dir = split_dir / dir_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                for src_file in src_dir.iterdir():
                    if not src_file.is_file():
                        continue
                    if src_file.stem in stems:
                        shutil.copy2(src_file, dst_dir / src_file.name)

            self._write_samples_index(split_dir, files)

        shutil.copy2(manifest_path, output_dir / "manifest.json")
        self._write_dataset_config(output_dir, manifest, profile.default_paradigm)

        return PreprocessResult(
            output_dir=output_dir,
            num_train=len(split_map["train"]),
            num_val=len(split_map["val"]),
            num_test=len(split_map["test"]),
            class_distribution={name: 0 for name in manifest.classes},
            success=True,
            message=f"时序预处理完成: {n} 个样本",
        )

    def _preprocess_pre_split(
        self,
        input_dir: Path,
        output_dir: Path,
        manifest: DatasetManifest,
    ) -> PreprocessResult:
        for split_name in ("train", "val", "test"):
            src_dir = input_dir / split_name
            dst_dir = output_dir / split_name
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

        shutil.copy2(input_dir / "manifest.json", output_dir / "manifest.json")
        self._write_dataset_config(output_dir, manifest, self.config.get("paradigm", ""))

        counts = {
            split_name: self._count_files(output_dir / split_name)
            for split_name in ("train", "val", "test")
        }
        return PreprocessResult(
            output_dir=output_dir,
            num_train=counts["train"],
            num_val=counts["val"],
            num_test=counts["test"],
            class_distribution={name: 0 for name in manifest.classes},
            success=True,
            message="时序预分割数据已归档",
        )

    def _collect_primary_files(self, input_dir: Path, task_type: str) -> list[Path]:
        candidate_dirs = {
            "acoustic_time_frequency_anomaly": ("waveforms",),
            "multivariate_sensor_anomaly": ("timeseries",),
            "equipment_health_prediction": ("timeseries",),
            "health_index_calibration": ("timeseries",),
            "action_event_sequence_recognition": ("events",),
        }.get(task_type, ())
        for dir_name in candidate_dirs:
            src_dir = input_dir / dir_name
            if src_dir.exists():
                return [path for path in sorted(src_dir.iterdir()) if path.is_file()]
        return []

    def _write_samples_index(self, split_dir: Path, files: list[Path]) -> None:
        rows = [
            {
                "sample_id": path.stem,
                "file_name": path.name,
                "relative_path": path.name,
            }
            for path in files
        ]
        with open(split_dir / "samples.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_dataset_config(
        self,
        output_dir: Path,
        manifest: DatasetManifest,
        paradigm: str,
    ) -> None:
        payload = {
            "plugin_id": manifest.plugin_id,
            "task_type": manifest.task_type,
            "input_modality": manifest.input_modality,
            "paradigm": paradigm,
            "sample_rate_hz": manifest.sample_rate_hz,
            "sequence_length": manifest.sequence_length,
            "window_size": manifest.window_size,
            "prediction_horizon": manifest.prediction_horizon,
            "feature_views": manifest.feature_views,
            "sensor_columns": manifest.sensor_columns,
            "event_types": manifest.event_types,
            "target_schema": manifest.target_schema,
        }
        with open(output_dir / "dataset_config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _count_files(self, split_dir: Path) -> int:
        total = 0
        for child in split_dir.iterdir():
            if child.is_dir():
                total += sum(1 for path in child.iterdir() if path.is_file())
        return total
