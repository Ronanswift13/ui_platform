"""
多模态融合预处理器

使用 samples.jsonl 作为对齐索引，支持部分模态缺失。
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from training.multimodal_fusion import validate_multimodal_sample_record
from training.schemas.dataset_manifest import DatasetManifest

from .base_preprocessor import BasePreprocessor, PreprocessResult


class MultimodalPreprocessor(BasePreprocessor):
    """多模态融合数据预处理"""

    def preprocess(self, input_dir: Path, output_dir: Path) -> PreprocessResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = input_dir / "manifest.json"
        samples_index_path = input_dir / "samples.jsonl"
        if not manifest_path.exists():
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message=f"manifest.json 不存在: {manifest_path}",
            )
        if not samples_index_path.exists():
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message=f"samples.jsonl 不存在: {samples_index_path}",
            )

        manifest = DatasetManifest.from_json(manifest_path)
        records = self._load_records(samples_index_path)
        if not records:
            return PreprocessResult(
                output_dir=output_dir,
                success=False,
                message="samples.jsonl 中无合法样本记录",
            )

        split_ratio = manifest.split or self.split_ratio
        rng = random.Random(self.config.get("random_seed", 42))
        rng.shuffle(records)
        n = len(records)
        n_train = int(n * split_ratio.get("train", 0.8))
        n_val = int(n * split_ratio.get("val", 0.15))
        split_map = {
            "train": records[:n_train],
            "val": records[n_train : n_train + n_val],
            "test": records[n_train + n_val :],
        }

        for split_name, rows in split_map.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            self._write_split_index(split_dir / "samples.jsonl", rows)
            self._copy_modalities(input_dir, split_dir, rows)
            self._copy_auxiliary_files(input_dir, split_dir)

        shutil.copy2(manifest_path, output_dir / "manifest.json")
        self._write_dataset_config(output_dir, manifest)

        return PreprocessResult(
            output_dir=output_dir,
            num_train=len(split_map["train"]),
            num_val=len(split_map["val"]),
            num_test=len(split_map["test"]),
            class_distribution={name: 0 for name in manifest.classes},
            success=True,
            message=f"多模态预处理完成: {len(records)} 个样本",
        )

    def _load_records(self, path: Path) -> list[dict]:
        records: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if validate_multimodal_sample_record(record):
                    continue
                records.append(record)
        return records

    def _write_split_index(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _copy_modalities(self, input_dir: Path, split_dir: Path, rows: list[dict]) -> None:
        copied: set[tuple[str, str]] = set()
        for row in rows:
            modality_paths = row.get("modality_paths", {})
            for modality, rel_path in modality_paths.items():
                if not rel_path:
                    continue
                src = input_dir / rel_path
                if not src.exists():
                    continue
                dst = split_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                cache_key = (modality, rel_path)
                if cache_key in copied:
                    continue
                shutil.copy2(src, dst)
                copied.add(cache_key)

    def _copy_auxiliary_files(self, input_dir: Path, split_dir: Path) -> None:
        for dir_name in ("labels", "metadata"):
            src_dir = input_dir / dir_name
            if src_dir.exists():
                dst_dir = split_dir / dir_name
                if dst_dir.exists():
                    continue
                shutil.copytree(src_dir, dst_dir)

    def _write_dataset_config(self, output_dir: Path, manifest: DatasetManifest) -> None:
        payload = {
            "plugin_id": manifest.plugin_id,
            "task_type": manifest.task_type,
            "fusion_strategy": manifest.fusion_strategy,
            "supported_modalities": manifest.supported_modalities,
            "required_modalities": manifest.required_modalities,
            "missing_modality_policy": manifest.missing_modality_policy,
            "alignment_tolerance_ms": manifest.alignment_tolerance_ms,
            "diagnosis_target_schema": manifest.diagnosis_target_schema,
            "diagnostic_rule_pack": manifest.diagnostic_rule_pack,
        }
        with open(output_dir / "dataset_config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
