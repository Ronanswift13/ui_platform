"""
训练数据上传校验器 (完整版)

六大校验维度:
1. 文件完整性    — manifest 存在、图像/标签配对、必要目录结构
2. 标签合法性    — 按 task_type 逐文件校验标签格式
3. 类别映射一致性 — manifest.classes vs 实际标签中的 class_id / 类名
4. 重复样本检测   — 基于文件名 + 文件大小 + 可选 hash 去重
5. 模糊/坏图过滤  — Laplacian 方差检测 + 可读性检查
6. train/val/test 划分检查 — pre_split 目录一致性 / 比例合理性
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.schemas.dataset_manifest import (
    DatasetManifest,
    LEGACY_ALIAS_MAP,
    validate_manifest,
)
from training.schemas.label_schemas import get_schema_for_task_type

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTS_TEXT = {".txt", ".xml", ".json", ".csv"}
LABEL_EXTS_MASK = {".png"}


# =============================================================================
# 校验报告
# =============================================================================


@dataclass
class CheckItem:
    """单项校验结果"""

    category: str  # 校验类别
    check_name: str  # 校验项名称
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadValidationReport:
    """上传校验总报告"""

    valid: bool = True
    checks: list[CheckItem] = field(default_factory=list)
    manifest: DatasetManifest | None = None

    # 汇总
    total_errors: int = 0
    total_warnings: int = 0

    def add_check(self, item: CheckItem) -> None:
        self.checks.append(item)
        self.total_errors += len(item.errors)
        self.total_warnings += len(item.warnings)
        if not item.passed:
            self.valid = False

    def summary(self) -> str:
        status = "✅ 通过" if self.valid else "❌ 失败"
        lines = [
            f"上传校验结果: {status}",
            f"  错误: {self.total_errors}  警告: {self.total_warnings}",
            "",
        ]
        for item in self.checks:
            icon = "✓" if item.passed else "✗"
            lines.append(f"  [{icon}] {item.category} / {item.check_name}")
            for e in item.errors[:5]:
                lines.append(f"      ✗ {e}")
            if len(item.errors) > 5:
                lines.append(f"      ... 及其他 {len(item.errors) - 5} 个错误")
            for w in item.warnings[:3]:
                lines.append(f"      ⚠ {w}")
            if item.stats:
                lines.append(f"      统计: {item.stats}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "checks": [
                {
                    "category": c.category,
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "errors": c.errors,
                    "warnings": c.warnings,
                    "stats": c.stats,
                }
                for c in self.checks
            ],
        }


# =============================================================================
# 校验器
# =============================================================================


class UploadValidator:
    """
    训练数据上传校验器

    用法:
        validator = UploadValidator()
        report = validator.validate("/path/to/upload_package")
        print(report.summary())
    """

    def __init__(
        self,
        label_sample_size: int = 50,
        image_sample_size: int = 30,
        blur_threshold: float = 100.0,
        min_image_size: tuple[int, int] = (32, 32),
        hash_dedup: bool = False,
    ):
        self.label_sample_size = label_sample_size
        self.image_sample_size = image_sample_size
        self.blur_threshold = blur_threshold
        self.min_image_size = min_image_size
        self.hash_dedup = hash_dedup

    def validate(self, dataset_dir: str | Path) -> UploadValidationReport:
        """
        执行完整六维校验

        Args:
            dataset_dir: 上传数据包根目录

        Returns:
            UploadValidationReport
        """
        dataset_dir = Path(dataset_dir)
        report = UploadValidationReport()

        # ── 0. Manifest 加载 ──
        manifest = self._load_manifest(dataset_dir, report)
        if manifest is None:
            return report
        report.manifest = manifest

        # ── 1. 文件完整性 ──
        image_files, label_files = self._check_file_integrity(
            dataset_dir, manifest, report
        )

        # ── 2. 标签合法性 ──
        self._check_label_validity(label_files, manifest, report)

        # ── 3. 类别映射一致性 ──
        self._check_class_consistency(label_files, manifest, report)

        # ── 4. 重复样本 ──
        self._check_duplicates(image_files, report)

        # ── 5. 模糊/坏图过滤 ──
        self._check_image_quality(image_files, report)

        # ── 6. train/val/test 划分 ──
        self._check_split(dataset_dir, manifest, image_files, report)

        return report

    # ─────────────────────────────────────────────────────────────
    # 0. Manifest 加载与结构校验
    # ─────────────────────────────────────────────────────────────

    def _load_manifest(
        self, dataset_dir: Path, report: UploadValidationReport
    ) -> DatasetManifest | None:
        check = CheckItem(category="manifest", check_name="manifest.json 存在与合法性")

        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            check.passed = False
            check.errors.append(f"manifest.json 不存在: {manifest_path}")
            report.add_check(check)
            return None

        try:
            manifest = DatasetManifest.from_json(manifest_path)
        except Exception as e:
            check.passed = False
            check.errors.append(f"manifest.json 解析失败: {e}")
            report.add_check(check)
            return None

        errs = validate_manifest(manifest)
        if errs:
            check.passed = False
            check.errors = errs
        else:
            check.stats = {
                "plugin_id": manifest.plugin_id,
                "task_type": manifest.task_type,
                "classes": manifest.classes,
                "num_images": manifest.num_images,
            }

        report.add_check(check)
        return manifest if check.passed else None

    # ─────────────────────────────────────────────────────────────
    # 1. 文件完整性
    # ─────────────────────────────────────────────────────────────

    def _check_file_integrity(
        self,
        dataset_dir: Path,
        manifest: DatasetManifest,
        report: UploadValidationReport,
    ) -> tuple[list[Path], list[Path]]:
        check = CheckItem(category="文件完整性", check_name="图像/标签配对与目录结构")

        image_files: list[Path] = []
        label_files: list[Path] = []

        # 查找图像目录
        image_dir = self._find_dir(dataset_dir, ("images", "train/images", "JPEGImages"))
        if image_dir is None:
            # 分类任务可能没有独立 images 目录
            if manifest.task_type == "classification":
                image_dir = dataset_dir
            else:
                check.passed = False
                check.errors.append("未找到图像目录 (images/)")
                report.add_check(check)
                return [], []

        image_files = self._collect_files(image_dir, IMAGE_EXTS, recursive=True)

        # 查找标签
        if manifest.task_type == "segmentation" and manifest.mask_format in (
            "png_index",
            "png_color",
        ):
            label_dir = self._find_dir(dataset_dir, ("masks", "labels", "SegmentationClass"))
            if label_dir:
                label_files = self._collect_files(label_dir, LABEL_EXTS_MASK)
        elif manifest.task_type == "classification" and manifest.format == "image_folder":
            label_files = []  # 文件夹即标签
        else:
            label_dir = self._find_dir(
                dataset_dir, ("labels", "Annotations", "annotations", "train/labels")
            )
            if label_dir:
                label_files = self._collect_files(label_dir, LABEL_EXTS_TEXT | LABEL_EXTS_MASK)

        check.stats["actual_images"] = len(image_files)
        check.stats["actual_labels"] = len(label_files)

        # 配对检查 (detection / segmentation 要求 1:1)
        if manifest.task_type in ("detection", "segmentation"):
            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}
            missing_labels = image_stems - label_stems
            orphan_labels = label_stems - image_stems

            if missing_labels:
                n = len(missing_labels)
                samples = sorted(missing_labels)[:5]
                check.warnings.append(
                    f"{n} 张图像无对应标签: {samples}{'...' if n > 5 else ''}"
                )
            if orphan_labels:
                n = len(orphan_labels)
                samples = sorted(orphan_labels)[:5]
                check.warnings.append(
                    f"{n} 个标签无对应图像 (孤立标签): {samples}{'...' if n > 5 else ''}"
                )

        # 声明数量 vs 实际
        if manifest.num_images > 0:
            diff = abs(len(image_files) - manifest.num_images)
            if diff > max(5, manifest.num_images * 0.05):
                check.warnings.append(
                    f"声明图像数 {manifest.num_images} vs 实际 {len(image_files)}"
                )

        if not image_files:
            check.passed = False
            check.errors.append("数据包中无图像文件")

        report.add_check(check)
        return image_files, label_files

    # ─────────────────────────────────────────────────────────────
    # 2. 标签合法性
    # ─────────────────────────────────────────────────────────────

    def _check_label_validity(
        self,
        label_files: list[Path],
        manifest: DatasetManifest,
        report: UploadValidationReport,
    ) -> None:
        check = CheckItem(category="标签合法性", check_name="标签格式校验 (抽样)")

        if not label_files:
            if manifest.task_type not in ("classification",):
                check.warnings.append("无标签文件可校验")
            report.add_check(check)
            return

        schema_kwargs = {}
        if manifest.task_type == "segmentation":
            schema_kwargs["mask_format"] = manifest.mask_format
            schema_kwargs["ignore_index"] = manifest.ignore_index

        try:
            schema = get_schema_for_task_type(manifest.task_type, **schema_kwargs)
        except ValueError as e:
            check.passed = False
            check.errors.append(str(e))
            report.add_check(check)
            return

        num_classes = len(manifest.classes)
        sample = random.sample(
            label_files, min(self.label_sample_size, len(label_files))
        )
        check.stats["sampled"] = len(sample)

        for lf in sample:
            errs = schema.validate_label_file(lf, num_classes)
            for e in errs:
                check.errors.append(f"[{lf.name}] {e}")

        if check.errors:
            check.passed = False

        report.add_check(check)

    # ─────────────────────────────────────────────────────────────
    # 3. 类别映射一致性
    # ─────────────────────────────────────────────────────────────

    def _check_class_consistency(
        self,
        label_files: list[Path],
        manifest: DatasetManifest,
        report: UploadValidationReport,
    ) -> None:
        check = CheckItem(category="类别映射", check_name="manifest.classes vs 实际标签 class_id")

        if manifest.task_type in ("classification",) or not label_files:
            report.add_check(check)
            return

        num_classes = len(manifest.classes)
        seen_class_ids: set[int] = set()
        sample = random.sample(
            label_files, min(self.label_sample_size, len(label_files))
        )

        for lf in sample:
            if manifest.task_type == "detection" and lf.suffix == ".txt":
                self._extract_yolo_class_ids(lf, seen_class_ids)
            elif lf.suffix == ".json":
                self._extract_json_class_ids(lf, seen_class_ids)

        # 超出范围的 class_id
        out_of_range = {cid for cid in seen_class_ids if cid < 0 or cid >= num_classes}
        if out_of_range:
            check.passed = False
            check.errors.append(
                f"标签中出现超范围 class_id: {sorted(out_of_range)}，"
                f"合法范围 [0, {num_classes - 1}] (classes={manifest.classes})"
            )

        # 未使用的类别
        used = seen_class_ids & set(range(num_classes))
        unused = set(range(num_classes)) - used
        if unused and len(sample) >= 20:
            unused_names = [manifest.classes[i] for i in sorted(unused)]
            check.warnings.append(
                f"以下类别在抽样中未出现: {unused_names} "
                f"(抽样 {len(sample)} 个标签文件)"
            )

        check.stats["seen_class_ids"] = sorted(seen_class_ids)
        check.stats["declared_classes"] = num_classes

        report.add_check(check)

    def _extract_yolo_class_ids(self, path: Path, ids: set[int]) -> None:
        try:
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        ids.add(int(parts[0]))
        except (ValueError, IOError):
            pass

    def _extract_json_class_ids(self, path: Path, ids: set[int]) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            anns = data.get("annotations", data if isinstance(data, list) else [])
            for ann in anns:
                if "category_id" in ann:
                    ids.add(int(ann["category_id"]))
        except (json.JSONDecodeError, IOError, ValueError):
            pass

    # ─────────────────────────────────────────────────────────────
    # 4. 重复样本检测
    # ─────────────────────────────────────────────────────────────

    def _check_duplicates(
        self, image_files: list[Path], report: UploadValidationReport
    ) -> None:
        check = CheckItem(category="重复检测", check_name="重复样本检测")

        if not image_files:
            report.add_check(check)
            return

        # 阶段 1: 文件名去重
        name_counter = Counter(f.name for f in image_files)
        dup_names = {name: cnt for name, cnt in name_counter.items() if cnt > 1}
        if dup_names:
            check.warnings.append(
                f"文件名重复: {len(dup_names)} 个 "
                f"(如 {list(dup_names.keys())[:3]})"
            )

        # 阶段 2: 文件大小去重 (快速近似)
        size_map: dict[int, list[Path]] = {}
        for f in image_files:
            try:
                sz = f.stat().st_size
                size_map.setdefault(sz, []).append(f)
            except OSError:
                pass

        dup_size_groups = {sz: paths for sz, paths in size_map.items() if len(paths) > 1}

        # 阶段 3: 可选 hash 精确去重
        exact_dups: list[list[str]] = []
        if self.hash_dedup and dup_size_groups:
            for sz, paths in dup_size_groups.items():
                hash_map: dict[str, list[Path]] = {}
                for p in paths:
                    h = self._file_hash(p)
                    hash_map.setdefault(h, []).append(p)
                for h, group in hash_map.items():
                    if len(group) > 1:
                        exact_dups.append([str(p) for p in group])

        if exact_dups:
            check.warnings.append(f"精确重复 (hash 相同): {len(exact_dups)} 组")
        elif dup_size_groups:
            check.warnings.append(
                f"疑似重复 (文件大小相同): {len(dup_size_groups)} 组 "
                f"(启用 hash_dedup=True 以精确判定)"
            )

        check.stats["total_images"] = len(image_files)
        check.stats["duplicate_name_count"] = sum(dup_names.values()) - len(dup_names)
        check.stats["suspect_size_dup_groups"] = len(dup_size_groups)

        report.add_check(check)

    def _file_hash(self, path: Path, chunk_size: int = 8192) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    # ─────────────────────────────────────────────────────────────
    # 5. 模糊/坏图过滤
    # ─────────────────────────────────────────────────────────────

    def _check_image_quality(
        self, image_files: list[Path], report: UploadValidationReport
    ) -> None:
        check = CheckItem(category="图像质量", check_name="模糊/坏图/尺寸异常检测 (抽样)")

        if not image_files:
            report.add_check(check)
            return

        sample = random.sample(
            image_files, min(self.image_sample_size, len(image_files))
        )
        check.stats["sampled"] = len(sample)

        corrupt_count = 0
        blur_count = 0
        tiny_count = 0

        try:
            from PIL import Image
            pil_available = True
        except ImportError:
            pil_available = False
            check.warnings.append("Pillow 未安装，跳过图像质量检查")

        try:
            import cv2
            import numpy as np
            cv2_available = True
        except ImportError:
            cv2_available = False

        if not pil_available:
            report.add_check(check)
            return

        for img_path in sample:
            # 5a. 可读性
            try:
                img = Image.open(img_path)
                img.verify()
                img = Image.open(img_path)  # verify 后需重新打开
            except Exception:
                corrupt_count += 1
                check.errors.append(f"坏图 (无法解码): {img_path.name}")
                continue

            # 5b. 尺寸
            w, h = img.size
            if w < self.min_image_size[0] or h < self.min_image_size[1]:
                tiny_count += 1
                check.warnings.append(
                    f"尺寸过小 ({w}×{h}): {img_path.name}"
                )

            # 5c. 模糊检测 (Laplacian 方差)
            if cv2_available:
                try:
                    cv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if cv_img is not None:
                        lap_var = cv2.Laplacian(cv_img, cv2.CV_64F).var()
                        if lap_var < self.blur_threshold:
                            blur_count += 1
                            check.warnings.append(
                                f"疑似模糊 (Laplacian={lap_var:.1f}): {img_path.name}"
                            )
                except Exception:
                    pass

        check.stats.update({
            "corrupt": corrupt_count,
            "blurry": blur_count,
            "tiny": tiny_count,
        })

        if corrupt_count > 0:
            check.passed = False

        report.add_check(check)

    # ─────────────────────────────────────────────────────────────
    # 6. train/val/test 划分检查
    # ─────────────────────────────────────────────────────────────

    def _check_split(
        self,
        dataset_dir: Path,
        manifest: DatasetManifest,
        all_images: list[Path],
        report: UploadValidationReport,
    ) -> None:
        check = CheckItem(category="数据划分", check_name="train/val/test 划分校验")

        if manifest.pre_split:
            # 预分割模式: 检查子目录是否存在
            split_stats: dict[str, int] = {}
            for split_name in ("train", "val", "test"):
                split_img_dir = dataset_dir / split_name / "images"
                if not split_img_dir.exists():
                    # 尝试直接 split_name 目录
                    split_img_dir = dataset_dir / split_name
                if split_img_dir.exists():
                    count = len(self._collect_files(split_img_dir, IMAGE_EXTS, recursive=True))
                    split_stats[split_name] = count
                else:
                    check.errors.append(
                        f"pre_split=true 但 '{split_name}/' 目录不存在"
                    )

            total = sum(split_stats.values())
            if total > 0:
                for s, n in split_stats.items():
                    ratio = n / total
                    check.stats[f"{s}_count"] = n
                    check.stats[f"{s}_ratio"] = round(ratio, 3)

                # 比例合理性警告
                train_ratio = split_stats.get("train", 0) / total
                if train_ratio < 0.5:
                    check.warnings.append(
                        f"训练集比例过低: {train_ratio:.1%} (建议 ≥ 60%)"
                    )
                val_count = split_stats.get("val", 0)
                if val_count == 0:
                    check.passed = False
                    check.errors.append("验证集为空 (val/ 无图像)")
        else:
            # 比例模式: 检查 split 比例
            total = sum(manifest.split.values())
            if abs(total - 1.0) > 0.01:
                check.passed = False
                check.errors.append(f"split 比例之和 {total:.3f} ≠ 1.0")

            train_ratio = manifest.split.get("train", 0)
            val_ratio = manifest.split.get("val", 0)
            if train_ratio < 0.5:
                check.warnings.append(f"训练集比例过低: {train_ratio:.1%}")
            if val_ratio <= 0:
                check.passed = False
                check.errors.append("val 比例为 0，必须预留验证集")

            check.stats = {
                "mode": "auto_split",
                "split_config": manifest.split,
                "total_images": len(all_images),
            }

        if check.errors:
            check.passed = False

        report.add_check(check)

    # ─────────────────────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _find_dir(root: Path, candidates: tuple[str, ...]) -> Path | None:
        for name in candidates:
            p = root / name
            if p.exists() and p.is_dir():
                return p
        return None

    @staticmethod
    def _collect_files(
        directory: Path, exts: set[str], recursive: bool = False
    ) -> list[Path]:
        if not directory.exists():
            return []
        pattern = "**/*" if recursive else "*"
        return [
            f
            for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in exts
        ]
