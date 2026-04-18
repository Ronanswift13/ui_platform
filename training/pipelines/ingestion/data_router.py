"""
数据路由器

根据 manifest.json 中的 plugin_id 和 task_type，
将数据路由到正确的预处理 → 训练流水线。
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from training.multimodal_fusion import MULTIMODAL_DATASET_FAMILY, is_multimodal_task
from training.schemas.dataset_manifest import DatasetManifest
from training.temporal_anomaly import TEMPORAL_DATASET_FAMILY, is_temporal_task

logger = logging.getLogger(__name__)

# 训练根目录 (相对于本文件)
TRAINING_ROOT = Path(__file__).parent.parent.parent


@dataclass
class RouteResult:
    """路由结果"""

    plugin_id: str
    task_type: str
    destination: Path  # 数据将存放的目标路径
    storage_family: str  # visual_defect / temporal_anomaly
    preprocessor: str  # 预处理器模块路径
    trainer: str  # 训练器模块路径
    copy_dirs: tuple[str, ...] = ()
    success: bool = True
    message: str = ""


class DataRouter:
    """
    数据路由器

    路由规则:
    - 视觉任务       → datasets/visual_defect/{plugin_id}/{task_type}/
    - 时序任务       → datasets/temporal_anomaly/{plugin_id}/{task_type}/{version}/
    - 多模态任务     → datasets/multimodal_fusion/{plugin_id}/{task_type}/{version}/
    """

    # task_type → (storage_family, 子目录, preprocessor, trainer, copy_dirs)
    ROUTE_TABLE = {
        "detection": (
            "visual_defect",
            "detection",
            "pipelines.preprocessing.detection_preprocessor.DetectionPreprocessor",
            "pipelines.training.detection_trainer.DetectionTrainer",
            ("images", "labels", "annotations"),
        ),
        "segmentation": (
            "visual_defect",
            "segmentation",
            "pipelines.preprocessing.segmentation_preprocessor.SegmentationPreprocessor",
            "pipelines.training.segmentation_trainer.SegmentationTrainer",
            ("images", "labels", "annotations", "masks"),
        ),
        "classification": (
            "visual_defect",
            "classification",
            "pipelines.preprocessing.classification_preprocessor.ClassificationPreprocessor",
            "pipelines.training.classification_trainer.ClassificationTrainer",
            ("images", "labels", "annotations"),
        ),
        "ocr": (
            "visual_defect",
            "ocr",
            "pipelines.preprocessing.ocr_preprocessor.OCRPreprocessor",
            "pipelines.training.ocr_trainer.OCRTrainer",
            ("images", "labels", "annotations"),
        ),
        "thermal_anomaly": (
            "visual_defect",
            "thermal",
            "pipelines.preprocessing.thermal_preprocessor.ThermalPreprocessor",
            "pipelines.training.thermal_trainer.ThermalTrainer",
            ("images", "labels", "annotations"),
        ),
        "hyperspectral_classification": (
            "visual_defect",
            "hyperspectral",
            "pipelines.preprocessing.hyperspectral_preprocessor.HyperspectralPreprocessor",
            "pipelines.training.hyperspectral_trainer.HyperspectralTrainer",
            ("images", "labels", "annotations"),
        ),
        "hyperspectral_anomaly": (
            "visual_defect",
            "hyperspectral",
            "pipelines.preprocessing.hyperspectral_preprocessor.HyperspectralPreprocessor",
            "pipelines.training.hyperspectral_trainer.HyperspectralTrainer",
            ("images", "labels", "annotations"),
        ),
        "acoustic_time_frequency_anomaly": (
            TEMPORAL_DATASET_FAMILY,
            "acoustic",
            "pipelines.preprocessing.temporal_preprocessor.TemporalPreprocessor",
            "pipelines.training.temporal_trainer.TemporalTrainer",
            ("waveforms", "spectrograms", "labels", "metadata"),
        ),
        "multivariate_sensor_anomaly": (
            TEMPORAL_DATASET_FAMILY,
            "sensor_anomaly",
            "pipelines.preprocessing.temporal_preprocessor.TemporalPreprocessor",
            "pipelines.training.temporal_trainer.TemporalTrainer",
            ("timeseries", "labels", "metadata"),
        ),
        "equipment_health_prediction": (
            TEMPORAL_DATASET_FAMILY,
            "health_prediction",
            "pipelines.preprocessing.temporal_preprocessor.TemporalPreprocessor",
            "pipelines.training.temporal_trainer.TemporalTrainer",
            ("timeseries", "labels", "metadata"),
        ),
        "health_index_calibration": (
            TEMPORAL_DATASET_FAMILY,
            "health_calibration",
            "pipelines.preprocessing.temporal_preprocessor.TemporalPreprocessor",
            "pipelines.training.temporal_trainer.TemporalTrainer",
            ("timeseries", "labels", "metadata"),
        ),
        "action_event_sequence_recognition": (
            TEMPORAL_DATASET_FAMILY,
            "action_sequence",
            "pipelines.preprocessing.temporal_preprocessor.TemporalPreprocessor",
            "pipelines.training.temporal_trainer.TemporalTrainer",
            ("events", "labels", "metadata"),
        ),
        "multimodal_feature_fusion": (
            MULTIMODAL_DATASET_FAMILY,
            "feature_fusion",
            "pipelines.preprocessing.multimodal_preprocessor.MultimodalPreprocessor",
            "pipelines.training.multimodal_trainer.MultimodalTrainer",
            (
                "visual",
                "thermal",
                "acoustic",
                "ultrasonic",
                "gas",
                "hyperspectral",
                "vibration",
                "labels",
                "metadata",
            ),
        ),
        "multimodal_decision_fusion": (
            MULTIMODAL_DATASET_FAMILY,
            "decision_fusion",
            "pipelines.preprocessing.multimodal_preprocessor.MultimodalPreprocessor",
            "pipelines.training.multimodal_trainer.MultimodalTrainer",
            (
                "visual",
                "thermal",
                "acoustic",
                "ultrasonic",
                "gas",
                "hyperspectral",
                "vibration",
                "labels",
                "metadata",
            ),
        ),
    }

    def __init__(self, training_root: Path | None = None):
        self.training_root = training_root or TRAINING_ROOT

    def resolve_plugin_id(self, plugin_id_or_alias: str) -> str:
        """将旧名称映射到新 plugin_id"""
        alias_map_path = self.training_root / "registry" / "plugin_training_mapping.json"
        if alias_map_path.exists():
            with open(alias_map_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            legacy_map = mapping.get("legacy_alias_map", {})
            if plugin_id_or_alias in legacy_map:
                resolved = legacy_map[plugin_id_or_alias]
                logger.info(f"别名映射: {plugin_id_or_alias} → {resolved}")
                return resolved
        return plugin_id_or_alias

    def route(self, manifest: DatasetManifest) -> RouteResult:
        """
        根据 manifest 决定数据目标路径和处理流水线

        Args:
            manifest: 已通过校验的数据集 manifest

        Returns:
            RouteResult
        """
        plugin_id = self.resolve_plugin_id(manifest.plugin_id)
        task_type = manifest.task_type

        if task_type not in self.ROUTE_TABLE:
            return RouteResult(
                plugin_id=plugin_id,
                task_type=task_type,
                destination=Path(""),
                storage_family="",
                preprocessor="",
                trainer="",
                success=False,
                message=f"未知 task_type: '{task_type}'",
            )

        storage_family, subdir, preprocessor, trainer, copy_dirs = self.ROUTE_TABLE[task_type]
        destination = self.training_root / "datasets" / storage_family / plugin_id / subdir

        # 如有电压等级，进一步细分
        if manifest.voltage_level:
            destination = destination / manifest.voltage_level
        if is_temporal_task(task_type) or is_multimodal_task(task_type):
            destination = destination / manifest.version

        return RouteResult(
            plugin_id=plugin_id,
            task_type=task_type,
            destination=destination,
            storage_family=storage_family,
            preprocessor=preprocessor,
            trainer=trainer,
            copy_dirs=copy_dirs,
            success=True,
            message=f"路由: {plugin_id}/{task_type} ({storage_family}) → {destination}",
        )

    def ingest(self, source_dir: Path, manifest: DatasetManifest, copy: bool = True) -> RouteResult:
        """
        执行数据摄入: 校验 → 路由 → 复制/链接到目标目录

        Args:
            source_dir: 源数据目录
            manifest: 数据集 manifest
            copy: True=复制, False=符号链接

        Returns:
            RouteResult
        """
        result = self.route(manifest)
        if not result.success:
            return result

        result.destination.mkdir(parents=True, exist_ok=True)

        if copy:
            for subdir_name in result.copy_dirs:
                src = source_dir / subdir_name
                if src.exists():
                    dst = result.destination / subdir_name
                    if dst.exists():
                        logger.warning(f"目标已存在，跳过: {dst}")
                    else:
                        shutil.copytree(src, dst)
                        logger.info(f"已复制: {src} → {dst}")
            for file_name in (
                "samples.jsonl",
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "targets.jsonl",
                "README.md",
            ):
                src_file = source_dir / file_name
                if src_file.exists():
                    shutil.copy2(src_file, result.destination / file_name)
        else:
            # 创建符号链接
            link = result.destination / "source"
            if not link.exists():
                link.symlink_to(source_dir.resolve())
                logger.info(f"已链接: {source_dir} → {link}")

        # 复制 manifest
        manifest.save(result.destination / "manifest.json")

        return result
