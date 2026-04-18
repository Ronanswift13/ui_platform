"""
训练数据上传批次服务

负责 training 内部独立的数据上传、解包、校验、标准化与批次注册。
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import tarfile
import threading
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from training.multimodal_fusion import is_multimodal_task
from training.pipelines.ingestion.manifest_validator import (
    ManifestValidator,
    ValidationReport,
)
from training.pipelines.ingestion.upload_validator import (
    CheckItem,
    UploadValidationReport,
    UploadValidator,
)
from training.registry import get_plugin_config, get_registry, resolve_alias
from training.schemas.dataset_manifest import DatasetManifest
from training.temporal_anomaly import is_temporal_task

logger = logging.getLogger(__name__)

TRAINING_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = TRAINING_ROOT / "datasets"
INCOMING_ROOT = DATASETS_ROOT / "incoming"
STAGING_ROOT = DATASETS_ROOT / "staging"
STANDARDIZED_ROOT = DATASETS_ROOT / "standardized"
REGISTRY_ROOT = TRAINING_ROOT / "registry"
UPLOAD_BATCHES_FILE = REGISTRY_ROOT / "upload_batches.json"
LEGACY_UPLOAD_RECORDS_FILE = TRAINING_ROOT / "data" / "upload_records.json"
COMPAT_PROCESSED_ROOT = TRAINING_ROOT / "data" / "processed"

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_EXTS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
}
AUDIO_EXTS = {
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".wma",
    ".opus",
}
LABEL_EXTS = {
    ".txt",
    ".xml",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
}


PLUGIN_CLASSES: dict[str, list[str]] = {
    "transformer": [
        "oil_leak",
        "oil_level_low",
        "oil_level_high",
        "rust",
        "corrosion",
        "crack",
        "damage",
        "foreign_object",
        "bird_nest",
        "overheating",
        "insulator_damage",
        "bushing_crack",
        "normal",
    ],
    "switch": [
        "breaker_open",
        "breaker_closed",
        "breaker_fault",
        "isolator_open",
        "isolator_closed",
        "isolator_fault",
        "contact_wear",
        "arc_damage",
        "mechanism_fault",
        "overheating",
        "rust",
        "normal",
    ],
    "busbar": [
        "pin_missing",
        "pin_loose",
        "crack",
        "corrosion",
        "foreign_object",
        "bird_nest",
        "overheating",
        "insulator_damage",
        "normal",
    ],
    "capacitor": [
        "tilt",
        "collapse",
        "missing",
        "oil_leak",
        "bulge",
        "crack",
        "normal",
    ],
    "meter": [
        "digit_0",
        "digit_1",
        "digit_2",
        "digit_3",
        "digit_4",
        "digit_5",
        "digit_6",
        "digit_7",
        "digit_8",
        "digit_9",
    ],
    "bird": ["sparrow", "crow", "pigeon", "eagle", "unknown_bird"],
    "acoustic": [
        "normal",
        "partial_discharge",
        "corona",
        "mechanical_fault",
        "abnormal",
    ],
    "gas": [
        "sf6_normal",
        "sf6_leak",
        "oil_gas_normal",
        "oil_gas_abnormal",
        "alarm",
    ],
    "hyperspectral": ["normal", "contamination", "aging", "damage", "abnormal"],
    "slam": ["obstacle", "path", "equipment", "marker", "unknown"],
    "fusion": [
        "defect_confirmed",
        "defect_suspected",
        "normal",
        "needs_review",
        "thermal_anomaly",
        "visual_anomaly",
    ],
    "indoor_fence": ["person", "intrusion", "equipment", "normal"],
}

SUPPORTED_FORMATS = {
    "images": sorted(IMAGE_EXTS),
    "videos": sorted(VIDEO_EXTS),
    "audio": sorted(AUDIO_EXTS),
    "labels": sorted(LABEL_EXTS),
    "archives": [".zip", ".tar", ".tar.gz", ".tgz"],
}

MANUAL_PLUGIN_NORMALIZE = {
    "transformer_inspection": "transformer_inspection",
    "transformer": "transformer_inspection",
    "switch_inspection": "switch_inspection",
    "switch": "switch_inspection",
    "busbar_inspection": "busbar_inspection",
    "busbar": "busbar_inspection",
    "capacitor_inspection": "capacitor_inspection",
    "capacitor": "capacitor_inspection",
    "meter_reading": "meter_reading",
    "meter": "meter_reading",
    "bird_monitoring": "bird_monitoring",
    "bird": "bird_monitoring",
    "animal_detection": "animal_detection",
    "acoustic_monitoring": "acoustic_monitoring",
    "acoustic": "acoustic_monitoring",
    "gas_detection": "gas_detection",
    "gas": "gas_detection",
    "hyperspectral_detection": "hyperspectral_detection",
    "hyperspectral": "hyperspectral_detection",
    "temperature_monitoring": "temperature_monitoring",
    "thermal": "thermal",
    "multimodal_fusion": "multimodal_fusion",
    "fusion": "multimodal_fusion",
    "action_event_monitoring": "action_event_monitoring",
    "device_monitoring": "device_monitoring",
    "fire_detection": "fire_detection",
    "indoor_fence": "indoor_fence",
}

CANONICAL_TO_LEGACY = {
    "transformer_inspection": "transformer",
    "switch_inspection": "switch",
    "busbar_inspection": "busbar",
    "capacitor_inspection": "capacitor",
    "meter_reading": "meter",
    "bird_monitoring": "bird",
    "acoustic_monitoring": "acoustic",
    "gas_detection": "gas",
    "hyperspectral_detection": "hyperspectral",
    "multimodal_fusion": "fusion",
    "indoor_fence": "indoor_fence",
    "temperature_monitoring": "thermal",
    "thermal": "thermal",
}

VOLTAGE_NORMALIZE = {
    "220kv": "HV_220kV",
    "110kv": "HV_110kV",
    "35kv": "MV_35kV",
    "10kv": "LV_10kV",
    "500kv": "EHV_500kV",
    "330kv": "EHV_330kV",
    "1000kv": "UHV_1000kV_AC",
    "800kv": "UHV_800kV_DC",
    "HV_220kV": "HV_220kV",
    "HV_110kV": "HV_110kV",
    "MV_35kV": "MV_35kV",
    "LV_10kV": "LV_10kV",
    "EHV_500kV": "EHV_500kV",
    "EHV_330kV": "EHV_330kV",
    "UHV_1000kV_AC": "UHV_1000kV_AC",
    "UHV_800kV_DC": "UHV_800kV_DC",
}


class UploadBatchStatus(str, Enum):
    UPLOADED = "uploaded"
    UNPACKING = "unpacking"
    VALIDATING = "validating"
    VALIDATION_FAILED = "validation_failed"
    STANDARDIZED = "standardized"
    READY_FOR_TRAINING = "ready_for_training"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


def infer_task_type_for_plugin(plugin_id: str) -> str:
    """根据 plugin_id 推断默认 task_type。"""
    config = get_plugin_config(plugin_id) or {}
    primary_task = config.get("primary_task")
    if isinstance(primary_task, str) and primary_task:
        return primary_task
    task_types = config.get("task_types") or []
    if task_types:
        return str(task_types[0])
    return "detection"


@dataclass
class UploadBatch:
    batch_id: str
    dataset_name: str
    plugin_id: str
    task_type: str
    upload_path: str
    unpack_path: str
    status: str
    progress: float
    validation_report: dict[str, Any]
    created_at: str
    updated_at: str
    voltage_level: str = ""
    legacy_plugin: str = ""
    archive_name: str = ""
    source_type: str = "archive"
    standardized_path: str = ""
    compat_data_yaml: str = ""
    message: str = ""
    deleted: bool = False
    deleted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UploadBatchRegistry:
    """上传批次注册表"""

    def __init__(
        self,
        registry_file: Path = UPLOAD_BATCHES_FILE,
        legacy_records_file: Path = LEGACY_UPLOAD_RECORDS_FILE,
    ):
        self.registry_file = registry_file
        self.legacy_records_file = legacy_records_file
        self._lock = threading.RLock()
        self._batches: dict[str, UploadBatch] = {}
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.legacy_records_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.registry_file.exists():
                self._save()
                return
            try:
                data = json.loads(self.registry_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("upload_batches.json 已损坏，重建为空注册表")
                self._batches = {}
                self._save()
                return

            self._batches = {}
            for raw in data.get("batches", []):
                try:
                    batch = UploadBatch(**raw)
                except TypeError:
                    continue
                self._batches[batch.batch_id] = batch
            if not self._batches:
                self._migrate_legacy_records()

    def _migrate_legacy_records(self) -> None:
        if not self.legacy_records_file.exists():
            return
        try:
            data = json.loads(self.legacy_records_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return

        migrated = False
        for raw in data.get("records", []):
            batch_id = str(raw.get("id", "")).strip()
            if not batch_id or batch_id in self._batches:
                continue
            plugins = raw.get("plugins_raw") or raw.get("plugins") or [""]
            plugin_id = MANUAL_PLUGIN_NORMALIZE.get(str(plugins[0]).lower(), str(plugins[0]))
            legacy_plugin = str((raw.get("plugins") or [""])[0] or self.to_legacy_plugin(plugin_id))
            file_count = int(raw.get("file_count", 0) or 0)
            total_size = int(raw.get("total_size", 0) or 0)
            image_count = int(raw.get("image_count", 0) or 0)
            video_count = int(raw.get("video_count", 0) or 0)
            audio_count = int(raw.get("audio_count", 0) or 0)
            label_count = int(raw.get("label_count", 0) or 0)
            created_at = str(raw.get("created_at") or datetime.now().isoformat())
            task_type = infer_task_type_for_plugin(plugin_id)
            data_path = str(raw.get("data_path") or "")
            dataset_name = Path(data_path).name if data_path else batch_id

            batch = UploadBatch(
                batch_id=batch_id,
                dataset_name=dataset_name,
                plugin_id=plugin_id,
                task_type=task_type,
                upload_path=data_path,
                unpack_path=data_path,
                status=str(raw.get("status") or UploadBatchStatus.COMPLETED.value),
                progress=100.0,
                validation_report={
                    "valid": True,
                    "summary": str(raw.get("message") or "legacy upload record"),
                    "manifest": {
                        "valid": True,
                        "errors": [],
                        "warnings": [],
                        "stats": {},
                        "summary": "legacy upload record",
                        "dataset_manifest": {},
                    },
                    "labels": {
                        "valid": True,
                        "errors": [],
                        "warnings": [],
                        "stats": {},
                        "checks": [],
                    },
                    "files": {
                        "total_files": file_count,
                        "total_size_bytes": total_size,
                        "images": image_count,
                        "videos": video_count,
                        "audio": audio_count,
                        "labels": label_count,
                        "archives": 0,
                        "other": 0,
                        "by_extension": {},
                    },
                },
                created_at=created_at,
                updated_at=created_at,
                voltage_level=str(raw.get("voltage_level") or ""),
                legacy_plugin=legacy_plugin,
                message=str(raw.get("message") or ""),
            )
            self._batches[batch.batch_id] = batch
            migrated = True

        if migrated:
            self._save()

    def _save(self) -> None:
        payload = {
            "updated_at": datetime.now().isoformat(),
            "batches": [b.to_dict() for b in self.list_all(limit=None, include_deleted=True)],
        }
        self.registry_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._sync_legacy_records()

    def _sync_legacy_records(self) -> None:
        records = []
        seen_ids: set[str] = set()
        for batch in self.list_all(include_deleted=False):
            file_stats = (batch.validation_report or {}).get("files", {})
            seen_ids.add(batch.batch_id)
            labels = file_stats.get("labels", 0)
            records.append(
                {
                    "id": batch.batch_id,
                    "created_at": batch.created_at,
                    "voltage_level": batch.voltage_level,
                    "voltage_level_raw": batch.voltage_level,
                    "plugins": [batch.legacy_plugin or batch.plugin_id],
                    "plugins_raw": [batch.plugin_id],
                    "file_count": file_stats.get("total_files", 0),
                    "total_size": file_stats.get("total_size_bytes", 0),
                    "status": batch.status,
                    "data_path": batch.upload_path,
                    "message": batch.message,
                    "image_count": file_stats.get("images", 0),
                    "video_count": file_stats.get("videos", 0),
                    "audio_count": file_stats.get("audio", 0),
                    "label_count": labels,
                }
            )
        if self.legacy_records_file.exists():
            try:
                data = json.loads(self.legacy_records_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            for record in data.get("records", []):
                record_id = str(record.get("id", "")).strip()
                if record_id and record_id not in seen_ids:
                    records.append(record)
        payload = {
            "updated_at": datetime.now().isoformat(),
            "records": records,
        }
        self.legacy_records_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create(self, batch: UploadBatch) -> UploadBatch:
        with self._lock:
            self._batches[batch.batch_id] = batch
            self._save()
        return batch

    def update(self, batch_id: str, **kwargs: Any) -> UploadBatch | None:
        with self._lock:
            batch = self._batches.get(batch_id)
            if not batch:
                return None
            for key, value in kwargs.items():
                if hasattr(batch, key):
                    setattr(batch, key, value)
            batch.updated_at = datetime.now().isoformat()
            self._save()
            return batch

    def get(self, batch_id: str) -> UploadBatch | None:
        with self._lock:
            return self._batches.get(batch_id)

    def list_all(
        self,
        limit: int | None = 100,
        *,
        include_deleted: bool = False,
    ) -> list[UploadBatch]:
        with self._lock:
            batches = [
                batch
                for batch in self._batches.values()
                if include_deleted or not batch.deleted
            ]
        batches.sort(key=lambda item: item.created_at, reverse=True)
        if limit is None:
            return batches
        return batches[:limit]

    def delete(self, batch_id: str) -> bool:
        with self._lock:
            if batch_id not in self._batches:
                return False
            del self._batches[batch_id]
            self._save()
        return True


class UploadBatchService:
    """上传批次生命周期服务"""

    def __init__(self, training_root: Path = TRAINING_ROOT):
        self.training_root = training_root
        self.incoming_root = INCOMING_ROOT
        self.staging_root = STAGING_ROOT
        self.standardized_root = STANDARDIZED_ROOT
        self.compat_processed_root = COMPAT_PROCESSED_ROOT
        self.registry = UploadBatchRegistry()
        self.manifest_validator = ManifestValidator()
        self.upload_validator = UploadValidator()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for path in (
            self.incoming_root,
            self.staging_root,
            self.standardized_root,
            REGISTRY_ROOT,
            self.compat_processed_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def generate_batch_id(self) -> str:
        return f"upload_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def normalize_voltage_level(self, voltage_level: str) -> str:
        if not voltage_level:
            return ""
        return VOLTAGE_NORMALIZE.get(voltage_level, VOLTAGE_NORMALIZE.get(voltage_level.lower(), voltage_level))

    def normalize_plugin_id(self, plugin_id: str) -> str:
        if not plugin_id:
            return ""
        plugin_id = plugin_id.strip()
        registry = get_registry()
        alias_map = registry.get("legacy_alias_map", {})
        if plugin_id in alias_map:
            return alias_map[plugin_id]
        lowered = plugin_id.lower()
        if lowered in alias_map:
            return alias_map[lowered]
        if lowered in MANUAL_PLUGIN_NORMALIZE:
            return MANUAL_PLUGIN_NORMALIZE[lowered]
        return resolve_alias(plugin_id)

    def to_legacy_plugin(self, plugin_id: str) -> str:
        if not plugin_id:
            return ""
        return CANONICAL_TO_LEGACY.get(plugin_id, plugin_id)

    def infer_task_type(self, plugin_id: str, task_type: str | None = None) -> str:
        if task_type:
            return task_type
        return infer_task_type_for_plugin(plugin_id)

    def list_plugin_types(self) -> list[dict[str, Any]]:
        registry = get_registry()
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for plugin_id, config in (registry.get("plugins") or {}).items():
            legacy = self.to_legacy_plugin(plugin_id)
            if legacy in seen:
                continue
            seen.add(legacy)
            classes = PLUGIN_CLASSES.get(legacy, [])
            result.append(
                {
                    "id": legacy,
                    "plugin_id": plugin_id,
                    "name": config.get("display_name", plugin_id),
                    "classes": len(classes),
                    "task_types": config.get("task_types", []),
                    "primary_task": config.get("primary_task", ""),
                }
            )
        result.sort(key=lambda item: item["name"])
        return result

    def preflight_validate(
        self,
        voltage_level: str,
        plugins: list[str],
        files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        warnings: list[str] = []
        errors: list[str] = []

        normalized_voltage = self.normalize_voltage_level(voltage_level)
        plugin_id = self.normalize_plugin_id(plugins[0]) if plugins else ""
        inferred_task_type = self.infer_task_type(plugin_id) if plugin_id else ""

        images = videos = audio = labels = archives = manifests = unsupported = 0
        total_size = 0

        for file_info in files:
            name = str(file_info.get("name", ""))
            size = int(file_info.get("size", 0) or 0)
            total_size += size
            category = self.get_file_category(name)
            if category == "images":
                images += 1
            elif category == "videos":
                videos += 1
            elif category == "audio":
                audio += 1
            elif category == "labels":
                labels += 1
            elif category == "archives":
                archives += 1
            elif category == "manifests":
                manifests += 1
            else:
                unsupported += 1
                warnings.append(f"不支持的文件格式: {name}")

        if not normalized_voltage:
            errors.append("缺少电压等级")
        if not plugin_id:
            errors.append("缺少目标插件")
        if files and archives == 0:
            warnings.append("当前训练包建议上传单个 zip 或 tar.gz，以保留 manifest 与目录层级")
        if archives > 1:
            warnings.append("检测到多个压缩包，系统会依次解包到同一 staging 区")
        if images > 0 and labels == 0 and archives == 0:
            warnings.append("未检测到标注文件；若目录层级在压缩包内，请改为上传 zip/tar.gz")

        valid = len(errors) == 0
        return {
            "valid": valid,
            "message": "验证通过" if valid else "验证失败，请检查错误信息",
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "voltage_level": normalized_voltage,
                "plugins": [plugin_id] if plugin_id else [],
                "task_type": inferred_task_type,
                "image_count": images,
                "video_count": videos,
                "audio_count": audio,
                "label_count": labels,
                "manifest_count": manifests,
                "archive_count": archives,
                "unsupported_count": unsupported,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            },
        }

    def create_batch_from_uploaded_files(
        self,
        *,
        files: Iterable[Any],
        voltage_level: str,
        plugins: list[str],
        task_type: str | None = None,
        dataset_name: str | None = None,
    ) -> UploadBatch:
        primary_plugin = self.normalize_plugin_id(plugins[0] if plugins else "")
        normalized_voltage = self.normalize_voltage_level(voltage_level)
        inferred_task_type = self.infer_task_type(primary_plugin, task_type)
        batch_id = self.generate_batch_id()
        upload_dir = self.incoming_root / batch_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        stored_names: list[str] = []
        archive_name = ""
        source_type = "files"

        for upload in files:
            filename = self._sanitize_filename(getattr(upload, "filename", "") or "upload.bin")
            target = upload_dir / filename
            if target.exists():
                stem = target.stem
                suffix = target.suffix
                target = upload_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            stored_names.append(target.name)
            if self.is_archive_path(target):
                source_type = "archive"
                if not archive_name:
                    archive_name = target.name

        if not stored_names:
            raise ValueError("没有收到可保存的上传文件")

        fallback_dataset_name = dataset_name or self._derive_dataset_name(stored_names[0], batch_id)
        batch = UploadBatch(
            batch_id=batch_id,
            dataset_name=fallback_dataset_name,
            plugin_id=primary_plugin,
            task_type=inferred_task_type,
            upload_path=str(upload_dir),
            unpack_path=str(self.staging_root / batch_id),
            status=UploadBatchStatus.UPLOADED.value,
            progress=5.0,
            validation_report={},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            voltage_level=normalized_voltage,
            legacy_plugin=self.to_legacy_plugin(primary_plugin),
            archive_name=archive_name,
            source_type=source_type,
            message=f"已接收 {len(stored_names)} 个文件，等待解包校验",
        )
        self.registry.create(batch)
        return batch

    def import_local_directory(
        self,
        *,
        source_path: Path,
        voltage_level: str,
        plugins: list[str],
        task_type: str | None = None,
        dataset_name: str | None = None,
    ) -> UploadBatch:
        if not source_path.exists():
            raise FileNotFoundError(f"源路径不存在: {source_path}")

        primary_plugin = self.normalize_plugin_id(plugins[0] if plugins else "")
        normalized_voltage = self.normalize_voltage_level(voltage_level)
        inferred_task_type = self.infer_task_type(primary_plugin, task_type)
        batch_id = self.generate_batch_id()
        upload_dir = self.incoming_root / batch_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        if source_path.is_dir():
            shutil.copytree(source_path, upload_dir / source_path.name)
            source_type = "directory"
        else:
            shutil.copy2(source_path, upload_dir / source_path.name)
            source_type = "archive" if self.is_archive_path(source_path) else "file"

        batch = UploadBatch(
            batch_id=batch_id,
            dataset_name=dataset_name or self._derive_dataset_name(source_path.name, batch_id),
            plugin_id=primary_plugin,
            task_type=inferred_task_type,
            upload_path=str(upload_dir),
            unpack_path=str(self.staging_root / batch_id),
            status=UploadBatchStatus.UPLOADED.value,
            progress=5.0,
            validation_report={},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            voltage_level=normalized_voltage,
            legacy_plugin=self.to_legacy_plugin(primary_plugin),
            archive_name=source_path.name if source_type == "archive" else "",
            source_type=source_type,
            message="本地数据已登记，等待解包校验",
        )
        self.registry.create(batch)
        return batch

    def process_uploaded_batch(self, batch_id: str, *, auto_standardize: bool = False) -> UploadBatch:
        batch = self.registry.get(batch_id)
        if not batch:
            raise ValueError(f"批次不存在: {batch_id}")

        try:
            self.registry.update(
                batch_id,
                status=UploadBatchStatus.UNPACKING.value,
                progress=15.0,
                message="正在解包到 staging",
            )
            dataset_root = self._prepare_staging_dataset(batch)

            self.registry.update(
                batch_id,
                unpack_path=str(dataset_root),
                status=UploadBatchStatus.VALIDATING.value,
                progress=35.0,
                message="正在执行 manifest 与标签校验",
            )

            manifest, report = self._validate_dataset(dataset_root)
            update_payload: dict[str, Any] = {
                "validation_report": report,
                "progress": 60.0 if report.get("valid") else 100.0,
            }

            if manifest is not None:
                update_payload["dataset_name"] = manifest.name or batch.dataset_name
                update_payload["plugin_id"] = self.normalize_plugin_id(manifest.plugin_id) or batch.plugin_id
                update_payload["task_type"] = manifest.task_type or batch.task_type
                update_payload["legacy_plugin"] = self.to_legacy_plugin(
                    self.normalize_plugin_id(manifest.plugin_id) or batch.plugin_id
                )
                update_payload["voltage_level"] = (
                    self.normalize_voltage_level(manifest.voltage_level) or batch.voltage_level
                )

            if report.get("valid"):
                update_payload["status"] = UploadBatchStatus.UPLOADED.value
                update_payload["message"] = "校验通过，等待标准化"
            else:
                update_payload["status"] = UploadBatchStatus.VALIDATION_FAILED.value
                update_payload["message"] = report.get("summary", "校验失败")
            self.registry.update(batch_id, **update_payload)

            if report.get("valid") and auto_standardize:
                return self.standardize_batch(batch_id)
            refreshed = self.registry.get(batch_id)
            if refreshed is None:
                raise ValueError(f"批次不存在: {batch_id}")
            return refreshed
        except Exception as exc:
            logger.error("批次处理失败: %s", exc, exc_info=True)
            self.registry.update(
                batch_id,
                status=UploadBatchStatus.FAILED.value,
                progress=100.0,
                message=f"处理失败: {exc}",
            )
            batch = self.registry.get(batch_id)
            if batch is None:
                raise
            return batch

    def standardize_batch(self, batch_id: str) -> UploadBatch:
        batch = self.registry.get(batch_id)
        if not batch:
            raise ValueError(f"批次不存在: {batch_id}")
        if batch.deleted:
            raise ValueError(f"批次已软删除: {batch_id}")

        validation = batch.validation_report or {}
        if not validation.get("valid"):
            raise ValueError("批次尚未通过校验，不能标准化")

        dataset_root = Path(batch.unpack_path)
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json 不存在: {manifest_path}")
        manifest = DatasetManifest.from_json(manifest_path)

        self.registry.update(
            batch_id,
            status=UploadBatchStatus.STANDARDIZED.value,
            progress=70.0,
            message="正在生成 standardized 数据集",
        )

        standardized_dir = self.standardized_root / batch_id
        if standardized_dir.exists():
            shutil.rmtree(standardized_dir)
        standardized_dir.mkdir(parents=True, exist_ok=True)

        standardization_report: dict[str, Any]
        if is_temporal_task(manifest.task_type) or is_multimodal_task(manifest.task_type):
            self._copy_tree_contents(dataset_root, standardized_dir)
            manifest.save(standardized_dir / "manifest.json")
            standardization_report = {
                "mode": "passthrough",
                "supported_for_training": False,
                "message": "当前版本仅打通视觉类 ready_for_training；时序/多模态先完成 standardized 落盘",
            }
            self.registry.update(
                batch_id,
                standardized_path=str(standardized_dir),
                validation_report=self._with_standardization(validation, standardization_report),
                message=standardization_report["message"],
            )
            refreshed = self.registry.get(batch_id)
            if refreshed is None:
                raise ValueError(f"批次不存在: {batch_id}")
            return refreshed

        standardization_report = self._standardize_visual_dataset(
            dataset_root=dataset_root,
            standardized_dir=standardized_dir,
            manifest=manifest,
        )
        compat_data_yaml = self._write_compat_data_yaml(
            manifest=manifest,
            batch_id=batch.batch_id,
            standardized_dir=standardized_dir,
            voltage_level=self.normalize_voltage_level(manifest.voltage_level) or batch.voltage_level,
            plugin_id=self.normalize_plugin_id(manifest.plugin_id) or batch.plugin_id,
        )
        standardization_report["compat_data_yaml"] = str(compat_data_yaml)
        standardization_report["supported_for_training"] = True

        self.registry.update(
            batch_id,
            standardized_path=str(standardized_dir),
            compat_data_yaml=str(compat_data_yaml),
            status=UploadBatchStatus.READY_FOR_TRAINING.value,
            progress=100.0,
            message="标准化完成，可直接用于训练",
            validation_report=self._with_standardization(validation, standardization_report),
        )
        refreshed = self.registry.get(batch_id)
        if refreshed is None:
            raise ValueError(f"批次不存在: {batch_id}")
        return refreshed

    def soft_delete_failed_batch(self, batch_id: str, *, purge_files: bool = False) -> UploadBatch:
        batch = self.registry.get(batch_id)
        if not batch:
            raise ValueError(f"批次不存在: {batch_id}")
        if batch.status not in {
            UploadBatchStatus.VALIDATION_FAILED.value,
            UploadBatchStatus.FAILED.value,
        }:
            raise ValueError("仅允许软删除失败批次")

        if purge_files:
            for raw_path in (batch.upload_path, batch.unpack_path, batch.standardized_path):
                if raw_path:
                    path = Path(raw_path)
                    if path.exists():
                        shutil.rmtree(path)
        self.registry.update(
            batch_id,
            deleted=True,
            deleted_at=datetime.now().isoformat(),
            message="批次已软删除",
        )
        refreshed = self.registry.get(batch_id)
        if refreshed is None:
            raise ValueError(f"批次不存在: {batch_id}")
        return refreshed

    def list_batches(self, limit: int = 100) -> list[UploadBatch]:
        return self.registry.list_all(limit=limit, include_deleted=False)

    def get_batch(self, batch_id: str) -> UploadBatch | None:
        return self.registry.get(batch_id)

    def get_stats(self) -> dict[str, Any]:
        batches = self.list_batches(limit=1000)
        by_status: dict[str, int] = {}
        by_voltage_level: dict[str, int] = {}
        total_files = 0
        total_size = 0
        ready_count = 0
        failed_count = 0

        for batch in batches:
            file_stats = (batch.validation_report or {}).get("files", {})
            total_files += int(file_stats.get("total_files", 0))
            total_size += int(file_stats.get("total_size_bytes", 0))
            by_status[batch.status] = by_status.get(batch.status, 0) + 1
            by_voltage_level[batch.voltage_level] = by_voltage_level.get(batch.voltage_level, 0) + 1
            if batch.status == UploadBatchStatus.READY_FOR_TRAINING.value:
                ready_count += 1
            if batch.status in {
                UploadBatchStatus.VALIDATION_FAILED.value,
                UploadBatchStatus.FAILED.value,
            }:
                failed_count += 1

        return {
            "total_uploads": len(batches),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "ready_for_training": ready_count,
            "failed_batches": failed_count,
            "by_status": by_status,
            "by_voltage_level": by_voltage_level,
        }

    def get_file_category(self, filename: str) -> str:
        lower_name = filename.lower()
        if lower_name.endswith("manifest.json"):
            return "manifests"
        ext = Path(lower_name).suffix
        if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz") or ext == ".zip" or ext == ".tar":
            return "archives"
        if ext in IMAGE_EXTS:
            return "images"
        if ext in VIDEO_EXTS:
            return "videos"
        if ext in AUDIO_EXTS:
            return "audio"
        if ext in LABEL_EXTS:
            return "labels"
        return "unknown"

    def is_archive_path(self, path: Path) -> bool:
        try:
            return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)
        except (OSError, tarfile.TarError):
            return False

    def _prepare_staging_dataset(self, batch: UploadBatch) -> Path:
        upload_dir = Path(batch.upload_path)
        staging_dir = self.staging_root / batch.batch_id
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        archives = [path for path in upload_dir.iterdir() if path.is_file() and self.is_archive_path(path)]
        if archives:
            payload_root = staging_dir / "payload"
            payload_root.mkdir(parents=True, exist_ok=True)
            for archive in archives:
                self._extract_archive_safe(archive, payload_root)
        else:
            payload_root = staging_dir / "payload"
            payload_root.mkdir(parents=True, exist_ok=True)
            self._copy_tree_contents(upload_dir, payload_root)

        return self._detect_dataset_root(payload_root)

    def _validate_dataset(self, dataset_root: Path) -> tuple[DatasetManifest | None, dict[str, Any]]:
        manifest_obj: DatasetManifest | None = None
        manifest_errors: list[str] = []
        manifest_path = dataset_root / "manifest.json"
        if manifest_path.exists():
            try:
                manifest_obj = DatasetManifest.from_json(manifest_path)
            except Exception as exc:
                manifest_errors.append(f"manifest.json 解析失败: {exc}")
        else:
            manifest_errors.append("manifest.json 不存在")

        manifest_report = self.manifest_validator.validate(dataset_root)
        upload_report = self.upload_validator.validate(dataset_root)

        if manifest_obj and manifest_obj.plugin_id:
            manifest_obj.plugin_id = self.normalize_plugin_id(manifest_obj.plugin_id)
        file_stats = self._scan_dataset_files(dataset_root)
        label_report = self._build_label_report(upload_report)

        summary_parts = []
        if manifest_report.summary():
            summary_parts.append(manifest_report.summary())
        if upload_report.summary():
            summary_parts.append(upload_report.summary())

        combined = {
            "valid": manifest_report.valid and upload_report.valid and not manifest_errors,
            "summary": "\n\n".join(summary_parts) if summary_parts else "",
            "manifest": self._serialize_manifest_report(manifest_report, manifest_errors, manifest_obj),
            "labels": label_report,
            "files": file_stats,
            "checks": upload_report.to_dict(),
        }
        return manifest_obj, combined

    def _serialize_manifest_report(
        self,
        report: ValidationReport,
        parse_errors: list[str],
        manifest: DatasetManifest | None,
    ) -> dict[str, Any]:
        errors = list(parse_errors)
        errors.extend(report.manifest_errors)
        return {
            "valid": report.valid and not parse_errors,
            "errors": errors,
            "warnings": report.warnings,
            "stats": report.stats,
            "summary": report.summary(),
            "dataset_manifest": manifest.to_dict() if manifest else {},
        }

    def _build_label_report(self, upload_report: UploadValidationReport) -> dict[str, Any]:
        label_checks = [
            check
            for check in upload_report.checks
            if check.category in {"标签合法性", "类别映射"}
        ]
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}
        valid = True
        for check in label_checks:
            errors.extend([f"[{check.check_name}] {msg}" for msg in check.errors])
            warnings.extend([f"[{check.check_name}] {msg}" for msg in check.warnings])
            stats[check.check_name] = check.stats
            if not check.passed:
                valid = False
        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "checks": [self._serialize_check_item(check) for check in label_checks],
        }

    def _serialize_check_item(self, check: CheckItem) -> dict[str, Any]:
        return {
            "category": check.category,
            "check_name": check.check_name,
            "passed": check.passed,
            "errors": check.errors,
            "warnings": check.warnings,
            "stats": check.stats,
        }

    def _scan_dataset_files(self, root: Path) -> dict[str, Any]:
        total_files = 0
        total_size = 0
        images = videos = audio = labels = manifests = archives = other = 0
        by_extension: dict[str, int] = {}

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            total_files += 1
            total_size += path.stat().st_size
            category = self.get_file_category(path.name)
            ext_key = path.suffix.lower() or "<none>"
            by_extension[ext_key] = by_extension.get(ext_key, 0) + 1
            if category == "images":
                images += 1
            elif category == "videos":
                videos += 1
            elif category == "audio":
                audio += 1
            elif category == "labels":
                labels += 1
            elif category == "archives":
                archives += 1
            elif category == "manifests":
                manifests += 1
            else:
                other += 1

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "images": images,
            "videos": videos,
            "audio": audio,
            "labels": labels,
            "manifests": manifests,
            "archives": archives,
            "other": other,
            "by_extension": dict(sorted(by_extension.items())),
        }

    def _extract_archive_safe(self, archive_path: Path, destination: Path) -> None:
        if zipfile.is_zipfile(archive_path):
            self._extract_zip_safe(archive_path, destination)
            return
        if tarfile.is_tarfile(archive_path):
            self._extract_tar_safe(archive_path, destination)
            return
        raise ValueError(f"不支持的压缩包格式: {archive_path.name}")

    def _extract_zip_safe(self, archive_path: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.infolist():
                target = self._safe_target_path(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError(f"压缩包包含符号链接，已拒绝: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)

    def _extract_tar_safe(self, archive_path: Path, destination: Path) -> None:
        with tarfile.open(archive_path) as tf:
            for member in tf.getmembers():
                target = self._safe_target_path(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if member.issym() or member.islnk():
                    raise ValueError(f"压缩包包含链接文件，已拒绝: {member.name}")
                if not member.isfile():
                    continue
                source = tf.extractfile(member)
                if source is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)

    def _safe_target_path(self, root: Path, member_name: str) -> Path:
        normalized = member_name.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute():
            raise ValueError(f"压缩包路径非法(绝对路径): {member_name}")
        parts = [part for part in pure.parts if part not in {"", "."}]
        if any(part == ".." for part in parts):
            raise ValueError(f"压缩包路径非法(路径穿越): {member_name}")
        if parts and ":" in parts[0]:
            raise ValueError(f"压缩包路径非法(盘符路径): {member_name}")
        if not parts:
            return root
        target = (root / Path(*parts)).resolve()
        root_resolved = root.resolve()
        if target != root_resolved and root_resolved not in target.parents:
            raise ValueError(f"压缩包路径越界: {member_name}")
        return target

    def _detect_dataset_root(self, root: Path) -> Path:
        manifest_here = root / "manifest.json"
        if manifest_here.exists():
            return root

        manifests = sorted(root.rglob("manifest.json"), key=lambda path: (len(path.relative_to(root).parts), str(path)))
        if manifests:
            return manifests[0].parent

        children = [path for path in root.iterdir()]
        dirs = [path for path in children if path.is_dir()]
        files = [path for path in children if path.is_file()]
        if len(dirs) == 1 and not files:
            return self._detect_dataset_root(dirs[0])
        return root

    def _standardize_visual_dataset(
        self,
        *,
        dataset_root: Path,
        standardized_dir: Path,
        manifest: DatasetManifest,
    ) -> dict[str, Any]:
        image_files, label_map = self._collect_visual_assets(dataset_root, manifest)
        if not image_files:
            raise ValueError("未找到可标准化的视觉样本")

        splits = self._resolve_visual_splits(dataset_root, manifest, image_files)
        for split_name in ("train", "val", "test"):
            (standardized_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (standardized_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        copied_labels = 0
        split_counts: dict[str, int] = {}
        split_label_counts: dict[str, int] = {}

        for split_name, files in splits.items():
            split_counts[split_name] = len(files)
            split_label_counts[split_name] = 0
            for image_path in files:
                target_image = standardized_dir / "images" / split_name / image_path.name
                shutil.copy2(image_path, target_image)

                label_path = label_map.get(image_path.stem)
                if label_path and label_path.exists():
                    target_label = standardized_dir / "labels" / split_name / label_path.name
                    shutil.copy2(label_path, target_label)
                    copied_labels += 1
                    split_label_counts[split_name] += 1

        manifest.save(standardized_dir / "manifest.json")
        self._write_data_yaml(standardized_dir / "data.yaml", standardized_dir, manifest.classes)

        return {
            "mode": "visual_v1",
            "split_counts": split_counts,
            "split_label_counts": split_label_counts,
            "copied_images": sum(split_counts.values()),
            "copied_labels": copied_labels,
            "standardized_path": str(standardized_dir),
        }

    def _collect_visual_assets(
        self,
        dataset_root: Path,
        manifest: DatasetManifest,
    ) -> tuple[list[Path], dict[str, Path]]:
        excluded_dirs = {
            "labels",
            "annotations",
            "metadata",
            "masks",
            "__macosx",
        }
        image_files = [
            path
            for path in dataset_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTS
            and not any(part.lower() in excluded_dirs for part in path.relative_to(dataset_root).parts[:-1])
        ]
        image_files = sorted(dict.fromkeys(image_files), key=str)

        label_map: dict[str, Path] = {}
        for path in dataset_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in LABEL_EXTS and path.suffix.lower() != ".png":
                continue
            rel_parts = [part.lower() for part in path.relative_to(dataset_root).parts[:-1]]
            if not any(part in {"labels", "annotations", "masks", "metadata"} for part in rel_parts):
                continue
            label_map.setdefault(path.stem, path)

        return image_files, label_map

    def _resolve_visual_splits(
        self,
        dataset_root: Path,
        manifest: DatasetManifest,
        image_files: list[Path],
    ) -> dict[str, list[Path]]:
        if manifest.pre_split:
            result = {"train": [], "val": [], "test": []}
            for image_path in image_files:
                parts = [part.lower() for part in image_path.relative_to(dataset_root).parts]
                if "train" in parts:
                    result["train"].append(image_path)
                elif "val" in parts or "valid" in parts:
                    result["val"].append(image_path)
                elif "test" in parts:
                    result["test"].append(image_path)
                else:
                    result["train"].append(image_path)
            return result

        shuffled = sorted(image_files, key=str)
        seed = f"{manifest.name}:{manifest.version}:{manifest.plugin_id}:{manifest.task_type}"
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        train_ratio = float((manifest.split or {}).get("train", 0.8))
        val_ratio = float((manifest.split or {}).get("val", 0.15))
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        return {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    def _write_compat_data_yaml(
        self,
        *,
        manifest: DatasetManifest,
        batch_id: str,
        standardized_dir: Path,
        voltage_level: str,
        plugin_id: str,
    ) -> Path:
        legacy_plugin = self.to_legacy_plugin(plugin_id)
        compat_dir = self.compat_processed_root / voltage_level / legacy_plugin
        compat_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = compat_dir / "data.yaml"
        self._write_data_yaml(data_yaml, standardized_dir, manifest.classes)

        meta = {
            "batch_id": batch_id,
            "plugin_id": plugin_id,
            "legacy_plugin": legacy_plugin,
            "task_type": manifest.task_type,
            "voltage_level": voltage_level,
            "standardized_path": str(standardized_dir),
            "updated_at": datetime.now().isoformat(),
        }
        (compat_dir / "batch_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return data_yaml

    def _write_data_yaml(self, path: Path, dataset_root: Path, classes: list[str]) -> None:
        classes_json = ", ".join(json.dumps(name, ensure_ascii=False) for name in classes)
        content = (
            f"# Auto-generated by training upload batch service\n"
            f"path: {dataset_root}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"nc: {len(classes)}\n"
            f"names: [{classes_json}]\n"
        )
        path.write_text(content, encoding="utf-8")

    def _with_standardization(
        self,
        validation_report: dict[str, Any],
        standardization_report: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(validation_report)
        merged["standardization"] = standardization_report
        return merged

    def _copy_tree_contents(self, source: Path, destination: Path) -> None:
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    def _sanitize_filename(self, filename: str) -> str:
        name = Path(filename).name.strip()
        return name or f"upload_{uuid.uuid4().hex[:8]}"

    def _derive_dataset_name(self, filename: str, batch_id: str) -> str:
        lower = filename.lower()
        if lower.endswith(".tar.gz"):
            return filename[:-7]
        if lower.endswith(".tgz"):
            return filename[:-4]
        stem = Path(filename).stem
        return stem or batch_id


batch_service = UploadBatchService()
record_manager = batch_service.registry
