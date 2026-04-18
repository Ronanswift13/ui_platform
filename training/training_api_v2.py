"""
统一训练入口 API (v2)

基于 FastAPI，提供:
1. 数据上传与校验 (manifest.json 必检)
2. 训练任务创建 (按 plugin_id + task_type 路由)
3. 训练状态查询
4. 模型评估触发
5. 模型导出

与现有 training_api.py 兼容:
- 旧接口继续工作 (通过 legacy_alias_map 映射)
- 新接口使用 /v2/ 前缀

生命周期:
  created → validating → ingesting → preprocessing
  → training → evaluating → exporting → registering → completed

路由设计:
  POST   /v2/datasets/validate         校验数据包
  GET    /v2/datasets                   列出已有数据集

  POST   /v2/train                     创建并启动训练任务
  GET    /v2/train/{task_id}           查询训练状态
  POST   /v2/train/{task_id}/cancel    取消训练
  GET    /v2/train                     列出所有任务

  GET    /v2/plugins                   列出支持的训练插件
  GET    /v2/plugins/{plugin_id}       插件训练配置详情
  GET    /v2/task-types                列出支持的 task_type

  GET    /v2/capabilities              系统能力概览
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---- 依赖导入 (延迟，允许单独使用) ----
try:
    from fastapi import APIRouter, HTTPException, UploadFile, File
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from training.registry import get_plugin_config, resolve_alias, list_supported_plugins
from training.schemas.dataset_manifest import DatasetManifest, validate_manifest
from training.multimodal_fusion import MULTIMODAL_DATASET_FAMILY
from training.pipelines.ingestion.manifest_validator import ManifestValidator
from training.pipelines.ingestion.data_router import DataRouter
from training.temporal_anomaly import TEMPORAL_DATASET_FAMILY

TRAINING_ROOT = Path(__file__).parent


# ============================================================
# 生命周期阶段 (统一 11 状态)
# ============================================================

class TaskStatus(str, Enum):
    CREATED = "created"
    VALIDATING = "validating"
    INGESTING = "ingesting"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    REGISTERING = "registering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# 请求/响应模型
# ============================================================

if FASTAPI_AVAILABLE:

    class TrainRequest(BaseModel):
        """训练请求"""
        plugin_id: str = ""
        task_type: str = ""
        dataset_path: str = ""
        voltage_level: str = ""

        # 超参覆盖 (可选)
        epochs: int | None = None
        batch_size: int | None = None
        learning_rate: float | None = None
        backbone: str | None = None
        image_size: int | None = None
        paradigm: str | None = None

        # 高光谱专用
        pca_components: int | None = None
        model_type: str | None = None

        # 时序专用
        sequence_length: int | None = None
        window_size: float | None = None
        prediction_horizon: float | None = None
        sample_rate_hz: int | None = None
        feature_views: list[str] | None = None
        sensor_columns: list[str] | None = None
        event_types: list[str] | None = None

        # 多模态专用
        fusion_strategy: str | None = None
        supported_modalities: list[str] | None = None
        required_modalities: list[str] | None = None
        missing_modality_policy: str | None = None

else:

    class TrainRequest:
        """训练请求 (无 FastAPI 时的占位)"""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# ============================================================
# 训练任务
# ============================================================

class TrainTask:
    """训练任务 (完整生命周期)"""

    def __init__(self, request: TrainRequest):
        self.task_id: str = str(uuid.uuid4())[:8]
        self.plugin_id: str = resolve_alias(getattr(request, "plugin_id", ""))
        self.task_type: str = getattr(request, "task_type", "")
        self.status: TaskStatus = TaskStatus.CREATED
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = self.created_at
        self.request = request
        self.result: dict[str, Any] = {}
        self.error: str = ""
        self.error_category: str = ""
        self.progress: float = 0.0
        self.current_phase: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "plugin_id": self.plugin_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "progress": self.progress,
            "current_phase": self.current_phase,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "error_category": self.error_category,
        }


# ============================================================
# 训练管理器 V2 (单例)
# ============================================================

class TrainingManagerV2:
    """训练任务管理器 (v2) — 驱动 TaskOrchestrator"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tasks: dict[str, TrainTask] = {}
            cls._instance._validator = ManifestValidator()
            cls._instance._router = DataRouter()
            cls._instance._orchestrators: dict[str, Any] = {}
        return cls._instance

    def validate_dataset(self, dataset_dir: str | Path) -> dict:
        """校验数据包"""
        report = self._validator.validate(dataset_dir)
        return {
            "valid": report.valid,
            "summary": report.summary(),
            "stats": report.stats,
        }

    def create_task(self, request: TrainRequest) -> TrainTask:
        """创建训练任务"""
        plugin_id = resolve_alias(getattr(request, "plugin_id", ""))

        # 兼容旧插件名
        legacy_map = {
            "transformer": "transformer_inspection",
            "switch": "switch_inspection",
            "busbar": "busbar_inspection",
            "capacitor": "capacitor_inspection",
            "meter": "meter_reading",
            "bird": "bird_monitoring",
            "acoustic": "acoustic_monitoring",
            "gas": "gas_detection",
        }
        if plugin_id in legacy_map:
            plugin_id = legacy_map[plugin_id]
        plugin_id = resolve_alias(plugin_id)

        config = get_plugin_config(plugin_id)
        if not config:
            raise ValueError(
                f"未知插件: {getattr(request, 'plugin_id', '')} (解析后: {plugin_id})"
            )

        task_type = getattr(request, "task_type", "") or config.get(
            "primary_task", "detection"
        )
        valid_tasks = config.get("task_types", [])
        if task_type not in valid_tasks:
            raise ValueError(
                f"插件 {plugin_id} 不支持 task_type '{task_type}'，合法值: {valid_tasks}"
            )

        task = TrainTask(request)
        task.plugin_id = plugin_id
        task.task_type = task_type
        self._tasks[task.task_id] = task

        logger.info(f"创建训练任务: {task.task_id} ({plugin_id}/{task_type})")
        return task

    def get_task(self, task_id: str) -> TrainTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[dict]:
        return [t.to_dict() for t in self._tasks.values()]

    def cancel_task(self, task_id: str) -> bool:
        """取消训练任务"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            return False
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.now().isoformat()

        orchestrator = self._orchestrators.get(task_id)
        if orchestrator:
            orchestrator.cancel()

        return True

    def run_task_async(self, task_id: str) -> None:
        """在后台线程中运行训练流水线"""
        thread = threading.Thread(
            target=self._run_task_sync,
            args=(task_id,),
            daemon=True,
        )
        thread.start()

    def _run_task_sync(self, task_id: str) -> None:
        """
        同步执行训练流水线 (在后台线程调用)。
        通过 TaskOrchestrator 驱动全部 7 阶段。
        """
        from training.services.task_orchestrator import TaskOrchestrator, TaskEvent

        task = self._tasks.get(task_id)
        if not task:
            return

        dataset_path = getattr(task.request, "dataset_path", "")
        dataset_dir = Path(dataset_path) if dataset_path else None

        if not dataset_dir or not dataset_dir.exists():
            task.status = TaskStatus.FAILED
            task.error = f"数据集路径不存在: {dataset_path}"
            task.error_category = "data_missing"
            task.updated_at = datetime.now().isoformat()
            return

        # 构建配置覆盖
        config: dict[str, Any] = {}
        for key in (
            "epochs",
            "batch_size",
            "learning_rate",
            "backbone",
            "image_size",
            "paradigm",
            "pca_components",
            "model_type",
            "sequence_length",
            "window_size",
            "prediction_horizon",
            "sample_rate_hz",
            "feature_views",
            "sensor_columns",
            "event_types",
            "fusion_strategy",
            "supported_modalities",
            "required_modalities",
            "missing_modality_policy",
        ):
            val = getattr(task.request, key, None)
            if val is not None:
                config[key] = val

        # 进度回调 → 更新 task 状态
        def on_progress(event: TaskEvent):
            try:
                task.status = TaskStatus(event.phase.value)
            except ValueError:
                pass
            task.progress = event.progress
            task.current_phase = event.phase.value
            task.updated_at = datetime.now().isoformat()
            if event.message:
                logger.info(f"[Task {task_id}] {event.phase.value}: {event.message}")

        orchestrator = TaskOrchestrator(TRAINING_ROOT)
        self._orchestrators[task_id] = orchestrator

        try:
            result = orchestrator.run(dataset_dir, config, on_progress=on_progress)

            task.result = result.to_dict()
            task.updated_at = datetime.now().isoformat()

            if result.success:
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
            elif result.phase_reached.value == "cancelled":
                task.status = TaskStatus.CANCELLED
            else:
                task.status = TaskStatus.FAILED
                task.error = result.error_message
                task.error_category = (
                    result.error_category.value if result.error_category else ""
                )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.error_category = "unknown"
            task.updated_at = datetime.now().isoformat()
            logger.error(f"任务 {task_id} 异常: {e}", exc_info=True)

        finally:
            self._orchestrators.pop(task_id, None)


# ============================================================
# FastAPI 路由
# ============================================================

def create_v2_router() -> "APIRouter":
    """创建 v2 API 路由"""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI 未安装")

    router = APIRouter(prefix="/v2", tags=["training-v2"])
    manager = TrainingManagerV2()

    # ── 插件与任务类型 ─────────────────────────────────────

    @router.get("/plugins")
    async def list_plugins():
        """列出支持的训练插件"""
        plugins = list_supported_plugins()
        result = []
        for pid in plugins:
            config = get_plugin_config(pid)
            result.append({
                "plugin_id": pid,
                "display_name": config.get("display_name", pid) if config else pid,
                "task_types": config.get("task_types", []) if config else [],
                "primary_task": config.get("primary_task", "") if config else "",
                "input_modality": config.get("input_modality", "") if config else "",
            })
        return {"plugins": result, "count": len(result)}

    @router.get("/plugins/{plugin_id}")
    async def get_plugin_detail(plugin_id: str):
        """获取插件训练配置详情"""
        resolved = resolve_alias(plugin_id)
        config = get_plugin_config(resolved)
        if not config:
            raise HTTPException(404, f"未知插件: {plugin_id}")
        return config

    @router.get("/task-types")
    async def list_task_types():
        """列出支持的 task_type"""
        from training.registry import get_registry

        registry = get_registry()
        return {"task_types": registry.get("task_types", {})}

    # ── 数据集 ─────────────────────────────────────────────

    @router.post("/datasets/validate")
    async def validate_dataset(dataset_path: str):
        """校验数据集"""
        return manager.validate_dataset(dataset_path)

    @router.get("/datasets")
    async def list_datasets():
        """列出已有数据集"""
        result = []
        for storage_family in (
            "visual_defect",
            TEMPORAL_DATASET_FAMILY,
            MULTIMODAL_DATASET_FAMILY,
        ):
            datasets_dir = TRAINING_ROOT / "datasets" / storage_family
            if not datasets_dir.exists():
                continue
            for plugin_dir in sorted(datasets_dir.iterdir()):
                if plugin_dir.is_dir() and not plugin_dir.name.startswith((".", "_")):
                    for task_dir in sorted(plugin_dir.iterdir()):
                        if not task_dir.is_dir():
                            continue
                        version_dirs = [task_dir]
                        if storage_family == TEMPORAL_DATASET_FAMILY:
                            version_dirs = [
                                d for d in sorted(task_dir.iterdir()) if d.is_dir()
                            ] or [task_dir]
                        for dataset_dir in version_dirs:
                            manifest = dataset_dir / "manifest.json"
                            result.append({
                                "plugin_id": plugin_dir.name,
                                "task_type": task_dir.name,
                                "storage_family": storage_family,
                                "version": (
                                    dataset_dir.name
                                    if dataset_dir != task_dir
                                    else ""
                                ),
                                "has_manifest": manifest.exists(),
                                "path": str(dataset_dir),
                            })
        return {"datasets": result, "count": len(result)}

    # ── 训练任务 ───────────────────────────────────────────

    @router.post("/train")
    async def create_train_task(request: TrainRequest):
        """创建并启动训练任务"""
        try:
            task = manager.create_task(request)
            manager.run_task_async(task.task_id)
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "plugin_id": task.plugin_id,
                "task_type": task.task_type,
                "message": "训练任务已创建并启动",
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @router.get("/train/{task_id}")
    async def get_train_status(task_id: str):
        """查询训练状态"""
        task = manager.get_task(task_id)
        if not task:
            raise HTTPException(404, f"任务不存在: {task_id}")
        return task.to_dict()

    @router.post("/train/{task_id}/cancel")
    async def cancel_train_task(task_id: str):
        """取消训练任务"""
        success = manager.cancel_task(task_id)
        if not success:
            raise HTTPException(
                400, f"无法取消任务 {task_id} (可能已完成/已取消/不存在)"
            )
        return {"task_id": task_id, "status": "cancelled", "message": "训练已取消"}

    @router.get("/train")
    async def list_train_tasks():
        """列出所有训练任务"""
        return {"tasks": manager.list_tasks()}

    # ── 系统能力 ───────────────────────────────────────────

    @router.get("/capabilities")
    async def system_capabilities():
        """返回 v2 系统能力概览"""
        from training.registry import get_registry

        registry = get_registry()
        plugins = list_supported_plugins()

        deps = {}
        for name in ("ultralytics", "torch", "torchvision", "onnx", "PIL"):
            try:
                __import__(name)
                deps[name] = True
            except ImportError:
                deps[name] = False

        return {
            "api_version": "v2",
            "lifecycle_stages": [s.value for s in TaskStatus],
            "supported_plugins": len(plugins),
            "supported_task_types": len(registry.get("task_types", {})),
            "dependencies": deps,
            "training_root": str(TRAINING_ROOT),
        }

    return router
