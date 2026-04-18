"""
训练任务编排器

将 ingestion → preprocessing → training → evaluation → export → registration
串联为完整闭环，支持进度回调与结构化错误分类。

使用方式:
    orchestrator = TaskOrchestrator()
    result = orchestrator.run(dataset_dir, config)

设计原则:
- 不直接实现具体算法，仅负责阶段调度与生命周期管理
- 通过 DataRouter.ROUTE_TABLE 动态加载 Preprocessor / Trainer
- 每个阶段的输入输出均为确定性数据结构
"""

from __future__ import annotations

import importlib
import json
import logging
import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

TRAINING_ROOT = Path(__file__).parent.parent


# ============================================================
# 生命周期阶段与事件
# ============================================================

class TaskPhase(str, Enum):
    """训练任务的完整生命周期阶段"""
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


class ErrorCategory(str, Enum):
    """结构化错误分类"""
    DATA_MISSING = "data_missing"
    DATA_INVALID = "data_invalid"
    MANIFEST_ERROR = "manifest_error"
    PREPROCESSING_ERROR = "preprocessing_error"
    TRAINING_DIVERGED = "training_diverged"
    TRAINING_OOM = "training_oom"
    TRAINING_RUNTIME = "training_runtime"
    EVALUATION_ERROR = "evaluation_error"
    EXPORT_ERROR = "export_error"
    REGISTRATION_ERROR = "registration_error"
    DEPENDENCY_MISSING = "dependency_missing"
    UNKNOWN = "unknown"


@dataclass
class TaskEvent:
    """阶段事件 (用于进度回调)"""
    phase: TaskPhase
    progress: float  # 0.0 ~ 1.0
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    """编排器执行结果"""
    success: bool = False
    phase_reached: TaskPhase = TaskPhase.CREATED
    error_category: Optional[ErrorCategory] = None
    error_message: str = ""
    error_traceback: str = ""
    total_time_seconds: float = 0.0

    # 各阶段产物
    validation_report: dict[str, Any] = field(default_factory=dict)
    ingestion_result: dict[str, Any] = field(default_factory=dict)
    preprocess_result: dict[str, Any] = field(default_factory=dict)
    train_result: dict[str, Any] = field(default_factory=dict)
    eval_result: dict[str, Any] = field(default_factory=dict)
    export_result: dict[str, Any] = field(default_factory=dict)
    registration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phase_reached"] = self.phase_reached.value
        if self.error_category:
            d["error_category"] = self.error_category.value
        return d


# ============================================================
# 动态模块加载
# ============================================================

def _import_class(module_path: str) -> type:
    """
    从字符串路径加载类。
    例如: "pipelines.preprocessing.detection_preprocessor.DetectionPreprocessor"
    → 从 training/pipelines/preprocessing/detection_preprocessor.py 导入 DetectionPreprocessor
    """
    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"无效的模块路径: {module_path}")

    module_dotpath, class_name = parts
    # 转换为 training.pipelines... 前缀
    full_module = f"training.{module_dotpath}"
    try:
        mod = importlib.import_module(full_module)
    except ModuleNotFoundError:
        # 尝试直接导入
        mod = importlib.import_module(module_dotpath)
    return getattr(mod, class_name)


# ============================================================
# 核心编排器
# ============================================================

class TaskOrchestrator:
    """
    端到端训练任务编排器

    生命周期:
      created → validating → ingesting → preprocessing
      → training → evaluating → exporting → registering → completed

    任何阶段失败均进入 failed，可通过 error_category 定位根因。
    """

    def __init__(self, training_root: Path | None = None):
        self.training_root = training_root or TRAINING_ROOT
        self._cancelled = False

    def cancel(self):
        """标记取消 (异步安全)"""
        self._cancelled = True

    def run(
        self,
        dataset_dir: Path,
        config: dict[str, Any],
        on_progress: Callable[[TaskEvent], None] | None = None,
    ) -> OrchestratorResult:
        """
        执行完整训练流水线。

        Args:
            dataset_dir: 数据集根目录 (含 manifest.json)
            config: 训练配置覆盖 (epochs, batch_size, learning_rate 等)
            on_progress: 进度回调

        Returns:
            OrchestratorResult
        """
        self._cancelled = False
        result = OrchestratorResult()
        start = time.time()

        def emit(phase: TaskPhase, progress: float, msg: str = "", **kw):
            result.phase_reached = phase
            if on_progress:
                on_progress(TaskEvent(phase=phase, progress=progress, message=msg, detail=kw))

        try:
            # ── 阶段 1: 校验 ──────────────────────────────────
            emit(TaskPhase.VALIDATING, 0.0, "开始数据校验")
            manifest, validation = self._phase_validate(dataset_dir)
            result.validation_report = validation
            if not validation.get("valid", False):
                result.error_category = ErrorCategory.DATA_INVALID
                result.error_message = validation.get("summary", "数据校验失败")
                result.phase_reached = TaskPhase.FAILED
                return result
            emit(TaskPhase.VALIDATING, 1.0, "数据校验通过")

            if self._cancelled:
                result.phase_reached = TaskPhase.CANCELLED
                return result

            # ── 阶段 2: 数据摄入 (路由 + 复制) ────────────────
            emit(TaskPhase.INGESTING, 0.0, "开始数据摄入")
            route_result = self._phase_ingest(dataset_dir, manifest)
            result.ingestion_result = {
                "plugin_id": route_result.plugin_id,
                "task_type": route_result.task_type,
                "destination": str(route_result.destination),
                "storage_family": route_result.storage_family,
                "preprocessor": route_result.preprocessor,
                "trainer": route_result.trainer,
            }
            if not route_result.success:
                result.error_category = ErrorCategory.DATA_INVALID
                result.error_message = route_result.message
                result.phase_reached = TaskPhase.FAILED
                return result
            emit(TaskPhase.INGESTING, 1.0, route_result.message)

            if self._cancelled:
                result.phase_reached = TaskPhase.CANCELLED
                return result

            plugin_id = route_result.plugin_id
            task_type = route_result.task_type

            # ── 阶段 3: 预处理 ────────────────────────────────
            emit(TaskPhase.PREPROCESSING, 0.0, "开始数据预处理")
            preprocess_out = self._phase_preprocess(
                route_result, config
            )
            result.preprocess_result = {
                "output_dir": str(preprocess_out.output_dir),
                "num_train": preprocess_out.num_train,
                "num_val": preprocess_out.num_val,
                "num_test": preprocess_out.num_test,
                "class_distribution": preprocess_out.class_distribution,
                "success": preprocess_out.success,
                "message": preprocess_out.message,
            }
            if not preprocess_out.success:
                result.error_category = ErrorCategory.PREPROCESSING_ERROR
                result.error_message = preprocess_out.message
                result.phase_reached = TaskPhase.FAILED
                return result
            emit(
                TaskPhase.PREPROCESSING,
                1.0,
                f"预处理完成: train={preprocess_out.num_train}, "
                f"val={preprocess_out.num_val}, test={preprocess_out.num_test}",
            )

            if self._cancelled:
                result.phase_reached = TaskPhase.CANCELLED
                return result

            # ── 阶段 4: 训练 ──────────────────────────────────
            emit(TaskPhase.TRAINING, 0.0, "开始模型训练")
            train_out = self._phase_train(
                route_result, preprocess_out.output_dir, config
            )
            result.train_result = {
                "plugin_id": train_out.plugin_id,
                "task_type": train_out.task_type,
                "model_path": str(train_out.model_path) if train_out.model_path else "",
                "best_checkpoint": str(train_out.best_checkpoint) if train_out.best_checkpoint else "",
                "epochs_completed": train_out.epochs_completed,
                "metrics": train_out.metrics,
                "training_time_seconds": train_out.training_time_seconds,
                "success": train_out.success,
                "message": train_out.message,
            }
            if not train_out.success:
                result.error_category = self._classify_train_error(train_out.message)
                result.error_message = train_out.message
                result.phase_reached = TaskPhase.FAILED
                return result
            emit(TaskPhase.TRAINING, 1.0, train_out.message)

            if self._cancelled:
                result.phase_reached = TaskPhase.CANCELLED
                return result

            # ── 阶段 5: 评估 ──────────────────────────────────
            emit(TaskPhase.EVALUATING, 0.0, "开始模型评估")
            eval_out = self._phase_evaluate(
                plugin_id, task_type, train_out, preprocess_out.output_dir
            )
            result.eval_result = {
                "metrics": eval_out.metrics,
                "per_class_metrics": eval_out.per_class_metrics,
                "qualified": eval_out.qualified,
                "qualification_reason": eval_out.qualification_reason,
                "eval_time_seconds": eval_out.eval_time_seconds,
            }
            emit(
                TaskPhase.EVALUATING,
                1.0,
                f"评估完成: qualified={eval_out.qualified}, "
                f"reason={eval_out.qualification_reason}",
            )

            if self._cancelled:
                result.phase_reached = TaskPhase.CANCELLED
                return result

            # ── 阶段 6: 导出 ──────────────────────────────────
            emit(TaskPhase.EXPORTING, 0.0, "开始模型导出")
            export_out = self._phase_export(
                plugin_id, task_type, train_out, eval_out.metrics, manifest
            )
            result.export_result = {
                "format": export_out.format,
                "output_path": str(export_out.output_path),
                "file_size_mb": export_out.file_size_mb,
                "sha256": export_out.sha256,
                "success": export_out.success,
                "message": export_out.message,
            }
            if not export_out.success:
                # 导出失败不阻塞流程，降级警告
                logger.warning(f"模型导出失败 (非致命): {export_out.message}")
            emit(TaskPhase.EXPORTING, 1.0, export_out.message)

            # ── 阶段 7: 注册 ──────────────────────────────────
            emit(TaskPhase.REGISTERING, 0.0, "注册模型到插件")
            reg = self._phase_register(
                plugin_id, task_type, train_out, eval_out, export_out
            )
            result.registration = reg
            emit(TaskPhase.REGISTERING, 1.0, "模型注册完成")

            # ── 完成 ──────────────────────────────────────────
            result.success = True
            result.phase_reached = TaskPhase.COMPLETED
            emit(TaskPhase.COMPLETED, 1.0, "训练流水线全部完成")

        except Exception as e:
            result.success = False
            result.phase_reached = TaskPhase.FAILED
            result.error_category = ErrorCategory.UNKNOWN
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            logger.error(f"编排器异常: {e}", exc_info=True)

        finally:
            result.total_time_seconds = time.time() - start
            logger.info(
                f"编排完成: phase={result.phase_reached.value}, "
                f"success={result.success}, time={result.total_time_seconds:.1f}s"
            )

        return result

    # ================================================================
    # 各阶段实现
    # ================================================================

    def _phase_validate(self, dataset_dir: Path):
        """阶段1: manifest 校验"""
        from training.pipelines.ingestion import ManifestValidator
        from training.schemas.dataset_manifest import DatasetManifest

        validator = ManifestValidator()
        manifest_path = dataset_dir / "manifest.json"

        # 如果不存在 manifest.json，尝试从目录结构推断
        if not manifest_path.exists():
            inferred = self._infer_manifest(dataset_dir)
            if inferred:
                inferred.save(manifest_path)
                logger.info(f"已从目录结构推断 manifest: {manifest_path}")
            else:
                return None, {
                    "valid": False,
                    "summary": f"manifest.json 不存在且无法推断: {dataset_dir}",
                }

        report = validator.validate(dataset_dir)
        manifest = DatasetManifest.from_json(manifest_path)

        return manifest, {
            "valid": report.valid,
            "summary": report.summary(),
            "stats": report.stats,
        }

    def _infer_manifest(self, dataset_dir: Path):
        """从目录结构推断 manifest (YOLO 格式兜底)"""
        from training.schemas.dataset_manifest import DatasetManifest

        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        if not images_dir.exists():
            # 也检查 train/images 结构
            for sub in ("train", "val"):
                candidate = dataset_dir / sub / "images"
                if candidate.exists():
                    images_dir = candidate
                    break

        if not images_dir.exists():
            return None

        # 扫描 yaml 获取类别
        classes = []
        for yaml_path in dataset_dir.glob("*.yaml"):
            try:
                import yaml

                with open(yaml_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict) and "names" in cfg:
                    classes = cfg["names"]
                    break
            except Exception:
                pass

        # 统计图片数
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        num_images = sum(
            1
            for f in images_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in img_exts
        )

        if num_images == 0:
            return None

        return DatasetManifest(
            name=dataset_dir.name,
            plugin_id="unknown",
            task_type="detection",
            version="v1.0",
            num_images=num_images,
            classes=classes if classes else ["defect", "normal"],
            format="yolo",
        )

    def _phase_ingest(self, dataset_dir: Path, manifest):
        """阶段2: 数据路由与摄入"""
        from training.pipelines.ingestion import DataRouter

        router = DataRouter(self.training_root)
        return router.ingest(dataset_dir, manifest, copy=True)

    def _phase_preprocess(self, route_result, config: dict):
        """阶段3: 动态加载预处理器并执行"""
        PreprocessorClass = _import_class(route_result.preprocessor)
        preprocessor = PreprocessorClass(config=config)

        # 输出到 processed 目录
        output_dir = (
            self.training_root
            / "data"
            / "processed"
            / route_result.plugin_id
            / route_result.task_type
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        return preprocessor.preprocess(route_result.destination, output_dir)

    def _phase_train(self, route_result, preprocessed_dir: Path, config: dict):
        """阶段4: 动态加载训练器并执行"""
        TrainerClass = _import_class(route_result.trainer)

        # 合并配置: 注册表默认值 + 用户覆盖
        from training.registry import get_plugin_config

        plugin_config = get_plugin_config(route_result.plugin_id) or {}
        merged = {**plugin_config, **config}

        trainer = TrainerClass(route_result.plugin_id, config=merged)

        output_dir = self.training_root / "checkpoints"
        return trainer.train(preprocessed_dir, output_dir)

    def _phase_evaluate(self, plugin_id, task_type, train_result, test_data_dir):
        """阶段5: 模型评估"""
        from training.pipelines.evaluation import BaseEvaluator

        evaluator = BaseEvaluator(plugin_id, task_type)

        model_path = train_result.best_checkpoint or train_result.model_path
        if model_path is None:
            model_path = Path(".")

        # 测试数据位于 preprocessed_dir/test/
        test_dir = test_data_dir / "test"
        if not test_dir.exists():
            test_dir = test_data_dir

        return evaluator.evaluate(
            model_path=model_path,
            test_data_dir=test_dir,
            metrics_from_training=train_result.metrics,
        )

    def _phase_export(self, plugin_id, task_type, train_result, metrics, manifest):
        """阶段6: 模型导出"""
        from training.pipelines.export import ModelExporter

        exporter = ModelExporter(self.training_root / "exports")

        model_path = train_result.best_checkpoint or train_result.model_path
        if model_path is None or not Path(model_path).exists():
            from training.pipelines.export.model_exporter import ExportResult

            return ExportResult(
                plugin_id=plugin_id,
                task_type=task_type,
                format="none",
                output_path=Path("."),
                success=False,
                message="无可导出的模型文件",
            )

        model_path = Path(model_path)

        # 尝试 ONNX 导出
        if model_path.suffix in (".pt", ".pth"):
            return exporter.export_onnx(
                model_path=model_path,
                plugin_id=plugin_id,
                task_type=task_type,
            )

        # 对于非 PyTorch 格式，使用 bundle 导出
        version = manifest.version if manifest else "v1.0"
        return exporter.export_bundle(
            plugin_id=plugin_id,
            task_type=task_type,
            version=version,
            model_path=model_path,
            modality=manifest.input_modality if manifest else "visual",
            metrics=metrics,
            labels=manifest.classes if manifest else [],
        )

    def _phase_register(self, plugin_id, task_type, train_result, eval_result, export_result):
        """阶段7: 将训练结果注册到模型注册表"""
        registry_path = self.training_root / "results" / "model_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载现有注册表
        registry = {}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if "models" not in registry:
            registry["models"] = []

        entry = {
            "plugin_id": plugin_id,
            "task_type": task_type,
            "model_path": str(train_result.best_checkpoint or train_result.model_path or ""),
            "export_path": str(export_result.output_path) if export_result.success else "",
            "metrics": train_result.metrics,
            "eval_metrics": eval_result.metrics,
            "qualified": eval_result.qualified,
            "epochs": train_result.epochs_completed,
            "training_time_s": train_result.training_time_seconds,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        registry["models"].append(entry)
        registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2, default=str)

        logger.info(
            f"模型已注册: {plugin_id}/{task_type}, "
            f"qualified={eval_result.qualified}"
        )
        return entry

    # ================================================================
    # 错误分类
    # ================================================================

    @staticmethod
    def _classify_train_error(message: str) -> ErrorCategory:
        """根据训练错误消息分类"""
        msg = message.lower()
        if "oom" in msg or "out of memory" in msg or "cuda out" in msg:
            return ErrorCategory.TRAINING_OOM
        if "nan" in msg or "diverge" in msg or "inf" in msg:
            return ErrorCategory.TRAINING_DIVERGED
        if "not found" in msg or "不存在" in msg or "missing" in msg:
            return ErrorCategory.DATA_MISSING
        if "import" in msg or "未安装" in msg or "not installed" in msg:
            return ErrorCategory.DEPENDENCY_MISSING
        return ErrorCategory.TRAINING_RUNTIME
