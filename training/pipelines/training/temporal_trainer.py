"""
时序异常训练器

当前版本以工程交付为目标:
- 固化任务画像与训练计划
- 产出检查点目录与元数据
- 为后续真实训练框架接入保留统一入口
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from training.schemas.dataset_manifest import DatasetManifest
from training.temporal_anomaly import get_temporal_task_profile

from .base_trainer import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class TemporalTrainer(BaseTrainer):
    """时序异常统一训练器"""

    def __init__(
        self,
        plugin_id: str,
        task_type: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(plugin_id, task_type=task_type, config=config)
        self.paradigm = str(self.config.get("paradigm", "")).strip()

    def train(self, dataset_dir: Path, output_dir: Path) -> TrainResult:
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                success=False,
                message=f"manifest.json 不存在: {manifest_path}",
            )

        manifest = DatasetManifest.from_json(manifest_path)
        profile = get_temporal_task_profile(self.task_type)
        if profile is None:
            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                success=False,
                message=f"未知时序 task_type: {self.task_type}",
            )

        ckpt_dir = self.get_checkpoint_dir(output_dir)
        best_checkpoint = ckpt_dir / "best.ckpt"
        model_path = ckpt_dir / "model.pt"
        training_plan_path = ckpt_dir / "training_plan.json"

        effective_paradigm = self.paradigm or profile.default_paradigm
        training_plan = {
            "plugin_id": self.plugin_id,
            "task_type": self.task_type,
            "paradigm": effective_paradigm,
            "model_family": self.config.get("model_family", profile.default_model_family),
            "supported_paradigms": list(profile.supported_paradigms),
            "input_modality": manifest.input_modality,
            "feature_views": manifest.feature_views,
            "sensor_columns": manifest.sensor_columns,
            "event_types": manifest.event_types,
            "target_schema": manifest.target_schema,
            "output_heads": [
                {
                    "name": head.name,
                    "type": head.head_type,
                    "description": head.description,
                    "range_hint": head.range_hint,
                }
                for head in profile.output_heads
            ],
            "training_hparams": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "sequence_length": self.config.get("sequence_length", manifest.sequence_length),
                "window_size": self.config.get("window_size", manifest.window_size),
                "prediction_horizon": self.config.get(
                    "prediction_horizon", manifest.prediction_horizon
                ),
            },
            "evaluation_metrics": list(profile.evaluation_metrics),
        }

        with open(training_plan_path, "w", encoding="utf-8") as f:
            json.dump(training_plan, f, ensure_ascii=False, indent=2)

        # 当前阶段先写出占位检查点，保证工程链路可执行。
        best_checkpoint.write_text("temporal-checkpoint-placeholder\n", encoding="utf-8")
        model_path.write_text("temporal-model-placeholder\n", encoding="utf-8")

        metrics = self._build_placeholder_metrics(profile.evaluation_metrics)
        return TrainResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=model_path,
            best_checkpoint=best_checkpoint,
            epochs_completed=self.epochs,
            best_epoch=max(1, min(self.epochs, 1)),
            metrics=metrics,
            training_time_seconds=time.time() - start_time,
            success=True,
            message=f"时序训练计划已生成: {effective_paradigm}",
        )

    def _build_placeholder_metrics(self, metric_names: tuple[str, ...]) -> dict[str, float]:
        metric_defaults = {
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "f1": 0.0,
            "false_alarm_rate": 1.0,
            "latency_ms": 0.0,
            "health_mae": 0.0,
            "health_rmse": 0.0,
            "anomaly_auc": 0.0,
            "failure_f1": 0.0,
            "failure_recall": 0.0,
            "lead_time_hours": 0.0,
            "sequence_accuracy": 0.0,
            "macro_f1": 0.0,
            "event_recall": 0.0,
            "edit_distance": 0.0,
            "false_positive_rate": 1.0,
            "lead_time_minutes": 0.0,
        }
        return {name: metric_defaults.get(name, 0.0) for name in metric_names}
