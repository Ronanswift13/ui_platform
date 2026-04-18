"""
多模态融合训练器

第一阶段目标:
- 固化特征级 / 决策级训练计划
- 记录模态缺失补偿策略
- 生成可导出的训练元数据
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from training.multimodal_fusion import get_multimodal_task_profile
from training.schemas.dataset_manifest import DatasetManifest

from .base_trainer import BaseTrainer, TrainResult


class MultimodalTrainer(BaseTrainer):
    """多模态融合训练器"""

    def __init__(
        self,
        plugin_id: str,
        task_type: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(plugin_id, task_type=task_type, config=config)
        self.fusion_strategy = str(self.config.get("fusion_strategy", "hybrid"))
        self.runtime_mode = str(self.config.get("runtime_mode", "rule_model_hybrid"))
        self.missing_modality_policy = str(
            self.config.get("missing_modality_policy", "mask_and_gate")
        )

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
        profile = get_multimodal_task_profile(self.task_type)
        if profile is None:
            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                success=False,
                message=f"未知多模态 task_type: {self.task_type}",
            )

        ckpt_dir = self.get_checkpoint_dir(output_dir)
        best_checkpoint = ckpt_dir / "best.ckpt"
        model_path = ckpt_dir / "model.pt"
        plan_path = ckpt_dir / "multimodal_training_plan.json"

        plan = {
            "plugin_id": self.plugin_id,
            "task_type": self.task_type,
            "fusion_strategy": manifest.fusion_strategy or profile.fusion_strategy,
            "runtime_mode": self.runtime_mode,
            "missing_modality_policy": manifest.missing_modality_policy
            or self.missing_modality_policy,
            "supported_modalities": manifest.supported_modalities,
            "required_modalities": manifest.required_modalities,
            "diagnosis_target_schema": manifest.diagnosis_target_schema,
            "diagnostic_rule_pack": manifest.diagnostic_rule_pack,
            "model_family": self.config.get("model_family", profile.default_model_family),
            "training_paradigm": self.config.get(
                "paradigm",
                profile.supported_training_paradigms[0]
                if profile.supported_training_paradigms
                else "",
            ),
            "export_outputs": [head.name for head in profile.output_heads],
            "evaluation_metrics": list(profile.evaluation_metrics),
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        best_checkpoint.write_text("multimodal-checkpoint-placeholder\n", encoding="utf-8")
        model_path.write_text("multimodal-model-placeholder\n", encoding="utf-8")

        metrics = {name: 0.0 for name in profile.evaluation_metrics}
        return TrainResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=model_path,
            best_checkpoint=best_checkpoint,
            epochs_completed=self.epochs,
            best_epoch=1 if self.epochs else 0,
            metrics=metrics,
            training_time_seconds=time.time() - start_time,
            success=True,
            message=f"多模态训练计划已生成: {plan['fusion_strategy']}",
        )
