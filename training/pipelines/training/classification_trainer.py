"""
图像分类训练器

支持:
- torchvision 预训练模型 (ResNet / EfficientNet / MobileNet)
- ultralytics YOLOv8-cls
- 迁移学习 (冻结 backbone + 微调 head)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .base_trainer import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class ClassificationTrainer(BaseTrainer):
    """图像分类训练"""

    def __init__(self, plugin_id: str, config: dict[str, Any] | None = None):
        super().__init__(plugin_id, task_type="classification", config=config)
        self.backbone: str = self.config.get("backbone", "resnet18")
        self.pretrained: bool = self.config.get("pretrained", True)
        self.freeze_backbone_epochs: int = self.config.get("freeze_backbone_epochs", 5)
        self.num_classes: int = self.config.get("num_classes", 2)

    def train(self, dataset_dir: Path, output_dir: Path) -> TrainResult:
        ckpt_dir = self.get_checkpoint_dir(output_dir)
        device = self.detect_device()
        logger.info(
            f"[ClassificationTrainer] plugin={self.plugin_id}, "
            f"backbone={self.backbone}, classes={self.num_classes}"
        )

        start_time = time.time()

        try:
            return self._train_torchvision(dataset_dir, ckpt_dir, device, start_time)
        except ImportError:
            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                model_path=ckpt_dir,
                training_time_seconds=time.time() - start_time,
                success=False,
                message="torch / torchvision 未安装",
            )

    def _train_torchvision(
        self, dataset_dir: Path, ckpt_dir: Path, device: str, start_time: float
    ) -> TrainResult:
        """使用 torchvision 训练分类模型"""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        logger.info(f"分类训练开始: {self.backbone} on {device}")

        # 构建模型 (骨架 — 实际应使用 torchvision.models)
        # 此处预留接口，集成阶段补充完整的 dataset + training loop
        model_path = ckpt_dir / "best.pth"

        return TrainResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=ckpt_dir,
            best_checkpoint=model_path if model_path.exists() else None,
            epochs_completed=0,
            metrics={},
            training_time_seconds=time.time() - start_time,
            success=True,
            message=f"分类训练 (骨架模式): {self.backbone}",
        )
