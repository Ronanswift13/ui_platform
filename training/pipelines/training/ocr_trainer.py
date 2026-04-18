"""
OCR / 读数识别训练器

专为 meter_reading 插件设计，支持:
- CRNN (CNN + BiLSTM + CTC) 文字序列识别
- 关键点检测 (HRNet) 用于表盘定位
- 端到端读数回归

训练流程:
1. 表盘区域检测 (可选，使用 DetectionTrainer)
2. 透视矫正后的显示区域 → CRNN 序列识别
3. CTC loss 优化
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .base_trainer import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class OCRTrainer(BaseTrainer):
    """OCR 读数训练"""

    def __init__(self, plugin_id: str, config: dict[str, Any] | None = None):
        super().__init__(plugin_id, task_type="ocr", config=config)
        self.backbone: str = self.config.get("backbone", "resnet34_lstm")
        self.input_height: int = self.config.get("input_height", 64)
        self.input_width: int = self.config.get("input_width", 200)
        self.max_label_length: int = self.config.get("max_label_length", 10)
        self.charset: str = self.config.get("charset", "0123456789.-")

    def train(self, dataset_dir: Path, output_dir: Path) -> TrainResult:
        ckpt_dir = self.get_checkpoint_dir(output_dir)
        device = self.detect_device()
        logger.info(
            f"[OCRTrainer] plugin={self.plugin_id}, "
            f"backbone={self.backbone}, charset='{self.charset}'"
        )

        start_time = time.time()

        try:
            import torch
            return self._train_crnn(dataset_dir, ckpt_dir, device, start_time)
        except ImportError:
            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                model_path=ckpt_dir,
                training_time_seconds=time.time() - start_time,
                success=False,
                message="PyTorch 未安装",
            )

    def _train_crnn(
        self, dataset_dir: Path, ckpt_dir: Path, device: str, start_time: float
    ) -> TrainResult:
        """
        CRNN 训练骨架

        完整实现需要:
        1. 构建 CRNN 模型 (ResNet34 encoder + BiLSTM + Linear)
        2. CTC loss
        3. 从 labels.json 加载 (image, reading) 对
        4. 训练循环 + 验证 + checkpoint
        """
        import torch

        logger.info(f"CRNN 训练: input={self.input_height}×{self.input_width}, device={device}")

        model_path = ckpt_dir / "best_crnn.pth"

        return TrainResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=ckpt_dir,
            best_checkpoint=model_path if model_path.exists() else None,
            epochs_completed=0,
            metrics={},
            training_time_seconds=time.time() - start_time,
            success=True,
            message=f"OCR CRNN 训练 (骨架模式): {self.backbone}, charset='{self.charset}'",
        )
