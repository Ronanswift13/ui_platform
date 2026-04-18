"""
训练器基类

所有 task_type 的训练器继承此基类。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """训练结果"""

    plugin_id: str = ""
    task_type: str = ""
    model_path: Path | None = None
    best_checkpoint: Path | None = None

    # 训练指标
    epochs_completed: int = 0
    best_epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0

    success: bool = True
    message: str = ""


class BaseTrainer(ABC):
    """训练器基类"""

    def __init__(
        self,
        plugin_id: str,
        task_type: str,
        config: dict[str, Any] | None = None,
    ):
        self.plugin_id = plugin_id
        self.task_type = task_type
        self.config = config or {}

        # 通用超参
        self.epochs: int = self.config.get("epochs", 100)
        self.batch_size: int = self.config.get("batch_size", 16)
        self.learning_rate: float = self.config.get("learning_rate", 0.001)
        self.image_size: int = self.config.get("image_size", 640)
        self.device: str = self.config.get("device", "auto")

    def detect_device(self) -> str:
        """自动检测最优计算设备"""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @abstractmethod
    def train(self, dataset_dir: Path, output_dir: Path) -> TrainResult:
        """
        执行训练

        Args:
            dataset_dir:  预处理后的数据集目录
            output_dir:   检查点 / 模型输出目录

        Returns:
            TrainResult
        """
        ...

    def get_checkpoint_dir(self, output_dir: Path) -> Path:
        """获取检查点保存目录"""
        ckpt_dir = output_dir / "checkpoints" / self.plugin_id / self.task_type
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir
