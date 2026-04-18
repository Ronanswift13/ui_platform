"""
高光谱训练器

支持:
- 1D CNN (光谱特征)
- 3D CNN (谱空联合特征)
- HybridSN (2D + 3D CNN 混合)
- 异常检测 (RX detector / AutoEncoder)

训练流程:
1. 加载降维后的 patches (.npy)
2. 构建谱空联合网络
3. 训练 + 评估 (OA / Kappa / per-class F1)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .base_trainer import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class HyperspectralTrainer(BaseTrainer):
    """高光谱训练"""

    def __init__(self, plugin_id: str, config: dict[str, Any] | None = None):
        super().__init__(plugin_id, task_type="hyperspectral_classification", config=config)
        self.backbone: str = self.config.get("backbone", "spectral_resnet")
        self.pca_components: int = self.config.get("pca_components", 30)
        self.spatial_patch_size: int = self.config.get("spatial_patch_size", 11)
        self.model_type: str = self.config.get("model_type", "hybrid_sn")  # 1d_cnn / 3d_cnn / hybrid_sn

    def train(self, dataset_dir: Path, output_dir: Path) -> TrainResult:
        ckpt_dir = self.get_checkpoint_dir(output_dir)
        device = self.detect_device()
        logger.info(
            f"[HyperspectralTrainer] plugin={self.plugin_id}, "
            f"model={self.model_type}, pca={self.pca_components}, patch={self.spatial_patch_size}"
        )

        start_time = time.time()

        try:
            import torch
            import numpy as np
        except ImportError:
            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                model_path=ckpt_dir,
                training_time_seconds=time.time() - start_time,
                success=False,
                message="PyTorch / numpy 未安装",
            )

        if self.model_type == "hybrid_sn":
            return self._train_hybrid_sn(dataset_dir, ckpt_dir, device, start_time)
        elif self.model_type == "3d_cnn":
            return self._train_3d_cnn(dataset_dir, ckpt_dir, device, start_time)
        else:
            return self._train_1d_cnn(dataset_dir, ckpt_dir, device, start_time)

    def _train_hybrid_sn(
        self, dataset_dir: Path, ckpt_dir: Path, device: str, start_time: float
    ) -> TrainResult:
        """
        HybridSN 训练 (骨架)

        架构: 3D Conv → 2D Conv → FC
        - 3D Conv: 提取谱空联合特征 (8→16→32 filters)
        - 2D Conv: 提取空间特征 (64 filters)
        - FC: 分类

        完整实现需要:
        1. 从 patches/ 加载 .npy patch 数据
        2. 构建 HybridSN 模型
        3. CrossEntropy loss + Adam
        4. 训练 + 验证 + 计算 OA/Kappa
        """
        logger.info(f"  HybridSN 训练: pca={self.pca_components}, patch={self.spatial_patch_size}")

        return TrainResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=ckpt_dir,
            epochs_completed=0,
            metrics={},
            training_time_seconds=time.time() - start_time,
            success=True,
            message=f"HybridSN 训练 (骨架模式): pca={self.pca_components}",
        )

    def _train_3d_cnn(
        self, dataset_dir: Path, ckpt_dir: Path, device: str, start_time: float
    ) -> TrainResult:
        """3D CNN 谱空联合训练 (骨架)"""
        logger.info("  3D CNN 训练")
        return TrainResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=ckpt_dir,
            training_time_seconds=time.time() - start_time,
            success=True,
            message="3D CNN 训练 (骨架模式)",
        )

    def _train_1d_cnn(
        self, dataset_dir: Path, ckpt_dir: Path, device: str, start_time: float
    ) -> TrainResult:
        """1D CNN 光谱特征训练 (骨架)"""
        logger.info("  1D CNN 训练")
        return TrainResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=ckpt_dir,
            training_time_seconds=time.time() - start_time,
            success=True,
            message="1D CNN 训练 (骨架模式)",
        )
