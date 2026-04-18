"""
预处理基类

所有 task_type 的预处理器继承此基类，实现统一接口。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """预处理结果"""

    output_dir: Path
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    success: bool = True
    message: str = ""


class BasePreprocessor(ABC):
    """预处理器基类"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.image_size: int = self.config.get("image_size", 640)
        self.split_ratio: dict[str, float] = self.config.get(
            "split_ratio", {"train": 0.8, "val": 0.15, "test": 0.05}
        )

    @abstractmethod
    def preprocess(self, input_dir: Path, output_dir: Path) -> PreprocessResult:
        """
        执行预处理

        Args:
            input_dir:  原始数据目录 (含 images/ 和 labels/)
            output_dir: 预处理后的输出目录

        Returns:
            PreprocessResult
        """
        ...

    def split_dataset(self, image_files: list[Path], output_dir: Path) -> dict[str, list[Path]]:
        """
        按比例分割数据集

        Returns:
            {"train": [...], "val": [...], "test": [...]}
        """
        import random

        random.shuffle(image_files)
        n = len(image_files)
        n_train = int(n * self.split_ratio["train"])
        n_val = int(n * self.split_ratio["val"])

        splits = {
            "train": image_files[:n_train],
            "val": image_files[n_train : n_train + n_val],
            "test": image_files[n_train + n_val :],
        }

        for split_name, files in splits.items():
            split_dir = output_dir / split_name
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
            logger.info(f"  {split_name}: {len(files)} 张")

        return splits

    def generate_yaml_config(self, output_dir: Path, classes: list[str], dataset_name: str) -> Path:
        """生成 YOLO 格式的数据集 YAML 配置"""
        yaml_path = output_dir / f"{dataset_name}.yaml"
        lines = [
            f"# 自动生成的训练数据集配置 - {dataset_name}",
            f"path: {output_dir}",
            f"train: train/images",
            f"val: val/images",
            f"test: test/images",
            "",
            f"nc: {len(classes)}",
            f"names: {classes}",
        ]
        yaml_path.write_text("\n".join(lines), encoding="utf-8")
        return yaml_path
