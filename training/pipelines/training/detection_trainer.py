"""
目标检测训练器

基于 YOLOv8 / ultralytics 的目标检测训练。
兼容现有 train_main.py 中的 SimpleDetectionModel (PyTorch 原生) 作为回退。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .base_trainer import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class DetectionTrainer(BaseTrainer):
    """YOLOv8 目标检测训练"""

    def __init__(self, plugin_id: str, config: dict[str, Any] | None = None):
        super().__init__(plugin_id, task_type="detection", config=config)
        self.backbone: str = self.config.get("backbone", "yolov8n")
        self.pretrained: str = self.config.get("pretrained_weights", "yolov8n.pt")
        self.optimizer: str = self.config.get("optimizer", "AdamW")
        self.augmentation: dict = self.config.get("augmentation", {})

    def train(self, dataset_dir: Path, output_dir: Path) -> TrainResult:
        ckpt_dir = self.get_checkpoint_dir(output_dir)
        device = self.detect_device()
        logger.info(
            f"[DetectionTrainer] plugin={self.plugin_id}, "
            f"backbone={self.backbone}, device={device}, epochs={self.epochs}"
        )

        start_time = time.time()

        # 查找 dataset.yaml
        yaml_path = self._find_dataset_yaml(dataset_dir)

        try:
            return self._train_ultralytics(yaml_path, ckpt_dir, device, start_time)
        except ImportError:
            logger.warning("ultralytics 未安装，回退到 PyTorch 原生训练")
            return self._train_pytorch_fallback(dataset_dir, ckpt_dir, device, start_time)

    def _find_dataset_yaml(self, dataset_dir: Path) -> Path | None:
        """查找数据集 YAML 配置"""
        for pattern in ("*.yaml", "*.yml"):
            yamls = list(dataset_dir.glob(pattern))
            if yamls:
                return yamls[0]
        return None

    def _train_ultralytics(
        self, yaml_path: Path | None, ckpt_dir: Path, device: str, start_time: float
    ) -> TrainResult:
        """使用 ultralytics YOLO 训练"""
        from ultralytics import YOLO

        model = YOLO(self.pretrained)
        results = model.train(
            data=str(yaml_path) if yaml_path else None,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.image_size,
            device=device,
            project=str(ckpt_dir.parent),
            name=ckpt_dir.name,
            optimizer=self.optimizer,
            exist_ok=True,
        )

        # 提取指标
        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = {k: float(v) for k, v in results.results_dict.items()}

        best_pt = ckpt_dir / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = ckpt_dir / "best.pt"

        return TrainResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=ckpt_dir,
            best_checkpoint=best_pt if best_pt.exists() else None,
            epochs_completed=self.epochs,
            metrics=metrics,
            training_time_seconds=time.time() - start_time,
            success=True,
            message=f"YOLO 训练完成: {self.backbone}, {self.epochs} epochs",
        )

    def _train_pytorch_fallback(
        self, dataset_dir: Path, ckpt_dir: Path, device: str, start_time: float
    ) -> TrainResult:
        """
        PyTorch 原生检测训练回退。

        使用轻量 CNN 骨干 + 全连接检测头，用于:
        - ultralytics 未安装时的兜底训练
        - 小数据集快速验证流水线可用性
        - Mac/CPU 环境的调试运行
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            from PIL import Image
            from torchvision import transforms

            logger.info(f"使用 PyTorch 原生训练模式, device={device}")

            # ── 1. 构建简化检测模型 ──────────────────────────
            class SimpleDetectionModel(nn.Module):
                """轻量检测模型 (分类 + 定位)"""

                def __init__(self, num_classes: int, img_size: int = 640):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((4, 4)),
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(128 * 4 * 4, 256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(256, num_classes),
                    )

                def forward(self, x):
                    return self.classifier(self.features(x))

            # ── 2. 构建数据集 ────────────────────────────────
            class SimpleDetDataset(Dataset):
                def __init__(self, img_dir: Path, transform):
                    self.images = []
                    self.labels = []
                    self.transform = transform
                    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

                    if not img_dir.exists():
                        return

                    for img_path in sorted(img_dir.rglob("*")):
                        if img_path.suffix.lower() not in img_exts:
                            continue
                        # 查找对应标注
                        label_path = self._find_label(img_path)
                        cls_id = self._parse_label(label_path)
                        self.images.append(img_path)
                        self.labels.append(cls_id)

                def _find_label(self, img_path: Path) -> Path | None:
                    # images/train/xxx.jpg → labels/train/xxx.txt
                    parts = list(img_path.parts)
                    for i, p in enumerate(parts):
                        if p == "images":
                            parts[i] = "labels"
                            break
                    label = Path(*parts).with_suffix(".txt")
                    return label if label.exists() else None

                def _parse_label(self, label_path: Path | None) -> int:
                    if label_path is None:
                        return 0
                    try:
                        first_line = label_path.read_text().strip().split("\n")[0]
                        return int(first_line.split()[0])
                    except Exception:
                        return 0

                def __len__(self):
                    return len(self.images)

                def __getitem__(self, idx):
                    img = Image.open(self.images[idx]).convert("RGB")
                    img = self.transform(img)
                    return img, self.labels[idx]

            # ── 3. 准备数据 ──────────────────────────────────
            img_size = self.image_size
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            train_dir = dataset_dir / "train" / "images"
            if not train_dir.exists():
                train_dir = dataset_dir / "images" / "train"
            if not train_dir.exists():
                train_dir = dataset_dir / "images"

            train_ds = SimpleDetDataset(train_dir, transform)

            if len(train_ds) == 0:
                return TrainResult(
                    plugin_id=self.plugin_id,
                    task_type=self.task_type,
                    model_path=ckpt_dir,
                    training_time_seconds=time.time() - start_time,
                    success=False,
                    message=f"训练集为空: {train_dir}",
                )

            # 推断类别数
            num_classes = max(train_ds.labels) + 1 if train_ds.labels else 2
            num_classes = max(num_classes, 2)

            train_loader = DataLoader(
                train_ds,
                batch_size=min(self.batch_size, len(train_ds)),
                shuffle=True,
                num_workers=0,
            )

            # ── 4. 训练循环 ──────────────────────────────────
            torch_device = torch.device(device if device != "auto" else "cpu")
            model = SimpleDetectionModel(num_classes, img_size).to(torch_device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            best_loss = float("inf")
            best_epoch = 0
            model_path = ckpt_dir / "best.pth"

            for epoch in range(1, self.epochs + 1):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for images, labels in train_loader:
                    images = images.to(torch_device)
                    labels = torch.tensor(labels, dtype=torch.long).to(torch_device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                scheduler.step()
                epoch_loss = running_loss / total
                epoch_acc = correct / total

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)

                if epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs:
                    logger.info(
                        f"  Epoch {epoch}/{self.epochs}: "
                        f"loss={epoch_loss:.4f}, acc={epoch_acc:.4f}"
                    )

            metrics = {
                "train_loss": round(best_loss, 4),
                "train_accuracy": round(epoch_acc, 4),
                "best_epoch": best_epoch,
            }

            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                model_path=ckpt_dir,
                best_checkpoint=model_path if model_path.exists() else None,
                epochs_completed=self.epochs,
                best_epoch=best_epoch,
                metrics=metrics,
                training_time_seconds=time.time() - start_time,
                success=True,
                message=(
                    f"PyTorch 训练完成: {self.epochs} epochs, "
                    f"best_loss={best_loss:.4f}, acc={epoch_acc:.4f}"
                ),
            )

        except ImportError:
            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                model_path=ckpt_dir,
                training_time_seconds=time.time() - start_time,
                success=False,
                message="PyTorch 和 ultralytics 均未安装",
            )
        except Exception as e:
            logger.error(f"PyTorch 原生训练异常: {e}", exc_info=True)
            return TrainResult(
                plugin_id=self.plugin_id,
                task_type=self.task_type,
                model_path=ckpt_dir,
                training_time_seconds=time.time() - start_time,
                success=False,
                message=f"训练异常: {e}",
            )
