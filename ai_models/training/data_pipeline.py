"""
数据标准化与训练管道
==================

基于UPDATE.md优化方案实现:
- 统一数据格式 (JSON/YAML)
- 自动标注与人机协同
- MLOps流水线
- 数据闭环与主动学习

适用于: 数据导入、模型训练、版本管理
"""

from __future__ import annotations
import logging
import json
import time
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class AnnotationType(Enum):
    """标注类型"""
    BOUNDING_BOX = "bbox"
    POLYGON = "polygon"
    KEYPOINT = "keypoint"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    TIMESERIES = "timeseries"


class DataStatus(Enum):
    """数据状态"""
    RAW = "raw"
    PREPROCESSED = "preprocessed"
    ANNOTATED = "annotated"
    VERIFIED = "verified"
    TRAINING = "training"
    ARCHIVED = "archived"


class LabelSource(Enum):
    """标注来源"""
    MANUAL = "manual"
    AUTO = "auto"
    SEMI_AUTO = "semi_auto"
    IMPORTED = "imported"


@dataclass
class BoundingBoxAnnotation:
    """边界框标注"""
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float = 1.0
    source: LabelSource = LabelSource.MANUAL
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolygonAnnotation:
    """多边形标注"""
    points: List[Tuple[float, float]]
    label: str
    confidence: float = 1.0
    source: LabelSource = LabelSource.MANUAL


@dataclass
class KeypointAnnotation:
    """关键点标注"""
    keypoints: List[Tuple[float, float, float]]  # x, y, visibility
    label: str
    confidence: float = 1.0
    source: LabelSource = LabelSource.MANUAL


@dataclass
class StandardAnnotation:
    """标准化标注格式"""
    id: str
    image_path: str
    image_width: int
    image_height: int
    annotations: List[Union[BoundingBoxAnnotation, PolygonAnnotation, KeypointAnnotation]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: DataStatus = DataStatus.RAW
    created_at: str = ""
    updated_at: str = ""
    version: int = 1


@dataclass
class DatasetMetadata:
    """数据集元数据"""
    name: str
    version: str
    description: str
    task_type: str
    num_classes: int
    class_names: List[str]
    num_images: int
    num_annotations: int
    split_ratio: Dict[str, float]
    created_at: str
    updated_at: str
    source: str
    license: str = "proprietary"


# =============================================================================
# 标准数据格式转换器
# =============================================================================
class DataFormatConverter:
    """
    数据格式转换器

    支持与常见标注格式的相互转换:
    - COCO
    - YOLO
    - Pascal VOC
    - LabelMe
    - CVAT
    """

    @staticmethod
    def from_coco(coco_data: Dict) -> List[StandardAnnotation]:
        """从COCO格式转换"""
        annotations = []
        images = {img["id"]: img for img in coco_data.get("images", [])}
        categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

        # 按图像分组标注
        image_annots = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in image_annots:
                image_annots[img_id] = []
            image_annots[img_id].append(ann)

        for img_id, img_info in images.items():
            img_annots = image_annots.get(img_id, [])
            boxes = []

            for ann in img_annots:
                bbox = ann.get("bbox", [0, 0, 0, 0])
                cat_id = ann.get("category_id", 0)

                boxes.append(BoundingBoxAnnotation(
                    x=bbox[0] / img_info["width"],
                    y=bbox[1] / img_info["height"],
                    width=bbox[2] / img_info["width"],
                    height=bbox[3] / img_info["height"],
                    label=categories.get(cat_id, f"class_{cat_id}"),
                    source=LabelSource.IMPORTED,
                ))

            annotations.append(StandardAnnotation(
                id=str(uuid.uuid4()),
                image_path=img_info.get("file_name", ""),
                image_width=img_info.get("width", 0),
                image_height=img_info.get("height", 0),
                annotations=boxes,
                status=DataStatus.ANNOTATED,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            ))

        return annotations

    @staticmethod
    def to_coco(annotations: List[StandardAnnotation], class_names: List[str]) -> Dict:
        """转换为COCO格式"""
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
        }

        ann_id = 1
        for img_idx, std_ann in enumerate(annotations):
            coco["images"].append({
                "id": img_idx + 1,
                "file_name": std_ann.image_path,
                "width": std_ann.image_width,
                "height": std_ann.image_height,
            })

            for ann in std_ann.annotations:
                if isinstance(ann, BoundingBoxAnnotation):
                    cat_id = class_names.index(ann.label) if ann.label in class_names else 0
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_idx + 1,
                        "category_id": cat_id,
                        "bbox": [
                            ann.x * std_ann.image_width,
                            ann.y * std_ann.image_height,
                            ann.width * std_ann.image_width,
                            ann.height * std_ann.image_height,
                        ],
                        "area": ann.width * std_ann.image_width * ann.height * std_ann.image_height,
                        "iscrowd": 0,
                    })
                    ann_id += 1

        return coco

    @staticmethod
    def from_yolo(
        labels_dir: Path,
        images_dir: Path,
        class_names: List[str],
    ) -> List[StandardAnnotation]:
        """从YOLO格式转换"""
        annotations = []

        for label_file in labels_dir.glob("*.txt"):
            image_name = label_file.stem
            # 查找对应图像
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = images_dir / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = str(candidate)
                    break

            if image_path is None:
                continue

            boxes = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])

                        boxes.append(BoundingBoxAnnotation(
                            x=cx - w / 2,
                            y=cy - h / 2,
                            width=w,
                            height=h,
                            label=class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                            source=LabelSource.IMPORTED,
                        ))

            # 获取图像尺寸 (简化处理)
            img_width, img_height = 640, 640

            annotations.append(StandardAnnotation(
                id=str(uuid.uuid4()),
                image_path=image_path,
                image_width=img_width,
                image_height=img_height,
                annotations=boxes,
                status=DataStatus.ANNOTATED,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            ))

        return annotations

    @staticmethod
    def to_yolo(
        annotations: List[StandardAnnotation],
        class_names: List[str],
        output_dir: Path,
    ) -> None:
        """转换为YOLO格式"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for std_ann in annotations:
            label_file = output_dir / f"{Path(std_ann.image_path).stem}.txt"
            lines = []

            for ann in std_ann.annotations:
                if isinstance(ann, BoundingBoxAnnotation):
                    class_id = class_names.index(ann.label) if ann.label in class_names else 0
                    cx = ann.x + ann.width / 2
                    cy = ann.y + ann.height / 2
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {ann.width:.6f} {ann.height:.6f}")

            with open(label_file, "w") as f:
                f.write("\n".join(lines))


# =============================================================================
# 自动标注器
# =============================================================================
class AutoAnnotator:
    """
    自动标注器

    使用预训练模型进行预标注,支持:
    - 目标检测预标注
    - 分割预标注
    - 关键点预标注
    """

    def __init__(self, model_registry=None):
        self.model_registry = model_registry
        self._detectors = {}

    def auto_annotate(
        self,
        image: np.ndarray,
        task_type: str,
        confidence_threshold: float = 0.5,
    ) -> List[Union[BoundingBoxAnnotation, PolygonAnnotation]]:
        """
        自动标注

        Args:
            image: 输入图像
            task_type: 任务类型
            confidence_threshold: 置信度阈值

        Returns:
            标注列表
        """
        annotations = []

        # 使用YOLOv8-ViT进行检测
        try:
            from ai_models.deep_learning.yolov8_vit import YOLOv8ViTDetector, YOLOv8ViTConfig, DetectionTask

            task_map = {
                "transformer": DetectionTask.TRANSFORMER_DEFECT,
                "switch": DetectionTask.SWITCH_STATE,
                "busbar": DetectionTask.BUSBAR_DEFECT,
                "capacitor": DetectionTask.CAPACITOR_DEFECT,
                "bird": DetectionTask.BIRD_DETECTION,
            }

            task = task_map.get(task_type, DetectionTask.GENERAL)
            config = YOLOv8ViTConfig(task=task, confidence_threshold=confidence_threshold)
            detector = YOLOv8ViTDetector(config)
            detector.load()

            result = detector.detect(image)

            for det in result.detections:
                annotations.append(BoundingBoxAnnotation(
                    x=det.bbox["x"],
                    y=det.bbox["y"],
                    width=det.bbox["width"],
                    height=det.bbox["height"],
                    label=det.class_name,
                    confidence=det.confidence,
                    source=LabelSource.AUTO,
                ))

        except Exception as e:
            logger.warning(f"[AutoAnnotator] 自动标注失败: {e}")

        return annotations

    def batch_annotate(
        self,
        images: List[np.ndarray],
        task_type: str,
        confidence_threshold: float = 0.5,
    ) -> List[List[Union[BoundingBoxAnnotation, PolygonAnnotation]]]:
        """批量自动标注"""
        return [
            self.auto_annotate(img, task_type, confidence_threshold)
            for img in images
        ]


# =============================================================================
# 训练数据管道
# =============================================================================
@dataclass
class TrainingConfig:
    """训练配置"""
    model_type: str
    model_config: Dict[str, Any]
    dataset_path: str
    output_dir: str
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    validation_split: float = 0.2
    augmentation: bool = True
    early_stopping: bool = True
    patience: int = 10
    save_best_only: bool = True


@dataclass
class TrainingRun:
    """训练运行记录"""
    run_id: str
    config: TrainingConfig
    status: str                       # pending, running, completed, failed
    metrics: Dict[str, float]
    start_time: str
    end_time: Optional[str]
    model_path: Optional[str]
    logs: List[str]


class TrainingPipeline:
    """
    训练管道

    实现MLOps流水线:
    - 数据准备
    - 模型训练
    - 评估验证
    - 模型版本管理
    - 自动部署
    """

    def __init__(self, data_dir: Path, model_dir: Path):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self._runs: Dict[str, TrainingRun] = {}

        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(
        self,
        annotations: List[StandardAnnotation],
        output_format: str = "yolo",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Path:
        """
        准备数据集

        Args:
            annotations: 标注数据
            output_format: 输出格式 (yolo, coco)
            split_ratio: 训练/验证/测试比例

        Returns:
            数据集路径
        """
        # 创建数据集目录
        dataset_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        dataset_dir = self.data_dir / f"dataset_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 随机打乱并分割
        np.random.shuffle(annotations)
        n = len(annotations)
        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))

        splits = {
            "train": annotations[:train_end],
            "val": annotations[train_end:val_end],
            "test": annotations[val_end:],
        }

        # 收集所有类别
        class_names = set()
        for ann in annotations:
            for a in ann.annotations:
                if hasattr(a, 'label'):
                    class_names.add(a.label)
        class_names = sorted(list(class_names))

        # 转换格式
        for split_name, split_data in splits.items():
            split_dir = dataset_dir / split_name
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            if output_format == "yolo":
                DataFormatConverter.to_yolo(split_data, class_names, labels_dir)
            elif output_format == "coco":
                coco_data = DataFormatConverter.to_coco(split_data, class_names)
                with open(split_dir / "annotations.json", "w") as f:
                    json.dump(coco_data, f, indent=2)

        # 保存数据集配置
        config = {
            "format": output_format,
            "classes": class_names,
            "num_classes": len(class_names),
            "splits": {name: len(data) for name, data in splits.items()},
            "created_at": datetime.now().isoformat(),
        }

        if YAML_AVAILABLE:
            with open(dataset_dir / "dataset.yaml", "w") as f:
                yaml.dump(config, f)
        else:
            with open(dataset_dir / "dataset.json", "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"[TrainingPipeline] 数据集已准备: {dataset_dir}")
        return dataset_dir

    def start_training(self, config: TrainingConfig) -> str:
        """
        启动训练

        Args:
            config: 训练配置

        Returns:
            运行ID
        """
        run_id = str(uuid.uuid4())[:8]

        run = TrainingRun(
            run_id=run_id,
            config=config,
            status="pending",
            metrics={},
            start_time=datetime.now().isoformat(),
            end_time=None,
            model_path=None,
            logs=[],
        )

        self._runs[run_id] = run

        # 这里应该启动异步训练任务
        # 简化版直接模拟
        run.status = "running"
        run.logs.append(f"[{datetime.now().isoformat()}] 训练开始")

        logger.info(f"[TrainingPipeline] 训练已启动: {run_id}")
        return run_id

    def get_run_status(self, run_id: str) -> Optional[TrainingRun]:
        """获取训练状态"""
        return self._runs.get(run_id)

    def list_runs(self) -> List[TrainingRun]:
        """列出所有训练运行"""
        return list(self._runs.values())


# =============================================================================
# 主动学习采样器
# =============================================================================
class ActiveLearningSampler:
    """
    主动学习采样器

    实现多种采样策略:
    - 不确定性采样
    - 多样性采样
    - 混合采样
    """

    def __init__(self, strategy: str = "uncertainty"):
        self.strategy = strategy

    def select_samples(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int,
    ) -> List[int]:
        """
        选择需要标注的样本

        Args:
            predictions: 模型预测结果
            n_samples: 选择样本数

        Returns:
            选中样本的索引列表
        """
        if self.strategy == "uncertainty":
            return self._uncertainty_sampling(predictions, n_samples)
        elif self.strategy == "diversity":
            return self._diversity_sampling(predictions, n_samples)
        else:
            return self._random_sampling(predictions, n_samples)

    def _uncertainty_sampling(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int,
    ) -> List[int]:
        """不确定性采样 - 选择模型最不确定的样本"""
        # 计算不确定性分数 (置信度越低越不确定)
        scores = []
        for pred in predictions:
            if "detections" in pred:
                if pred["detections"]:
                    avg_conf = np.mean([d.get("confidence", 1.0) for d in pred["detections"]])
                else:
                    avg_conf = 0.5  # 无检测结果
            else:
                avg_conf = pred.get("confidence", 0.5)

            # 不确定性 = 1 - 置信度
            scores.append(1 - avg_conf)

        # 选择不确定性最高的样本
        indices = np.argsort(scores)[::-1][:n_samples]
        return indices.tolist()

    def _diversity_sampling(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int,
    ) -> List[int]:
        """多样性采样 - 选择最具代表性的样本"""
        # 简化: 随机选择
        return self._random_sampling(predictions, n_samples)

    def _random_sampling(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int,
    ) -> List[int]:
        """随机采样"""
        n = len(predictions)
        return np.random.choice(n, min(n_samples, n), replace=False).tolist()


# =============================================================================
# 数据管理器 (统一入口)
# =============================================================================
class DataManager:
    """
    数据管理器

    统一管理数据导入、标注、训练的完整流程
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"

        self.converter = DataFormatConverter()
        self.auto_annotator = AutoAnnotator()
        self.pipeline = TrainingPipeline(self.data_dir, self.models_dir)
        self.active_learner = ActiveLearningSampler()

        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def import_data(
        self,
        source_path: Path,
        format_type: str,
        module_type: str,
    ) -> List[StandardAnnotation]:
        """
        导入数据

        Args:
            source_path: 数据源路径
            format_type: 格式类型 (coco, yolo, raw)
            module_type: 功能模块类型

        Returns:
            标准化标注数据
        """
        source_path = Path(source_path)

        if format_type == "coco":
            with open(source_path, "r") as f:
                coco_data = json.load(f)
            annotations = self.converter.from_coco(coco_data)

        elif format_type == "yolo":
            # 假设YOLO目录结构
            labels_dir = source_path / "labels"
            images_dir = source_path / "images"
            class_names = self._get_class_names(module_type)
            annotations = self.converter.from_yolo(labels_dir, images_dir, class_names)

        else:
            # 原始图像,需要自动标注
            annotations = self._import_raw_images(source_path, module_type)

        logger.info(f"[DataManager] 导入 {len(annotations)} 条数据")
        return annotations

    def _import_raw_images(
        self,
        images_dir: Path,
        module_type: str,
    ) -> List[StandardAnnotation]:
        """导入原始图像并自动标注"""
        annotations = []

        for img_path in images_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                # 读取图像
                try:
                    import cv2
                    image = cv2.imread(str(img_path))
                    h, w = image.shape[:2]
                except Exception:
                    image = None
                    h, w = 640, 640

                # 自动标注
                if image is not None:
                    auto_anns = self.auto_annotator.auto_annotate(image, module_type)
                else:
                    auto_anns = []

                annotations.append(StandardAnnotation(
                    id=str(uuid.uuid4()),
                    image_path=str(img_path),
                    image_width=w,
                    image_height=h,
                    annotations=auto_anns,
                    status=DataStatus.RAW if not auto_anns else DataStatus.ANNOTATED,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    metadata={"module": module_type, "auto_annotated": len(auto_anns) > 0}
                ))

        return annotations

    def _get_class_names(self, module_type: str) -> List[str]:
        """获取模块类别名称"""
        class_map = {
            "transformer": ["oil_leak", "rust", "damage", "foreign_object", "crack"],
            "switch": ["breaker_open", "breaker_closed", "isolator_open", "isolator_closed"],
            "busbar": ["pin_missing", "crack", "foreign_object", "corrosion"],
            "capacitor": ["tilt", "collapse", "missing", "oil_leak", "bulge"],
            "bird": ["sparrow", "crow", "pigeon", "eagle", "unknown_bird"],
        }
        return class_map.get(module_type, [])

    def export_for_training(
        self,
        annotations: List[StandardAnnotation],
        output_format: str = "yolo",
    ) -> Path:
        """导出用于训练的数据集"""
        return self.pipeline.prepare_dataset(annotations, output_format)

    def start_training(self, config: TrainingConfig) -> str:
        """启动训练"""
        return self.pipeline.start_training(config)

    def get_uncertain_samples(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int = 100,
    ) -> List[int]:
        """获取需要人工标注的不确定样本"""
        return self.active_learner.select_samples(predictions, n_samples)


# =============================================================================
# 便捷函数
# =============================================================================

def create_data_manager(base_dir: str = "data") -> DataManager:
    """创建数据管理器"""
    return DataManager(Path(base_dir))
