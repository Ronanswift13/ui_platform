"""
训练流水线

按 task_type 提供不同的训练器:
- DetectionTrainer         (YOLOv8 目标检测)
- ClassificationTrainer    (图像分类)
- OCRTrainer              (CRNN 读数识别)
- ThermalTrainer          (热像异常检测 CNN)
- HyperspectralTrainer    (谱空联合训练)
- TemporalTrainer         (时序/数值异常)
- MultimodalTrainer       (多模态融合)
"""

from .base_trainer import BaseTrainer, TrainResult
from .detection_trainer import DetectionTrainer
from .classification_trainer import ClassificationTrainer
from .ocr_trainer import OCRTrainer
from .thermal_trainer import ThermalTrainer
from .hyperspectral_trainer import HyperspectralTrainer
from .temporal_trainer import TemporalTrainer
from .multimodal_trainer import MultimodalTrainer

__all__ = [
    "BaseTrainer",
    "TrainResult",
    "DetectionTrainer",
    "ClassificationTrainer",
    "OCRTrainer",
    "ThermalTrainer",
    "HyperspectralTrainer",
    "TemporalTrainer",
    "MultimodalTrainer",
]
