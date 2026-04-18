"""
预处理流水线

按 task_type 提供不同的数据预处理器:
- DetectionPreprocessor       (目标检测)
- SegmentationPreprocessor    (语义/实例分割)
- ClassificationPreprocessor  (图像分类)
- OCRPreprocessor             (OCR/读数)
- ThermalPreprocessor         (热像异常)
- HyperspectralPreprocessor   (高光谱)
- TemporalPreprocessor        (时序/数值异常)
- MultimodalPreprocessor      (多模态融合)
"""

from .base_preprocessor import BasePreprocessor
from .detection_preprocessor import DetectionPreprocessor
from .segmentation_preprocessor import SegmentationPreprocessor
from .classification_preprocessor import ClassificationPreprocessor
from .ocr_preprocessor import OCRPreprocessor
from .thermal_preprocessor import ThermalPreprocessor
from .hyperspectral_preprocessor import HyperspectralPreprocessor
from .temporal_preprocessor import TemporalPreprocessor
from .multimodal_preprocessor import MultimodalPreprocessor

__all__ = [
    "BasePreprocessor",
    "DetectionPreprocessor",
    "SegmentationPreprocessor",
    "ClassificationPreprocessor",
    "OCRPreprocessor",
    "ThermalPreprocessor",
    "HyperspectralPreprocessor",
    "TemporalPreprocessor",
    "MultimodalPreprocessor",
]
