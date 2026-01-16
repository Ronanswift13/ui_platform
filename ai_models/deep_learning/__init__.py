"""
深度学习模型模块
输变电激光星芒破夜绘明监测平台 - 算法迭代优化版

包含:
- YOLOv8-ViT: 改进版目标检测模型
- GL-TransLSTM: 气体浓度预测模型
- CNN-LSTM: 声学异常检测模型
- DeepSORT: 多目标跟踪
- KeypointNet: 表计关键点检测
"""

from .yolov8_vit import YOLOv8ViTDetector, YOLOv8ViTConfig
from .gl_translstm import GLTransLSTM, GLTransLSTMConfig
from .acoustic_cnn_lstm import AcousticCNNLSTM, AcousticModelConfig
from .deep_sort_tracker import DeepSORTTracker, TrackerConfig
from .keypoint_detector import KeypointDetector, KeypointConfig

__all__ = [
    "YOLOv8ViTDetector",
    "YOLOv8ViTConfig",
    "GLTransLSTM",
    "GLTransLSTMConfig",
    "AcousticCNNLSTM",
    "AcousticModelConfig",
    "DeepSORTTracker",
    "TrackerConfig",
    "KeypointDetector",
    "KeypointConfig",
]
