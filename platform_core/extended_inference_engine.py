"""
扩展推理引擎模块
输变电站全自动AI巡检方案 - 多模态深度学习支持

新增支持:
1. 点云(Point Cloud)推理引擎 - 用于3D LiDAR SLAM和点云分割
2. 音频(Audio)推理引擎 - 用于声学异常检测
3. 时序(TimeSeries)推理引擎 - 用于气体浓度预测
4. 多模态融合(Multimodal)推理引擎 - 用于多传感器数据融合

作者: AI巡检系统增强模块
版本: 2.0.0
"""

from __future__ import annotations
import os
import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

logger = logging.getLogger(__name__)


# =============================================================================
# 扩展模型类型枚举
# =============================================================================
class ExtendedModelType(Enum):
    """扩展模型类型"""
    # 原有类型
    YOLOV8 = "yolov8"
    RTDETR = "rtdetr"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    KEYPOINT = "keypoint"
    OCR = "ocr"
    ANOMALY_DETECTION = "anomaly_detection"
    REGRESSION = "regression"
    
    # 新增类型
    POINT_CLOUD = "point_cloud"           # 点云处理
    POINT_CLOUD_SLAM = "point_cloud_slam" # 点云SLAM
    POINT_CLOUD_SEG = "point_cloud_seg"   # 点云语义分割
    AUDIO = "audio"                       # 音频处理
    AUDIO_ANOMALY = "audio_anomaly"       # 音频异常检测
    TIME_SERIES = "time_series"           # 时间序列
    TIME_SERIES_FORECAST = "time_series_forecast"  # 时序预测
    MULTIMODAL = "multimodal"             # 多模态融合
    HYPERSPECTRAL = "hyperspectral"       # 高光谱图像


class ExtendedInferenceBackend(Enum):
    """扩展推理后端"""
    ONNX_CPU = "onnx_cpu"
    ONNX_CUDA = "onnx_cuda"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    OPEN3D = "open3d"           # Open3D点云处理
    PCL = "pcl"                 # PCL点云库
    PYTORCH = "pytorch"         # PyTorch直接推理
    TFLITE = "tflite"          # TensorFlow Lite


# =============================================================================
# 扩展模型配置
# =============================================================================
@dataclass
class ExtendedModelConfig:
    """扩展模型配置 - 支持多种输入类型"""
    model_id: str
    model_path: str
    model_type: str
    input_format: str = "default"
    output_format: str = "default"
    backend: str = "onnx_cpu"
    device_id: int = 0
    
    # 图像模型参数
    input_size: Optional[Tuple[int, int]] = None
    
    # 点云模型参数
    point_cloud_format: str = "xyz"       # xyz, xyzrgb, xyzi
    max_points: int = 100000
    voxel_size: float = 0.05
    
    # 音频模型参数
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    audio_duration: float = 2.0           # 音频片段长度(秒)
    
    # 时序模型参数
    sequence_length: int = 100            # 序列长度
    input_features: int = 1               # 输入特征数
    prediction_horizon: int = 10          # 预测步长
    
    # 通用参数
    classes: Optional[List[str]] = None
    confidence_threshold: float = 0.5
    description: str = ""
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, model_id: str, config: Dict) -> 'ExtendedModelConfig':
        """从字典创建配置"""
        return cls(
            model_id=model_id,
            model_path=config.get("model_path", ""),
            model_type=config.get("model_type", ""),
            input_format=config.get("input_format", "default"),
            output_format=config.get("output_format", "default"),
            backend=config.get("backend", "onnx_cpu"),
            device_id=config.get("device_id", 0),
            input_size=tuple(config["input_size"]) if "input_size" in config else None,
            point_cloud_format=config.get("point_cloud_format", "xyz"),
            max_points=config.get("max_points", 100000),
            voxel_size=config.get("voxel_size", 0.05),
            sample_rate=config.get("sample_rate", 16000),
            n_mels=config.get("n_mels", 128),
            n_fft=config.get("n_fft", 2048),
            hop_length=config.get("hop_length", 512),
            audio_duration=config.get("audio_duration", 2.0),
            sequence_length=config.get("sequence_length", 100),
            input_features=config.get("input_features", 1),
            prediction_horizon=config.get("prediction_horizon", 10),
            classes=config.get("classes"),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            description=config.get("description", ""),
            extra_config={k: v for k, v in config.items() 
                         if k not in cls.__dataclass_fields__}
        )


@dataclass
class ExtendedInferenceResult:
    """扩展推理结果 - 支持多种输出类型"""
    success: bool
    model_id: str
    inference_time_ms: float
    error_message: Optional[str] = None
    
    # 通用输出
    outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # 点云输出
    point_cloud: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None           # [x, y, z, qw, qx, qy, qz]
    semantic_labels: Optional[np.ndarray] = None
    
    # 音频输出
    audio_embedding: Optional[np.ndarray] = None
    anomaly_score: float = 0.0
    anomaly_type: Optional[str] = None
    
    # 时序输出
    predictions: Optional[np.ndarray] = None
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # 多模态输出
    fused_features: Optional[np.ndarray] = None
    modality_weights: Optional[Dict[str, float]] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 扩展推理引擎基类
# =============================================================================
class ExtendedBaseInferenceEngine(ABC):
    """扩展推理引擎基类 - 支持多种输入类型"""
    
    def __init__(self, config: ExtendedModelConfig):
        self.config = config
        self._loaded = False
        self._lock = threading.Lock()
        self._inference_count = 0
        self._total_inference_time = 0.0
    
    @abstractmethod
    def load(self) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """卸载模型"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass
    
    def infer(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
        """
        统一推理接口
        
        Args:
            inputs: 输入数据字典,可包含:
                - "image": np.ndarray (H, W, C)
                - "point_cloud": np.ndarray (N, 3/4/6)
                - "audio": np.ndarray (samples,) 或 (channels, samples)
                - "time_series": np.ndarray (seq_len, features)
                - 其他自定义输入
        
        Returns:
            ExtendedInferenceResult: 推理结果
        """
        if not self.is_loaded():
            return ExtendedInferenceResult(
                success=False,
                model_id=self.config.model_id,
                inference_time_ms=0,
                error_message="模型未加载"
            )
        
        start_time = time.perf_counter()
        try:
            result = self._do_infer(inputs)
            inference_time = (time.perf_counter() - start_time) * 1000
            result.inference_time_ms = inference_time
            
            self._inference_count += 1
            self._total_inference_time += inference_time
            
            return result
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return ExtendedInferenceResult(
                success=False,
                model_id=self.config.model_id,
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    @abstractmethod
    def _do_infer(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
        """实际推理实现"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "model_id": self.config.model_id,
            "model_type": self.config.model_type,
            "loaded": self._loaded,
            "inference_count": self._inference_count,
            "avg_inference_time_ms": (
                self._total_inference_time / self._inference_count 
                if self._inference_count > 0 else 0
            )
        }


# =============================================================================
# 点云推理引擎
# =============================================================================
class PointCloudInferenceEngine(ExtendedBaseInferenceEngine):
    """
    点云推理引擎
    
    支持:
    - 点云语义分割 (PointNet++, KPConv, PointTransformer)
    - 点云目标检测 (PointPillars, VoxelNet)
    - SLAM位姿估计
    """
    
    def __init__(self, config: ExtendedModelConfig):
        super().__init__(config)
        self._session = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
    
    def load(self) -> bool:
        """加载点云模型"""
        try:
            if ort is None:
                logger.error("onnxruntime未安装")
                return False
            
            model_path = self.config.model_path
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 配置执行提供者
            providers = self._get_providers()
            
            # 会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 创建会话
            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            
            self._loaded = True
            logger.info(f"点云模型加载成功: {self.config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"点云模型加载失败: {e}")
            return False
    
    def _get_providers(self) -> List[str]:
        """获取执行提供者"""
        backend = self.config.backend
        if backend == "onnx_cuda":
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif backend == "tensorrt":
            return ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    def unload(self) -> None:
        """卸载模型"""
        self._session = None
        self._loaded = False
    
    def is_loaded(self) -> bool:
        return self._loaded and self._session is not None
    
    def _preprocess_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        点云预处理
        
        Args:
            points: 原始点云 (N, 3/4/6)
        
        Returns:
            处理后的点云
        """
        # 1. 限制点数
        max_points = self.config.max_points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        # 2. 体素下采样 (可选)
        if self.config.voxel_size > 0 and o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd = pcd.voxel_down_sample(self.config.voxel_size)
            points_downsampled = np.asarray(pcd.points)
            
            # 如果原始点云有额外特征,需要重新采样
            if points.shape[1] > 3:
                # 简单起见,只保留xyz
                points = points_downsampled
            else:
                points = points_downsampled
        
        # 3. 中心化
        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] -= centroid
        
        # 4. 归一化
        max_dist = np.max(np.linalg.norm(points[:, :3], axis=1))
        if max_dist > 0:
            points[:, :3] /= max_dist
        
        # 5. 填充到固定大小
        if len(points) < max_points:
            padding = np.zeros((max_points - len(points), points.shape[1]))
            points = np.vstack([points, padding])
        
        return points.astype(np.float32)
    
    def _do_infer(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
        """执行点云推理"""
        point_cloud = inputs.get("point_cloud")
        if point_cloud is None:
            return ExtendedInferenceResult(
                success=False,
                model_id=self.config.model_id,
                inference_time_ms=0,
                error_message="缺少点云输入"
            )
        
        # 预处理
        processed_points = self._preprocess_point_cloud(point_cloud)
        
        # 准备输入
        input_data = {
            self._input_names[0]: processed_points[np.newaxis, ...].astype(np.float32)
        }
        
        # 执行推理
        outputs = self._session.run(self._output_names, input_data)
        
        # 解析输出
        result = ExtendedInferenceResult(
            success=True,
            model_id=self.config.model_id,
            inference_time_ms=0,
            outputs={name: out for name, out in zip(self._output_names, outputs)}
        )
        
        # 根据模型类型解析特定输出
        model_type = self.config.model_type
        if model_type == "point_cloud_seg":
            result.semantic_labels = outputs[0].squeeze()
        elif model_type == "point_cloud_slam":
            if len(outputs) >= 2:
                result.pose = outputs[0].squeeze()  # [x, y, z, qw, qx, qy, qz]
                result.point_cloud = outputs[1].squeeze()  # 变换后的点云
        
        return result


# =============================================================================
# 音频推理引擎
# =============================================================================
class AudioInferenceEngine(ExtendedBaseInferenceEngine):
    """
    音频推理引擎
    
    支持:
    - 声学异常检测 (Autoencoder, Contrastive Learning)
    - 局部放电识别
    - 机械故障诊断
    """
    
    def __init__(self, config: ExtendedModelConfig):
        super().__init__(config)
        self._session = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
    
    def load(self) -> bool:
        """加载音频模型"""
        try:
            if ort is None:
                logger.error("onnxruntime未安装")
                return False
            
            model_path = self.config.model_path
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.config.backend == "onnx_cuda" else ['CPUExecutionProvider']
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            
            self._loaded = True
            logger.info(f"音频模型加载成功: {self.config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"音频模型加载失败: {e}")
            return False
    
    def unload(self) -> None:
        self._session = None
        self._loaded = False
    
    def is_loaded(self) -> bool:
        return self._loaded and self._session is not None
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        计算梅尔频谱图
        
        Args:
            audio: 音频波形 (samples,)
        
        Returns:
            梅尔频谱图 (n_mels, time_frames)
        """
        try:
            import librosa
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio.astype(np.float32),
                sr=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            
            # 转换为对数刻度
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 归一化到 [0, 1]
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            return mel_spec_norm.astype(np.float32)
            
        except ImportError:
            # 如果没有librosa,使用简化实现
            return self._simple_spectrogram(audio)
    
    def _simple_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """简化的频谱图计算(不依赖librosa)"""
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        n_mels = self.config.n_mels
        
        # 分帧
        frames = []
        for i in range(0, len(audio) - n_fft, hop_length):
            frame = audio[i:i + n_fft]
            # 加窗
            window = np.hanning(n_fft)
            frame = frame * window
            # FFT
            spectrum = np.abs(np.fft.rfft(frame))
            frames.append(spectrum)
        
        spectrogram = np.array(frames).T  # (freq_bins, time_frames)
        
        # 简化的梅尔滤波器组
        freq_bins = spectrogram.shape[0]
        mel_filters = np.zeros((n_mels, freq_bins))
        mel_points = np.linspace(0, freq_bins - 1, n_mels + 2).astype(int)
        
        for i in range(n_mels):
            mel_filters[i, mel_points[i]:mel_points[i+2]] = 1.0
        
        mel_spec = np.dot(mel_filters, spectrogram)
        
        # 对数变换和归一化
        mel_spec = np.log(mel_spec + 1e-8)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        
        return mel_spec.astype(np.float32)
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        音频预处理
        
        Args:
            audio: 原始音频 (samples,) 或 (channels, samples)
        
        Returns:
            处理后的特征 (1, n_mels, time_frames) 或 (1, samples)
        """
        # 转换为单声道
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        
        # 重采样到目标采样率 (简化处理)
        target_samples = int(self.config.audio_duration * self.config.sample_rate)
        
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # 根据模型输入格式处理
        input_format = self.config.input_format
        if input_format == "mel_spectrogram":
            features = self._compute_mel_spectrogram(audio)
            return features[np.newaxis, np.newaxis, ...]  # (1, 1, n_mels, time)
        elif input_format == "raw_waveform":
            return audio[np.newaxis, np.newaxis, ...].astype(np.float32)  # (1, 1, samples)
        else:
            # 默认使用梅尔频谱
            features = self._compute_mel_spectrogram(audio)
            return features[np.newaxis, np.newaxis, ...]
    
    def _do_infer(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
        """执行音频推理"""
        audio = inputs.get("audio")
        if audio is None:
            return ExtendedInferenceResult(
                success=False,
                model_id=self.config.model_id,
                inference_time_ms=0,
                error_message="缺少音频输入"
            )
        
        # 预处理
        processed_audio = self._preprocess_audio(audio)
        
        # 准备输入
        input_data = {self._input_names[0]: processed_audio}
        
        # 执行推理
        outputs = self._session.run(self._output_names, input_data)
        
        # 解析输出
        result = ExtendedInferenceResult(
            success=True,
            model_id=self.config.model_id,
            inference_time_ms=0,
            outputs={name: out for name, out in zip(self._output_names, outputs)}
        )
        
        # 根据模型类型解析
        model_type = self.config.model_type
        if model_type == "audio_anomaly":
            # 异常检测模型通常输出重建误差或异常分数
            if len(outputs) >= 1:
                output = outputs[0].squeeze()
                if output.ndim == 0:  # 标量异常分数
                    result.anomaly_score = float(output)
                else:  # 重建误差
                    result.anomaly_score = float(np.mean(np.abs(output)))
                
                # 判断异常类型
                threshold = self.config.confidence_threshold
                if result.anomaly_score > threshold:
                    result.anomaly_type = self._classify_anomaly(result.anomaly_score)
        
        return result
    
    def _classify_anomaly(self, score: float) -> str:
        """根据异常分数分类异常类型"""
        if score > 0.8:
            return "severe_anomaly"
        elif score > 0.6:
            return "moderate_anomaly"
        elif score > 0.4:
            return "mild_anomaly"
        else:
            return "normal"


# =============================================================================
# 时序推理引擎
# =============================================================================
class TimeSeriesInferenceEngine(ExtendedBaseInferenceEngine):
    """
    时序推理引擎
    
    支持:
    - 气体浓度预测 (LSTM, Transformer)
    - 设备健康趋势预测
    - 异常检测
    """
    
    def __init__(self, config: ExtendedModelConfig):
        super().__init__(config)
        self._session = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._history_buffer: List[np.ndarray] = []
    
    def load(self) -> bool:
        """加载时序模型"""
        try:
            if ort is None:
                logger.error("onnxruntime未安装")
                return False
            
            model_path = self.config.model_path
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.config.backend == "onnx_cuda" else ['CPUExecutionProvider']
            
            self._session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            
            self._loaded = True
            logger.info(f"时序模型加载成功: {self.config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"时序模型加载失败: {e}")
            return False
    
    def unload(self) -> None:
        self._session = None
        self._loaded = False
        self._history_buffer.clear()
    
    def is_loaded(self) -> bool:
        return self._loaded and self._session is not None
    
    def _preprocess_time_series(self, data: np.ndarray) -> np.ndarray:
        """
        时序数据预处理
        
        Args:
            data: 时序数据 (seq_len,) 或 (seq_len, features)
        
        Returns:
            处理后的数据 (1, seq_len, features)
        """
        seq_len = self.config.sequence_length
        n_features = self.config.input_features
        
        # 确保是2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # 调整序列长度
        if len(data) > seq_len:
            data = data[-seq_len:]
        elif len(data) < seq_len:
            padding = np.zeros((seq_len - len(data), data.shape[1]))
            data = np.vstack([padding, data])
        
        # 标准化 (简单的z-score)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        data = (data - mean) / std
        
        return data[np.newaxis, ...].astype(np.float32)
    
    def _do_infer(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
        """执行时序推理"""
        time_series = inputs.get("time_series")
        if time_series is None:
            return ExtendedInferenceResult(
                success=False,
                model_id=self.config.model_id,
                inference_time_ms=0,
                error_message="缺少时序输入"
            )
        
        # 预处理
        processed_data = self._preprocess_time_series(time_series)
        
        # 准备输入
        input_data = {self._input_names[0]: processed_data}
        
        # 执行推理
        outputs = self._session.run(self._output_names, input_data)
        
        # 解析输出
        result = ExtendedInferenceResult(
            success=True,
            model_id=self.config.model_id,
            inference_time_ms=0,
            outputs={name: out for name, out in zip(self._output_names, outputs)}
        )
        
        # 预测结果
        if len(outputs) >= 1:
            result.predictions = outputs[0].squeeze()
            
            # 如果有置信区间输出
            if len(outputs) >= 3:
                result.confidence_intervals = (
                    outputs[1].squeeze(),  # lower bound
                    outputs[2].squeeze()   # upper bound
                )
        
        return result
    
    def update_history(self, new_data: np.ndarray) -> None:
        """更新历史数据缓冲区"""
        self._history_buffer.append(new_data)
        max_history = self.config.sequence_length * 2
        if len(self._history_buffer) > max_history:
            self._history_buffer = self._history_buffer[-max_history:]
    
    def get_history(self) -> np.ndarray:
        """获取历史数据"""
        if not self._history_buffer:
            return np.array([])
        return np.concatenate(self._history_buffer, axis=0)


# =============================================================================
# 多模态融合推理引擎
# =============================================================================
class MultimodalFusionEngine(ExtendedBaseInferenceEngine):
    """
    多模态融合推理引擎
    
    支持:
    - 特征级融合 (Early Fusion)
    - 决策级融合 (Late Fusion)
    - 注意力融合 (Attention Fusion)
    """
    
    def __init__(self, config: ExtendedModelConfig):
        super().__init__(config)
        self._session = None
        self._modality_weights: Dict[str, float] = {}
        self._sub_engines: Dict[str, ExtendedBaseInferenceEngine] = {}
    
    def load(self) -> bool:
        """加载多模态融合模型"""
        try:
            if ort is None:
                logger.error("onnxruntime未安装")
                return False
            
            model_path = self.config.model_path
            if not os.path.exists(model_path):
                logger.warning(f"融合模型文件不存在,使用规则融合: {model_path}")
                self._loaded = True
                return True
            
            providers = ['CPUExecutionProvider']
            self._session = ort.InferenceSession(model_path, providers=providers)
            
            self._loaded = True
            logger.info(f"多模态融合模型加载成功: {self.config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"多模态融合模型加载失败: {e}")
            return False
    
    def unload(self) -> None:
        self._session = None
        self._loaded = False
        for engine in self._sub_engines.values():
            engine.unload()
        self._sub_engines.clear()
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def register_modality_engine(self, modality: str, engine: ExtendedBaseInferenceEngine) -> None:
        """注册模态子引擎"""
        self._sub_engines[modality] = engine
    
    def set_modality_weights(self, weights: Dict[str, float]) -> None:
        """设置模态权重"""
        self._modality_weights = weights
    
    def _do_infer(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
        """执行多模态融合推理"""
        # 收集各模态特征
        modality_features: Dict[str, np.ndarray] = {}
        modality_results: Dict[str, ExtendedInferenceResult] = {}
        
        for modality, engine in self._sub_engines.items():
            if modality in inputs:
                result = engine.infer({modality: inputs[modality]})
                if result.success:
                    modality_results[modality] = result
                    # 提取特征
                    if result.outputs:
                        modality_features[modality] = list(result.outputs.values())[0]
        
        if not modality_features:
            return ExtendedInferenceResult(
                success=False,
                model_id=self.config.model_id,
                inference_time_ms=0,
                error_message="没有有效的模态输入"
            )
        
        # 融合策略
        fusion_type = self.config.extra_config.get("fusion_type", "late")
        
        if fusion_type == "early":
            fused = self._early_fusion(modality_features)
        elif fusion_type == "attention":
            fused = self._attention_fusion(modality_features)
        else:  # late fusion
            fused = self._late_fusion(modality_results)
        
        return ExtendedInferenceResult(
            success=True,
            model_id=self.config.model_id,
            inference_time_ms=0,
            fused_features=fused,
            modality_weights=self._modality_weights,
            metadata={"modality_results": modality_results}
        )
    
    def _early_fusion(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """特征级融合 - 连接所有特征"""
        feature_list = []
        for modality, feat in features.items():
            # 展平特征
            flat_feat = feat.flatten()
            feature_list.append(flat_feat)
        
        return np.concatenate(feature_list)
    
    def _late_fusion(self, results: Dict[str, ExtendedInferenceResult]) -> np.ndarray:
        """决策级融合 - 加权投票"""
        predictions = []
        weights = []
        
        for modality, result in results.items():
            weight = self._modality_weights.get(modality, 1.0)
            weights.append(weight)
            
            # 提取预测结果
            if result.outputs:
                pred = list(result.outputs.values())[0]
                predictions.append(pred.flatten() * weight)
        
        if not predictions:
            return np.array([])
        
        # 加权平均
        total_weight = sum(weights)
        fused = sum(predictions) / total_weight
        
        return fused
    
    def _attention_fusion(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """注意力融合 - 学习自适应权重"""
        if self._session is None:
            # 回退到late fusion
            return self._early_fusion(features)
        
        # 使用融合模型计算注意力权重并融合
        # 这里假设融合模型接受连接的特征并输出融合结果
        combined = self._early_fusion(features)
        input_data = {"input": combined[np.newaxis, ...].astype(np.float32)}
        
        outputs = self._session.run(None, input_data)
        return outputs[0].squeeze()


# =============================================================================
# 引擎工厂
# =============================================================================
class InferenceEngineFactory:
    """推理引擎工厂"""
    
    @staticmethod
    def create(config: ExtendedModelConfig) -> ExtendedBaseInferenceEngine:
        """根据配置创建对应的推理引擎"""
        model_type = config.model_type
        
        if model_type in ["point_cloud", "point_cloud_slam", "point_cloud_seg"]:
            return PointCloudInferenceEngine(config)
        elif model_type in ["audio", "audio_anomaly"]:
            return AudioInferenceEngine(config)
        elif model_type in ["time_series", "time_series_forecast"]:
            return TimeSeriesInferenceEngine(config)
        elif model_type == "multimodal":
            return MultimodalFusionEngine(config)
        else:
            # 默认使用点云引擎(可替换为通用ONNX引擎)
            return PointCloudInferenceEngine(config)
