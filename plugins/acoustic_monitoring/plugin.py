"""
声学异常检测插件
输变电站全自动AI巡检方案 - 声学监测与故障诊断

功能:
1. 局部放电检测 (Partial Discharge Detection)
2. 机械故障声学诊断
3. 异常声音分类
4. 实时声学监控

依赖:
- 麦克风阵列/单麦克风音频数据
- 振动传感器数据(可选)

版本: 2.0.0
"""

from __future__ import annotations
import os
import time
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================
class AnomalyType(Enum):
    """异常类型"""
    NORMAL = "normal"
    PARTIAL_DISCHARGE = "partial_discharge"       # 局部放电
    MECHANICAL_FAULT = "mechanical_fault"         # 机械故障
    CORONA_DISCHARGE = "corona_discharge"         # 电晕放电
    ARCING = "arcing"                             # 电弧
    LOOSE_CONNECTION = "loose_connection"         # 接触不良
    TRANSFORMER_HUM = "transformer_hum"           # 变压器异常嗡鸣
    COOLING_FAN_FAULT = "cooling_fan_fault"       # 冷却风扇故障
    UNKNOWN_ANOMALY = "unknown_anomaly"           # 未知异常


@dataclass
class AudioFrame:
    """音频帧"""
    samples: np.ndarray          # 音频采样数据
    sample_rate: int = 16000
    timestamp: float = 0.0
    channel_count: int = 1
    
    @property
    def duration(self) -> float:
        """音频时长(秒)"""
        return len(self.samples) / self.sample_rate


@dataclass
class AcousticFeatures:
    """声学特征"""
    mel_spectrogram: Optional[np.ndarray] = None      # 梅尔频谱图
    mfcc: Optional[np.ndarray] = None                 # MFCC特征
    spectral_centroid: Optional[float] = None         # 频谱质心
    spectral_bandwidth: Optional[float] = None        # 频谱带宽
    spectral_rolloff: Optional[float] = None          # 频谱滚降点
    zero_crossing_rate: Optional[float] = None        # 过零率
    rms_energy: Optional[float] = None                # RMS能量
    spectral_flatness: Optional[float] = None         # 频谱平坦度
    peak_frequency: Optional[float] = None            # 峰值频率


@dataclass
class AcousticAnomalyResult:
    """声学异常检测结果"""
    is_anomaly: bool = False
    anomaly_type: AnomalyType = AnomalyType.NORMAL
    anomaly_score: float = 0.0
    confidence: float = 0.0
    features: Optional[AcousticFeatures] = None
    timestamp: float = 0.0
    description: str = ""
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# 音频特征提取
# =============================================================================
class AudioFeatureExtractor:
    """音频特征提取器"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 n_mfcc: int = 20):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # 梅尔滤波器组
        self._mel_filterbank = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """创建梅尔滤波器组"""
        # 频率到梅尔转换
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # 梅尔刻度边界
        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # 转换为FFT bin索引
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # 创建滤波器组
        n_bins = self.n_fft // 2 + 1
        filterbank = np.zeros((self.n_mels, n_bins))
        
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # 三角滤波器
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)
        
        return filterbank
    
    def extract_features(self, audio: np.ndarray) -> AcousticFeatures:
        """
        提取全部声学特征
        
        Args:
            audio: 音频信号 (samples,)
        
        Returns:
            声学特征
        """
        features = AcousticFeatures()
        
        # 确保音频是float类型
        audio = audio.astype(np.float32)
        
        # 归一化
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # 1. 计算STFT
        stft = self._compute_stft(audio)
        magnitude = np.abs(stft)
        
        # 2. 梅尔频谱图
        features.mel_spectrogram = self._compute_mel_spectrogram(magnitude)
        
        # 3. MFCC
        features.mfcc = self._compute_mfcc(features.mel_spectrogram)
        
        # 4. 频谱特征
        features.spectral_centroid = self._compute_spectral_centroid(magnitude)
        features.spectral_bandwidth = self._compute_spectral_bandwidth(magnitude)
        features.spectral_rolloff = self._compute_spectral_rolloff(magnitude)
        features.spectral_flatness = self._compute_spectral_flatness(magnitude)
        
        # 5. 时域特征
        features.zero_crossing_rate = self._compute_zcr(audio)
        features.rms_energy = self._compute_rms(audio)
        
        # 6. 峰值频率
        features.peak_frequency = self._find_peak_frequency(magnitude)
        
        return features
    
    def _compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """计算短时傅里叶变换"""
        # 分帧
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        frames = np.zeros((n_frames, self.n_fft))
        
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = audio[start:start + self.n_fft]
        
        # 加窗
        window = np.hanning(self.n_fft)
        frames = frames * window
        
        # FFT
        stft = np.fft.rfft(frames, axis=1)
        
        return stft.T  # (freq_bins, time_frames)
    
    def _compute_mel_spectrogram(self, magnitude: np.ndarray) -> np.ndarray:
        """计算梅尔频谱图"""
        # 应用梅尔滤波器组
        mel_spec = np.dot(self._mel_filterbank, magnitude)
        
        # 对数变换
        mel_spec = np.log(mel_spec + 1e-8)
        
        return mel_spec
    
    def _compute_mfcc(self, mel_spec: np.ndarray) -> np.ndarray:
        """计算MFCC"""
        # DCT-II
        n_frames = mel_spec.shape[1]
        mfcc = np.zeros((self.n_mfcc, n_frames))
        
        for i in range(self.n_mfcc):
            for j in range(self.n_mels):
                mfcc[i] += mel_spec[j] * np.cos(np.pi * i * (2*j + 1) / (2 * self.n_mels))
        
        mfcc *= np.sqrt(2.0 / self.n_mels)
        
        return mfcc
    
    def _compute_spectral_centroid(self, magnitude: np.ndarray) -> float:
        """计算频谱质心"""
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sample_rate)
        
        # 对每帧计算质心,然后取平均
        centroids = []
        for frame in magnitude.T:
            if np.sum(frame) > 0:
                centroid = np.sum(freqs * frame) / np.sum(frame)
                centroids.append(centroid)
        
        return np.mean(centroids) if centroids else 0.0
    
    def _compute_spectral_bandwidth(self, magnitude: np.ndarray) -> float:
        """计算频谱带宽"""
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sample_rate)
        centroid = self._compute_spectral_centroid(magnitude)
        
        bandwidths = []
        for frame in magnitude.T:
            if np.sum(frame) > 0:
                bw = np.sqrt(np.sum(((freqs - centroid) ** 2) * frame) / np.sum(frame))
                bandwidths.append(bw)
        
        return np.mean(bandwidths) if bandwidths else 0.0
    
    def _compute_spectral_rolloff(self, magnitude: np.ndarray, 
                                   percentile: float = 0.85) -> float:
        """计算频谱滚降点"""
        rolloffs = []
        
        for frame in magnitude.T:
            total_energy = np.sum(frame)
            if total_energy > 0:
                cumsum = np.cumsum(frame)
                rolloff_idx = np.searchsorted(cumsum, percentile * total_energy)
                rolloff_freq = rolloff_idx * self.sample_rate / self.n_fft
                rolloffs.append(rolloff_freq)
        
        return np.mean(rolloffs) if rolloffs else 0.0
    
    def _compute_spectral_flatness(self, magnitude: np.ndarray) -> float:
        """计算频谱平坦度"""
        flatness_values = []
        
        for frame in magnitude.T:
            frame = frame + 1e-8  # 避免log(0)
            geo_mean = np.exp(np.mean(np.log(frame)))
            arith_mean = np.mean(frame)
            flatness = geo_mean / arith_mean if arith_mean > 0 else 0
            flatness_values.append(flatness)
        
        return np.mean(flatness_values) if flatness_values else 0.0
    
    def _compute_zcr(self, audio: np.ndarray) -> float:
        """计算过零率"""
        signs = np.sign(audio)
        sign_changes = np.abs(np.diff(signs))
        return np.mean(sign_changes) / 2
    
    def _compute_rms(self, audio: np.ndarray) -> float:
        """计算RMS能量"""
        return np.sqrt(np.mean(audio ** 2))
    
    def _find_peak_frequency(self, magnitude: np.ndarray) -> float:
        """找到峰值频率"""
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sample_rate)
        avg_magnitude = np.mean(magnitude, axis=1)
        peak_idx = np.argmax(avg_magnitude)
        return freqs[peak_idx]


# =============================================================================
# 基于规则的异常检测
# =============================================================================
class RuleBasedAnomalyDetector:
    """基于规则的声学异常检测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 阈值配置
        self.thresholds = {
            "rms_energy_low": self.config.get("rms_energy_low", 0.01),
            "rms_energy_high": self.config.get("rms_energy_high", 0.3),
            "spectral_centroid_pd": self.config.get("spectral_centroid_pd", 15000),  # 局放高频
            "spectral_flatness_noise": self.config.get("spectral_flatness_noise", 0.5),
            "zcr_high": self.config.get("zcr_high", 0.3),
        }
        
        # 特征频率范围 (Hz)
        self.frequency_ranges = {
            AnomalyType.PARTIAL_DISCHARGE: (10000, 100000),   # 局放: 超声频段
            AnomalyType.CORONA_DISCHARGE: (100, 10000),       # 电晕: 音频频段
            AnomalyType.TRANSFORMER_HUM: (100, 300),          # 变压器嗡鸣: 工频谐波
            AnomalyType.MECHANICAL_FAULT: (500, 5000),        # 机械故障: 中频
            AnomalyType.COOLING_FAN_FAULT: (200, 2000),       # 风扇故障: 中低频
        }
    
    def detect(self, features: AcousticFeatures) -> AcousticAnomalyResult:
        """
        基于规则检测异常
        
        Args:
            features: 声学特征
        
        Returns:
            异常检测结果
        """
        result = AcousticAnomalyResult(features=features, timestamp=time.time())
        
        anomaly_scores = {}
        
        # 1. 检测局部放电 (高频成分)
        if features.spectral_centroid and features.peak_frequency:
            pd_range = self.frequency_ranges[AnomalyType.PARTIAL_DISCHARGE]
            if features.peak_frequency > pd_range[0]:
                pd_score = min((features.peak_frequency - pd_range[0]) / 
                              (pd_range[1] - pd_range[0]), 1.0)
                anomaly_scores[AnomalyType.PARTIAL_DISCHARGE] = pd_score
        
        # 2. 检测机械故障 (异常频谱模式)
        if features.spectral_bandwidth and features.spectral_flatness:
            # 机械故障通常表现为窄带周期性噪声
            if features.spectral_flatness < 0.3 and features.rms_energy > 0.1:
                mf_score = (1 - features.spectral_flatness) * features.rms_energy
                anomaly_scores[AnomalyType.MECHANICAL_FAULT] = min(mf_score, 1.0)
        
        # 3. 检测变压器异常嗡鸣 (工频谐波增强)
        if features.peak_frequency:
            # 检查是否在工频谐波附近 (50/100/150/200/250 Hz)
            harmonics = [50 * i for i in range(1, 6)]
            for h in harmonics:
                if abs(features.peak_frequency - h) < 10:  # 10Hz容差
                    if features.rms_energy > self.thresholds["rms_energy_high"]:
                        anomaly_scores[AnomalyType.TRANSFORMER_HUM] = \
                            features.rms_energy / self.thresholds["rms_energy_high"]
                    break
        
        # 4. 检测电晕放电 (噼啪声)
        if features.zero_crossing_rate and features.spectral_flatness:
            if features.zero_crossing_rate > self.thresholds["zcr_high"]:
                corona_score = features.zero_crossing_rate * features.rms_energy
                anomaly_scores[AnomalyType.CORONA_DISCHARGE] = min(corona_score, 1.0)
        
        # 5. 确定最可能的异常类型
        if anomaly_scores:
            max_anomaly = max(anomaly_scores, key=anomaly_scores.get)
            max_score = anomaly_scores[max_anomaly]
            
            if max_score > 0.3:  # 阈值
                result.is_anomaly = True
                result.anomaly_type = max_anomaly
                result.anomaly_score = max_score
                result.confidence = min(max_score / 0.5, 1.0)  # 置信度
                result.description = self._get_anomaly_description(max_anomaly)
                result.recommendations = self._get_recommendations(max_anomaly)
        
        return result
    
    def _get_anomaly_description(self, anomaly_type: AnomalyType) -> str:
        """获取异常描述"""
        descriptions = {
            AnomalyType.PARTIAL_DISCHARGE: "检测到局部放电信号,可能存在绝缘缺陷",
            AnomalyType.MECHANICAL_FAULT: "检测到机械异常声音,可能存在松动或磨损",
            AnomalyType.CORONA_DISCHARGE: "检测到电晕放电声音,可能存在高压放电",
            AnomalyType.TRANSFORMER_HUM: "检测到变压器异常嗡鸣,可能存在铁芯或绕组问题",
            AnomalyType.ARCING: "检测到电弧声音,存在严重的放电风险",
            AnomalyType.LOOSE_CONNECTION: "检测到接触不良噪声",
            AnomalyType.COOLING_FAN_FAULT: "检测到冷却风扇异常声音",
        }
        return descriptions.get(anomaly_type, "检测到未知类型的声学异常")
    
    def _get_recommendations(self, anomaly_type: AnomalyType) -> List[str]:
        """获取处理建议"""
        recommendations = {
            AnomalyType.PARTIAL_DISCHARGE: [
                "建议进行绝缘检测",
                "检查油中气体分析",
                "考虑安排停电检修"
            ],
            AnomalyType.MECHANICAL_FAULT: [
                "检查机械部件紧固情况",
                "检测振动水平",
                "润滑相关部件"
            ],
            AnomalyType.CORONA_DISCHARGE: [
                "检查高压导体表面状态",
                "检查均压环/屏蔽装置",
                "清洁绝缘子表面"
            ],
            AnomalyType.TRANSFORMER_HUM: [
                "检查铁芯紧固螺栓",
                "检测磁路异常",
                "评估负载情况"
            ],
            AnomalyType.COOLING_FAN_FAULT: [
                "检查风扇叶片",
                "更换风扇轴承",
                "清洁风道"
            ],
        }
        return recommendations.get(anomaly_type, ["建议进一步检查"])


# =============================================================================
# 基于深度学习的异常检测
# =============================================================================
class DeepLearningAnomalyDetector:
    """基于深度学习的声学异常检测器"""
    
    def __init__(self, model_registry=None, model_id: str = "audio_anomaly_detector"):
        self._model_registry = model_registry
        self._model_id = model_id
        self._feature_extractor = AudioFeatureExtractor()
        
        # 异常检测阈值
        self.anomaly_threshold = 0.5
    
    def set_model_registry(self, registry) -> None:
        """设置模型注册中心"""
        self._model_registry = registry
    
    def detect(self, audio: np.ndarray, 
               sample_rate: int = 16000) -> AcousticAnomalyResult:
        """
        使用深度学习模型检测异常
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
        
        Returns:
            异常检测结果
        """
        # 提取特征
        features = self._feature_extractor.extract_features(audio)
        
        result = AcousticAnomalyResult(
            features=features,
            timestamp=time.time()
        )
        
        if self._model_registry is None:
            result.description = "模型注册中心未配置"
            return result
        
        # 准备模型输入
        mel_spec = features.mel_spectrogram
        if mel_spec is None:
            result.description = "特征提取失败"
            return result
        
        # 执行推理
        try:
            infer_result = self._model_registry.infer(
                self._model_id,
                {"audio": audio}
            )
            
            if infer_result.success:
                result.anomaly_score = infer_result.anomaly_score
                result.is_anomaly = result.anomaly_score > self.anomaly_threshold
                
                if infer_result.anomaly_type:
                    result.anomaly_type = AnomalyType(infer_result.anomaly_type)
                
                result.confidence = min(result.anomaly_score / self.anomaly_threshold, 1.0)
            else:
                result.description = f"推理失败: {infer_result.error_message}"
                
        except Exception as e:
            result.description = f"推理异常: {str(e)}"
        
        return result


# =============================================================================
# 声学监控插件
# =============================================================================
class AcousticMonitoringPlugin:
    """
    声学监控插件
    
    整合规则和深度学习方法的声学异常检测
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 组件
        self._feature_extractor = AudioFeatureExtractor(
            sample_rate=self.config.get("sample_rate", 16000),
            n_fft=self.config.get("n_fft", 2048),
            n_mels=self.config.get("n_mels", 128)
        )
        
        self._rule_detector = RuleBasedAnomalyDetector(self.config)
        self._dl_detector = DeepLearningAnomalyDetector()
        
        # 历史记录
        self._history: deque = deque(maxlen=100)
        self._anomaly_count = 0
        
        # 模型注册中心
        self._model_registry = None
        self._use_deep_learning = False
    
    def set_model_registry(self, registry) -> None:
        """设置模型注册中心"""
        self._model_registry = registry
        self._dl_detector.set_model_registry(registry)
        self._use_deep_learning = registry is not None
    
    def process_audio(self, audio: np.ndarray, 
                      sample_rate: int = 16000,
                      timestamp: float = None) -> AcousticAnomalyResult:
        """
        处理音频数据
        
        Args:
            audio: 音频采样数据
            sample_rate: 采样率
            timestamp: 时间戳
        
        Returns:
            异常检测结果
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 创建音频帧
        frame = AudioFrame(
            samples=audio,
            sample_rate=sample_rate,
            timestamp=timestamp
        )
        
        # 提取特征
        features = self._feature_extractor.extract_features(audio)
        
        # 深度学习检测
        if self._use_deep_learning:
            dl_result = self._dl_detector.detect(audio, sample_rate)
            if dl_result.is_anomaly:
                # 如果深度学习检测到异常,使用其结果
                dl_result.features = features
                self._record_result(dl_result)
                return dl_result
        
        # 规则检测
        rule_result = self._rule_detector.detect(features)
        rule_result.timestamp = timestamp
        
        # 记录结果
        self._record_result(rule_result)
        
        return rule_result
    
    def _record_result(self, result: AcousticAnomalyResult) -> None:
        """记录检测结果"""
        self._history.append(result)
        if result.is_anomaly:
            self._anomaly_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self._history)
        anomalies = sum(1 for r in self._history if r.is_anomaly)
        
        # 按类型统计
        type_counts = {}
        for r in self._history:
            if r.is_anomaly:
                t = r.anomaly_type.value
                type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total_processed": total,
            "total_anomalies": anomalies,
            "anomaly_rate": anomalies / total if total > 0 else 0,
            "anomaly_by_type": type_counts,
            "use_deep_learning": self._use_deep_learning
        }
    
    def inspect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行巡检(插件统一接口)
        
        Args:
            data: 包含音频数据的字典
        
        Returns:
            巡检结果
        """
        audio = data.get("audio")
        if audio is None:
            return {"success": False, "error": "缺少音频数据"}
        
        sample_rate = data.get("sample_rate", 16000)
        timestamp = data.get("timestamp")
        
        result = self.process_audio(audio, sample_rate, timestamp)
        
        return {
            "success": True,
            "is_anomaly": result.is_anomaly,
            "anomaly_type": result.anomaly_type.value,
            "anomaly_score": result.anomaly_score,
            "confidence": result.confidence,
            "description": result.description,
            "recommendations": result.recommendations,
            "features": {
                "rms_energy": result.features.rms_energy if result.features else None,
                "spectral_centroid": result.features.spectral_centroid if result.features else None,
                "peak_frequency": result.features.peak_frequency if result.features else None
            }
        }


# =============================================================================
# 检测器增强版
# =============================================================================
class AcousticDetectorEnhanced:
    """声学检测器增强版"""
    
    def __init__(self, config: Dict[str, Any] = None, model_registry=None):
        self.config = config or {}
        self._model_registry = model_registry
        
        # 核心组件
        self.plugin = AcousticMonitoringPlugin(config)
        
        if model_registry:
            self.plugin.set_model_registry(model_registry)
    
    def set_model_registry(self, registry) -> None:
        """设置模型注册中心"""
        self._model_registry = registry
        self.plugin.set_model_registry(registry)
    
    def detect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行检测"""
        return self.plugin.inspect(data)
    
    def detect_partial_discharge(self, audio: np.ndarray, 
                                  sample_rate: int = 16000) -> Dict[str, Any]:
        """
        专门检测局部放电
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
        
        Returns:
            检测结果
        """
        result = self.plugin.process_audio(audio, sample_rate)
        
        is_pd = result.anomaly_type == AnomalyType.PARTIAL_DISCHARGE
        
        return {
            "detected": is_pd,
            "score": result.anomaly_score if is_pd else 0.0,
            "confidence": result.confidence if is_pd else 0.0,
            "peak_frequency": result.features.peak_frequency if result.features else None
        }
    
    def continuous_monitoring(self, audio_stream, 
                              callback=None,
                              window_size: float = 2.0,
                              hop_size: float = 0.5) -> None:
        """
        连续监控模式
        
        Args:
            audio_stream: 音频流生成器
            callback: 检测到异常时的回调函数
            window_size: 分析窗口大小(秒)
            hop_size: 窗口滑动步长(秒)
        """
        buffer = []
        sample_rate = 16000  # 假设采样率
        window_samples = int(window_size * sample_rate)
        hop_samples = int(hop_size * sample_rate)
        
        for chunk in audio_stream:
            buffer.extend(chunk)
            
            while len(buffer) >= window_samples:
                window = np.array(buffer[:window_samples])
                buffer = buffer[hop_samples:]
                
                result = self.plugin.process_audio(window, sample_rate)
                
                if result.is_anomaly and callback:
                    callback(result)
