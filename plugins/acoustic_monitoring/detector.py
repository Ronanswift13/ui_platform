#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声学检测器模块
支持传统信号处理和深度学习方法

功能:
1. 梅尔频谱特征提取
2. MFCC特征提取
3. 传统信号处理检测
4. 深度学习模型推理
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 音频特征提取器
# =============================================================================
class AudioFeatureExtractor:
    """音频特征提取器"""
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128,
                 n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # 预计算梅尔滤波器组
        self.mel_filterbank = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """创建梅尔滤波器组"""
        # 频率到梅尔转换
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # 计算梅尔频率点
        low_freq = 0
        high_freq = self.sample_rate / 2
        mel_low = hz_to_mel(low_freq)
        mel_high = hz_to_mel(high_freq)
        mel_points = np.linspace(mel_low, mel_high, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # 转换为FFT bin索引
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # 创建滤波器组
        n_freq = self.n_fft // 2 + 1
        filterbank = np.zeros((self.n_mels, n_freq))
        
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # 上升沿
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)
            
            # 下降沿
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)
        
        return filterbank
    
    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """计算短时傅里叶变换频谱图"""
        # 分帧
        num_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        frames = np.zeros((num_frames, self.n_fft))
        
        window = np.hanning(self.n_fft)
        
        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            frames[i] = frame * window
        
        # FFT
        spectrogram = np.abs(np.fft.rfft(frames, axis=1))
        return spectrogram.T  # (freq_bins, time_frames)
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """计算梅尔频谱图"""
        spectrogram = self.compute_spectrogram(audio)
        mel_spec = np.dot(self.mel_filterbank, spectrogram)
        
        # 对数变换
        mel_spec = np.log(mel_spec + 1e-8)
        
        return mel_spec  # (n_mels, time_frames)
    
    def compute_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """计算MFCC特征"""
        mel_spec = self.compute_mel_spectrogram(audio)
        
        # DCT (Type-II)
        n_frames = mel_spec.shape[1]
        mfcc = np.zeros((n_mfcc, n_frames))
        
        for k in range(n_mfcc):
            for n in range(self.n_mels):
                mfcc[k] += mel_spec[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * self.n_mels))
        
        return mfcc
    
    def compute_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """计算频谱统计特征"""
        spectrogram = self.compute_spectrogram(audio)
        
        # 频率轴
        freqs = np.linspace(0, self.sample_rate / 2, spectrogram.shape[0])
        
        # 每帧计算特征，然后取平均
        spectral_centroid = []
        spectral_bandwidth = []
        spectral_rolloff = []
        spectral_flatness = []
        
        for frame in spectrogram.T:
            frame_sum = np.sum(frame) + 1e-8
            
            # 频谱质心
            centroid = np.sum(freqs * frame) / frame_sum
            spectral_centroid.append(centroid)
            
            # 频谱带宽
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * frame) / frame_sum)
            spectral_bandwidth.append(bandwidth)
            
            # 频谱滚降点 (85%)
            cumsum = np.cumsum(frame)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
            spectral_rolloff.append(rolloff)
            
            # 频谱平坦度
            geometric_mean = np.exp(np.mean(np.log(frame + 1e-8)))
            arithmetic_mean = np.mean(frame)
            flatness = geometric_mean / (arithmetic_mean + 1e-8)
            spectral_flatness.append(flatness)
        
        return {
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_centroid_std": np.std(spectral_centroid),
            "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
            "spectral_bandwidth_std": np.std(spectral_bandwidth),
            "spectral_rolloff_mean": np.mean(spectral_rolloff),
            "spectral_flatness_mean": np.mean(spectral_flatness),
            "spectral_flatness_std": np.std(spectral_flatness)
        }
    
    def compute_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """计算时域特征"""
        # 零交叉率
        sign_changes = np.diff(np.sign(audio))
        zcr = np.sum(np.abs(sign_changes) > 0) / len(audio)
        
        # RMS能量
        rms = np.sqrt(np.mean(audio ** 2))
        
        # 峰值因子
        peak = np.max(np.abs(audio))
        crest_factor = peak / (rms + 1e-8)
        
        # 短时能量变化
        frame_length = self.n_fft
        hop = self.hop_length
        num_frames = 1 + (len(audio) - frame_length) // hop
        
        energies = []
        for i in range(num_frames):
            start = i * hop
            frame = audio[start:start + frame_length]
            energies.append(np.sum(frame ** 2))
        
        energy_std = np.std(energies) if energies else 0
        
        return {
            "zero_crossing_rate": float(zcr),
            "rms_energy": float(rms),
            "peak_amplitude": float(peak),
            "crest_factor": float(crest_factor),
            "energy_std": float(energy_std)
        }


# =============================================================================
# 传统声学检测器
# =============================================================================
class AcousticDetector:
    """
    传统信号处理声学检测器
    使用频谱特征和规则进行异常检测
    """
    
    def __init__(self, config):
        self.config = config
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
        
        # 异常阈值 (可调整)
        self.thresholds = {
            "partial_discharge": {
                "high_freq_energy_ratio": 0.3,    # 高频能量占比
                "impulse_density": 0.1             # 脉冲密度
            },
            "corona_discharge": {
                "hiss_freq_range": (5000, 15000),  # 嘶嘶声频率范围
                "continuous_energy_ratio": 0.2
            },
            "bearing_fault": {
                "periodic_component_strength": 0.15,
                "base_freq_range": (20, 200)
            },
            "transformer_hum": {
                "harmonic_freq": 100,              # 基频100Hz (50Hz工频的二次谐波)
                "harmonic_strength": 0.3
            }
        }
    
    def detect(self, audio: np.ndarray, sample_rate: int = None) -> Dict[str, Any]:
        """
        检测声学异常
        
        Args:
            audio: 音频波形
            sample_rate: 采样率
        
        Returns:
            检测结果字典
        """
        if sample_rate and sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate, self.config.sample_rate)
        
        # 预处理
        audio = self._preprocess(audio)
        
        # 提取特征
        mel_spec = self.feature_extractor.compute_mel_spectrogram(audio)
        spectral_features = self.feature_extractor.compute_spectral_features(audio)
        temporal_features = self.feature_extractor.compute_temporal_features(audio)
        
        # 各类异常检测
        anomaly_scores = {}
        
        # 1. 局部放电检测 (高频脉冲)
        pd_score = self._detect_partial_discharge(audio, mel_spec)
        anomaly_scores["partial_discharge"] = pd_score
        
        # 2. 电晕放电检测 (嘶嘶声)
        corona_score = self._detect_corona_discharge(audio, spectral_features)
        anomaly_scores["corona_discharge"] = corona_score
        
        # 3. 轴承故障检测 (周期性)
        bearing_score = self._detect_bearing_fault(audio)
        anomaly_scores["bearing_fault"] = bearing_score
        
        # 4. 变压器异常嗡鸣检测
        hum_score = self._detect_transformer_hum(audio)
        anomaly_scores["transformer_hum"] = hum_score
        
        # 5. 机械故障检测
        mech_score = self._detect_mechanical_fault(audio, temporal_features)
        anomaly_scores["mechanical_fault"] = mech_score
        
        # 确定最可能的异常类型
        max_score = 0
        anomaly_type = "normal"
        for atype, score in anomaly_scores.items():
            if score > max_score and score > self.config.anomaly_threshold:
                max_score = score
                anomaly_type = atype
        
        return {
            "anomaly_type": anomaly_type,
            "anomaly_score": max_score if anomaly_type != "normal" else 0.0,
            "confidence": max_score,
            "all_scores": anomaly_scores,
            "spectrogram": mel_spec,
            "features": {
                "spectral": spectral_features,
                "temporal": temporal_features
            }
        }
    
    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """音频预处理"""
        # 转单声道
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # 去除直流分量
        audio = audio - np.mean(audio)
        
        return audio.astype(np.float32)
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样"""
        if orig_sr == target_sr:
            return audio
        
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        return np.interp(
            np.linspace(0, len(audio), new_length),
            np.arange(len(audio)),
            audio
        )
    
    def _detect_partial_discharge(self, audio: np.ndarray, mel_spec: np.ndarray) -> float:
        """检测局部放电 (特征: 高频短脉冲)"""
        # 1. 高频能量占比
        high_freq_bins = mel_spec.shape[0] * 2 // 3  # 上1/3频率
        high_freq_energy = np.sum(mel_spec[high_freq_bins:])
        total_energy = np.sum(mel_spec) + 1e-8
        high_freq_ratio = high_freq_energy / total_energy
        
        # 2. 脉冲检测 (短时能量突变)
        frame_energies = np.sum(mel_spec, axis=0)
        energy_diff = np.abs(np.diff(frame_energies))
        impulse_count = np.sum(energy_diff > np.mean(energy_diff) * 3)
        impulse_density = impulse_count / len(energy_diff)
        
        # 综合评分
        score = 0.5 * min(high_freq_ratio / 0.3, 1.0) + 0.5 * min(impulse_density / 0.1, 1.0)
        
        return float(np.clip(score, 0, 1))
    
    def _detect_corona_discharge(self, audio: np.ndarray, spectral_features: Dict) -> float:
        """检测电晕放电 (特征: 持续高频嘶嘶声)"""
        # 频谱质心在高频区域
        centroid = spectral_features["spectral_centroid_mean"]
        
        # 电晕的频谱质心通常在5-15kHz
        if 5000 < centroid < 15000:
            centroid_score = 1.0
        elif centroid > 15000:
            centroid_score = 0.5
        else:
            centroid_score = centroid / 5000
        
        # 频谱平坦度高 (嘶嘶声较为平坦)
        flatness = spectral_features["spectral_flatness_mean"]
        flatness_score = min(flatness / 0.3, 1.0)
        
        score = 0.6 * centroid_score + 0.4 * flatness_score
        
        return float(np.clip(score, 0, 1))
    
    def _detect_bearing_fault(self, audio: np.ndarray) -> float:
        """检测轴承故障 (特征: 周期性冲击)"""
        # 计算自相关函数
        n = len(audio)
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[n-1:] / (autocorr[n-1] + 1e-8)
        
        # 寻找周期性峰值
        # 轴承故障频率通常在20-200Hz
        min_lag = int(self.config.sample_rate / 200)
        max_lag = int(self.config.sample_rate / 20)
        
        if max_lag < len(autocorr):
            search_range = autocorr[min_lag:max_lag]
            peak_value = np.max(search_range)
            
            # 周期性强度
            score = min(peak_value / 0.15, 1.0)
        else:
            score = 0
        
        return float(np.clip(score, 0, 1))
    
    def _detect_transformer_hum(self, audio: np.ndarray) -> float:
        """检测变压器异常嗡鸣 (特征: 100/120Hz及谐波)"""
        # FFT分析
        n = len(audio)
        fft_result = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(n, 1 / self.config.sample_rate)
        
        # 检测100Hz和200Hz分量
        harmonic_freqs = [100, 200, 300, 400]  # 基频及谐波
        harmonic_powers = []
        
        for hf in harmonic_freqs:
            # 找到最接近的频率bin
            idx = np.argmin(np.abs(freqs - hf))
            # 取周围几个bin的平均
            power = np.mean(fft_result[max(0, idx-2):min(len(fft_result), idx+3)])
            harmonic_powers.append(power)
        
        # 谐波能量占比
        total_power = np.sum(fft_result) + 1e-8
        harmonic_ratio = sum(harmonic_powers) / total_power
        
        # 如果100Hz分量显著强于正常值
        score = min(harmonic_ratio / 0.3, 1.0)
        
        return float(np.clip(score, 0, 1))
    
    def _detect_mechanical_fault(self, audio: np.ndarray, temporal_features: Dict) -> float:
        """检测机械故障 (特征: 能量波动大、峰值因子高)"""
        crest_factor = temporal_features["crest_factor"]
        energy_std = temporal_features["energy_std"]
        
        # 机械故障通常有较高的峰值因子
        cf_score = min((crest_factor - 3) / 5, 1.0)  # 正常约3, 故障可能达到8+
        cf_score = max(cf_score, 0)
        
        # 能量变化大
        energy_score = min(energy_std / 0.1, 1.0)
        
        score = 0.6 * cf_score + 0.4 * energy_score
        
        return float(np.clip(score, 0, 1))


# =============================================================================
# 增强版声学检测器 (支持深度学习)
# =============================================================================
class AcousticDetectorEnhanced(AcousticDetector):
    """
    增强版声学检测器
    结合传统方法和深度学习
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._model_registry = None
        self._use_deep_learning = False
        self._dl_model = None
    
    def set_model_registry(self, model_registry):
        """设置模型注册中心"""
        self._model_registry = model_registry
        
        # 检查深度学习模型是否可用
        try:
            transformer_id = self.config.model_ids.get("transformer")
            if transformer_id and model_registry.is_model_loaded(transformer_id):
                self._use_deep_learning = True
                logger.info("声学检测器: 深度学习模型已启用")
        except Exception as e:
            logger.warning(f"深度学习模型不可用: {e}")
            self._use_deep_learning = False
    
    def detect(self, audio: np.ndarray, sample_rate: int = None) -> Dict[str, Any]:
        """
        检测声学异常 (融合传统方法和深度学习)
        """
        # 1. 传统方法检测
        traditional_result = super().detect(audio, sample_rate)
        
        # 2. 深度学习检测 (如果可用)
        if self._use_deep_learning and self._model_registry:
            dl_result = self._detect_by_deep_learning(audio, sample_rate)
            
            if dl_result and dl_result.get("success"):
                # 融合两种方法的结果
                return self._fuse_results(traditional_result, dl_result)
        
        return traditional_result
    
    def _detect_by_deep_learning(self, audio: np.ndarray, sample_rate: int = None) -> Dict:
        """使用深度学习模型检测"""
        try:
            # 预处理
            audio = self._preprocess(audio)
            
            if sample_rate and sample_rate != self.config.sample_rate:
                audio = self._resample(audio, sample_rate, self.config.sample_rate)
            
            # 确保音频长度
            target_samples = int(self.config.sample_rate * self.config.audio_duration)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            elif len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            
            # 计算梅尔频谱
            mel_spec = self.feature_extractor.compute_mel_spectrogram(audio)
            
            # 准备输入
            input_data = mel_spec[np.newaxis, np.newaxis, ...]  # (1, 1, n_mels, time)
            
            # 模型推理
            model_id = self.config.model_ids.get("transformer")
            result = self._model_registry.infer(model_id, {"input": input_data})
            
            if result.get("success"):
                outputs = result.get("outputs", {})
                
                # 解析输出
                anomaly_score = outputs.get("anomaly_score", [[0]])[0][0]
                anomaly_logits = outputs.get("anomaly_logits", [[]])[0]
                
                if len(anomaly_logits) > 0:
                    anomaly_type_idx = np.argmax(anomaly_logits)
                    anomaly_types = [
                        "normal", "partial_discharge", "corona_discharge",
                        "transformer_hum", "mechanical_fault"
                    ]
                    anomaly_type = anomaly_types[min(anomaly_type_idx, len(anomaly_types) - 1)]
                else:
                    anomaly_type = "normal" if anomaly_score < 0.5 else "unknown"
                
                return {
                    "success": True,
                    "anomaly_type": anomaly_type,
                    "anomaly_score": float(anomaly_score),
                    "confidence": float(np.max(anomaly_logits)) if len(anomaly_logits) > 0 else anomaly_score,
                    "logits": anomaly_logits
                }
        
        except Exception as e:
            logger.warning(f"深度学习检测失败: {e}")
        
        return {"success": False}
    
    def _fuse_results(self, traditional: Dict, dl: Dict) -> Dict:
        """融合传统方法和深度学习结果"""
        trad_score = traditional.get("anomaly_score", 0)
        dl_score = dl.get("anomaly_score", 0)
        
        # 加权融合 (深度学习权重更高)
        fused_score = 0.3 * trad_score + 0.7 * dl_score
        
        # 确定异常类型
        if fused_score > self.config.anomaly_threshold:
            # 优先使用深度学习的分类结果
            anomaly_type = dl.get("anomaly_type", traditional.get("anomaly_type", "normal"))
        else:
            anomaly_type = "normal"
        
        return {
            "anomaly_type": anomaly_type,
            "anomaly_score": fused_score,
            "confidence": max(dl.get("confidence", 0), traditional.get("confidence", 0)),
            "traditional_result": traditional,
            "dl_result": dl,
            "spectrogram": traditional.get("spectrogram"),
            "features": traditional.get("features", {})
        }
