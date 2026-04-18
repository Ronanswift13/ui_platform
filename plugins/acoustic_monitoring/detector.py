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

    def compute_ultrasonic_features(self, audio: np.ndarray,
                                     freq_min: int = 20000,
                                     freq_max: int = 100000) -> Dict[str, Any]:
        """超声波频段特征提取 (线性/对数频带能量)

        当 sample_rate > 48000 时启用。使用线性频带而非 Mel 频带,
        因为 Mel 尺度针对人耳听觉设计, 不适用于超声波分析。
        """
        if self.sample_rate <= 48000:
            return {}

        spectrogram = self.compute_spectrogram(audio)
        freqs = np.linspace(0, self.sample_rate / 2, spectrogram.shape[0])

        mask = (freqs >= freq_min) & (freqs <= freq_max)
        us_spec = spectrogram[mask, :]
        us_freqs = freqs[mask]

        if us_spec.size == 0:
            return {"ultrasonic_available": False}

        # 线性等宽频带能量
        n_bands = 8
        band_edges = np.linspace(freq_min, freq_max, n_bands + 1)
        band_energies = []
        for i in range(n_bands):
            band_mask = (us_freqs >= band_edges[i]) & (us_freqs < band_edges[i + 1])
            energy = float(np.sum(us_spec[band_mask, :] ** 2))
            band_energies.append(energy)

        total_energy = sum(band_energies) + 1e-8
        band_ratios = [e / total_energy for e in band_energies]

        # 对数频带能量 (低频段分辨率更高)
        log_edges = np.logspace(np.log10(max(freq_min, 1)), np.log10(freq_max), n_bands + 1)
        log_band_energies = []
        for i in range(n_bands):
            band_mask = (us_freqs >= log_edges[i]) & (us_freqs < log_edges[i + 1])
            energy = float(np.sum(us_spec[band_mask, :] ** 2))
            log_band_energies.append(energy)

        return {
            "ultrasonic_available": True,
            "linear_band_energies": band_energies,
            "linear_band_ratios": band_ratios,
            "log_band_energies": log_band_energies,
            "ultrasonic_total_energy": float(total_energy),
            "band_edges_linear": band_edges.tolist(),
            "band_edges_log": log_edges.tolist(),
        }

    def compute_hf_impulse_analysis(self, audio: np.ndarray,
                                      freq_min: int = 20000) -> Dict[str, Any]:
        """高频脉冲时间分辨率分析

        检测超声频段中的短时脉冲事件 (局部放电特征),
        使用 3σ 阈值检测脉冲, 并分析脉冲的时间间距和周期性。
        """
        if self.sample_rate <= 48000:
            return {}

        spectrogram = self.compute_spectrogram(audio)
        freqs = np.linspace(0, self.sample_rate / 2, spectrogram.shape[0])

        hf_mask = freqs >= freq_min
        hf_energy_over_time = np.sum(spectrogram[hf_mask, :] ** 2, axis=0)

        if len(hf_energy_over_time) == 0:
            return {}

        # 3σ 阈值脉冲检测
        mean_hf = np.mean(hf_energy_over_time)
        std_hf = np.std(hf_energy_over_time)
        threshold = mean_hf + 3 * std_hf

        impulse_frames = np.where(hf_energy_over_time > threshold)[0]
        impulse_count = len(impulse_frames)
        impulse_density = impulse_count / max(len(hf_energy_over_time), 1)

        # 脉冲时间间距分析
        if len(impulse_frames) > 1:
            spacing = np.diff(impulse_frames) * (self.hop_length / self.sample_rate)
            mean_spacing = float(np.mean(spacing))
            std_spacing = float(np.std(spacing))
            periodicity = 1.0 - min(std_spacing / (mean_spacing + 1e-8), 1.0)
        else:
            mean_spacing = 0.0
            std_spacing = 0.0
            periodicity = 0.0

        return {
            "hf_impulse_count": impulse_count,
            "hf_impulse_density": float(impulse_density),
            "hf_impulse_mean_spacing_sec": mean_spacing,
            "hf_impulse_periodicity": periodicity,
            "hf_energy_profile": hf_energy_over_time.tolist(),
        }

    def compute_spectral_kurtosis(self, audio: np.ndarray) -> Dict[str, Any]:
        """计算谱峰度 — 每个频率 bin 跨帧的功率谱峰度

        高谱峰度 → 该频率存在瞬态/脉冲能量 (PD 特征)
        低谱峰度 → 该频率为稳态信号 (背景噪声)
        """
        spectrogram = self.compute_spectrogram(audio)  # (freq_bins, time_frames)
        power_spec = spectrogram ** 2

        n_frames = power_spec.shape[1]
        if n_frames < 4:
            return {"spectral_kurtosis_mean": 0.0, "spectral_kurtosis_max": 0.0,
                    "spectral_kurtosis_profile": []}

        # 每个频率 bin 的峰度
        mean_power = np.mean(power_spec, axis=1, keepdims=True) + 1e-12
        std_power = np.std(power_spec, axis=1, keepdims=True) + 1e-12
        sk = np.mean(((power_spec - mean_power) / std_power) ** 4, axis=1) - 3.0

        return {
            "spectral_kurtosis_mean": float(np.mean(sk)),
            "spectral_kurtosis_max": float(np.max(sk)),
            "spectral_kurtosis_profile": sk.tolist(),
        }

    def compute_hjorth_parameters(self, audio: np.ndarray) -> Dict[str, float]:
        """计算 Hjorth 参数 — 紧凑的时域特征描述

        Activity: 信号方差 (信号强度)
        Mobility: 一阶导数标准差 / 信号标准差 (主频率)
        Complexity: 二阶导 mobility / 一阶导 mobility (频带宽度)
        """
        # Activity
        activity = float(np.var(audio))

        # 一阶差分
        d1 = np.diff(audio)
        d1_var = np.var(d1) + 1e-12

        # 二阶差分
        d2 = np.diff(d1)
        d2_var = np.var(d2) + 1e-12

        # Mobility
        mobility = float(np.sqrt(d1_var / (activity + 1e-12)))

        # Complexity
        d1_mobility = np.sqrt(d2_var / d1_var)
        complexity = float(d1_mobility / (mobility + 1e-12))

        return {
            "hjorth_activity": activity,
            "hjorth_mobility": mobility,
            "hjorth_complexity": complexity,
        }

    def compute_mfcc_deltas(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """计算 MFCC 及其一阶、二阶差分 (39维特征矩阵)

        用于时间动态捕捉, 供序列模型 (LSTM/Transformer) 使用。
        返回: (39, T) 特征矩阵
        """
        mfcc = self.compute_mfcc(audio, n_mfcc)  # (n_mfcc, T)

        # 一阶差分 (Δ)
        if mfcc.shape[1] > 1:
            delta = np.zeros_like(mfcc)
            delta[:, 1:-1] = (mfcc[:, 2:] - mfcc[:, :-2]) / 2.0
            delta[:, 0] = mfcc[:, 1] - mfcc[:, 0]
            delta[:, -1] = mfcc[:, -1] - mfcc[:, -2]
        else:
            delta = np.zeros_like(mfcc)

        # 二阶差分 (ΔΔ)
        if delta.shape[1] > 1:
            delta2 = np.zeros_like(delta)
            delta2[:, 1:-1] = (delta[:, 2:] - delta[:, :-2]) / 2.0
            delta2[:, 0] = delta[:, 1] - delta[:, 0]
            delta2[:, -1] = delta[:, -1] - delta[:, -2]
        else:
            delta2 = np.zeros_like(delta)

        # 堆叠: [MFCC; Δ; ΔΔ] → (39, T)
        return np.vstack([mfcc, delta, delta2])

    def compute_training_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """聚合所有高级特征为标准化训练输出

        返回:
        - mfcc_deltas: (39, T) — 序列模型输入
        - mel_spectrogram: (n_mels, T) — CNN/ViT 输入
        - global_features: dict — 全局统计特征
        """
        mfcc_deltas = self.compute_mfcc_deltas(audio)
        mel_spec = self.compute_mel_spectrogram(audio)
        spectral = self.compute_spectral_features(audio)
        temporal = self.compute_temporal_features(audio)
        hjorth = self.compute_hjorth_parameters(audio)

        # 全局特征向量 (10维)
        global_features = {
            "spectral_centroid": spectral["spectral_centroid_mean"],
            "spectral_bandwidth": spectral["spectral_bandwidth_mean"],
            "spectral_rolloff": spectral["spectral_rolloff_mean"],
            "spectral_flatness": spectral["spectral_flatness_mean"],
            "zero_crossing_rate": temporal["zero_crossing_rate"],
            "rms_energy": temporal["rms_energy"],
            "crest_factor": temporal["crest_factor"],
            "hjorth_activity": hjorth["hjorth_activity"],
            "hjorth_mobility": hjorth["hjorth_mobility"],
            "hjorth_complexity": hjorth["hjorth_complexity"],
        }

        return {
            "mfcc_deltas": mfcc_deltas.tolist(),
            "mel_spectrogram": mel_spec.tolist(),
            "global_features": global_features,
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

        # 超声波特征 (sample_rate > 48kHz 时启用)
        ultrasonic_features = {}
        hf_impulse_features = {}
        if sample_rate and sample_rate > 48000:
            ultrasonic_features = self.feature_extractor.compute_ultrasonic_features(
                audio, self.config.ultrasonic_freq_min, self.config.ultrasonic_freq_max
            )
            hf_impulse_features = self.feature_extractor.compute_hf_impulse_analysis(
                audio, self.config.ultrasonic_freq_min
            )

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
        
        # 训练特征导出 (配置门控, 默认关闭)
        training_features = None
        if self.config.export_training_features:
            try:
                training_features = self.feature_extractor.compute_training_features(audio)
            except Exception as e:
                logger.warning(f"训练特征导出失败: {e}")

        return {
            "anomaly_type": anomaly_type,
            "anomaly_score": max_score if anomaly_type != "normal" else 0.0,
            "confidence": max_score,
            "all_scores": anomaly_scores,
            "spectrogram": mel_spec,
            "training_features": training_features,
            "features": {
                "spectral": spectral_features,
                "temporal": temporal_features,
                "ultrasonic": ultrasonic_features,
                "hf_impulse": hf_impulse_features,
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
        """抗混叠多相重采样 (自定义 Kaiser FIR 滤波器)"""
        if orig_sr == target_sr:
            return audio
        from math import gcd
        from scipy.signal import resample_poly, firwin
        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        # 自定义 FIR: Kaiser 窗, beta=8, ~80dB 阻带衰减
        max_rate = max(orig_sr, target_sr)
        cutoff = 0.5 * min(orig_sr, target_sr) / max_rate
        fir = firwin(101, cutoff, window=('kaiser', 8.0))
        return resample_poly(audio, up, down, window=fir).astype(audio.dtype)
    
    def _detect_partial_discharge(self, audio: np.ndarray, mel_spec: np.ndarray) -> float:
        """检测局部放电 (特征: 高频短脉冲)

        局部放电产生短暂的高频脉冲，特征是:
        1. 高频能量占比显著高于背景噪声
        2. 存在明显的能量突变 (脉冲)
        两个条件必须同时满足才会给出高分。

        注: 高频分析使用功率梅尔频谱 (非对数) 以正确计算能量比。
        脉冲检测使用时域短窗分析, 因为梅尔帧 (128ms) 太粗,
        无法捕捉微秒级 PD 脉冲。
        """
        # 1. 高频能量占比 (使用功率梅尔频谱)
        power_mel = np.exp(mel_spec)  # 撤销 log 变换
        high_freq_bins = power_mel.shape[0] * 2 // 3  # 上1/3频率
        high_freq_energy = np.sum(power_mel[high_freq_bins:])
        total_energy = np.sum(power_mel) + 1e-8
        high_freq_ratio = high_freq_energy / total_energy

        # 2. 时域脉冲检测 (高通滤波 → 短窗能量突变)
        # 先高通滤波去除工频背景 (50/100Hz), 仅保留 PD 脉冲成分
        # 再用短窗分析检测能量突变
        try:
            from scipy.signal import butter, sosfiltfilt
            nyq = self.config.sample_rate / 2.0
            hp_cutoff = min(2000.0, nyq * 0.5) / nyq  # 2kHz 高通
            if 0 < hp_cutoff < 1.0:
                sos_hp = butter(4, hp_cutoff, btype='high', output='sos')
                hf_audio = sosfiltfilt(sos_hp, audio)
            else:
                hf_audio = audio
        except Exception:
            hf_audio = audio

        short_window = max(int(self.config.sample_rate * 0.002), 32)  # ~2ms 窗口
        n_short_frames = max(len(hf_audio) // short_window, 1)
        short_energies = np.array([
            np.sum(hf_audio[i*short_window:(i+1)*short_window] ** 2)
            for i in range(n_short_frames)
        ])
        energy_diff = np.abs(np.diff(short_energies))
        mean_diff = np.mean(energy_diff) + 1e-8
        impulse_count = np.sum(energy_diff > mean_diff * self.config.pd_impulse_multiplier)
        impulse_density = impulse_count / max(len(energy_diff), 1)

        # 高频能量占比阈值，减少白噪声误报
        hf_score = min(high_freq_ratio / self.config.pd_high_freq_energy_ratio, 1.0)
        imp_score = min(impulse_density / self.config.pd_impulse_density, 1.0)

        # 两个特征使用乘法融合，要求两者同时存在
        score = np.sqrt(hf_score * imp_score)

        # 包络解调鉴别: 高采样率时区分 PD 与宽带机械噪声
        if self.config.sample_rate > 48000 and score > 0.2:
            env_result = self._envelope_demodulation_analysis(audio, self.config.sample_rate)
            if env_result["envelope_kurtosis"] < self.config.pd_envelope_kurtosis_gate:
                # 包络平滑 → 更可能是机械噪声而非 PD, 降低分数
                score *= 0.5

        return float(np.clip(score, 0, 1))

    def _detect_corona_discharge(self, audio: np.ndarray, spectral_features: Dict) -> float:
        """检测电晕放电 (特征: 持续高频嘶嘶声)

        电晕放电产生 5-15kHz 的持续嘶嘶声，频谱质心必须在高频区域。
        低于 3kHz 的质心表示信号以低频为主 (如工频谐波)，不可能是电晕。
        """
        centroid = spectral_features["spectral_centroid_mean"]

        # 质心低于阈值时，信号以低频为主，不可能是电晕放电
        if centroid < self.config.corona_centroid_min:
            return 0.0

        # 电晕的频谱质心通常在最优频率范围内
        if self.config.corona_optimal_freq_low <= centroid <= self.config.corona_optimal_freq_high:
            centroid_score = 1.0
        elif centroid > self.config.corona_optimal_freq_high:
            centroid_score = 0.5
        else:
            # 线性过渡
            centroid_score = (centroid - self.config.corona_centroid_min) / (self.config.corona_optimal_freq_low - self.config.corona_centroid_min)

        # 频谱平坦度高 (嘶嘶声较为平坦)
        flatness = spectral_features["spectral_flatness_mean"]
        flatness_score = min(flatness / self.config.corona_flatness_threshold, 1.0)

        # 两个特征都必须存在 (乘法融合)
        score = centroid_score * flatness_score

        return float(np.clip(score, 0, 1))
    
    def _detect_bearing_fault(self, audio: np.ndarray) -> float:
        """检测轴承故障 (特征: 周期性冲击)

        轴承故障产生周期性冲击脉冲，具有高峰度 (kurtosis > 3)。
        正常的工频信号 (50Hz 正弦波) 峰度约 1.5，不应触发此检测。
        使用峰度作为前置条件，避免将变电站正常工频信号误判为轴承故障。
        """
        # 峰度检查: 轴承故障的冲击脉冲使信号峰度显著高于正弦波
        # 正弦波峰度 ~1.5, 高斯噪声 ~3.0, 冲击脉冲 > 4.0
        n = len(audio)
        mean = np.mean(audio)
        std = np.std(audio) + 1e-8
        kurtosis = np.mean(((audio - mean) / std) ** 4)

        # 峰度低于阈值表示信号平滑无冲击，不可能是轴承故障
        if kurtosis < self.config.bearing_kurtosis_gate:
            return 0.0

        # Hilbert 包络自相关: 先提取共振频段包络, 再分析周期性
        sr = self.config.sample_rate
        try:
            from scipy.signal import butter, sosfiltfilt, hilbert
            nyq = sr / 2.0
            bp_low = self.config.bearing_bandpass_low / nyq
            bp_high = min(self.config.bearing_bandpass_high, nyq * 0.95) / nyq

            if 0 < bp_low < bp_high < 1.0:
                sos = butter(4, [bp_low, bp_high], btype='band', output='sos')
                filtered = sosfiltfilt(sos, audio)
                analytic = hilbert(filtered)
                envelope = np.abs(analytic)
                envelope -= np.mean(envelope)
            else:
                # 带通范围无效时退回原始信号
                envelope = audio - np.mean(audio)
        except Exception:
            envelope = audio - np.mean(audio)

        # 对包络计算自相关
        autocorr = np.correlate(envelope, envelope, mode='full')
        autocorr = autocorr[n-1:] / (autocorr[n-1] + 1e-8)

        # 寻找周期性峰值 (轴承故障频率范围)
        min_lag = int(sr / self.config.bearing_freq_max)
        max_lag = int(sr / self.config.bearing_freq_min)

        if max_lag < len(autocorr):
            search_range = autocorr[min_lag:max_lag]
            peak_value = np.max(search_range)

            # 结合峰度和周期性评分
            kurtosis_score = min((kurtosis - self.config.bearing_kurtosis_gate) / self.config.bearing_kurtosis_scaling, 1.0)
            periodicity_score = min(peak_value / self.config.bearing_periodicity_threshold, 1.0)
            score = 0.5 * kurtosis_score + 0.5 * periodicity_score
        else:
            score = 0

        return float(np.clip(score, 0, 1))
    
    def _detect_transformer_hum(self, audio: np.ndarray) -> float:
        """检测变压器异常嗡鸣 (特征: 100/120Hz及谐波)

        变压器异常嗡鸣表现为 100Hz (50Hz工频二次谐波) 及其倍频 (200/300/400Hz)
        的能量显著增强。使用能量 (幅度平方) 而非幅度进行比较，以准确反映频率集中度。
        """
        n = len(audio)
        fft_magnitude = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(n, 1 / self.config.sample_rate)
        fft_energy = fft_magnitude ** 2

        # 检测谐波能量
        harmonic_freqs = self.config.transformer_harmonic_freqs
        harmonic_energy = 0.0

        for hf in harmonic_freqs:
            idx = np.argmin(np.abs(freqs - hf))
            # 取峰值附近 +/- bandwidth bin 的能量之和
            low = max(0, idx - self.config.transformer_bin_bandwidth)
            high = min(len(fft_energy), idx + self.config.transformer_bin_bandwidth + 1)
            harmonic_energy += np.sum(fft_energy[low:high])

        # 使用能量比 (平方和)
        total_energy = np.sum(fft_energy) + 1e-8
        harmonic_ratio = harmonic_energy / total_energy

        # 谐波能量比评分
        score = min(harmonic_ratio / self.config.transformer_ratio_threshold, 1.0)

        return float(np.clip(score, 0, 1))
    
    def _detect_mechanical_fault(self, audio: np.ndarray, temporal_features: Dict) -> float:
        """检测机械故障 (特征: 能量波动大、峰值因子高)

        机械故障产生宽带振动和冲击，具有高峰值因子 (>4) 和能量波动。
        峰值因子是核心指标，能量变化作为辅助验证。
        """
        crest_factor = temporal_features["crest_factor"]
        rms_energy = temporal_features["rms_energy"]
        energy_std = temporal_features["energy_std"]

        # 机械故障通常有较高的峰值因子 (正常约2-3, 故障可能达到6+)
        cf_score = min((crest_factor - self.config.mechanical_crest_factor_threshold) / self.config.mechanical_crest_factor_threshold, 1.0)
        cf_score = max(cf_score, 0)

        # 能量变化使用相对变异系数 (相对于均值)
        mean_energy = rms_energy ** 2 * self.config.n_fft
        energy_cv = energy_std / (mean_energy + 1e-8)  # 变异系数
        energy_score = min(energy_cv / self.config.mechanical_energy_cv_threshold, 1.0)

        # 峰值因子是核心指标
        score = self.config.mechanical_cf_weight * cf_score + self.config.mechanical_energy_weight * energy_score

        return float(np.clip(score, 0, 1))

    def _envelope_demodulation_analysis(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """包络解调分析 — 区分 PD 脉冲与宽带机械噪声

        PD 脉冲 → 尖锐包络 (kurtosis >> 3)
        机械噪声 → 平滑包络 (kurtosis ≈ 3)
        """
        from scipy.signal import butter, sosfiltfilt, hilbert

        nyq = sr / 2.0
        low = self.config.pd_bandpass_low / nyq
        high = min(self.config.pd_bandpass_high, nyq * 0.95) / nyq

        # 安全检查: 带通范围必须有效
        if low >= high or low >= 1.0 or high <= 0:
            return {"envelope_kurtosis": 0.0, "envelope_peak_count": 0}

        try:
            sos = butter(4, [low, high], btype='band', output='sos')
            filtered = sosfiltfilt(sos, audio)
            analytic = hilbert(filtered)
            envelope = np.abs(analytic)

            # 包络峰度
            env_mean = np.mean(envelope)
            env_std = np.std(envelope) + 1e-8
            envelope_kurtosis = float(np.mean(((envelope - env_mean) / env_std) ** 4))

            # 包络峰值计数 (3σ 阈值)
            threshold = env_mean + 3 * env_std
            envelope_peak_count = int(np.sum(envelope > threshold))

            return {
                "envelope_kurtosis": envelope_kurtosis,
                "envelope_peak_count": envelope_peak_count,
            }
        except Exception as e:
            logger.warning(f"包络解调分析失败: {e}")
            return {"envelope_kurtosis": 0.0, "envelope_peak_count": 0}


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
