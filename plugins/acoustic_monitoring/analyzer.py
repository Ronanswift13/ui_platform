#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声学分析器模块
提供详细的音频频谱分析功能

功能:
1. 频率成分分析
2. 时频分析
3. 异常特征提取
4. 诊断报告生成
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class AcousticAnalyzer:
    """
    声学信号详细分析器
    
    提供:
    1. 频率成分分析
    2. 谐波分析
    3. 时频特性分析
    4. 异常诊断建议
    """
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        
        # 电力设备特征频率参考
        self.reference_frequencies = {
            "power_line": 50,           # 工频
            "transformer_hum": [100, 200, 300],  # 变压器嗡鸣
            "corona": (5000, 20000),     # 电晕放电频率范围
            "partial_discharge": (20000, 100000),  # 局放超声频率
        }
    
    def analyze(self, audio: np.ndarray, sample_rate: int,
                features: Dict = None) -> Dict[str, Any]:
        """
        全面分析音频信号
        
        Args:
            audio: 音频波形
            sample_rate: 采样率
            features: 已提取的特征 (可选)
        
        Returns:
            分析结果字典
        """
        # 预处理
        audio = self._preprocess(audio)
        
        # 频率分析
        freq_analysis = self._analyze_frequency_components(audio, sample_rate)
        
        # 时域分析
        time_analysis = self._analyze_temporal_characteristics(audio, sample_rate)
        
        # 谐波分析
        harmonic_analysis = self._analyze_harmonics(audio, sample_rate)
        
        # 特殊频段分析
        band_analysis = self._analyze_frequency_bands(audio, sample_rate)
        
        # 生成诊断摘要
        summary = self._generate_summary(
            freq_analysis, time_analysis, harmonic_analysis, band_analysis
        )
        
        return {
            "frequency_analysis": freq_analysis,
            "time_analysis": time_analysis,
            "harmonic_analysis": harmonic_analysis,
            "band_analysis": band_analysis,
            "summary": summary,
            "diagnosis_suggestions": self._generate_diagnosis_suggestions(
                freq_analysis, harmonic_analysis, band_analysis
            )
        }
    
    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """预处理"""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        audio = audio - np.mean(audio)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio.astype(np.float32)
    
    def _analyze_frequency_components(self, audio: np.ndarray, 
                                      sample_rate: int) -> Dict[str, Any]:
        """频率成分分析"""
        n = len(audio)
        fft_result = np.fft.rfft(audio)
        fft_magnitude = np.abs(fft_result) / n
        freqs = np.fft.rfftfreq(n, 1 / sample_rate)
        
        # 找到主要频率成分
        peak_indices = self._find_peaks(fft_magnitude, num_peaks=10)
        dominant_frequencies = [
            {
                "frequency": float(freqs[idx]),
                "magnitude": float(fft_magnitude[idx]),
                "relative_power": float(fft_magnitude[idx] / np.max(fft_magnitude))
            }
            for idx in peak_indices
        ]
        
        # 频带能量分布
        bands = {
            "sub_bass": (0, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 4000),
            "presence": (4000, 6000),
            "brilliance": (6000, 20000),
            "ultrasonic": (20000, sample_rate // 2)
        }
        
        band_energies = {}
        total_energy = np.sum(fft_magnitude ** 2)
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(fft_magnitude[mask] ** 2)
            band_energies[band_name] = {
                "energy": float(band_energy),
                "percentage": float(band_energy / total_energy * 100) if total_energy > 0 else 0
            }
        
        return {
            "dominant_frequencies": dominant_frequencies,
            "band_energies": band_energies,
            "total_energy": float(total_energy),
            "spectral_peak_frequency": float(freqs[np.argmax(fft_magnitude)]),
            "frequency_resolution": float(sample_rate / n)
        }
    
    def _find_peaks(self, data: np.ndarray, num_peaks: int = 10, 
                    min_distance: int = 5) -> List[int]:
        """查找频谱峰值"""
        peaks = []
        data_smoothed = np.convolve(data, np.ones(3)/3, mode='same')
        
        for i in range(1, len(data_smoothed) - 1):
            if (data_smoothed[i] > data_smoothed[i-1] and 
                data_smoothed[i] > data_smoothed[i+1]):
                
                # 检查与已有峰值的距离
                if all(abs(i - p) >= min_distance for p in peaks):
                    peaks.append(i)
        
        # 按幅度排序，取前N个
        peaks.sort(key=lambda x: data[x], reverse=True)
        return peaks[:num_peaks]
    
    def _analyze_temporal_characteristics(self, audio: np.ndarray,
                                          sample_rate: int) -> Dict[str, Any]:
        """时域特性分析"""
        # 包络检测
        analytic_signal = self._hilbert_transform(audio)
        envelope = np.abs(analytic_signal)
        
        # 短时能量
        frame_size = int(0.02 * sample_rate)  # 20ms
        hop_size = int(0.01 * sample_rate)    # 10ms
        
        num_frames = (len(audio) - frame_size) // hop_size + 1
        short_time_energy = np.array([
            np.sum(audio[i*hop_size:i*hop_size+frame_size] ** 2)
            for i in range(num_frames)
        ])
        
        # 能量变化率
        energy_diff = np.abs(np.diff(short_time_energy))
        
        # 瞬态检测
        transient_threshold = np.mean(energy_diff) + 2 * np.std(energy_diff)
        transient_count = np.sum(energy_diff > transient_threshold)
        
        return {
            "duration_seconds": float(len(audio) / sample_rate),
            "rms_level": float(np.sqrt(np.mean(audio ** 2))),
            "peak_level": float(np.max(np.abs(audio))),
            "crest_factor": float(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-8)),
            "envelope_mean": float(np.mean(envelope)),
            "envelope_std": float(np.std(envelope)),
            "energy_variation_coefficient": float(np.std(short_time_energy) / (np.mean(short_time_energy) + 1e-8)),
            "transient_count": int(transient_count),
            "transient_rate": float(transient_count / (len(audio) / sample_rate))
        }
    
    def _hilbert_transform(self, x: np.ndarray) -> np.ndarray:
        """希尔伯特变换计算解析信号"""
        n = len(x)
        X = np.fft.fft(x)
        h = np.zeros(n)
        
        if n % 2 == 0:
            h[0] = h[n//2] = 1
            h[1:n//2] = 2
        else:
            h[0] = 1
            h[1:(n+1)//2] = 2
        
        return np.fft.ifft(X * h)
    
    def _analyze_harmonics(self, audio: np.ndarray, 
                           sample_rate: int) -> Dict[str, Any]:
        """谐波分析"""
        n = len(audio)
        fft_result = np.fft.rfft(audio)
        fft_magnitude = np.abs(fft_result) / n
        freqs = np.fft.rfftfreq(n, 1 / sample_rate)
        
        # 基频检测 (使用自相关)
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[n-1:]
        
        # 查找第一个峰值 (排除零点)
        min_period = int(sample_rate / 500)  # 最高500Hz
        max_period = int(sample_rate / 30)   # 最低30Hz
        
        fundamental_freq = 0
        if max_period < len(autocorr):
            search_range = autocorr[min_period:max_period]
            if len(search_range) > 0:
                peak_idx = np.argmax(search_range) + min_period
                fundamental_freq = sample_rate / peak_idx
        
        # 检测谐波
        harmonics = []
        if fundamental_freq > 0:
            for h in range(1, 11):  # 前10个谐波
                target_freq = fundamental_freq * h
                if target_freq > sample_rate / 2:
                    break
                
                # 找最近的FFT bin
                idx = np.argmin(np.abs(freqs - target_freq))
                harmonics.append({
                    "harmonic_number": h,
                    "expected_frequency": float(target_freq),
                    "actual_frequency": float(freqs[idx]),
                    "magnitude": float(fft_magnitude[idx])
                })
        
        # 检测电力系统特征频率 (50Hz及谐波)
        power_harmonics = []
        for h in range(1, 11):
            target_freq = 50 * h
            if target_freq > sample_rate / 2:
                break
            
            idx = np.argmin(np.abs(freqs - target_freq))
            power_harmonics.append({
                "harmonic_number": h,
                "frequency": float(freqs[idx]),
                "magnitude": float(fft_magnitude[idx])
            })
        
        # 计算总谐波失真 (THD)
        if len(harmonics) > 1:
            fundamental_magnitude = harmonics[0]["magnitude"] if harmonics else 1e-8
            harmonic_sum = sum(h["magnitude"]**2 for h in harmonics[1:])
            thd = np.sqrt(harmonic_sum) / fundamental_magnitude * 100
        else:
            thd = 0
        
        return {
            "fundamental_frequency": float(fundamental_freq),
            "harmonics": harmonics,
            "power_frequency_harmonics": power_harmonics,
            "total_harmonic_distortion_percent": float(thd),
            "has_strong_power_frequency": any(h["magnitude"] > 0.1 for h in power_harmonics[:3])
        }
    
    def _analyze_frequency_bands(self, audio: np.ndarray,
                                 sample_rate: int) -> Dict[str, Any]:
        """特殊频段分析"""
        n = len(audio)
        fft_result = np.fft.rfft(audio)
        fft_magnitude = np.abs(fft_result) / n
        freqs = np.fft.rfftfreq(n, 1 / sample_rate)
        
        results = {}
        
        # 变压器嗡鸣频段 (100-400Hz)
        mask_hum = (freqs >= 100) & (freqs <= 400)
        hum_energy = np.sum(fft_magnitude[mask_hum] ** 2)
        total_energy = np.sum(fft_magnitude ** 2) + 1e-8
        
        results["transformer_hum_band"] = {
            "frequency_range": "100-400Hz",
            "energy": float(hum_energy),
            "percentage": float(hum_energy / total_energy * 100),
            "anomaly_indicator": hum_energy / total_energy > 0.3
        }
        
        # 电晕放电频段 (5-20kHz)
        if sample_rate >= 40000:
            mask_corona = (freqs >= 5000) & (freqs <= 20000)
            corona_energy = np.sum(fft_magnitude[mask_corona] ** 2)
            
            results["corona_band"] = {
                "frequency_range": "5-20kHz",
                "energy": float(corona_energy),
                "percentage": float(corona_energy / total_energy * 100),
                "anomaly_indicator": corona_energy / total_energy > 0.2
            }
        
        # 局部放电频段 (20kHz以上)
        if sample_rate >= 100000:
            mask_pd = freqs >= 20000
            pd_energy = np.sum(fft_magnitude[mask_pd] ** 2)
            
            results["partial_discharge_band"] = {
                "frequency_range": ">20kHz",
                "energy": float(pd_energy),
                "percentage": float(pd_energy / total_energy * 100),
                "anomaly_indicator": pd_energy / total_energy > 0.1
            }
        
        # 机械振动频段 (20-200Hz)
        mask_mech = (freqs >= 20) & (freqs <= 200)
        mech_energy = np.sum(fft_magnitude[mask_mech] ** 2)
        
        results["mechanical_band"] = {
            "frequency_range": "20-200Hz",
            "energy": float(mech_energy),
            "percentage": float(mech_energy / total_energy * 100),
            "anomaly_indicator": mech_energy / total_energy > 0.4
        }
        
        return results
    
    def _generate_summary(self, freq_analysis: Dict, time_analysis: Dict,
                          harmonic_analysis: Dict, band_analysis: Dict) -> str:
        """生成分析摘要"""
        summary_parts = []
        
        # 主频率
        if freq_analysis.get("dominant_frequencies"):
            top_freq = freq_analysis["dominant_frequencies"][0]["frequency"]
            summary_parts.append(f"主频率成分: {top_freq:.1f}Hz")
        
        # 能量特性
        crest = time_analysis.get("crest_factor", 0)
        if crest > 6:
            summary_parts.append(f"峰值因子较高({crest:.1f}), 可能存在冲击信号")
        
        # 谐波特性
        if harmonic_analysis.get("has_strong_power_frequency"):
            summary_parts.append("检测到显著的工频谐波成分")
        
        thd = harmonic_analysis.get("total_harmonic_distortion_percent", 0)
        if thd > 10:
            summary_parts.append(f"总谐波失真 {thd:.1f}%")
        
        # 频段异常
        for band_name, band_info in band_analysis.items():
            if band_info.get("anomaly_indicator"):
                summary_parts.append(f"{band_info['frequency_range']} 频段能量偏高")
        
        # 瞬态
        transient_rate = time_analysis.get("transient_rate", 0)
        if transient_rate > 5:
            summary_parts.append(f"检测到频繁瞬态信号 ({transient_rate:.1f}/秒)")
        
        return "; ".join(summary_parts) if summary_parts else "信号特性正常"
    
    def _generate_diagnosis_suggestions(self, freq_analysis: Dict,
                                        harmonic_analysis: Dict,
                                        band_analysis: Dict) -> List[str]:
        """生成诊断建议"""
        suggestions = []
        
        # 变压器嗡鸣
        hum_band = band_analysis.get("transformer_hum_band", {})
        if hum_band.get("anomaly_indicator"):
            suggestions.append(
                "变压器嗡鸣频段能量偏高，建议检查铁芯紧固情况和直流偏磁"
            )
        
        # 电晕放电
        corona_band = band_analysis.get("corona_band", {})
        if corona_band.get("anomaly_indicator"):
            suggestions.append(
                "高频嘶嘶声能量偏高，建议检查高压导体表面是否有尖端放电"
            )
        
        # 局部放电
        pd_band = band_analysis.get("partial_discharge_band", {})
        if pd_band.get("anomaly_indicator"):
            suggestions.append(
                "超声频段能量偏高，建议使用超声波检测仪进一步定位"
            )
        
        # 机械振动
        mech_band = band_analysis.get("mechanical_band", {})
        if mech_band.get("anomaly_indicator"):
            suggestions.append(
                "低频振动能量偏高，建议检查旋转设备轴承和紧固件"
            )
        
        # 工频谐波
        if harmonic_analysis.get("has_strong_power_frequency"):
            suggestions.append(
                "工频谐波显著，可能存在电磁干扰或设备异常振动"
            )
        
        if not suggestions:
            suggestions.append("当前声学特征在正常范围内，建议继续监测")
        
        return suggestions
