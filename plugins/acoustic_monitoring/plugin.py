#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声学监测插件 - 完整修复版
==========================

修复内容:
1. 添加 id 属性 (兼容 platform_core.plugin_manager)
2. 修复模块导入路径 (使用绝对导入)
3. 添加 set_status 方法
4. 支持多种构造函数签名
5. 增强频谱分析输出

作者: AI巡检系统
版本: 1.0.1
"""

from __future__ import annotations
import logging
import time
import importlib
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse as StarletteJSONResponse
import numpy as np

from darkbreaker_sdk.interfaces import HealthStatus, PluginStatus, PluginManifest
from darkbreaker_sdk.schemas import BoundingBox, RecognitionResult, Alarm, AlarmLevel
from plugins._sensor_contract import (
    build_common_metadata,
    build_time_window,
    build_unified_temporal_output,
    build_virtual_result,
    clamp_confidence,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 配置数据类
# =============================================================================
@dataclass
class AcousticConfig:
    """声学监测配置"""
    # === 基础参数 ===
    sample_rate: int = 16000
    audio_duration: float = 2.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    ultrasonic_sample_rate: int = 192000
    ultrasonic_freq_min: int = 20000
    ultrasonic_freq_max: int = 100000
    anomaly_threshold: float = 0.5
    alarm_accumulation_count: int = 3
    monitoring_window: float = 2.0
    hop_size: float = 0.5
    runtime_mode: str = "standalone"
    upgrade_placeholders: Dict[str, Any] = field(default_factory=lambda: {
        "prediction_model": "audio trend prediction hook",
        "anomaly_detection_model": "acoustic anomaly model hook",
        "protocol_adapter": "microphone/ultrasonic adapter hook",
        "online_learning": "online acoustic baseline hook",
    })
    model_ids: Dict[str, str] = field(default_factory=lambda: {
        "transformer": "audio_anomaly_transformer",
        "ultrasonic": "ultrasonic_pd_detector",
        "classifier": "audio_classifier"
    })

    # === 局部放电 (PD) 检测阈值 ===
    pd_high_freq_energy_ratio: float = 0.4      # 高频能量占比阈值
    pd_impulse_density: float = 0.1              # 脉冲密度阈值
    pd_impulse_multiplier: float = 4.0           # 脉冲判定倍数 (能量差 > 均值*此值)

    # === 电晕放电 (Corona) 检测阈值 ===
    corona_centroid_min: float = 3000.0          # 频谱质心最低门限 (Hz)
    corona_optimal_freq_low: float = 5000.0      # 最佳检测频率下限 (Hz)
    corona_optimal_freq_high: float = 15000.0    # 最佳检测频率上限 (Hz)
    corona_flatness_threshold: float = 0.5       # 频谱平坦度阈值

    # === 轴承故障 (Bearing) 检测阈值 ===
    bearing_kurtosis_gate: float = 3.5           # 峰度门限 (正弦~1.5, 噪声~3.0, 冲击>4.0)
    bearing_freq_min: float = 20.0               # 故障频率搜索下限 (Hz)
    bearing_freq_max: float = 200.0              # 故障频率搜索上限 (Hz)
    bearing_kurtosis_scaling: float = 4.0        # 峰度评分缩放因子
    bearing_periodicity_threshold: float = 0.3   # 周期性阈值

    # === 变压器嗡鸣 (Transformer) 检测阈值 ===
    transformer_harmonic_freqs: List[float] = field(default_factory=lambda: [100, 200, 300, 400])
    transformer_bin_bandwidth: int = 3           # FFT bin 搜索半径 (±N bins)
    transformer_ratio_threshold: float = 0.5     # 谐波能量占比阈值

    # === 机械故障 (Mechanical) 检测阈值 ===
    mechanical_crest_factor_threshold: float = 4.0  # 峰值因子门限
    mechanical_energy_cv_threshold: float = 0.5     # 能量变异系数阈值
    mechanical_cf_weight: float = 0.7               # 峰值因子权重
    mechanical_energy_weight: float = 0.3           # 能量变异权重

    # === PD 包络解调参数 (Phase 5) ===
    pd_bandpass_low: int = 30000                    # PD 带通滤波下截止频率 (Hz)
    pd_bandpass_high: int = 100000                  # PD 带通滤波上截止频率 (Hz)
    pd_envelope_kurtosis_gate: float = 5.0          # 包络峰度门限 (低于此值减半PD分数)

    # === 轴承 Hilbert 包络参数 (Phase 5) ===
    bearing_bandpass_low: float = 1000.0            # 轴承共振带通下限 (Hz)
    bearing_bandpass_high: float = 5000.0           # 轴承共振带通上限 (Hz)

    # === 训练数据导出 (Phase 5) ===
    export_training_features: bool = False           # 是否在 detect() 结果中导出训练特征


# =============================================================================
# 异常类型定义
# =============================================================================
class AcousticAnomalyType:
    """声学异常类型"""
    NORMAL = "normal"
    PARTIAL_DISCHARGE = "partial_discharge"
    CORONA_DISCHARGE = "corona_discharge"
    POOR_CONTACT = "poor_contact"
    BEARING_FAULT = "bearing_fault"
    TRANSFORMER_HUM = "transformer_hum"
    COOLING_FAN_FAULT = "cooling_fan_fault"
    MECHANICAL_FAULT = "mechanical_fault"
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [
            cls.NORMAL, cls.PARTIAL_DISCHARGE, cls.CORONA_DISCHARGE,
            cls.POOR_CONTACT, cls.BEARING_FAULT, cls.TRANSFORMER_HUM,
            cls.COOLING_FAN_FAULT, cls.MECHANICAL_FAULT
        ]
    
    @classmethod
    def get_severity(cls, anomaly_type: str) -> str:
        severity_map = {
            cls.NORMAL: "info",
            cls.PARTIAL_DISCHARGE: "critical",
            cls.CORONA_DISCHARGE: "error",
            cls.POOR_CONTACT: "error",
            cls.BEARING_FAULT: "warning",
            cls.TRANSFORMER_HUM: "warning",
            cls.COOLING_FAN_FAULT: "warning",
            cls.MECHANICAL_FAULT: "error"
        }
        return severity_map.get(anomaly_type, "info")


# =============================================================================
# 声学监测插件
# =============================================================================
class AcousticMonitoringPlugin:
    """
    声学监测插件 - 完整修复版
    
    功能:
    1. 实时音频流监测
    2. 多类型异常检测
    3. 频谱分析与可视化
    4. 异常事件告警
    """

    PLUGIN_ID = "acoustic_monitoring"
    PLUGIN_NAME = "声学监测"
    PLUGIN_VERSION = "1.0.1"

    def __init__(self, manifest=None, plugin_dir=None, config=None):
        """
        初始化声学监测插件
        
        支持多种初始化方式:
        1. AcousticMonitoringPlugin(manifest, plugin_dir) - platform_core 方式
        2. AcousticMonitoringPlugin(config=config) - 配置字典方式
        3. AcousticMonitoringPlugin() - 默认配置
        """
        # 保存 manifest 和 plugin_dir (用于 id 属性)
        self.manifest = manifest
        self.plugin_dir = plugin_dir if plugin_dir else Path(__file__).parent
        
        # === 状态管理 ===
        self._status: PluginStatus = PluginStatus.UNLOADED
        self._last_error: str = ""
        
        # 处理配置
        if isinstance(config, AcousticConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self._parse_config(config)
        elif manifest and hasattr(manifest, 'config_schema'):
            self.config = self._parse_config(manifest.config_schema or {})
        else:
            self.config = AcousticConfig()
        
        self._model_registry = None
        self._detector = None
        self._analyzer = None
        self._is_initialized = False
        
        # 告警缓冲区
        self._alarm_buffer: List[Dict] = []
        self._alarm_accumulation = {}

        # Standalone session manager (lazily created)
        self._audio_manager = None
        self._ws_clients = None

        logger.info(f"[{self.PLUGIN_NAME}] 实例已创建")
    
    def _parse_config(self, config_dict: Dict) -> AcousticConfig:
        """解析配置字典"""
        detection_params = config_dict.get("detection_params", {})
        pd = detection_params.get("partial_discharge", {})
        corona = detection_params.get("corona_discharge", {})
        bearing = detection_params.get("bearing_fault", {})
        transformer = detection_params.get("transformer_hum", {})
        mechanical = detection_params.get("mechanical_fault", {})
        thresholds = config_dict.get("thresholds", {})
        runtime = config_dict.get("runtime", {})

        return AcousticConfig(
            # 基础参数
            sample_rate=config_dict.get("sample_rate", 16000),
            audio_duration=config_dict.get("audio_duration", 2.0),
            n_mels=config_dict.get("n_mels", 128),
            n_fft=config_dict.get("n_fft", 2048),
            hop_length=config_dict.get("hop_length", 512),
            ultrasonic_sample_rate=config_dict.get("ultrasonic_sample_rate", 192000),
            ultrasonic_freq_min=config_dict.get("ultrasonic_freq_min", 20000),
            ultrasonic_freq_max=config_dict.get("ultrasonic_freq_max", 100000),
            anomaly_threshold=detection_params.get(
                "anomaly_threshold",
                thresholds.get("anomaly_score", config_dict.get("anomaly_threshold", 0.5)),
            ),
            alarm_accumulation_count=config_dict.get("alarm_accumulation_count", 3),
            monitoring_window=detection_params.get("monitoring_window", 2.0),
            hop_size=detection_params.get("hop_size", 0.5),
            runtime_mode=runtime.get("mode", config_dict.get("runtime_mode", "standalone")),
            upgrade_placeholders=config_dict.get(
                "upgrade_placeholders",
                AcousticConfig().upgrade_placeholders,
            ),
            # 局部放电阈值
            pd_high_freq_energy_ratio=pd.get("high_freq_energy_ratio", 0.4),
            pd_impulse_density=pd.get("impulse_density", 0.1),
            pd_impulse_multiplier=pd.get("impulse_multiplier", 4.0),
            # 电晕放电阈值
            corona_centroid_min=corona.get("centroid_min", 3000.0),
            corona_optimal_freq_low=corona.get("optimal_freq_low", 5000.0),
            corona_optimal_freq_high=corona.get("optimal_freq_high", 15000.0),
            corona_flatness_threshold=corona.get("flatness_threshold", 0.5),
            # 轴承故障阈值
            bearing_kurtosis_gate=bearing.get("kurtosis_gate", 3.5),
            bearing_freq_min=bearing.get("freq_min", 20.0),
            bearing_freq_max=bearing.get("freq_max", 200.0),
            bearing_kurtosis_scaling=bearing.get("kurtosis_scaling", 4.0),
            bearing_periodicity_threshold=bearing.get("periodicity_threshold", 0.3),
            # 变压器嗡鸣阈值
            transformer_harmonic_freqs=transformer.get("harmonic_freqs", [100, 200, 300, 400]),
            transformer_bin_bandwidth=transformer.get("bin_bandwidth", 3),
            transformer_ratio_threshold=transformer.get("ratio_threshold", 0.5),
            # 机械故障阈值
            mechanical_crest_factor_threshold=mechanical.get("crest_factor_threshold", 4.0),
            mechanical_energy_cv_threshold=mechanical.get("energy_cv_threshold", 0.5),
            mechanical_cf_weight=mechanical.get("cf_weight", 0.7),
            mechanical_energy_weight=mechanical.get("energy_weight", 0.3),
            # PD 包络解调参数
            pd_bandpass_low=pd.get("bandpass_low", 30000),
            pd_bandpass_high=pd.get("bandpass_high", 100000),
            pd_envelope_kurtosis_gate=pd.get("envelope_kurtosis_gate", 5.0),
            # 轴承 Hilbert 包络参数
            bearing_bandpass_low=bearing.get("bandpass_low", 1000.0),
            bearing_bandpass_high=bearing.get("bandpass_high", 5000.0),
            # 训练数据导出
            export_training_features=config_dict.get("export_training_features", False),
        )
    
    # =========================================================================
    # 关键属性 (修复 'id' 属性缺失问题)
    # =========================================================================
    
    @property
    def id(self) -> str:
        """插件ID - 兼容 platform_core.plugin_manager"""
        if self.manifest and hasattr(self.manifest, 'id'):
            return self.manifest.id
        return self.PLUGIN_ID
    
    @property
    def name(self) -> str:
        """插件名称"""
        if self.manifest and hasattr(self.manifest, 'name'):
            return self.manifest.name
        return self.PLUGIN_NAME
    
    @property
    def version(self) -> str:
        """插件版本"""
        if self.manifest and hasattr(self.manifest, 'version'):
            return self.manifest.version
        return self.PLUGIN_VERSION
    
    @property
    def code_hash(self) -> str:
        """代码哈希"""
        import hashlib
        h = hashlib.sha256()
        plugin_file = self.plugin_dir / "plugin.py"
        if plugin_file.exists():
            h.update(plugin_file.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"
    
    # =========================================================================
    # 状态管理方法
    # =========================================================================
    
    @property
    def status(self) -> PluginStatus:
        """获取插件状态"""
        return self._status
    
    @status.setter
    def status(self, value: PluginStatus):
        """设置插件状态"""
        self._status = value
    
    def set_status(self, status, error: str = "") -> None:
        """设置插件状态 - 兼容 platform_core"""
        if isinstance(status, str):
            try:
                status = PluginStatus(status)
            except ValueError:
                status = PluginStatus.ERROR
        
        self._status = status
        if error:
            self._last_error = error
            logger.error(f"[{self.PLUGIN_NAME}] 状态: {status.value}, 错误: {error}")
    
    def get_plugin_status(self) -> Dict[str, Any]:
        """获取插件状态详情"""
        return {
            'plugin_id': self.id,
            'name': self.name,
            'version': self.version,
            'status': self._status.value,
            'initialized': self._is_initialized,
            'last_error': self._last_error,
            'capabilities': [
                "partial_discharge_detection",
                "acoustic_monitoring",
                "frequency_analysis"
            ]
        }
    
    # =========================================================================
    # 初始化和关闭
    # =========================================================================
    
    def init(self, config_or_registry=None) -> bool:
        """
        初始化插件
        
        Args:
            config_or_registry: 配置字典或模型注册器
        """
        try:
            self.set_status(PluginStatus.LOADING)
            
            # 处理参数
            if isinstance(config_or_registry, dict):
                self.config = self._parse_config(config_or_registry)
            else:
                self._model_registry = config_or_registry
            
            # === 修复: 使用绝对导入加载检测器 ===
            self._detector = self._load_detector()
            self._analyzer = self._load_analyzer()
            
            self._is_initialized = True
            self.set_status(PluginStatus.READY)
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True
            
        except Exception as e:
            self.set_status(PluginStatus.ERROR, str(e))
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}")
            return False
    
    def _load_detector(self):
        """加载检测器 - 使用绝对导入"""
        try:
            # 方法1: 尝试绝对导入
            module_name = 'plugins.acoustic_monitoring.detector'
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                module = importlib.import_module(module_name)
            
            detector_class = getattr(module, 'AcousticDetectorEnhanced', None)
            if detector_class is None:
                detector_class = getattr(module, 'AcousticDetector', None)
            
            if detector_class:
                detector = detector_class(self.config)
                if self._model_registry and hasattr(detector, 'set_model_registry'):
                    detector.set_model_registry(self._model_registry)
                logger.info(f"[{self.PLUGIN_NAME}] 检测器加载成功")
                return detector
        except Exception as e:
            logger.warning(f"[{self.PLUGIN_NAME}] 检测器加载失败: {e}, 使用模拟模式")
        
        return None
    
    def _load_analyzer(self):
        """加载分析器 - 使用绝对导入"""
        try:
            module_name = 'plugins.acoustic_monitoring.analyzer'
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                module = importlib.import_module(module_name)
            
            analyzer_class = getattr(module, 'AcousticAnalyzer', None)
            if analyzer_class:
                analyzer = analyzer_class(self.config)
                logger.info(f"[{self.PLUGIN_NAME}] 分析器加载成功")
                return analyzer
        except Exception as e:
            logger.warning(f"[{self.PLUGIN_NAME}] 分析器加载失败: {e}, 使用模拟模式")
        
        return None
    
    def shutdown(self) -> bool:
        """关闭插件"""
        try:
            self._is_initialized = False
            self._alarm_buffer.clear()
            self._alarm_accumulation.clear()
            self.set_status(PluginStatus.UNLOADED)
            logger.info(f"[{self.PLUGIN_NAME}] 已关闭")
            return True
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 关闭失败: {e}")
            return False
    
    def cleanup(self) -> None:
        """清理资源 - 兼容 BasePlugin"""
        self.shutdown()
    
    # =========================================================================
    # 核心处理方法
    # =========================================================================
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理音频输入"""
        inputs = inputs or {}
        if not self._is_initialized:
            return self._contract_error(inputs, "插件未初始化")
        
        try:
            audio = inputs.get("audio")
            if audio is None and "audio_buffer" in inputs:
                audio = inputs.get("audio_buffer")
            if audio is None:
                audio = self._extract_audio_window(inputs)
            explicit_audio = (
                "audio" in inputs
                or "audio_buffer" in inputs
                or "sampled_sequence" in inputs
                or "sensor_window" in inputs
                or "structured_timeseries" in inputs
            )
            if explicit_audio and audio is None:
                return self._contract_error(inputs, "缺少音频缓冲数据")
            if explicit_audio:
                try:
                    if len(audio) == 0:
                        return self._contract_error(inputs, "音频缓冲为空")
                except TypeError:
                    pass

            sample_rate = inputs.get("sample_rate", inputs.get("sampling_rate", self.config.sample_rate))
            device_id = inputs.get("device_id", "unknown")
            data_source = inputs.get("data_source", "real")
            
            # 如果没有真实音频数据，生成模拟数据
            if audio is None:
                audio = self._generate_mock_audio(sample_rate)
            
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            if audio.size and audio.size < max(8, min(self.config.n_fft, sample_rate)):
                min_len = max(8, min(self.config.n_fft, sample_rate))
                audio = np.pad(audio.astype(np.float32), (0, min_len - audio.size))
            
            # 1. 频谱分析
            frequency_analysis = self._analyze_frequency(audio, sample_rate)
            
            # 2. 检测分析
            if self._detector:
                detection_result = self._detector.detect(audio, sample_rate)
            else:
                detection_result = self._mock_detection()
            
            # 3. 详细分析
            if self._analyzer:
                analysis_result = self._analyzer.analyze(
                    audio, sample_rate,
                    detection_result.get("features", {})
                )
            else:
                analysis_result = self._mock_analysis()
            
            # 4. 结果融合
            anomaly_type = detection_result.get("anomaly_type", AcousticAnomalyType.NORMAL)
            anomaly_score = detection_result.get("anomaly_score", 0.0)
            confidence = detection_result.get("confidence", 0.95)
            
            anomaly_detected = anomaly_score > self.config.anomaly_threshold
            
            # 5. 局部放电检测
            pd_detected = anomaly_type == AcousticAnomalyType.PARTIAL_DISCHARGE
            pd_level = "normal"
            if pd_detected:
                pd_level = "critical" if anomaly_score > 0.8 else ("alarm" if anomaly_score > 0.6 else "warning")
            
            # 6. 生成告警和建议
            alarms = self._generate_alarms(anomaly_detected, anomaly_type, anomaly_score, device_id)
            recommendations = self._generate_recommendations(anomaly_detected, anomaly_type, pd_detected, pd_level)
            
            # 7. Visualization data (downsampled for WebSocket transfer)
            waveform = audio[::max(1, len(audio) // 500)][:500].tolist()
            spectrum_data = frequency_analysis.get("spectrum", {})
            status = self._contract_status(anomaly_detected, anomaly_type, anomaly_score)
            label = self._contract_label(anomaly_detected, anomaly_type)
            metadata = self._build_contract_metadata(
                frequency_analysis=frequency_analysis,
                model_used=bool(self._detector),
            )
            time_window = build_time_window(
                inputs,
                window_size=self.config.monitoring_window,
                sample_interval=self.config.hop_size,
            )
            temporal_output = self._build_temporal_output(
                inputs=inputs,
                status=status,
                label=label,
                anomaly_detected=anomaly_detected,
                anomaly_type=anomaly_type,
                anomaly_score=float(anomaly_score),
                confidence=confidence,
                frequency_analysis=frequency_analysis,
                recommendations=recommendations,
                time_window=time_window,
            )
            virtual_result = build_virtual_result(
                payload=inputs,
                plugin_id=self.id,
                plugin_version=self.version,
                code_hash=self.code_hash,
                device_id=device_id,
                roi_id=inputs.get("roi_id") or inputs.get("channel_id") or device_id,
                label=label,
                value=float(anomaly_score),
                confidence=confidence,
                metadata=metadata,
                component_id="audio_channel",
            )

            return {
                "success": True,
                "status": status,
                "label": label,
                "value": float(anomaly_score),
                "confidence": clamp_confidence(confidence),
                "metadata": metadata,
                "results": [virtual_result],
                **temporal_output,
                "data_source": data_source,
                "device_id": device_id,
                "anomaly_detected": anomaly_detected,
                "anomaly_type": anomaly_type,
                "anomaly_score": float(anomaly_score),
                "confidence": float(confidence),
                "severity": AcousticAnomalyType.get_severity(anomaly_type),
                "pd_detected": pd_detected,
                "pd_level": pd_level,
                "pd_confidence": float(confidence) if pd_detected else 0.0,
                "frequency_analysis": frequency_analysis,
                "spectrogram": detection_result.get("spectrogram"),
                "time_analysis": analysis_result.get("time_analysis", {}),
                "alarms": alarms,
                "recommendations": recommendations,
                "waveform": waveform,
                "spectrum": {
                    "frequencies": spectrum_data.get("frequencies", []),
                    "magnitudes": spectrum_data.get("magnitude_normalized", []),
                },
                "all_scores": detection_result.get("all_scores", {}),
                "training_features": detection_result.get("training_features"),
            }
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return self._contract_error(inputs, str(e))
    
    # =========================================================================
    # 频谱分析方法
    # =========================================================================
    
    def _analyze_frequency(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """详细频谱分析"""
        try:
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            
            n_fft = min(self.config.n_fft, len(audio))
            fft_result = np.fft.fft(audio, n=n_fft)
            fft_magnitude = np.abs(fft_result[:n_fft//2])
            fft_freqs = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2]
            
            dominant_idx = np.argmax(fft_magnitude)
            dominant_frequency = float(fft_freqs[dominant_idx])
            
            # 找谐波
            sorted_indices = np.argsort(fft_magnitude)[::-1]
            harmonics = []
            for idx in sorted_indices[:10]:
                freq = float(fft_freqs[idx])
                if freq > 0 and freq not in harmonics:
                    harmonics.append(freq)
                if len(harmonics) >= 5:
                    break
            
            noise_level_db = float(20 * np.log10(np.std(audio) + 1e-10))
            
            # 频谱数据
            max_points = 256
            step = max(1, len(fft_freqs) // max_points)
            spectrum_freqs = fft_freqs[::step].tolist()
            spectrum_magnitude = fft_magnitude[::step].tolist()
            max_mag = max(spectrum_magnitude) if spectrum_magnitude else 1
            spectrum_magnitude_normalized = [m / max_mag for m in spectrum_magnitude]
            
            return {
                "dominant_frequency": dominant_frequency,
                "harmonics": harmonics[:4],
                "noise_level_db": noise_level_db,
                "spectrum": {
                    "frequencies": spectrum_freqs,
                    "magnitude": spectrum_magnitude,
                    "magnitude_normalized": spectrum_magnitude_normalized
                },
                "band_energy": self._calculate_band_energy(fft_freqs, fft_magnitude)
            }
        except Exception as e:
            logger.error(f"频谱分析失败: {e}")
            return {
                "dominant_frequency": 50,
                "harmonics": [100, 150, 200],
                "noise_level_db": -45
            }
    
    def _calculate_band_energy(self, freqs: np.ndarray, magnitude: np.ndarray) -> Dict[str, float]:
        """计算频带能量"""
        bands = {
            "low": (0, 200),
            "mid_low": (200, 1000),
            "mid": (1000, 4000),
            "mid_high": (4000, 8000),
            "high": (8000, 20000)
        }
        band_energy = {}
        total_energy = np.sum(magnitude ** 2)
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            energy = np.sum(magnitude[mask] ** 2) if np.any(mask) else 0
            band_energy[band_name] = float(energy / (total_energy + 1e-10) * 100)
        
        return band_energy
    
    # =========================================================================
    # 模拟和辅助方法
    # =========================================================================
    
    def _generate_mock_audio(self, sample_rate: int, anomaly_type: str = "normal",
                              intensity: float = 0.0) -> np.ndarray:
        """多模态声学模拟器 - 生成各类变电站音频信号

        支持模拟:
        - normal: 正常工频背景 (50Hz + 谐波 + 环境噪声)
        - partial_discharge: 局部放电 (超短时宽带脉冲 + 高频能量)
        - corona_discharge: 电晕放电 (5-15kHz 持续嘶嘶声)
        - bearing_fault: 轴承故障 (周期性冲击 + 谐振衰减)
        - transformer_hum: 变压器嗡鸣 (强谐波能量集中)
        - mechanical_fault: 机械故障 (高峰值因子冲击)
        """
        duration = self.config.audio_duration
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # 基底信号: 50Hz 工频 + 100Hz 谐波 + 环境噪声
        audio = 0.3 * np.sin(2 * np.pi * 50 * t)
        audio += 0.15 * np.sin(2 * np.pi * 100 * t)
        audio += 0.05 * np.random.randn(n_samples)

        if anomaly_type == "partial_discharge":
            # 局部放电: 高频衰减正弦脉冲 (非宽带 randn)
            # 策略: 脉冲能量集中在 Nyquist/2 ~ Nyquist 频段
            #   → 高频能量比高 (触发 PD 检测)
            #   → 整体峰值因子适中 (不触发机械检测)
            #   → 整体峰度适中 (不触发轴承检测)
            n_pulses = np.random.poisson(50 * max(intensity, 0.1) * duration)
            n_pulses = max(1, min(n_pulses, 500))
            power_period = sample_rate / 50.0
            pulse_freq = sample_rate * 0.38  # 脉冲载频 ≈ Nyquist*0.76
            pulse_dur = max(int(sample_rate * 0.001), 8)  # ~1ms

            for _ in range(n_pulses):
                # 50Hz 相位锁定
                cycle = np.random.randint(0, max(1, int(n_samples / power_period)))
                phase_offset = np.random.choice([0.25, 0.75])
                pos = int(cycle * power_period + phase_offset * power_period)
                pos = min(max(pos, 0), n_samples - 1)

                width = min(pulse_dur, n_samples - pos)
                if width > 0:
                    pt = np.arange(width) / sample_rate
                    # 高频衰减正弦 (能量集中在上1/3频段)
                    pulse = np.sin(2 * np.pi * pulse_freq * pt)
                    envelope = np.exp(-pt * 3000)  # 快速衰减
                    audio[pos:pos + width] += intensity * 0.8 * pulse * envelope

            # 间歇性高频突发 (增强帧间能量变化)
            hf_noise = np.random.randn(n_samples)
            hf_fft = np.fft.rfft(hf_noise)
            freqs = np.arange(len(hf_fft)) * sample_rate / (2 * len(hf_fft))
            hf_fft[freqs < sample_rate / 3] = 0  # 仅保留上1/3频段
            hf_noise = np.fft.irfft(hf_fft, n=n_samples)
            burst_envelope = np.zeros(n_samples)
            burst_dur_samples = max(int(sample_rate * 0.008), 16)
            for _ in range(n_pulses // 3):
                bpos = np.random.randint(0, max(n_samples - burst_dur_samples, 1))
                burst_envelope[bpos:bpos + burst_dur_samples] = 1.0
            audio += intensity * 0.4 * hf_noise * burst_envelope

        elif anomaly_type == "corona_discharge":
            # 电晕放电: Butterworth 带通滤波 (5-15kHz 嘶嘶声)
            from scipy.signal import butter, lfilter
            noise = np.random.randn(n_samples)
            nyq = sample_rate / 2.0
            low = 5000.0 / nyq
            high = min(15000.0, nyq * 0.95) / nyq
            if 0 < low < high < 1.0:
                b, a = butter(4, [low, high], btype='band')
                hiss = lfilter(b, a, noise)
            else:
                # 采样率不够高时退回 FFT 方法
                noise_fft = np.fft.rfft(noise)
                freqs = np.arange(len(noise_fft)) * sample_rate / (2 * len(noise_fft))
                mask = np.exp(-0.5 * ((freqs - 10000) / 5000) ** 2)
                noise_fft *= mask
                hiss = np.fft.irfft(noise_fft, n=n_samples)
            audio += intensity * 0.6 * hiss

        elif anomaly_type == "bearing_fault":
            # 轴承故障: 周期性冲击 + AR(1) 衰减 + 包络调制
            from scipy.signal import butter, sosfilt, lfilter
            bearing_freq = 45.0
            impulse_period = int(sample_rate / bearing_freq)
            # 1) 确定性周期冲击 (确保高峰度和可检测周期性)
            for i in range(0, n_samples, impulse_period):
                jitter = int(np.random.normal(0, impulse_period * 0.02))
                pos = max(0, min(i + jitter, n_samples - 1))
                ring_len = int(sample_rate * 0.005)
                end = min(pos + ring_len, n_samples)
                ring_t = np.arange(end - pos) / sample_rate
                ring_signal = np.sin(2 * np.pi * 2500.0 * ring_t)
                env = np.exp(-ring_t * 500)
                audio[pos:end] += intensity * 2.0 * ring_signal * env
            # 2) AR(1) 随机冲击成分 (增加宽带噪声纹理)
            impact_prob = intensity * bearing_freq / sample_rate
            impact_train = (np.random.rand(n_samples) < impact_prob).astype(float)
            impact_response = lfilter([1], [1, -0.99], impact_train)
            # 包络调制 (模拟转速波动)
            envelope_mod = 1.0 + 0.2 * np.sin(2 * np.pi * bearing_freq * t)
            # 结构共振带通 (1-5kHz)
            nyq = sample_rate / 2.0
            bp_low = 1000.0 / nyq
            bp_high = min(5000.0, nyq * 0.95) / nyq
            if 0 < bp_low < bp_high < 1.0:
                sos = butter(4, [bp_low, bp_high], btype='band', output='sos')
                modulated = sosfilt(sos, impact_response) * envelope_mod
            else:
                modulated = impact_response * envelope_mod
            audio += intensity * 0.5 * modulated

        elif anomaly_type == "transformer_hum":
            # 变压器嗡鸣: 强化工频谐波能量
            harmonic_freqs = self.config.transformer_harmonic_freqs
            for i, hf in enumerate(harmonic_freqs):
                amp = intensity * (1.0 - 0.2 * i)
                audio += amp * np.sin(2 * np.pi * hf * t)

        elif anomaly_type == "mechanical_fault":
            # 机械故障: 不规则 Tukey 冲击 (高峰值因子, 但非周期性)
            # 与轴承故障区分: 机械故障冲击间隔不规则, 幅度随机
            from scipy.signal.windows import tukey
            n_impacts = int(15 * max(intensity, 0.1) * duration)
            n_impacts = max(3, min(n_impacts, 100))
            # 随机位置 (非等间隔, 避免触发周期性检测)
            positions = np.sort(np.random.randint(0, max(n_samples - 100, 1), n_impacts))
            for pos in positions:
                width = int(sample_rate * np.random.uniform(0.003, 0.008))  # 3-8ms 随机宽度
                end = min(pos + width, n_samples)
                actual_width = end - pos
                # 随机幅度 (0.5x ~ 1.5x) — 增大能量变异
                amp = intensity * 5.0 * np.random.uniform(0.5, 1.5)
                if actual_width > 1:
                    window = tukey(actual_width, alpha=0.5)
                    audio[pos:end] += amp * window
                elif actual_width == 1:
                    audio[pos] += amp

        return audio.astype(np.float32)

    def _extract_audio_window(self, inputs: Dict[str, Any]) -> Optional[List[float]]:
        """Extract waveform values from unified sequence/window payloads."""
        source = inputs.get("sensor_window") or inputs.get("sampled_sequence")
        if isinstance(source, dict):
            for key in ("audio_buffer", "samples", "values", "waveform"):
                values = source.get(key)
                if isinstance(values, list):
                    if values and isinstance(values[0], dict):
                        return [sample.get("value", 0.0) for sample in values]
                    return values
        if isinstance(source, list):
            if source and isinstance(source[0], dict):
                return [sample.get("value", 0.0) for sample in source]
            return source

        structured = inputs.get("structured_timeseries")
        if isinstance(structured, dict):
            variables = structured.get("variables", structured)
            for key in ("audio", "audio_buffer", "amplitude", "waveform"):
                values = variables.get(key)
                if isinstance(values, list):
                    return values
        return None
    
    def _mock_detection(self) -> Dict[str, Any]:
        return {
            "anomaly_type": AcousticAnomalyType.NORMAL,
            "anomaly_score": 0.15,
            "confidence": 0.95,
            "features": {}
        }
    
    def _mock_analysis(self) -> Dict[str, Any]:
        return {"time_analysis": {"rms": 0.1, "peak": 0.3, "crest_factor": 3.0}}
    
    def _generate_alarms(self, anomaly_detected, anomaly_type, anomaly_score, device_id) -> List[Dict]:
        if not anomaly_detected:
            return []
        return [{
            "type": "acoustic_anomaly",
            "anomaly_type": anomaly_type,
            "level": AcousticAnomalyType.get_severity(anomaly_type),
            "score": anomaly_score,
            "device_id": device_id,
            "timestamp": time.time(),
            "message": f"检测到声学异常: {anomaly_type}"
        }]
    
    def _generate_recommendations(self, anomaly_detected, anomaly_type, pd_detected, pd_level) -> List[str]:
        recommendations = []
        if pd_detected:
            level_msgs = {
                "critical": "紧急: 检测到严重局部放电，建议立即停机检查",
                "alarm": "告警: 检测到局部放电，建议尽快安排检修",
                "warning": "注意: 检测到轻微局部放电，建议加强监测"
            }
            recommendations.append(level_msgs.get(pd_level, "检测到局部放电"))
        if not recommendations:
            recommendations.append("设备运行正常，建议定期监测")
        return recommendations

    def _threshold_snapshot(self) -> Dict[str, Any]:
        return {
            "anomaly_score": self.config.anomaly_threshold,
            "partial_discharge": {
                "high_freq_energy_ratio": self.config.pd_high_freq_energy_ratio,
                "impulse_density": self.config.pd_impulse_density,
                "impulse_multiplier": self.config.pd_impulse_multiplier,
            },
            "mechanical_fault": {
                "crest_factor_threshold": self.config.mechanical_crest_factor_threshold,
                "energy_cv_threshold": self.config.mechanical_energy_cv_threshold,
            },
        }

    def _build_contract_metadata(
        self,
        frequency_analysis: Optional[Dict[str, Any]] = None,
        model_used: bool = False,
    ) -> Dict[str, Any]:
        model_status = "available" if model_used else "unavailable"
        return build_common_metadata(
            modality="acoustic",
            sensor_type="audio_waveform",
            sampling_rate=self.config.sample_rate,
            sample_interval=self.config.hop_size,
            window_size=self.config.monitoring_window,
            threshold_snapshot=self._threshold_snapshot(),
            runtime_mode=self.config.runtime_mode,
            algorithm_stage="signal_features_with_model_fallback",
            model_status=model_status,
            fallback_level="model" if model_used else "rules",
            trend_prediction_available=False,
            upgrade_placeholders=self.config.upgrade_placeholders,
            extra={
                "frequency_summary": {
                    "dominant_frequency": (frequency_analysis or {}).get("dominant_frequency"),
                    "noise_level_db": (frequency_analysis or {}).get("noise_level_db"),
                }
            },
        )

    def _contract_status(self, anomaly_detected: bool, anomaly_type: str, anomaly_score: float) -> str:
        if not anomaly_detected:
            return "normal"
        if anomaly_type == AcousticAnomalyType.PARTIAL_DISCHARGE or anomaly_score >= 0.8:
            return "alarm"
        return "warning"

    def _contract_label(self, anomaly_detected: bool, anomaly_type: str) -> str:
        if not anomaly_detected:
            return "normal"
        if anomaly_type and anomaly_type != AcousticAnomalyType.NORMAL:
            return anomaly_type
        return "warning"

    def _build_temporal_output(
        self,
        *,
        inputs: Dict[str, Any],
        status: str,
        label: str,
        anomaly_detected: bool,
        anomaly_type: str,
        anomaly_score: float,
        confidence: float,
        frequency_analysis: Dict[str, Any],
        recommendations: List[str],
        time_window: Dict[str, Any],
    ) -> Dict[str, Any]:
        reason_codes = []
        anomaly_events = []
        abnormal_intervals = []
        severity = AcousticAnomalyType.get_severity(anomaly_type) if anomaly_detected else "normal"
        if anomaly_detected:
            reason = f"ACOUSTIC_{str(anomaly_type).upper()}"
            reason_codes.append(reason)
            anomaly_events.append({
                "event_id": f"{self.id}:{anomaly_type}:{time_window.get('start')}",
                "event_type": "acoustic_anomaly",
                "label": label,
                "severity": severity,
                "confidence": clamp_confidence(confidence),
                "reason_codes": [reason],
                "value": anomaly_score,
                "metric_name": "anomaly_score",
                "time_window": time_window,
                "evidence": {
                    "dominant_frequency": frequency_analysis.get("dominant_frequency"),
                    "noise_level_db": frequency_analysis.get("noise_level_db"),
                },
            })
            abnormal_intervals.append({
                "start": time_window.get("start"),
                "end": time_window.get("end"),
                "severity": severity,
                "reason_codes": [reason],
                "metrics": ["anomaly_score", "frequency_spectrum"],
            })
        else:
            reason_codes.append("ACOUSTIC_NORMAL_BASELINE")

        return build_unified_temporal_output(
            plugin_name=self.PLUGIN_NAME,
            task_type="anomaly_detection",
            payload=inputs,
            status=status,
            label=label,
            severity=severity,
            confidence=confidence,
            summary={
                "device_id": inputs.get("device_id", "unknown"),
                "status": status,
                "anomaly_type": anomaly_type,
                "anomaly_score": anomaly_score,
            },
            anomaly_events=anomaly_events,
            abnormal_intervals=abnormal_intervals,
            reason_codes=reason_codes,
            recommended_actions=recommendations,
            trend_diagnosis={
                "available": False,
                "direction": "unknown",
                "confidence": 0.0,
                "reason": "acoustic trend model reserved for second phase",
            },
            evidence=[
                {
                    "type": "frequency_analysis",
                    "dominant_frequency": frequency_analysis.get("dominant_frequency"),
                    "band_energy": frequency_analysis.get("band_energy", {}),
                }
            ],
            review_required=severity in ("warning", "error", "critical"),
            model_info={
                "model_status": "available" if self._detector else "unavailable",
                "algorithm_stage": "signal_features_with_model_fallback",
                "fallback_level": "model" if self._detector else "rules",
            },
            placeholders={
                "model_features_placeholder": ["spectrum", "band_energy", "time_analysis"],
                "sequence_embedding_placeholder": None,
                "temporal_pattern_placeholder": "audio_window_pattern",
                "anomaly_score_trace_placeholder": [anomaly_score],
                "root_cause_feature_placeholder": {
                    "dominant_frequency": frequency_analysis.get("dominant_frequency"),
                },
            },
            time_window=time_window,
            input_protocol={
                "metric_names": ["audio_amplitude", "anomaly_score", "frequency_spectrum"],
                "sampling_or_timestamp": inputs.get("sample_rate", inputs.get("sampling_rate", self.config.sample_rate)),
            },
        )

    def _contract_error(self, inputs: Dict[str, Any], message: str) -> Dict[str, Any]:
        device_id = inputs.get("device_id", "unknown") if isinstance(inputs, dict) else "unknown"
        metadata = self._build_contract_metadata(model_used=bool(self._detector))
        time_window = build_time_window(
            inputs if isinstance(inputs, dict) else {},
            window_size=self.config.monitoring_window,
            sample_interval=self.config.hop_size,
        )
        temporal_output = build_unified_temporal_output(
            plugin_name=self.PLUGIN_NAME,
            task_type="anomaly_detection",
            payload=inputs if isinstance(inputs, dict) else {},
            status="error",
            label="error",
            severity="error",
            confidence=0.0,
            summary={"device_id": device_id, "status": "error", "message": message},
            reason_codes=["INPUT_VALIDATION_ERROR"],
            recommended_actions=["检查 audio_buffer/sampled_sequence/sensor_window 与 timestamp 输入"],
            trend_diagnosis={"available": False, "direction": "unknown", "confidence": 0.0, "reason": message},
            evidence=[{"type": "validation_error", "message": message}],
            review_required=True,
            model_info={"model_status": "available" if self._detector else "unavailable", "fallback_level": "rules"},
            time_window=time_window,
        )
        virtual_result = build_virtual_result(
            payload=inputs if isinstance(inputs, dict) else {},
            plugin_id=self.id,
            plugin_version=self.version,
            code_hash=self.code_hash,
            device_id=device_id,
            roi_id=(inputs or {}).get("roi_id") or device_id if isinstance(inputs, dict) else device_id,
            label="error",
            value=None,
            confidence=0.0,
            metadata=metadata,
            component_id="audio_channel",
            failure_reason=message,
        )
        return {
            "success": False,
            "status": "error",
            "label": "error",
            "value": None,
            "confidence": 0.0,
            "metadata": metadata,
            "results": [virtual_result],
            **temporal_output,
            "error": message,
            "error_message": message,
            "anomaly_detected": False,
        }
    
    # =========================================================================
    # BasePlugin 兼容方法
    # =========================================================================
    
    def infer(self, frame, rois, context):
        """实现BasePlugin抽象方法"""
        return []

    def postprocess(self, results, rules):
        """实现BasePlugin抽象方法"""
        return []

    def healthcheck(self):
        """健康检查"""
        return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else "未初始化")

    @classmethod
    def create_standalone(cls, config=None):
        """Create plugin instance for standalone operation."""
        plugin_dir = Path(__file__).resolve().parent
        try:
            manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
        except ValueError:
            manifest = None
        instance = cls(manifest, plugin_dir)
        if config is None:
            from darkbreaker_sdk.utils import load_plugin_config
            config_path = plugin_dir / "configs" / "default.yaml"
            if config_path.exists():
                config = load_plugin_config(config_path)
            else:
                config = {}
        instance.init(config)
        return instance

    def _ensure_audio_manager(self):
        """Lazily create AudioSessionManager (needs runner's _ws_clients)."""
        if self._audio_manager is None:
            from plugins.acoustic_monitoring.standalone.audio_manager import AudioSessionManager
            self._audio_manager = AudioSessionManager(
                self, ws_clients=self._ws_clients
            )
        return self._audio_manager

    def get_standalone_routes(self) -> list:
        """Return additional standalone routes for acoustic monitoring.

        Provides 7 custom API routes:
          - /api/acoustic/simulation/start|stop  (POST)
          - /api/acoustic/monitoring/start|stop   (POST)
          - /api/acoustic/status                  (GET)
          - /api/acoustic/process                 (POST)
          - /api/acoustic/context                 (GET)
        """
        plugin = self  # capture for closures

        async def sim_start(request: StarletteRequest):
            body = {}
            try:
                body = await request.json()
            except Exception as exc:
                logger.debug("simulation start request has no JSON body: %s", exc)
            mgr = plugin._ensure_audio_manager()
            result = await mgr.start_simulation(body)
            return StarletteJSONResponse(result)

        async def sim_stop(request: StarletteRequest):
            mgr = plugin._ensure_audio_manager()
            result = await mgr.stop_simulation()
            return StarletteJSONResponse(result)

        async def mon_start(request: StarletteRequest):
            body = {}
            try:
                body = await request.json()
            except Exception as exc:
                logger.debug("monitoring start request has no JSON body: %s", exc)
            mgr = plugin._ensure_audio_manager()
            result = await mgr.start_monitoring(body)
            return StarletteJSONResponse(result)

        async def mon_stop(request: StarletteRequest):
            mgr = plugin._ensure_audio_manager()
            result = await mgr.stop_monitoring()
            return StarletteJSONResponse(result)

        async def acoustic_status(request: StarletteRequest):
            import json as _json
            mgr = plugin._ensure_audio_manager()
            status = mgr.get_status()
            # Sanitize: last_result may contain numpy types
            sanitized = _json.loads(_json.dumps(status, default=str))
            return StarletteJSONResponse(sanitized)

        async def process_audio_route(request: StarletteRequest):
            import json as _json
            body = await request.json()
            audio_list = body.get("audio", [])
            sr = body.get("sample_rate", plugin.config.sample_rate)
            device_id = body.get("device_id", "api")
            audio = np.array(audio_list, dtype=np.float32) if audio_list else None
            result = plugin.process({
                "audio": audio,
                "sample_rate": sr,
                "device_id": device_id,
            })
            # Sanitize numpy types for JSON serialization
            sanitized = _json.loads(_json.dumps(result, default=str))
            return StarletteJSONResponse(sanitized)

        async def acoustic_context(request: StarletteRequest):
            return StarletteJSONResponse({
                "sample_rate": plugin.config.sample_rate,
                "audio_duration": plugin.config.audio_duration,
                "anomaly_threshold": plugin.config.anomaly_threshold,
                "n_mels": plugin.config.n_mels,
                "n_fft": plugin.config.n_fft,
                "anomaly_types": [
                    "normal", "partial_discharge", "corona",
                    "bearing", "transformer_hum", "mechanical",
                ],
            })

        async def acoustic_smoke(request: StarletteRequest):
            body = {}
            try:
                body = await request.json()
            except Exception as exc:
                logger.debug("acoustic smoke request has no JSON body: %s", exc)
            sample = body or {
                "device_id": "acoustic_smoke_channel",
                "sample_rate": plugin.config.sample_rate,
                "context": {"task_id": "acoustic-smoke", "site_id": "standalone"},
            }
            result = plugin.process(sample)
            import json as _json
            return StarletteJSONResponse(_json.loads(_json.dumps(result, default=str)))

        return [
            {"path": "/api/acoustic/simulation/start", "endpoint": sim_start, "methods": ["POST"], "summary": "Start simulation"},
            {"path": "/api/acoustic/simulation/stop", "endpoint": sim_stop, "methods": ["POST"], "summary": "Stop simulation"},
            {"path": "/api/acoustic/monitoring/start", "endpoint": mon_start, "methods": ["POST"], "summary": "Start monitoring"},
            {"path": "/api/acoustic/monitoring/stop", "endpoint": mon_stop, "methods": ["POST"], "summary": "Stop monitoring"},
            {"path": "/api/acoustic/status", "endpoint": acoustic_status, "methods": ["GET"], "summary": "Get acoustic session status"},
            {"path": "/api/acoustic/process", "endpoint": process_audio_route, "methods": ["POST"], "summary": "Process single audio buffer"},
            {"path": "/api/acoustic/context", "endpoint": acoustic_context, "methods": ["GET"], "summary": "Get plugin config context"},
            {"path": "/api/acoustic/smoke", "endpoint": acoustic_smoke, "methods": ["GET", "POST"], "summary": "Run acoustic smoke sample"},
        ]

    @property
    def plugin_info(self) -> Dict:
        """插件信息"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": "基于深度学习的声学异常检测",
            "capabilities": ["partial_discharge_detection", "acoustic_monitoring", "frequency_analysis"]
        }


# Alias for standalone runner compatibility
Plugin = AcousticMonitoringPlugin
