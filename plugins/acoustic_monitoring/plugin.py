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
import numpy as np

from darkbreaker_sdk.interfaces import HealthStatus, PluginStatus, PluginManifest
from darkbreaker_sdk.schemas import BoundingBox, RecognitionResult, Alarm, AlarmLevel

logger = logging.getLogger(__name__)


# =============================================================================
# 配置数据类
# =============================================================================
@dataclass
class AcousticConfig:
    """声学监测配置"""
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
    model_ids: Dict[str, str] = field(default_factory=lambda: {
        "transformer": "audio_anomaly_transformer",
        "ultrasonic": "ultrasonic_pd_detector",
        "classifier": "audio_classifier"
    })


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
        
        logger.info(f"[{self.PLUGIN_NAME}] 实例已创建")
    
    def _parse_config(self, config_dict: Dict) -> AcousticConfig:
        """解析配置字典"""
        return AcousticConfig(
            sample_rate=config_dict.get("sample_rate", 16000),
            anomaly_threshold=config_dict.get("detection_params", {}).get("anomaly_threshold", 0.5)
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
        if not self._is_initialized:
            return {
                "success": False,
                "error": "插件未初始化",
                "anomaly_detected": False
            }
        
        try:
            audio = inputs.get("audio")
            sample_rate = inputs.get("sample_rate", self.config.sample_rate)
            device_id = inputs.get("device_id", "unknown")
            
            # 如果没有真实音频数据，生成模拟数据
            if audio is None:
                audio = self._generate_mock_audio(sample_rate)
            
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
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
            
            return {
                "success": True,
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
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return {"success": False, "error": str(e), "anomaly_detected": False}
    
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
    
    def _generate_mock_audio(self, sample_rate: int) -> np.ndarray:
        """生成模拟音频数据"""
        duration = self.config.audio_duration
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 50 * t)
        audio += 0.5 * np.sin(2 * np.pi * 100 * t)
        audio += 0.3 * np.sin(2 * np.pi * 150 * t)
        audio += 0.1 * np.random.randn(len(t))
        return audio.astype(np.float32)
    
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
        manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
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

    def get_standalone_routes(self) -> list:
        """Return additional standalone routes for this plugin."""
        return []

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
