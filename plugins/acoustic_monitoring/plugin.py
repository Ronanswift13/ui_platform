#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声学监测插件入口
输变电站全自动AI巡检方案

基于深度学习的声学异常检测：
- 音频Transformer异常检测
- 超声波局部放电检测
- 声谱特征分析
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# 导入插件状态混入类
try:
    from platform_core.plugin_mixin import PluginStatusMixin, PluginStatus
except ImportError:
    # 如果导入失败，内联定义
    from enum import Enum

    class PluginStatus(str, Enum):
        UNLOADED = "unloaded"
        LOADING = "loading"
        READY = "ready"
        RUNNING = "running"
        ERROR = "error"
        DISABLED = "disabled"

    class PluginStatusMixin:
        def __init_status__(self):
            self._status = PluginStatus.UNLOADED
            self._last_error = ""

        @property
        def status(self):
            return getattr(self, '_status', PluginStatus.UNLOADED)

        @status.setter
        def status(self, value):
            self._status = value

        def set_status(self, status, error: str = ""):
            if isinstance(status, str):
                try:
                    status = PluginStatus(status)
                except ValueError:
                    status = PluginStatus.ERROR
            self._status = status
            if error:
                self._last_error = error

        def get_plugin_status(self) -> dict:
            return {
                'plugin_id': getattr(self, 'PLUGIN_ID', 'unknown'),
                'name': getattr(self, 'PLUGIN_NAME', 'Unknown'),
                'version': getattr(self, 'PLUGIN_VERSION', '0.0.0'),
                'status': self._status.value if hasattr(self, '_status') else 'unknown',
                'initialized': getattr(self, '_is_initialized', False),
                'last_error': getattr(self, '_last_error', '')
            }

logger = logging.getLogger(__name__)


# =============================================================================
# 配置数据类
# =============================================================================
@dataclass
class AcousticConfig:
    """声学监测配置"""
    # 采样配置
    sample_rate: int = 16000
    audio_duration: float = 2.0
    
    # 频谱配置
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    
    # 超声波配置 (用于局放检测)
    ultrasonic_sample_rate: int = 192000
    ultrasonic_freq_min: int = 20000
    ultrasonic_freq_max: int = 100000
    
    # 检测配置
    anomaly_threshold: float = 0.5
    alarm_accumulation_count: int = 3
    monitoring_window: float = 2.0
    hop_size: float = 0.5
    
    # 模型ID映射
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
    PARTIAL_DISCHARGE = "partial_discharge"      # 局部放电
    CORONA_DISCHARGE = "corona_discharge"        # 电晕放电
    POOR_CONTACT = "poor_contact"                # 接触不良
    BEARING_FAULT = "bearing_fault"              # 轴承故障
    TRANSFORMER_HUM = "transformer_hum"          # 变压器异常嗡鸣
    COOLING_FAN_FAULT = "cooling_fan_fault"      # 冷却风扇故障
    MECHANICAL_FAULT = "mechanical_fault"        # 机械故障
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [
            cls.NORMAL, cls.PARTIAL_DISCHARGE, cls.CORONA_DISCHARGE,
            cls.POOR_CONTACT, cls.BEARING_FAULT, cls.TRANSFORMER_HUM,
            cls.COOLING_FAN_FAULT, cls.MECHANICAL_FAULT
        ]
    
    @classmethod
    def get_severity(cls, anomaly_type: str) -> str:
        """获取异常严重程度"""
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
class AcousticMonitoringPlugin(PluginStatusMixin):
    """
    声学监测插件

    功能:
    1. 实时音频流监测
    2. 多类型异常检测
    3. 与噪声分类器结合提升准确度
    4. 异常事件告警
    """

    PLUGIN_ID = "acoustic_monitoring"
    PLUGIN_NAME = "声学监测"
    PLUGIN_VERSION = "1.0.0"

    def __init__(self, manifest=None, plugin_dir=None):
        """
        初始化声学监测插件

        Args:
            manifest: 插件清单 (PluginManifest)
            plugin_dir: 插件目录 (Path)
        """
        # 初始化状态管理
        self.__init_status__()

        self.manifest = manifest
        self.plugin_dir = plugin_dir

        # 从 manifest 获取配置或使用默认值
        config = {}
        if manifest and hasattr(manifest, 'config_schema'):
            config = manifest.config_schema or {}
        self.config = AcousticConfig(**config)
        self._model_registry = None
        self._detector = None
        self._analyzer = None
        self._is_initialized = False
        
        # 告警缓冲区
        self._alarm_buffer: List[Dict] = []
        self._alarm_accumulation = {}
    
    def init(self, model_registry=None) -> bool:
        """初始化插件"""
        try:
            self._model_registry = model_registry
            
            # 初始化检测器
            from .detector import AcousticDetectorEnhanced
            self._detector = AcousticDetectorEnhanced(self.config)
            
            if model_registry:
                self._detector.set_model_registry(model_registry)
            
            # 初始化分析器
            from .analyzer import AcousticAnalyzer
            self._analyzer = AcousticAnalyzer(self.config)
            
            self._is_initialized = True
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}")
            return False
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理音频输入
        
        Args:
            inputs: {
                "audio": np.ndarray,           # 音频波形 (samples,) 或 (channels, samples)
                "sample_rate": int,             # 采样率
                "roi_id": str,                  # ROI标识 (可选)
                "device_id": str,               # 设备ID (可选)
                "timestamp": float              # 时间戳 (可选)
            }
        
        Returns:
            {
                "success": bool,
                "anomaly_detected": bool,
                "anomaly_type": str,
                "anomaly_score": float,
                "confidence": float,
                "spectrogram": np.ndarray,      # 频谱图 (可选)
                "details": Dict,
                "alarms": List[Dict]
            }
        """
        if not self._is_initialized:
            return {
                "success": False,
                "error": "插件未初始化",
                "anomaly_detected": False
            }
        
        try:
            audio = inputs.get("audio")
            sample_rate = inputs.get("sample_rate", self.config.sample_rate)
            
            if audio is None:
                return {
                    "success": False,
                    "error": "缺少音频输入",
                    "anomaly_detected": False
                }
            
            # 1. 使用检测器进行推理
            detection_result = self._detector.detect(audio, sample_rate)
            
            # 2. 使用分析器进行详细分析
            analysis_result = self._analyzer.analyze(
                audio, sample_rate,
                detection_result.get("features", {})
            )
            
            # 3. 融合结果
            anomaly_type = detection_result.get("anomaly_type", AcousticAnomalyType.NORMAL)
            anomaly_score = detection_result.get("anomaly_score", 0.0)
            confidence = detection_result.get("confidence", 0.0)
            
            # 4. 累积告警判断
            anomaly_detected = anomaly_score > self.config.anomaly_threshold
            alarms = []
            
            if anomaly_detected:
                alarms = self._process_alarm(
                    anomaly_type=anomaly_type,
                    anomaly_score=anomaly_score,
                    confidence=confidence,
                    device_id=inputs.get("device_id", "unknown"),
                    roi_id=inputs.get("roi_id"),
                    timestamp=inputs.get("timestamp"),
                    analysis_details=analysis_result
                )
            
            return {
                "success": True,
                "anomaly_detected": anomaly_detected,
                "anomaly_type": anomaly_type,
                "anomaly_score": float(anomaly_score),
                "confidence": float(confidence),
                "severity": AcousticAnomalyType.get_severity(anomaly_type),
                "spectrogram": detection_result.get("spectrogram"),
                "frequency_analysis": analysis_result.get("frequency_analysis", {}),
                "time_analysis": analysis_result.get("time_analysis", {}),
                "details": {
                    "detection": detection_result,
                    "analysis": analysis_result
                },
                "alarms": alarms
            }
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "anomaly_detected": False
            }
    
    def _process_alarm(self, anomaly_type: str, anomaly_score: float,
                       confidence: float, device_id: str,
                       roi_id: Optional[str], timestamp: Optional[float],
                       analysis_details: Dict) -> List[Dict]:
        """处理告警累积逻辑"""
        import time
        
        current_time = timestamp or time.time()
        alarm_key = f"{device_id}_{anomaly_type}"
        
        # 更新累积计数
        if alarm_key not in self._alarm_accumulation:
            self._alarm_accumulation[alarm_key] = {
                "count": 0,
                "first_time": current_time,
                "last_time": current_time,
                "max_score": anomaly_score
            }
        
        acc = self._alarm_accumulation[alarm_key]
        acc["count"] += 1
        acc["last_time"] = current_time
        acc["max_score"] = max(acc["max_score"], anomaly_score)
        
        # 检查是否达到告警阈值
        alarms = []
        if acc["count"] >= self.config.alarm_accumulation_count:
            alarm = {
                "alarm_id": f"ACOUSTIC_{alarm_key}_{int(current_time)}",
                "alarm_type": "acoustic_anomaly",
                "anomaly_type": anomaly_type,
                "severity": AcousticAnomalyType.get_severity(anomaly_type),
                "device_id": device_id,
                "roi_id": roi_id,
                "message": self._generate_alarm_message(anomaly_type, acc["max_score"]),
                "anomaly_score": acc["max_score"],
                "confidence": confidence,
                "accumulation_count": acc["count"],
                "first_detected": acc["first_time"],
                "last_detected": current_time,
                "analysis_summary": analysis_details.get("summary", ""),
                "recommended_action": self._get_recommended_action(anomaly_type)
            }
            alarms.append(alarm)
            
            # 重置累积
            self._alarm_accumulation[alarm_key] = {
                "count": 0,
                "first_time": current_time,
                "last_time": current_time,
                "max_score": 0
            }
        
        return alarms
    
    def _generate_alarm_message(self, anomaly_type: str, score: float) -> str:
        """生成告警消息"""
        messages = {
            AcousticAnomalyType.PARTIAL_DISCHARGE: f"检测到局部放电信号 (置信度: {score:.1%})",
            AcousticAnomalyType.CORONA_DISCHARGE: f"检测到电晕放电声音 (置信度: {score:.1%})",
            AcousticAnomalyType.POOR_CONTACT: f"检测到接触不良异常声 (置信度: {score:.1%})",
            AcousticAnomalyType.BEARING_FAULT: f"检测到轴承故障特征音 (置信度: {score:.1%})",
            AcousticAnomalyType.TRANSFORMER_HUM: f"检测到变压器异常嗡鸣 (置信度: {score:.1%})",
            AcousticAnomalyType.COOLING_FAN_FAULT: f"检测到冷却风扇异常 (置信度: {score:.1%})",
            AcousticAnomalyType.MECHANICAL_FAULT: f"检测到机械故障声音 (置信度: {score:.1%})"
        }
        return messages.get(anomaly_type, f"检测到声学异常 (置信度: {score:.1%})")
    
    def _get_recommended_action(self, anomaly_type: str) -> str:
        """获取推荐处置动作"""
        actions = {
            AcousticAnomalyType.PARTIAL_DISCHARGE: "建议立即安排巡检，使用超声波检测仪定位放电源",
            AcousticAnomalyType.CORONA_DISCHARGE: "建议检查高压导体表面状况，清除尖端毛刺",
            AcousticAnomalyType.POOR_CONTACT: "建议检查接线端子紧固情况，测量接触电阻",
            AcousticAnomalyType.BEARING_FAULT: "建议检查旋转设备轴承，安排振动分析",
            AcousticAnomalyType.TRANSFORMER_HUM: "建议检查变压器铁芯紧固情况，测量直流偏磁",
            AcousticAnomalyType.COOLING_FAN_FAULT: "建议检查冷却风扇叶片和电机状况",
            AcousticAnomalyType.MECHANICAL_FAULT: "建议对机械设备进行全面检查"
        }
        return actions.get(anomaly_type, "建议安排人工巡检确认")
    
    def process_stream(self, audio_stream, callback=None):
        """
        处理音频流 (实时监测)
        
        Args:
            audio_stream: 音频流迭代器
            callback: 检测结果回调函数
        """
        buffer = []
        buffer_samples = int(self.config.sample_rate * self.config.monitoring_window)
        hop_samples = int(self.config.sample_rate * self.config.hop_size)
        
        for chunk in audio_stream:
            buffer.extend(chunk)
            
            while len(buffer) >= buffer_samples:
                window = np.array(buffer[:buffer_samples])
                
                result = self.process({
                    "audio": window,
                    "sample_rate": self.config.sample_rate
                })
                
                if callback:
                    callback(result)
                
                # 滑动窗口
                buffer = buffer[hop_samples:]
    
    def shutdown(self) -> bool:
        """关闭插件"""
        try:
            self._is_initialized = False
            self._alarm_buffer.clear()
            self._alarm_accumulation.clear()
            logger.info(f"[{self.PLUGIN_NAME}] 已关闭")
            return True
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 关闭失败: {e}")
            return False
    
    def infer(self, frame, rois, context):
        """实现BasePlugin抽象方法 - 执行推理"""
        return []

    def postprocess(self, results, rules):
        """实现BasePlugin抽象方法 - 后处理"""
        return []

    def healthcheck(self):
        """实现BasePlugin抽象方法 - 健康检查"""
        try:
            from platform_core.plugin_manager.base import HealthStatus
            return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else "未初始化")
        except ImportError:
            return {"healthy": self._is_initialized, "message": "OK" if self._is_initialized else "未初始化"}

    @property
    def plugin_info(self) -> Dict:
        """插件信息"""
        return {
            "id": self.PLUGIN_ID,
            "name": self.PLUGIN_NAME,
            "version": self.PLUGIN_VERSION,
            "description": "基于深度学习的声学异常检测，支持局部放电、电晕、接触不良等异常类型",
            "author": "AI巡检系统",
            "capabilities": [
                "partial_discharge_detection",
                "corona_detection",
                "contact_fault_detection",
                "bearing_fault_detection",
                "transformer_hum_detection",
                "stream_monitoring"
            ],
            "models_required": list(self.config.model_ids.values()),
            "supported_sample_rates": [16000, 44100, 48000, 192000],
            "supported_anomaly_types": AcousticAnomalyType.get_all()
        }
