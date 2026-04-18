#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气体泄漏检测插件 - 完整修复版
==============================

修复内容:
1. 添加 id 属性 (兼容 platform_core.plugin_manager)
2. 修复模块导入路径 (使用绝对导入)
3. 添加 set_status 方法
4. 支持多种构造函数签名

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
from collections import deque
from enum import Enum
from pathlib import Path
import numpy as np
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse as StarletteJSONResponse

from darkbreaker_sdk.interfaces import HealthStatus
from plugins._sensor_contract import (
    build_common_metadata,
    build_time_window,
    build_unified_temporal_output,
    build_virtual_result,
    clamp_confidence,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 插件状态枚举
# =============================================================================
class PluginStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


# =============================================================================
# 气体类型定义
# =============================================================================
class GasType:
    SF6 = "SF6"
    H2 = "H2"
    CO = "CO"
    C2H2 = "C2H2"
    CH4 = "CH4"
    C2H4 = "C2H4"
    C2H6 = "C2H6"
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [cls.SF6, cls.H2, cls.CO, cls.C2H2, cls.CH4, cls.C2H4, cls.C2H6]


# =============================================================================
# 配置数据类
# =============================================================================
@dataclass
class GasThreshold:
    attention: float = 500
    warning: float = 800
    alarm: float = 1000
    critical: float = 1500


@dataclass
class GasDetectionConfig:
    thresholds: Dict[str, GasThreshold] = field(default_factory=lambda: {
        "SF6": GasThreshold(attention=800, warning=1000, alarm=1500, critical=2000),
        "H2": GasThreshold(attention=100, warning=150, alarm=300, critical=500),
        "CO": GasThreshold(attention=200, warning=300, alarm=600, critical=1000),
        "C2H2": GasThreshold(attention=1, warning=5, alarm=10, critical=50),
        "CH4": GasThreshold(attention=50, warning=100, alarm=200, critical=500),
        "C2H4": GasThreshold(attention=50, warning=100, alarm=200, critical=500),
        "C2H6": GasThreshold(attention=50, warning=100, alarm=150, critical=300),
    })
    history_buffer_size: int = 1000
    trend_window_hours: int = 24
    leak_detection_window: int = 10
    leak_rate_threshold: float = 5.0
    sample_interval_seconds: float = 3600.0
    window_size: int = 24
    history_length: int = 24
    prediction_horizon: int = 24
    trend_prediction_enabled: bool = True
    runtime_mode: str = "standalone"
    rule_confidence: float = 1.0
    upgrade_placeholders: Dict[str, Any] = field(default_factory=lambda: {
        "prediction_model": "GL-TransLSTM/Transformer model hook",
        "anomaly_detection_model": "multi-gas anomaly model hook",
        "protocol_adapter": "gas sensor protocol adapter hook",
        "online_learning": "online calibration hook",
    })
    model_ids: Dict[str, str] = field(default_factory=lambda: {
        "sf6_forecast": "sf6_forecast",
        "multi_gas_forecast": "multi_gas_forecast",
        "health_trend": "equipment_health_trend",
        "lstm": "sf6_forecast",
        "transformer": "multi_gas_forecast",
    })


# =============================================================================
# 气体泄漏检测插件
# =============================================================================
class GasDetectionPlugin:
    """气体泄漏检测插件 V3.0 - 集成GL-TransLSTM深度学习"""

    @classmethod
    def create_standalone(cls, config=None):
        """Create plugin instance for standalone operation."""
        plugin_dir = Path(__file__).resolve().parent
        instance = cls()
        if config is None:
            from darkbreaker_sdk.utils import load_plugin_config
            config = load_plugin_config(plugin_dir / "configs" / "default.yaml")
        if hasattr(instance, 'init'):
            instance.init(config)
        elif hasattr(instance, 'initialize'):
            instance.initialize(config)
        return instance

    PLUGIN_ID = "gas_detection"
    PLUGIN_NAME = "气体泄漏检测"
    PLUGIN_VERSION = "3.0.0"

    def __init__(self, manifest=None, plugin_dir=None, config=None):
        """
        初始化气体检测插件
        
        支持多种初始化方式:
        1. GasDetectionPlugin(manifest, plugin_dir) - platform_core 方式
        2. GasDetectionPlugin(config=config) - 配置字典方式
        3. GasDetectionPlugin() - 默认配置
        """
        self.manifest = manifest
        self.plugin_dir = plugin_dir if plugin_dir else Path(__file__).parent
        
        # 状态管理
        self._status: PluginStatus = PluginStatus.UNLOADED
        self._last_error: str = ""
        
        # 处理配置
        if isinstance(config, GasDetectionConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self._parse_config(config)
        elif manifest and hasattr(manifest, 'config_schema'):
            self.config = self._parse_config(manifest.config_schema or {})
        else:
            self.config = GasDetectionConfig()
        
        self._model_registry = None
        self._predictor = None
        self._analyzer = None
        self._is_initialized = False
        self._history_buffers: Dict[str, Dict[str, deque]] = {}
        self._alarm_states: Dict[str, Dict] = {}
        
        logger.info(f"[{self.PLUGIN_NAME}] 实例已创建")
    
    def _parse_config(self, config_dict: Dict) -> GasDetectionConfig:
        parsed = GasDetectionConfig()
        sampling = config_dict.get("sampling", {})
        window = config_dict.get("window", {})
        runtime = config_dict.get("runtime", {})
        model_cfg = config_dict.get("model", {})
        alarm_rules = config_dict.get("alarm_rules", {})

        parsed.sample_interval_seconds = float(
            sampling.get(
                "sample_interval_seconds",
                config_dict.get("sample_interval_seconds", parsed.sample_interval_seconds),
            )
        )
        parsed.window_size = int(
            window.get("size", config_dict.get("history_length", parsed.window_size))
        )
        parsed.history_length = int(
            window.get(
                "history_length",
                config_dict.get("history_length", parsed.window_size),
            )
        )
        parsed.prediction_horizon = int(
            model_cfg.get(
                "prediction_horizon",
                config_dict.get("prediction_horizon", parsed.prediction_horizon),
            )
        )
        parsed.trend_prediction_enabled = bool(
            model_cfg.get(
                "trend_prediction_enabled",
                config_dict.get("trend_prediction_enabled", parsed.trend_prediction_enabled),
            )
        )
        parsed.history_buffer_size = int(
            window.get(
                "history_buffer_size",
                config_dict.get("history_buffer_size", parsed.history_buffer_size),
            )
        )
        parsed.trend_window_hours = int(
            window.get(
                "trend_window_hours",
                config_dict.get("trend_window_hours", parsed.trend_window_hours),
            )
        )
        parsed.leak_detection_window = int(
            window.get(
                "leak_detection_window",
                config_dict.get("leak_detection_window", parsed.leak_detection_window),
            )
        )
        parsed.leak_rate_threshold = float(
            alarm_rules.get(
                "leak_rate_threshold",
                config_dict.get("leak_rate_threshold", parsed.leak_rate_threshold),
            )
        )
        parsed.runtime_mode = runtime.get("mode", config_dict.get("runtime_mode", parsed.runtime_mode))
        parsed.rule_confidence = float(
            runtime.get("rule_confidence", config_dict.get("rule_confidence", parsed.rule_confidence))
        )
        parsed.upgrade_placeholders = config_dict.get(
            "upgrade_placeholders",
            parsed.upgrade_placeholders,
        )
        if isinstance(model_cfg.get("ids"), dict):
            parsed.model_ids.update(model_cfg["ids"])
        parsed.model_ids.setdefault("lstm", parsed.model_ids.get("sf6_forecast", "sf6_forecast"))
        parsed.model_ids.setdefault("transformer", parsed.model_ids.get("multi_gas_forecast", "multi_gas_forecast"))
        if "thresholds" in config_dict:
            for gas, thresh in config_dict["thresholds"].items():
                if isinstance(thresh, dict):
                    parsed.thresholds[gas] = GasThreshold(**thresh)
        return parsed
    
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
        if self.manifest and hasattr(self.manifest, 'name'):
            return self.manifest.name
        return self.PLUGIN_NAME
    
    @property
    def version(self) -> str:
        if self.manifest and hasattr(self.manifest, 'version'):
            return self.manifest.version
        return self.PLUGIN_VERSION
    
    @property
    def code_hash(self) -> str:
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
        return self._status
    
    @status.setter
    def status(self, value: PluginStatus):
        self._status = value
    
    def set_status(self, status, error: str = "") -> None:
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
        return {
            'plugin_id': self.id,
            'name': self.name,
            'version': self.version,
            'status': self._status.value,
            'initialized': self._is_initialized,
            'last_error': self._last_error,
            'capabilities': ["gas_concentration_monitoring", "leakage_detection", "trend_prediction"]
        }
    
    # =========================================================================
    # 初始化和关闭
    # =========================================================================
    
    def init(self, config_or_registry=None) -> bool:
        try:
            self.set_status(PluginStatus.LOADING)
            
            if isinstance(config_or_registry, dict):
                self.config = self._parse_config(config_or_registry)
            else:
                self._model_registry = config_or_registry
            
            # 使用绝对导入加载预测器和分析器
            self._predictor = self._load_predictor()
            self._analyzer = self._load_analyzer()
            
            self._is_initialized = True
            self.set_status(PluginStatus.READY)
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True
            
        except Exception as e:
            self.set_status(PluginStatus.ERROR, str(e))
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}")
            return False
    
    def _load_predictor(self):
        """加载预测器 V3.0 - 集成GL-TransLSTM"""
        try:
            module_name = 'plugins.gas_detection.predictor'
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                module = importlib.import_module(module_name)

            predictor_class = getattr(module, 'GasConcentrationPredictor', None)
            if predictor_class:
                predictor = predictor_class(self.config)

                # V3.0: 调用initialize初始化GL-TransLSTM
                if hasattr(predictor, 'initialize'):
                    predictor.initialize()
                    logger.info(f"[{self.PLUGIN_NAME}] 预测器V3.0初始化完成 (GL-TransLSTM)")

                # 兼容旧版model_registry
                if self._model_registry and hasattr(predictor, 'set_model_registry'):
                    predictor.set_model_registry(self._model_registry)

                logger.info(f"[{self.PLUGIN_NAME}] 预测器加载成功")
                return predictor
        except Exception as e:
            logger.warning(f"[{self.PLUGIN_NAME}] 预测器加载失败: {e}, 使用模拟模式")
        return None
    
    def _load_analyzer(self):
        """加载分析器 - 使用绝对导入"""
        try:
            module_name = 'plugins.gas_detection.analyzer'
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                module = importlib.import_module(module_name)
            
            analyzer_class = getattr(module, 'GasDataAnalyzer', None)
            if analyzer_class:
                logger.info(f"[{self.PLUGIN_NAME}] 分析器加载成功")
                return analyzer_class(self.config)
        except Exception as e:
            logger.warning(f"[{self.PLUGIN_NAME}] 分析器加载失败: {e}, 使用模拟模式")
        return None
    
    def shutdown(self) -> bool:
        try:
            # V3.0: 清理预测器资源
            if self._predictor and hasattr(self._predictor, 'cleanup'):
                self._predictor.cleanup()

            self._history_buffers.clear()
            self._alarm_states.clear()
            self._is_initialized = False
            self.set_status(PluginStatus.UNLOADED)
            logger.info(f"[{self.PLUGIN_NAME}] 已关闭")
            return True
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 关闭失败: {e}")
            return False
    
    def cleanup(self) -> None:
        self.shutdown()
    
    # =========================================================================
    # 核心处理方法
    # =========================================================================
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_initialized:
            return self._contract_error(inputs or {}, "插件未初始化")
        
        try:
            inputs = inputs or {}
            device_id = inputs.get("device_id", "unknown")
            timestamp = inputs.get("timestamp", time.time())
            gas_readings = inputs.get("gas_readings")
            if gas_readings is None:
                gas_readings = inputs.get("readings")
            if isinstance(gas_readings, dict) and "gases" in gas_readings:
                gas_readings = gas_readings.get("gases")
            if gas_readings is None:
                gas_readings = self._extract_window_readings(inputs)
            gas_readings = gas_readings or {}
            environmental = inputs.get("environmental", {})
            
            if not gas_readings:
                return self._contract_error(inputs, "缺少气体读数数据")
            
            # 更新历史数据
            self._update_history(device_id, timestamp, gas_readings, environmental)
            
            # 评估状态
            gas_status = self._evaluate_gas_status(gas_readings)
            predictions = self._predict_trends(device_id)
            trend_analysis = self._analyze_trends(device_id, gas_readings)
            leak_detection = self._detect_leakage(device_id)
            overall_status = self._determine_overall_status(gas_status, predictions, leak_detection)
            
            # 生成告警和建议
            alarms = self._generate_alarms(device_id, gas_status, leak_detection, timestamp)
            recommendations = self._generate_recommendations(gas_status, leak_detection, overall_status)
            
            label = self._contract_label(overall_status, leak_detection)
            value = self._contract_value(gas_readings, gas_status)
            confidence = self.config.rule_confidence
            metadata = self._build_contract_metadata(
                gas_status=gas_status,
                predictions=predictions,
                trend_analysis=trend_analysis,
                leak_detection=leak_detection,
            )
            time_window = build_time_window(
                inputs,
                window_size=self.config.window_size,
                sample_interval=self.config.sample_interval_seconds,
                timestamp_value=timestamp,
            )
            temporal_output = self._build_temporal_output(
                inputs=inputs,
                status=overall_status,
                label=label,
                confidence=confidence,
                gas_status=gas_status,
                predictions=predictions,
                trend_analysis=trend_analysis,
                leak_detection=leak_detection,
                alarms=alarms,
                recommendations=recommendations,
                time_window=time_window,
                value=value,
            )
            virtual_result = build_virtual_result(
                payload=inputs,
                plugin_id=self.id,
                plugin_version=self.version,
                code_hash=self.code_hash,
                device_id=device_id,
                roi_id=inputs.get("roi_id") or inputs.get("channel_id") or device_id,
                label=label,
                value=value,
                confidence=confidence,
                metadata=metadata,
                component_id="gas_sensor",
            )

            return {
                "success": True,
                "status": overall_status,
                "label": label,
                "value": value,
                "confidence": clamp_confidence(confidence),
                "metadata": metadata,
                "results": [virtual_result],
                **temporal_output,
                "device_id": device_id,
                "timestamp": timestamp,
                "overall_status": overall_status,
                "gas_status": gas_status,
                "gas_levels": gas_status,
                "predictions": predictions,
                "trend_analysis": trend_analysis,
                "leak_detection": leak_detection,
                "alarms": alarms,
                "recommendations": recommendations,
                "health_index": self._calculate_health_index(gas_status, leak_detection)
            }
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return self._contract_error(inputs if isinstance(inputs, dict) else {}, str(e))
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _update_history(self, device_id: str, timestamp: float, gas_readings: Dict, environmental: Dict):
        if device_id not in self._history_buffers:
            self._history_buffers[device_id] = {
                "timestamps": deque(maxlen=self.config.history_buffer_size),
                "gas_data": {gas: deque(maxlen=self.config.history_buffer_size) for gas in GasType.get_all()},
                "environmental": {
                    "temperature": deque(maxlen=self.config.history_buffer_size),
                    "humidity": deque(maxlen=self.config.history_buffer_size),
                    "pressure": deque(maxlen=self.config.history_buffer_size)
                }
            }
        
        buffer = self._history_buffers[device_id]
        buffer["timestamps"].append(timestamp)
        
        for gas, value in gas_readings.items():
            if gas in buffer["gas_data"]:
                buffer["gas_data"][gas].append(value)
        
        for key, value in environmental.items():
            if key in buffer["environmental"]:
                buffer["environmental"][key].append(value)
    
    def _get_history(self, device_id: str) -> Dict:
        if device_id not in self._history_buffers:
            return {}
        buffer = self._history_buffers[device_id]
        return {
            "timestamps": list(buffer["timestamps"]),
            "gas_data": {gas: list(values) for gas, values in buffer["gas_data"].items()},
            "environmental": {key: list(values) for key, values in buffer["environmental"].items()}
        }

    def _extract_window_readings(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract latest gas readings from unified sequence/window payloads."""
        source = inputs.get("structured_timeseries")
        if isinstance(source, dict):
            variables = source.get("variables", source)
            readings = {}
            for gas in GasType.get_all():
                values = variables.get(gas)
                if isinstance(values, list) and values:
                    readings[gas] = values[-1]
                elif isinstance(values, (int, float)):
                    readings[gas] = values
            if readings:
                return readings

        source = inputs.get("sensor_window") or inputs.get("sampled_sequence")
        if isinstance(source, dict):
            for key in ("readings", "gas_readings", "samples"):
                values = source.get(key)
                if isinstance(values, list) and values:
                    last = values[-1]
                    if isinstance(last, dict):
                        return last.get("readings") or last.get("gas_readings") or last
                if isinstance(values, dict):
                    return values.get("readings") or values.get("gas_readings") or values
        if isinstance(source, list) and source:
            last = source[-1]
            if isinstance(last, dict):
                return last.get("readings") or last.get("gas_readings") or last
        return {}
    
    def _evaluate_gas_status(self, gas_readings: Dict) -> Dict[str, Dict]:
        status = {}
        for gas, value in gas_readings.items():
            if gas not in self.config.thresholds:
                continue
            
            threshold = self.config.thresholds[gas]
            
            if value >= threshold.critical:
                level = "critical"
            elif value >= threshold.alarm:
                level = "alarm"
            elif value >= threshold.warning:
                level = "warning"
            elif value >= threshold.attention:
                level = "attention"
            else:
                level = "normal"
            
            status[gas] = {
                "value": value,
                "unit": "ppm",
                "level": level,
                "status": level,
                "threshold": {
                    "attention": threshold.attention,
                    "warning": threshold.warning,
                    "alarm": threshold.alarm,
                    "critical": threshold.critical
                },
                "percentage_of_alarm": round(value / threshold.alarm * 100, 1)
            }
        return status
    
    def _predict_trends(self, device_id: str) -> Dict[str, Any]:
        if not self.config.trend_prediction_enabled:
            return {
                "available": False,
                "reason": "趋势预测已禁用",
                "next_24h": "unknown",
                "next_7d": "unknown",
            }

        history = self._get_history(device_id)
        required_samples = max(24, int(self.config.history_length))
        if not history or len(history.get("timestamps", [])) < required_samples:
            return {
                "available": False,
                "reason": f"历史数据不足，需要至少{required_samples}个样本",
                "next_24h": "unknown",
                "next_7d": "unknown",
                "required_samples": required_samples,
                "observed_samples": len(history.get("timestamps", [])) if history else 0,
            }
        
        if self._predictor:
            return self._normalize_prediction_contract(self._predictor.predict(history), history)
        
        return {
            "available": True,
            "method": "rule_placeholder",
            "prediction_horizon": self.config.prediction_horizon,
            "next_24h": "stable",
            "next_7d": "stable",
            "confidence": 0.85,
            "gas_predictions": {},
            "predicted_alarms": [],
        }

    def _normalize_prediction_contract(self, prediction: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize predictor output before it enters the platform contract."""
        prediction = dict(prediction or {})
        if prediction.get("success") is False and not prediction.get("available"):
            return {
                "available": False,
                "reason": prediction.get("reason", "趋势预测不可用"),
                "next_24h": "unknown",
                "next_7d": "unknown",
                "raw_prediction": prediction,
            }

        gas_predictions = prediction.get("gas_predictions", {})
        direction = self._prediction_direction(gas_predictions, history)
        prediction.setdefault("available", bool(gas_predictions) or prediction.get("available", False))
        prediction.setdefault("success", True)
        prediction.setdefault("method", "unknown")
        prediction.setdefault("prediction_horizon", self.config.prediction_horizon)
        prediction.setdefault("gas_predictions", gas_predictions)
        prediction.setdefault("predicted_alarms", [])
        prediction.setdefault("next_24h", direction)
        prediction.setdefault("next_7d", direction)
        prediction.setdefault("confidence", self._prediction_confidence(prediction))
        return prediction

    def _prediction_direction(self, gas_predictions: Dict[str, Any], history: Dict[str, Any]) -> str:
        slopes = []
        for gas, item in gas_predictions.items():
            if isinstance(item, dict) and "trend_slope" in item:
                slopes.append(float(item["trend_slope"]))
                continue
            values = item.get("values") if isinstance(item, dict) else None
            history_values = history.get("gas_data", {}).get(gas, [])
            if values and history_values:
                slopes.append(float(values[-1]) - float(history_values[-1]))
        if not slopes:
            return "stable"
        if max(slopes) > 0.1:
            return "increasing"
        if min(slopes) < -0.1:
            return "decreasing"
        return "stable"

    def _prediction_confidence(self, prediction: Dict[str, Any]) -> float:
        gas_predictions = prediction.get("gas_predictions", {})
        confidences = [
            item.get("confidence")
            for item in gas_predictions.values()
            if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float))
        ]
        if confidences:
            return clamp_confidence(float(sum(confidences) / len(confidences)), default=0.75)
        if prediction.get("method") in ("traditional", "rule_placeholder"):
            return 0.75
        return 0.0

    def _analyze_trends(self, device_id: str, gas_readings: Dict[str, Any]) -> Dict[str, Any]:
        """Run analyzer in the same main chain as predictor, with observable fallback."""
        history = self._get_history(device_id)
        observed_samples = len(history.get("timestamps", [])) if history else 0
        if self._analyzer is None:
            return {
                "available": False,
                "reason": "分析器不可用",
                "observed_samples": observed_samples,
            }
        try:
            analysis = self._analyzer.analyze_trends(history, gas_readings)
            gas_trends = analysis.get("gas_trends", {})
            return {
                "available": bool(gas_trends),
                "observed_samples": observed_samples,
                "source": "GasDataAnalyzer.analyze_trends",
                **analysis,
            }
        except Exception as e:
            logger.warning(f"[{self.PLUGIN_NAME}] 趋势分析失败: {e}")
            return {
                "available": False,
                "reason": str(e),
                "observed_samples": observed_samples,
            }
    
    def _detect_leakage(self, device_id: str) -> Dict[str, Any]:
        """泄漏检测 V3.0 - 优先使用GL-TransLSTM"""
        # V3.0: 优先使用预测器的泄漏检测
        if self._predictor and hasattr(self._predictor, 'detect_leak'):
            try:
                leak_result = self._predictor.detect_leak()
                if leak_result.get("success"):
                    return leak_result
            except Exception as e:
                logger.debug(f"[{self.PLUGIN_NAME}] GL-TransLSTM泄漏检测失败: {e}")

        # 回退到传统方法
        history = self._get_history(device_id)
        if not history or len(history.get("timestamps", [])) < self.config.leak_detection_window:
            return {"detected": False, "reason": "数据不足"}

        for gas in [GasType.SF6, GasType.H2]:
            if gas in history["gas_data"]:
                values = list(history["gas_data"][gas])[-self.config.leak_detection_window:]
                if len(values) >= 2:
                    rate = (values[-1] - values[0]) / len(values)
                    if rate > self.config.leak_rate_threshold:
                        return {"detected": True, "gas": gas, "rate": rate, "severity": "warning"}

        return {"detected": False, "reason": "未检测到异常"}
    
    def _determine_overall_status(self, gas_status: Dict, predictions: Dict, leak_detection: Dict) -> str:
        if leak_detection.get("detected"):
            return "alarm"
        
        levels = [s.get("level", "normal") for s in gas_status.values()]
        
        if "critical" in levels:
            return "critical"
        elif "alarm" in levels:
            return "alarm"
        elif "warning" in levels:
            return "warning"
        elif "attention" in levels:
            return "attention"
        return "normal"
    
    def _generate_alarms(self, device_id: str, gas_status: Dict, leak_detection: Dict, timestamp: float) -> List[Dict]:
        alarms = []
        for gas, status in gas_status.items():
            if status["level"] in ["warning", "alarm", "critical"]:
                alarms.append({
                    "type": "gas_threshold",
                    "gas": gas,
                    "level": status["level"],
                    "value": status["value"],
                    "device_id": device_id,
                    "timestamp": timestamp,
                    "message": f"{gas} 浓度 {status['value']} ppm 超过{status['level']}阈值"
                })
        
        if leak_detection.get("detected"):
            alarms.append({
                "type": "leak_detection",
                "gas": leak_detection.get("gas"),
                "level": leak_detection.get("severity", "warning"),
                "device_id": device_id,
                "timestamp": timestamp,
                "message": f"检测到 {leak_detection.get('gas')} 泄漏"
            })
        
        return alarms
    
    def _generate_recommendations(self, gas_status: Dict, leak_detection: Dict, overall_status: str) -> List[str]:
        recommendations = []
        
        status_msgs = {
            "critical": "紧急: 立即检查设备，考虑停机检修",
            "alarm": "建议: 尽快安排设备检查",
            "warning": "注意: 加强监测频率"
        }
        if overall_status in status_msgs:
            recommendations.append(status_msgs[overall_status])
        
        if leak_detection.get("detected"):
            recommendations.append(f"建议: 检查 {leak_detection.get('gas')} 密封状态")
        
        if not recommendations:
            recommendations.append("气体浓度正常，建议继续监测")
        
        return recommendations
    
    def _calculate_health_index(self, gas_status: Dict, leak_detection: Dict) -> float:
        base_score = 100.0
        for gas, status in gas_status.items():
            pct = status.get("percentage_of_alarm", 0)
            if pct > 100:
                base_score -= 20
            elif pct > 80:
                base_score -= 10
            elif pct > 60:
                base_score -= 5
        
        if leak_detection.get("detected"):
            base_score -= 15
        
        return max(0, min(100, base_score))

    def _threshold_snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            gas: {
                "attention": threshold.attention,
                "warning": threshold.warning,
                "alarm": threshold.alarm,
                "critical": threshold.critical,
            }
            for gas, threshold in self.config.thresholds.items()
        }

    def _model_status(self, predictions: Optional[Dict[str, Any]] = None) -> str:
        if self._predictor and getattr(self._predictor, "_dl_initialized", False):
            return "available"
        if predictions and predictions.get("available"):
            return "placeholder"
        return "unavailable"

    def _build_contract_metadata(
        self,
        gas_status: Optional[Dict[str, Any]] = None,
        predictions: Optional[Dict[str, Any]] = None,
        trend_analysis: Optional[Dict[str, Any]] = None,
        leak_detection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        predictions = predictions or {}
        trend_analysis = trend_analysis or {}
        model_status = self._model_status(predictions)
        trend_available = bool(predictions.get("available", False) or trend_analysis.get("available", False))
        return build_common_metadata(
            modality="gas",
            sensor_type="gas_concentration",
            sample_interval=self.config.sample_interval_seconds,
            window_size=self.config.window_size,
            threshold_snapshot=self._threshold_snapshot(),
            runtime_mode=self.config.runtime_mode,
            algorithm_stage="threshold_rules_with_predictor_and_analyzer",
            model_status=model_status,
            fallback_level="model" if model_status == "available" else "rules",
            trend_prediction_available=trend_available,
            upgrade_placeholders=self.config.upgrade_placeholders,
            extra={
                "gas_status": gas_status or {},
                "leak_detection": leak_detection or {"detected": False},
                "prediction_available": trend_available,
                "trend_analysis_available": bool(trend_analysis.get("available", False)),
            },
        )

    def _contract_label(self, overall_status: str, leak_detection: Dict[str, Any]) -> str:
        if leak_detection.get("detected"):
            return "gas_leak"
        if overall_status in ("critical", "alarm"):
            return "abnormal"
        if overall_status in ("warning", "attention"):
            return "warning"
        return "normal"

    def _contract_value(self, readings: Dict[str, Any], gas_status: Dict[str, Any]) -> Dict[str, Any]:
        if not readings:
            return {}
        highest = max(
            gas_status.items(),
            key=lambda item: item[1].get("percentage_of_alarm", 0),
            default=("", {}),
        )
        return {
            "readings": readings,
            "primary_gas": highest[0],
            "primary_value": highest[1].get("value"),
            "primary_unit": highest[1].get("unit", "ppm"),
        }

    def _build_temporal_output(
        self,
        *,
        inputs: Dict[str, Any],
        status: str,
        label: str,
        confidence: float,
        gas_status: Dict[str, Any],
        predictions: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        leak_detection: Dict[str, Any],
        alarms: List[Dict[str, Any]],
        recommendations: List[str],
        time_window: Dict[str, Any],
        value: Dict[str, Any],
    ) -> Dict[str, Any]:
        anomaly_events = []
        abnormal_intervals = []
        reason_codes = []
        for gas, item in gas_status.items():
            level = item.get("level", "normal")
            if level in ("warning", "alarm", "critical"):
                reason = f"GAS_THRESHOLD_{level.upper()}_{gas}"
                reason_codes.append(reason)
                anomaly_events.append({
                    "event_id": f"{self.id}:{gas}:{time_window.get('start')}",
                    "event_type": "gas_threshold",
                    "label": "gas_leak" if leak_detection.get("detected") else "abnormal",
                    "severity": level,
                    "confidence": clamp_confidence(confidence),
                    "reason_codes": [reason],
                    "value": item.get("value"),
                    "metric_name": gas,
                    "unit": item.get("unit", "ppm"),
                    "time_window": time_window,
                    "evidence": {"threshold": item.get("threshold", {})},
                })
                abnormal_intervals.append({
                    "start": time_window.get("start"),
                    "end": time_window.get("end"),
                    "severity": level,
                    "reason_codes": [reason],
                    "metrics": [gas],
                })
        if leak_detection.get("detected"):
            reason_codes.append("GAS_LEAK_RATE_EXCEEDED")
        if not predictions.get("available") and not trend_analysis.get("available"):
            reason_codes.append("GAS_TREND_HISTORY_INSUFFICIENT")
        trend_direction = predictions.get("next_24h", "unknown")
        if trend_direction == "unknown" and trend_analysis.get("gas_trends"):
            first_trend = next(iter(trend_analysis["gas_trends"].values()))
            trend_direction = first_trend.get("pattern", "unknown")

        return build_unified_temporal_output(
            plugin_name=self.PLUGIN_NAME,
            task_type="trend_analysis",
            payload=inputs,
            status=status,
            label=label,
            severity="critical" if status == "critical" else status,
            confidence=confidence,
            summary={
                "device_id": inputs.get("device_id", "unknown"),
                "status": status,
                "primary_metric": value.get("primary_gas"),
                "alarm_count": len(alarms),
            },
            anomaly_events=anomaly_events,
            abnormal_intervals=abnormal_intervals,
            reason_codes=reason_codes,
            recommended_actions=recommendations,
            trend_diagnosis={
                "available": bool(predictions.get("available") or trend_analysis.get("available")),
                "direction": trend_direction,
                "confidence": clamp_confidence(predictions.get("confidence", 0.0), default=0.0),
                "reason": predictions.get("reason", "placeholder trend result"),
                "prediction": predictions,
                "analysis": trend_analysis,
            },
            evidence=[
                {
                    "type": "gas_status",
                    "metrics": list(gas_status.keys()),
                    "threshold_snapshot": self._threshold_snapshot(),
                },
                {
                    "type": "trend_analysis",
                    "available": bool(trend_analysis.get("available")),
                    "abnormal_trends": trend_analysis.get("abnormal_trends", []),
                }
            ],
            review_required=status in ("alarm", "critical") or leak_detection.get("detected", False),
            model_info={
                "model_status": self._model_status(predictions),
                "algorithm_stage": "threshold_rules_with_predictor_and_analyzer",
                "fallback_level": "model" if self._model_status(predictions) == "available" else "rules",
            },
            placeholders={
                "model_features_placeholder": list(gas_status.keys()),
                "sequence_embedding_placeholder": None,
                "temporal_pattern_placeholder": predictions.get("next_24h", "unknown"),
                "anomaly_score_trace_placeholder": [
                    item.get("percentage_of_alarm", 0) / 100.0 for item in gas_status.values()
                ],
                "root_cause_feature_placeholder": leak_detection,
            },
            time_window=time_window,
            input_protocol={
                "metric_names": list(gas_status.keys()),
                "sampling_or_timestamp": self.config.sample_interval_seconds,
                "data_quality": {
                    "status": "ok" if time_window.get("timestamp_valid", True) else "degraded",
                    "issues": [] if time_window.get("timestamp_valid", True) else [time_window.get("timestamp_reason")],
                },
            },
        )

    def _contract_error(self, inputs: Dict[str, Any], message: str) -> Dict[str, Any]:
        device_id = inputs.get("device_id", "unknown") if isinstance(inputs, dict) else "unknown"
        metadata = self._build_contract_metadata()
        time_window = build_time_window(
            inputs if isinstance(inputs, dict) else {},
            window_size=self.config.window_size,
            sample_interval=self.config.sample_interval_seconds,
        )
        temporal_output = build_unified_temporal_output(
            plugin_name=self.PLUGIN_NAME,
            task_type="trend_analysis",
            payload=inputs if isinstance(inputs, dict) else {},
            status="error",
            label="error",
            severity="error",
            confidence=0.0,
            summary={"device_id": device_id, "status": "error", "message": message},
            anomaly_events=[],
            abnormal_intervals=[],
            reason_codes=["INPUT_VALIDATION_ERROR"],
            recommended_actions=["检查 device_id、timestamp 与 readings/gas_readings 输入"],
            trend_diagnosis={"available": False, "direction": "unknown", "confidence": 0.0, "reason": message},
            evidence=[{"type": "validation_error", "message": message}],
            review_required=True,
            model_info={"model_status": self._model_status(), "fallback_level": "rules"},
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
            component_id="gas_sensor",
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
        }
    
    # =========================================================================
    # BasePlugin 兼容方法
    # =========================================================================
    
    def infer(self, frame, rois, context):
        return []

    def postprocess(self, results, rules):
        return []

    def healthcheck(self):
        return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else "未初始化")

    def get_standalone_routes(self) -> list:
        plugin = self

        async def gas_smoke(request: StarletteRequest):
            body = {}
            try:
                body = await request.json()
            except Exception as exc:
                logger.debug("gas smoke request has no JSON body: %s", exc)
            sample = {
                "device_id": "gas_smoke_sensor",
                "timestamp": time.time(),
                "readings": {"SF6": 120.0, "H2": 30.0, "CO": 80.0},
                "context": {"task_id": "gas-smoke", "site_id": "standalone"},
            }
            sample.update(body)
            if not sample.get("readings") and not sample.get("gas_readings"):
                sample["readings"] = {"SF6": 120.0, "H2": 30.0, "CO": 80.0}
            return StarletteJSONResponse(plugin.process(sample))

        return [
            {"path": "/api/gas/smoke", "endpoint": gas_smoke, "methods": ["GET", "POST"], "summary": "Run gas smoke sample"},
        ]

    @property
    def plugin_info(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": "基于时序预测模型的气体泄漏检测",
            "capabilities": ["gas_concentration_monitoring", "leakage_detection", "trend_prediction"],
            "supported_gases": GasType.get_all()
        }


Plugin = GasDetectionPlugin
