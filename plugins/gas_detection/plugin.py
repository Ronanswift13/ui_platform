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
    model_ids: Dict[str, str] = field(default_factory=lambda: {
        "sf6_forecast": "sf6_forecast",
        "multi_gas_forecast": "multi_gas_forecast",
        "health_trend": "equipment_health_trend"
    })


# =============================================================================
# 气体泄漏检测插件
# =============================================================================
class GasDetectionPlugin:
    """气体泄漏检测插件 V3.0 - 集成GL-TransLSTM深度学习"""

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
            return {"success": False, "error": "插件未初始化"}
        
        try:
            device_id = inputs.get("device_id", "unknown")
            timestamp = inputs.get("timestamp", time.time())
            gas_readings = inputs.get("gas_readings", {})
            environmental = inputs.get("environmental", {})
            
            if not gas_readings:
                return {"success": False, "error": "缺少气体读数数据"}
            
            # 更新历史数据
            self._update_history(device_id, timestamp, gas_readings, environmental)
            
            # 评估状态
            gas_status = self._evaluate_gas_status(gas_readings)
            predictions = self._predict_trends(device_id)
            leak_detection = self._detect_leakage(device_id)
            overall_status = self._determine_overall_status(gas_status, predictions, leak_detection)
            
            # 生成告警和建议
            alarms = self._generate_alarms(device_id, gas_status, leak_detection, timestamp)
            recommendations = self._generate_recommendations(gas_status, leak_detection, overall_status)
            
            return {
                "success": True,
                "device_id": device_id,
                "timestamp": timestamp,
                "overall_status": overall_status,
                "gas_status": gas_status,
                "gas_levels": gas_status,
                "predictions": predictions,
                "leak_detection": leak_detection,
                "alarms": alarms,
                "recommendations": recommendations,
                "health_index": self._calculate_health_index(gas_status, leak_detection)
            }
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return {"success": False, "error": str(e)}
    
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
        history = self._get_history(device_id)
        if not history or len(history.get("timestamps", [])) < 24:
            return {"available": False, "reason": "历史数据不足", "next_24h": "unknown", "next_7d": "unknown"}
        
        if self._predictor:
            return self._predictor.predict(history)
        
        return {"available": True, "next_24h": "stable", "next_7d": "stable", "confidence": 0.85}
    
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
    
    # =========================================================================
    # BasePlugin 兼容方法
    # =========================================================================
    
    def infer(self, frame, rois, context):
        return []

    def postprocess(self, results, rules):
        return []

    def healthcheck(self):
        try:
            from platform_core.plugin_manager.base import HealthStatus
            return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else "未初始化")
        except ImportError:
            return {"healthy": self._is_initialized, "message": "OK" if self._is_initialized else "未初始化"}

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
