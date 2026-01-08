#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气体泄漏检测插件入口
输变电站全自动AI巡检方案

基于时序预测模型的气体监测：
- SF6/H2/CO/C2H2 浓度监测
- LSTM/Transformer 趋势预测
- 泄漏检测与告警
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 气体类型定义
# =============================================================================
class GasType:
    """气体类型"""
    SF6 = "SF6"      # 六氟化硫
    H2 = "H2"        # 氢气
    CO = "CO"        # 一氧化碳
    C2H2 = "C2H2"    # 乙炔
    CH4 = "CH4"      # 甲烷
    C2H4 = "C2H4"    # 乙烯
    C2H6 = "C2H6"    # 乙烷
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [cls.SF6, cls.H2, cls.CO, cls.C2H2, cls.CH4, cls.C2H4, cls.C2H6]
    
    @classmethod
    def get_oil_dissolved_gases(cls) -> List[str]:
        """油中溶解气体"""
        return [cls.H2, cls.CO, cls.C2H2, cls.CH4, cls.C2H4, cls.C2H6]


# =============================================================================
# 配置数据类
# =============================================================================
@dataclass
class GasThreshold:
    """单一气体阈值配置"""
    attention: float    # 注意值
    warning: float      # 警告值
    alarm: float        # 告警值
    critical: float     # 严重值


@dataclass
class GasDetectionConfig:
    """气体检测配置"""
    # 历史数据配置
    history_length: int = 168  # 历史序列长度 (7天*24小时)
    prediction_horizon: int = 24  # 预测时间步 (24小时)
    
    # 采样配置
    sample_interval_seconds: int = 3600  # 采样间隔 (1小时)
    
    # 阈值配置 (ppm)
    thresholds: Dict[str, GasThreshold] = field(default_factory=lambda: {
        GasType.SF6: GasThreshold(800, 1000, 1500, 2000),
        GasType.H2: GasThreshold(100, 150, 300, 500),
        GasType.CO: GasThreshold(200, 300, 600, 1000),
        GasType.C2H2: GasThreshold(3, 5, 10, 20),
        GasType.CH4: GasThreshold(30, 50, 100, 150),
        GasType.C2H4: GasThreshold(50, 100, 150, 200),
        GasType.C2H6: GasThreshold(50, 100, 150, 200),
    })
    
    # 泄漏检测配置
    min_leak_rate: float = 5.0  # 最小泄漏率阈值 (ppm/hour)
    leak_detection_window: int = 20  # 泄漏检测窗口 (数据点)
    leak_confidence_threshold: float = 0.7  # 泄漏判定置信度阈值
    
    # 模型配置
    model_ids: Dict[str, str] = field(default_factory=lambda: {
        "lstm": "sf6_forecast",
        "transformer": "multi_gas_forecast",
        "health_trend": "equipment_health_trend"
    })


# =============================================================================
# 气体泄漏检测插件
# =============================================================================
class GasDetectionPlugin:
    """
    气体泄漏检测插件

    功能:
    1. 多气体浓度实时监测
    2. 基于LSTM/Transformer的趋势预测
    3. 泄漏检测与告警
    4. 设备健康评估
    """

    PLUGIN_ID = "gas_detection"
    PLUGIN_NAME = "气体泄漏检测"
    PLUGIN_VERSION = "1.0.0"

    def __init__(self, manifest=None, plugin_dir=None):
        """
        初始化气体检测插件

        Args:
            manifest: 插件清单 (PluginManifest)
            plugin_dir: 插件目录 (Path)
        """
        self.manifest = manifest
        self.plugin_dir = plugin_dir

        # 从 manifest 获取配置或使用默认值
        config = {}
        if manifest and hasattr(manifest, 'config_schema'):
            config = manifest.config_schema or {}
        self.config = GasDetectionConfig(**config)
        self._model_registry = None
        self._predictor = None
        self._analyzer = None
        self._is_initialized = False
        
        # 历史数据缓冲区 (每个设备一个)
        self._history_buffers: Dict[str, Dict[str, deque]] = {}
        
        # 告警状态
        self._alarm_states: Dict[str, Dict] = {}
    
    def init(self, model_registry=None) -> bool:
        """初始化插件"""
        try:
            self._model_registry = model_registry
            
            # 初始化预测器
            from .predictor import GasConcentrationPredictor
            self._predictor = GasConcentrationPredictor(self.config)
            
            if model_registry:
                self._predictor.set_model_registry(model_registry)
            
            # 初始化分析器
            from .analyzer import GasDataAnalyzer
            self._analyzer = GasDataAnalyzer(self.config)
            
            self._is_initialized = True
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}")
            return False
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理气体数据输入
        
        Args:
            inputs: {
                "device_id": str,               # 设备ID
                "timestamp": float,             # 时间戳
                "gas_readings": {               # 气体读数
                    "SF6": float,               # SF6浓度 (ppm)
                    "H2": float,                # H2浓度 (ppm)
                    "CO": float,                # CO浓度 (ppm)
                    "C2H2": float               # C2H2浓度 (ppm)
                    ...
                },
                "environmental": {              # 环境参数 (可选)
                    "temperature": float,       # 温度 (℃)
                    "humidity": float,          # 湿度 (%)
                    "pressure": float           # 气压 (kPa)
                }
            }
        
        Returns:
            {
                "success": bool,
                "device_id": str,
                "status": str,                  # normal/attention/warning/alarm/critical
                "gas_status": Dict,             # 各气体状态
                "predictions": Dict,            # 预测结果
                "leak_detection": Dict,         # 泄漏检测结果
                "trend_analysis": Dict,         # 趋势分析
                "alarms": List[Dict],           # 告警列表
                "recommendations": List[str]    # 建议措施
            }
        """
        if not self._is_initialized:
            return {
                "success": False,
                "error": "插件未初始化"
            }
        
        try:
            device_id = inputs.get("device_id", "unknown")
            timestamp = inputs.get("timestamp", time.time())
            gas_readings = inputs.get("gas_readings", {})
            environmental = inputs.get("environmental", {})
            
            if not gas_readings:
                return {
                    "success": False,
                    "error": "缺少气体读数数据"
                }
            
            # 1. 更新历史数据
            self._update_history(device_id, timestamp, gas_readings, environmental)
            
            # 2. 当前状态评估
            gas_status = self._evaluate_gas_status(gas_readings)
            
            # 3. 趋势预测
            predictions = self._predict_trends(device_id)
            
            # 4. 泄漏检测
            leak_detection = self._detect_leakage(device_id)
            
            # 5. 趋势分析
            trend_analysis = self._analyzer.analyze_trends(
                self._get_history(device_id),
                gas_readings
            )
            
            # 6. 综合状态判定
            overall_status = self._determine_overall_status(
                gas_status, predictions, leak_detection
            )
            
            # 7. 生成告警
            alarms = self._generate_alarms(
                device_id, gas_status, predictions, leak_detection,
                trend_analysis, overall_status, timestamp
            )
            
            # 8. 生成建议
            recommendations = self._generate_recommendations(
                gas_status, leak_detection, trend_analysis
            )
            
            return {
                "success": True,
                "device_id": device_id,
                "timestamp": timestamp,
                "status": overall_status,
                "gas_status": gas_status,
                "predictions": predictions,
                "leak_detection": leak_detection,
                "trend_analysis": trend_analysis,
                "alarms": alarms,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_history(self, device_id: str, timestamp: float,
                        gas_readings: Dict, environmental: Dict):
        """更新历史数据缓冲区"""
        if device_id not in self._history_buffers:
            self._history_buffers[device_id] = {
                "timestamps": deque(maxlen=self.config.history_length),
                "gas_data": {gas: deque(maxlen=self.config.history_length) 
                            for gas in GasType.get_all()},
                "environmental": {
                    "temperature": deque(maxlen=self.config.history_length),
                    "humidity": deque(maxlen=self.config.history_length),
                    "pressure": deque(maxlen=self.config.history_length)
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
        """获取设备历史数据"""
        if device_id not in self._history_buffers:
            return {}
        
        buffer = self._history_buffers[device_id]
        return {
            "timestamps": list(buffer["timestamps"]),
            "gas_data": {gas: list(values) for gas, values in buffer["gas_data"].items()},
            "environmental": {key: list(values) for key, values in buffer["environmental"].items()}
        }
    
    def _evaluate_gas_status(self, gas_readings: Dict) -> Dict[str, Dict]:
        """评估各气体当前状态"""
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
                "threshold": {
                    "attention": threshold.attention,
                    "warning": threshold.warning,
                    "alarm": threshold.alarm,
                    "critical": threshold.critical
                },
                "percentage_of_alarm": value / threshold.alarm * 100
            }
        
        return status
    
    def _predict_trends(self, device_id: str) -> Dict[str, Any]:
        """预测气体浓度趋势"""
        history = self._get_history(device_id)
        
        if not history or len(history.get("timestamps", [])) < 24:
            return {
                "available": False,
                "reason": "历史数据不足 (需要至少24小时数据)"
            }
        
        return self._predictor.predict(history)
    
    def _detect_leakage(self, device_id: str) -> Dict[str, Any]:
        """检测气体泄漏"""
        history = self._get_history(device_id)
        
        if not history or len(history.get("timestamps", [])) < self.config.leak_detection_window:
            return {
                "detected": False,
                "reason": "数据不足"
            }
        
        results = {}
        
        for gas in [GasType.SF6]:  # 主要检测SF6泄漏
            gas_history = history["gas_data"].get(gas, [])
            if len(gas_history) < self.config.leak_detection_window:
                continue
            
            # 取最近的数据窗口
            window = list(gas_history)[-self.config.leak_detection_window:]
            timestamps = list(history["timestamps"])[-self.config.leak_detection_window:]
            
            # 计算泄漏率 (线性回归斜率)
            leak_rate, r_squared = self._calculate_leak_rate(timestamps, window)
            
            # 判断是否泄漏
            is_leaking = (
                leak_rate > self.config.min_leak_rate and
                r_squared > self.config.leak_confidence_threshold
            )
            
            results[gas] = {
                "detected": is_leaking,
                "leak_rate": float(leak_rate),
                "leak_rate_unit": "ppm/hour",
                "confidence": float(r_squared),
                "trend": "increasing" if leak_rate > 0 else "stable" if leak_rate == 0 else "decreasing",
                "estimated_time_to_alarm": self._estimate_time_to_alarm(
                    gas, window[-1], leak_rate
                ) if is_leaking else None
            }
        
        # 综合判断
        any_leak = any(r.get("detected", False) for r in results.values())
        
        return {
            "detected": any_leak,
            "gas_details": results,
            "overall_confidence": max(
                (r.get("confidence", 0) for r in results.values()), default=0
            )
        }
    
    def _calculate_leak_rate(self, timestamps: List[float], 
                            values: List[float]) -> Tuple[float, float]:
        """计算泄漏率 (线性回归)"""
        n = len(timestamps)
        if n < 2:
            return 0, 0
        
        # 转换时间戳为小时
        t0 = timestamps[0]
        hours = [(t - t0) / 3600 for t in timestamps]
        
        # 线性回归
        x = np.array(hours)
        y = np.array(values)
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0, 0
        
        slope = numerator / denominator
        
        # R²计算
        y_pred = slope * (x - x_mean) + y_mean
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        return slope, max(0, r_squared)
    
    def _estimate_time_to_alarm(self, gas: str, current_value: float,
                                leak_rate: float) -> Optional[float]:
        """估计达到告警阈值的时间 (小时)"""
        if leak_rate <= 0:
            return None
        
        threshold = self.config.thresholds.get(gas)
        if not threshold:
            return None
        
        if current_value >= threshold.alarm:
            return 0
        
        hours_to_alarm = (threshold.alarm - current_value) / leak_rate
        return float(hours_to_alarm)
    
    def _determine_overall_status(self, gas_status: Dict,
                                  predictions: Dict,
                                  leak_detection: Dict) -> str:
        """确定综合状态"""
        # 检查当前状态
        levels = [s.get("level", "normal") for s in gas_status.values()]
        
        if "critical" in levels:
            return "critical"
        if "alarm" in levels:
            return "alarm"
        
        # 检查泄漏
        if leak_detection.get("detected"):
            return "warning"
        
        if "warning" in levels:
            return "warning"
        if "attention" in levels:
            return "attention"
        
        # 检查预测
        if predictions.get("available"):
            predicted_alarms = predictions.get("predicted_alarms", [])
            if any(a.get("severity") == "critical" for a in predicted_alarms):
                return "warning"
            if predicted_alarms:
                return "attention"
        
        return "normal"
    
    def _generate_alarms(self, device_id: str, gas_status: Dict,
                         predictions: Dict, leak_detection: Dict,
                         trend_analysis: Dict, overall_status: str,
                         timestamp: float) -> List[Dict]:
        """生成告警列表"""
        alarms = []
        
        # 气体浓度告警
        for gas, status in gas_status.items():
            level = status.get("level", "normal")
            if level in ["warning", "alarm", "critical"]:
                alarms.append({
                    "alarm_id": f"GAS_{device_id}_{gas}_{int(timestamp)}",
                    "alarm_type": "gas_concentration",
                    "gas_type": gas,
                    "severity": level,
                    "device_id": device_id,
                    "current_value": status["value"],
                    "threshold": status["threshold"][level],
                    "message": f"{gas} 浓度 {status['value']:.1f} ppm 超过{level}阈值",
                    "timestamp": timestamp
                })
        
        # 泄漏告警
        if leak_detection.get("detected"):
            for gas, details in leak_detection.get("gas_details", {}).items():
                if details.get("detected"):
                    eta = details.get("estimated_time_to_alarm")
                    message = f"检测到 {gas} 泄漏，泄漏率 {details['leak_rate']:.2f} ppm/h"
                    if eta:
                        message += f"，预计 {eta:.1f} 小时后达到告警阈值"
                    
                    alarms.append({
                        "alarm_id": f"LEAK_{device_id}_{gas}_{int(timestamp)}",
                        "alarm_type": "gas_leakage",
                        "gas_type": gas,
                        "severity": "warning",
                        "device_id": device_id,
                        "leak_rate": details["leak_rate"],
                        "confidence": details["confidence"],
                        "estimated_time_to_alarm": eta,
                        "message": message,
                        "timestamp": timestamp
                    })
        
        # 预测告警
        if predictions.get("available"):
            for pred_alarm in predictions.get("predicted_alarms", []):
                alarms.append({
                    "alarm_id": f"PRED_{device_id}_{pred_alarm['gas']}_{int(timestamp)}",
                    "alarm_type": "predicted_threshold",
                    "gas_type": pred_alarm["gas"],
                    "severity": "attention",
                    "device_id": device_id,
                    "predicted_value": pred_alarm["predicted_value"],
                    "predicted_time": pred_alarm["predicted_time"],
                    "message": f"预测 {pred_alarm['gas']} 将在 {pred_alarm['hours_until']:.1f} 小时后超过阈值",
                    "timestamp": timestamp
                })
        
        return alarms
    
    def _generate_recommendations(self, gas_status: Dict,
                                  leak_detection: Dict,
                                  trend_analysis: Dict) -> List[str]:
        """生成建议措施"""
        recommendations = []
        
        # 基于气体状态
        for gas, status in gas_status.items():
            level = status.get("level", "normal")
            
            if level == "critical":
                recommendations.append(f"【紧急】{gas} 浓度严重超标，建议立即停运设备并撤离人员")
            elif level == "alarm":
                recommendations.append(f"【告警】{gas} 浓度超标，建议安排检修并加强通风")
            elif level == "warning":
                recommendations.append(f"【警告】{gas} 浓度偏高，建议增加监测频率")
        
        # 基于泄漏检测
        if leak_detection.get("detected"):
            recommendations.append("检测到气体泄漏趋势，建议检查密封件和管路连接")
            
            for gas, details in leak_detection.get("gas_details", {}).items():
                if details.get("detected"):
                    eta = details.get("estimated_time_to_alarm")
                    if eta and eta < 24:
                        recommendations.append(
                            f"预计 {eta:.1f} 小时后 {gas} 将达到告警值，"
                            "建议提前安排检修"
                        )
        
        # 基于趋势分析
        abnormal_trends = trend_analysis.get("abnormal_trends", [])
        for trend in abnormal_trends:
            recommendations.append(f"{trend['gas']} 呈{trend['pattern']}趋势，建议关注")
        
        if not recommendations:
            recommendations.append("各项指标正常，建议保持常规监测")
        
        return recommendations
    
    def get_device_summary(self, device_id: str) -> Dict[str, Any]:
        """获取设备气体监测摘要"""
        history = self._get_history(device_id)
        
        if not history:
            return {"available": False, "reason": "无历史数据"}
        
        summary = {
            "device_id": device_id,
            "available": True,
            "data_points": len(history.get("timestamps", [])),
            "time_range": {
                "start": min(history["timestamps"]) if history["timestamps"] else None,
                "end": max(history["timestamps"]) if history["timestamps"] else None
            },
            "gas_statistics": {}
        }
        
        for gas, values in history["gas_data"].items():
            if values:
                summary["gas_statistics"][gas] = {
                    "current": float(values[-1]),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
        
        return summary
    
    def shutdown(self) -> bool:
        """关闭插件"""
        try:
            self._history_buffers.clear()
            self._alarm_states.clear()
            self._is_initialized = False
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
            "description": "基于时序预测模型的气体泄漏检测，支持SF6/H2/CO/C2H2等多种气体监测",
            "author": "AI巡检系统",
            "capabilities": [
                "gas_concentration_monitoring",
                "leakage_detection",
                "trend_prediction",
                "multi_gas_support",
                "threshold_alerting"
            ],
            "supported_gases": GasType.get_all(),
            "models_required": list(self.config.model_ids.values())
        }
