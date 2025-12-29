"""
气体泄漏检测插件
输变电站全自动AI巡检方案 - SF6气体监测与异常检测

功能:
1. SF6气体浓度实时监测
2. 气体泄漏预测与报警
3. 时间序列异常检测
4. 泄漏源定位辅助

依赖:
- 气体传感器数据 (SF6, H2, O2, CO, CO2等)
- 环境传感器数据 (温度、湿度、气压)

版本: 2.0.0
"""

from __future__ import annotations
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
class GasType(Enum):
    """气体类型"""
    SF6 = "sf6"              # 六氟化硫
    H2 = "h2"                # 氢气
    O2 = "o2"                # 氧气
    CO = "co"                # 一氧化碳
    CO2 = "co2"              # 二氧化碳
    CH4 = "ch4"              # 甲烷
    C2H2 = "c2h2"            # 乙炔
    C2H4 = "c2h4"            # 乙烯
    C2H6 = "c2h6"            # 乙烷
    H2S = "h2s"              # 硫化氢


class AlarmLevel(Enum):
    """告警级别"""
    NORMAL = "normal"
    ATTENTION = "attention"    # 注意
    WARNING = "warning"        # 预警
    ALARM = "alarm"            # 告警
    CRITICAL = "critical"      # 紧急


@dataclass
class GasReading:
    """气体读数"""
    gas_type: GasType
    concentration: float       # 浓度 (ppm或%)
    timestamp: float
    temperature: Optional[float] = None    # 环境温度 (°C)
    humidity: Optional[float] = None       # 相对湿度 (%)
    pressure: Optional[float] = None       # 气压 (kPa)
    sensor_id: Optional[str] = None


@dataclass
class GasAlarm:
    """气体告警"""
    gas_type: GasType
    alarm_level: AlarmLevel
    current_value: float
    threshold: float
    trend: str                # "rising", "falling", "stable"
    timestamp: float
    location: Optional[str] = None
    description: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class LeakageAnalysis:
    """泄漏分析结果"""
    is_leaking: bool
    leak_rate: float           # 泄漏率 (ppm/hour)
    estimated_source: Optional[str] = None
    confidence: float = 0.0
    time_to_critical: Optional[float] = None  # 到达临界值的预估时间(小时)
    historical_trend: List[float] = field(default_factory=list)


# =============================================================================
# 时间序列特征提取
# =============================================================================
class TimeSeriesFeatureExtractor:
    """时间序列特征提取器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
    
    def extract_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        提取时间序列特征
        
        Args:
            data: 时间序列数据 (n,)
        
        Returns:
            特征字典
        """
        if len(data) < 2:
            return {}
        
        features = {}
        
        # 基本统计特征
        features["mean"] = np.mean(data)
        features["std"] = np.std(data)
        features["min"] = np.min(data)
        features["max"] = np.max(data)
        features["range"] = features["max"] - features["min"]
        
        # 趋势特征
        features["trend_slope"] = self._compute_trend_slope(data)
        features["trend_direction"] = 1 if features["trend_slope"] > 0 else -1
        
        # 变化率特征
        diff = np.diff(data)
        features["mean_change_rate"] = np.mean(np.abs(diff))
        features["max_change_rate"] = np.max(np.abs(diff))
        
        # 分布特征
        features["skewness"] = self._compute_skewness(data)
        features["kurtosis"] = self._compute_kurtosis(data)
        
        # 周期性特征 (简化)
        features["autocorr_lag1"] = self._compute_autocorr(data, lag=1)
        
        return features
    
    def _compute_trend_slope(self, data: np.ndarray) -> float:
        """计算线性趋势斜率"""
        x = np.arange(len(data))
        if len(data) < 2:
            return 0.0
        
        # 最小二乘拟合
        x_mean = np.mean(x)
        y_mean = np.mean(data)
        
        numerator = np.sum((x - x_mean) * (data - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_autocorr(self, data: np.ndarray, lag: int = 1) -> float:
        """计算自相关"""
        n = len(data)
        if n <= lag:
            return 0.0
        
        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return 0.0
        
        autocorr = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / ((n - lag) * var)
        return autocorr


# =============================================================================
# 时间序列预测模型
# =============================================================================
class TimeSeriesPredictor:
    """
    时间序列预测器
    
    支持多种预测方法:
    - 指数平滑
    - ARIMA (简化版)
    - LSTM (需要模型注册中心)
    """
    
    def __init__(self, model_registry=None, model_id: str = None):
        self._model_registry = model_registry
        self._model_id = model_id
        self._use_deep_learning = model_registry is not None and model_id is not None
        
        # 历史数据缓存
        self._history: Dict[str, deque] = {}
        self._max_history = 1000
    
    def update(self, gas_type: str, value: float, timestamp: float) -> None:
        """更新历史数据"""
        if gas_type not in self._history:
            self._history[gas_type] = deque(maxlen=self._max_history)
        
        self._history[gas_type].append((timestamp, value))
    
    def predict(self, gas_type: str, horizon: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测未来值
        
        Args:
            gas_type: 气体类型
            horizon: 预测步数
        
        Returns:
            predictions: 预测值
            confidence_intervals: 置信区间 (lower, upper)
        """
        if gas_type not in self._history or len(self._history[gas_type]) < 10:
            return np.array([]), (np.array([]), np.array([]))
        
        # 获取历史数据
        history = np.array([v for _, v in self._history[gas_type]])
        
        # 尝试深度学习预测
        if self._use_deep_learning:
            try:
                return self._predict_deep_learning(history, horizon)
            except Exception as e:
                logger.warning(f"深度学习预测失败,回退到传统方法: {e}")
        
        # 使用指数平滑预测
        return self._predict_exponential_smoothing(history, horizon)
    
    def _predict_exponential_smoothing(self, history: np.ndarray, 
                                        horizon: int,
                                        alpha: float = 0.3,
                                        beta: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Holt双指数平滑预测"""
        n = len(history)
        
        # 初始化
        level = history[0]
        trend = history[1] - history[0] if n > 1 else 0
        
        # 拟合历史数据
        for i in range(1, n):
            new_level = alpha * history[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
            trend = new_trend
        
        # 预测
        predictions = np.array([level + (i + 1) * trend for i in range(horizon)])
        
        # 置信区间 (基于历史残差)
        fitted = np.zeros(n)
        l, t = history[0], (history[1] - history[0]) if n > 1 else 0
        for i in range(n):
            fitted[i] = l + t
            new_l = alpha * history[i] + (1 - alpha) * (l + t)
            new_t = beta * (new_l - l) + (1 - beta) * t
            l, t = new_l, new_t
        
        residuals = history - fitted
        std = np.std(residuals)
        
        lower = predictions - 1.96 * std * np.sqrt(np.arange(1, horizon + 1))
        upper = predictions + 1.96 * std * np.sqrt(np.arange(1, horizon + 1))
        
        return predictions, (lower, upper)
    
    def _predict_deep_learning(self, history: np.ndarray, 
                               horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """深度学习预测"""
        if self._model_registry is None:
            raise RuntimeError("模型注册中心未配置")
        
        result = self._model_registry.infer(
            self._model_id,
            {"time_series": history}
        )
        
        if result.success and result.predictions is not None:
            predictions = result.predictions[:horizon]
            
            if result.confidence_intervals:
                lower, upper = result.confidence_intervals
                lower = lower[:horizon]
                upper = upper[:horizon]
            else:
                std = np.std(history) * 0.1
                lower = predictions - 1.96 * std
                upper = predictions + 1.96 * std
            
            return predictions, (lower, upper)
        
        raise RuntimeError(f"预测失败: {result.error_message}")


# =============================================================================
# 异常检测器
# =============================================================================
class GasAnomalyDetector:
    """气体异常检测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 告警阈值配置
        self.thresholds = {
            GasType.SF6: {
                AlarmLevel.ATTENTION: 100,   # ppm
                AlarmLevel.WARNING: 500,
                AlarmLevel.ALARM: 1000,
                AlarmLevel.CRITICAL: 2000
            },
            GasType.H2: {
                AlarmLevel.ATTENTION: 50,
                AlarmLevel.WARNING: 100,
                AlarmLevel.ALARM: 200,
                AlarmLevel.CRITICAL: 500
            },
            GasType.CO: {
                AlarmLevel.ATTENTION: 25,
                AlarmLevel.WARNING: 50,
                AlarmLevel.ALARM: 100,
                AlarmLevel.CRITICAL: 200
            },
            GasType.C2H2: {
                AlarmLevel.ATTENTION: 1,     # 乙炔非常危险,阈值很低
                AlarmLevel.WARNING: 5,
                AlarmLevel.ALARM: 10,
                AlarmLevel.CRITICAL: 20
            }
        }
        
        # 特征提取器
        self._feature_extractor = TimeSeriesFeatureExtractor()
        
        # 历史数据
        self._history: Dict[str, deque] = {}
        self._window_size = 100
    
    def check_threshold(self, reading: GasReading) -> Optional[GasAlarm]:
        """
        检查阈值告警
        
        Args:
            reading: 气体读数
        
        Returns:
            告警信息或None
        """
        gas_thresholds = self.thresholds.get(reading.gas_type)
        if gas_thresholds is None:
            return None
        
        # 确定告警级别
        alarm_level = AlarmLevel.NORMAL
        threshold = 0
        
        for level in [AlarmLevel.CRITICAL, AlarmLevel.ALARM, 
                      AlarmLevel.WARNING, AlarmLevel.ATTENTION]:
            if reading.concentration >= gas_thresholds[level]:
                alarm_level = level
                threshold = gas_thresholds[level]
                break
        
        if alarm_level == AlarmLevel.NORMAL:
            return None
        
        # 分析趋势
        trend = self._analyze_trend(reading.gas_type)
        
        return GasAlarm(
            gas_type=reading.gas_type,
            alarm_level=alarm_level,
            current_value=reading.concentration,
            threshold=threshold,
            trend=trend,
            timestamp=reading.timestamp,
            description=self._get_alarm_description(reading.gas_type, alarm_level),
            recommendations=self._get_recommendations(reading.gas_type, alarm_level)
        )
    
    def detect_anomaly(self, readings: List[GasReading]) -> List[GasAlarm]:
        """
        检测时序异常
        
        Args:
            readings: 气体读数序列
        
        Returns:
            告警列表
        """
        alarms = []
        
        # 按气体类型分组
        by_gas: Dict[GasType, List[float]] = {}
        for r in readings:
            if r.gas_type not in by_gas:
                by_gas[r.gas_type] = []
            by_gas[r.gas_type].append(r.concentration)
        
        # 对每种气体进行异常检测
        for gas_type, values in by_gas.items():
            if len(values) < 10:
                continue
            
            data = np.array(values)
            
            # 更新历史
            if gas_type not in self._history:
                self._history[gas_type] = deque(maxlen=self._window_size)
            self._history[gas_type].extend(values)
            
            # 提取特征
            features = self._feature_extractor.extract_features(data)
            
            # 检测异常模式
            anomaly_alarm = self._check_anomaly_patterns(gas_type, features, data)
            if anomaly_alarm:
                alarms.append(anomaly_alarm)
        
        return alarms
    
    def _check_anomaly_patterns(self, gas_type: GasType, 
                                 features: Dict[str, float],
                                 data: np.ndarray) -> Optional[GasAlarm]:
        """检测异常模式"""
        # 1. 快速上升检测
        if features.get("trend_slope", 0) > 0.5 and features.get("max_change_rate", 0) > 10:
            return GasAlarm(
                gas_type=gas_type,
                alarm_level=AlarmLevel.WARNING,
                current_value=data[-1],
                threshold=0,
                trend="rising",
                timestamp=time.time(),
                description=f"{gas_type.value}浓度快速上升,可能存在泄漏",
                recommendations=["立即检查气室密封", "准备气体回收设备"]
            )
        
        # 2. 异常波动检测
        if features.get("std", 0) > 50 and features.get("kurtosis", 0) > 3:
            return GasAlarm(
                gas_type=gas_type,
                alarm_level=AlarmLevel.ATTENTION,
                current_value=data[-1],
                threshold=0,
                trend="unstable",
                timestamp=time.time(),
                description=f"{gas_type.value}浓度波动异常",
                recommendations=["检查传感器状态", "确认设备运行状况"]
            )
        
        # 3. 持续偏高检测
        gas_thresholds = self.thresholds.get(gas_type)
        if gas_thresholds:
            attention_threshold = gas_thresholds.get(AlarmLevel.ATTENTION, float('inf'))
            if features.get("mean", 0) > attention_threshold * 0.8:
                return GasAlarm(
                    gas_type=gas_type,
                    alarm_level=AlarmLevel.ATTENTION,
                    current_value=data[-1],
                    threshold=attention_threshold,
                    trend="elevated",
                    timestamp=time.time(),
                    description=f"{gas_type.value}浓度持续偏高",
                    recommendations=["加强监测频率", "准备检修计划"]
                )
        
        return None
    
    def _analyze_trend(self, gas_type: GasType) -> str:
        """分析趋势"""
        if gas_type not in self._history or len(self._history[gas_type]) < 10:
            return "unknown"
        
        recent = list(self._history[gas_type])[-10:]
        slope = (recent[-1] - recent[0]) / len(recent)
        
        if slope > 1:
            return "rising"
        elif slope < -1:
            return "falling"
        else:
            return "stable"
    
    def _get_alarm_description(self, gas_type: GasType, level: AlarmLevel) -> str:
        """获取告警描述"""
        descriptions = {
            GasType.SF6: {
                AlarmLevel.ATTENTION: "SF6浓度略高,建议关注",
                AlarmLevel.WARNING: "SF6浓度偏高,可能存在微量泄漏",
                AlarmLevel.ALARM: "SF6浓度超标,存在泄漏风险",
                AlarmLevel.CRITICAL: "SF6严重泄漏,立即撤离并处置"
            },
            GasType.H2: {
                AlarmLevel.ATTENTION: "检测到微量氢气,建议检查",
                AlarmLevel.WARNING: "氢气浓度偏高,可能存在内部故障",
                AlarmLevel.ALARM: "氢气浓度超标,设备可能过热",
                AlarmLevel.CRITICAL: "氢气浓度危险,存在爆炸风险"
            },
            GasType.C2H2: {
                AlarmLevel.ATTENTION: "检测到微量乙炔,需要关注",
                AlarmLevel.WARNING: "乙炔浓度偏高,可能存在电弧放电",
                AlarmLevel.ALARM: "乙炔浓度超标,存在严重内部故障",
                AlarmLevel.CRITICAL: "乙炔浓度危险,立即停运检查"
            }
        }
        
        return descriptions.get(gas_type, {}).get(
            level, f"{gas_type.value}浓度异常,级别:{level.value}"
        )
    
    def _get_recommendations(self, gas_type: GasType, level: AlarmLevel) -> List[str]:
        """获取处理建议"""
        base_recommendations = {
            AlarmLevel.ATTENTION: ["增加监测频率", "记录数据变化趋势"],
            AlarmLevel.WARNING: ["现场巡检确认", "准备检修计划", "通知运维人员"],
            AlarmLevel.ALARM: ["立即现场确认", "启动应急预案", "准备停运"],
            AlarmLevel.CRITICAL: ["立即撤离区域", "切断电源", "启动紧急响应"]
        }
        
        return base_recommendations.get(level, ["联系运维人员"])


# =============================================================================
# 泄漏分析器
# =============================================================================
class LeakageAnalyzer:
    """泄漏分析器"""
    
    def __init__(self, model_registry=None, model_id: str = None):
        self._model_registry = model_registry
        self._model_id = model_id
        self._predictor = TimeSeriesPredictor(model_registry, model_id)
        
        # 历史数据
        self._history: Dict[str, List[Tuple[float, float]]] = {}
    
    def update(self, reading: GasReading) -> None:
        """更新读数"""
        key = f"{reading.gas_type.value}_{reading.sensor_id or 'default'}"
        if key not in self._history:
            self._history[key] = []
        
        self._history[key].append((reading.timestamp, reading.concentration))
        
        # 限制历史长度
        if len(self._history[key]) > 1000:
            self._history[key] = self._history[key][-1000:]
        
        # 更新预测器
        self._predictor.update(key, reading.concentration, reading.timestamp)
    
    def analyze(self, gas_type: GasType, 
                sensor_id: str = None,
                critical_threshold: float = None) -> LeakageAnalysis:
        """
        分析泄漏情况
        
        Args:
            gas_type: 气体类型
            sensor_id: 传感器ID
            critical_threshold: 临界阈值
        
        Returns:
            泄漏分析结果
        """
        key = f"{gas_type.value}_{sensor_id or 'default'}"
        
        if key not in self._history or len(self._history[key]) < 10:
            return LeakageAnalysis(is_leaking=False, leak_rate=0)
        
        # 获取历史数据
        history = self._history[key]
        timestamps = np.array([t for t, _ in history])
        values = np.array([v for _, v in history])
        
        # 计算泄漏率 (ppm/hour)
        if len(timestamps) >= 2:
            time_diff_hours = (timestamps[-1] - timestamps[0]) / 3600
            if time_diff_hours > 0:
                value_diff = values[-1] - values[0]
                leak_rate = value_diff / time_diff_hours
            else:
                leak_rate = 0
        else:
            leak_rate = 0
        
        # 判断是否泄漏
        is_leaking = leak_rate > 5  # 阈值: 5 ppm/hour
        
        # 预测到达临界值的时间
        time_to_critical = None
        if is_leaking and critical_threshold:
            current_value = values[-1]
            if current_value < critical_threshold and leak_rate > 0:
                time_to_critical = (critical_threshold - current_value) / leak_rate
        
        # 置信度评估
        confidence = min(len(history) / 100, 1.0)  # 数据越多置信度越高
        if np.std(values) > np.mean(values) * 0.5:  # 波动大时降低置信度
            confidence *= 0.7
        
        return LeakageAnalysis(
            is_leaking=is_leaking,
            leak_rate=leak_rate,
            confidence=confidence,
            time_to_critical=time_to_critical,
            historical_trend=list(values[-20:])  # 最近20个数据点
        )


# =============================================================================
# 气体检测插件
# =============================================================================
class GasDetectionPlugin:
    """
    气体检测插件
    
    整合阈值告警、异常检测和泄漏分析
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 组件
        self._anomaly_detector = GasAnomalyDetector(config)
        self._leakage_analyzer = LeakageAnalyzer()
        
        # 历史告警
        self._alarm_history: deque = deque(maxlen=1000)
        
        # 模型注册中心
        self._model_registry = None
    
    def set_model_registry(self, registry) -> None:
        """设置模型注册中心"""
        self._model_registry = registry
        # 为泄漏分析器设置预测模型
        self._leakage_analyzer = LeakageAnalyzer(
            registry, 
            "timeseries_inspection_gas_forecast"
        )
    
    def process_reading(self, reading: GasReading) -> Dict[str, Any]:
        """
        处理单个气体读数
        
        Args:
            reading: 气体读数
        
        Returns:
            处理结果
        """
        result = {
            "success": True,
            "gas_type": reading.gas_type.value,
            "concentration": reading.concentration,
            "timestamp": reading.timestamp,
            "alarms": [],
            "leakage_analysis": None
        }
        
        # 1. 阈值检查
        threshold_alarm = self._anomaly_detector.check_threshold(reading)
        if threshold_alarm:
            result["alarms"].append(self._alarm_to_dict(threshold_alarm))
            self._alarm_history.append(threshold_alarm)
        
        # 2. 更新泄漏分析器
        self._leakage_analyzer.update(reading)
        
        # 3. 泄漏分析
        leakage = self._leakage_analyzer.analyze(
            reading.gas_type,
            reading.sensor_id,
            critical_threshold=self._get_critical_threshold(reading.gas_type)
        )
        
        if leakage.is_leaking:
            result["leakage_analysis"] = {
                "is_leaking": leakage.is_leaking,
                "leak_rate": leakage.leak_rate,
                "confidence": leakage.confidence,
                "time_to_critical": leakage.time_to_critical
            }
        
        return result
    
    def process_batch(self, readings: List[GasReading]) -> Dict[str, Any]:
        """
        批量处理气体读数
        
        Args:
            readings: 气体读数列表
        
        Returns:
            处理结果
        """
        result = {
            "success": True,
            "processed_count": len(readings),
            "alarms": [],
            "anomalies": [],
            "leakage_summary": {}
        }
        
        # 处理每个读数
        for reading in readings:
            single_result = self.process_reading(reading)
            result["alarms"].extend(single_result.get("alarms", []))
            
            if single_result.get("leakage_analysis"):
                gas_key = reading.gas_type.value
                result["leakage_summary"][gas_key] = single_result["leakage_analysis"]
        
        # 时序异常检测
        anomaly_alarms = self._anomaly_detector.detect_anomaly(readings)
        for alarm in anomaly_alarms:
            result["anomalies"].append(self._alarm_to_dict(alarm))
            self._alarm_history.append(alarm)
        
        return result
    
    def predict_future(self, gas_type: GasType, 
                       horizon: int = 10) -> Dict[str, Any]:
        """
        预测未来气体浓度
        
        Args:
            gas_type: 气体类型
            horizon: 预测步数
        
        Returns:
            预测结果
        """
        key = f"{gas_type.value}_default"
        predictions, (lower, upper) = self._leakage_analyzer._predictor.predict(
            key, horizon
        )
        
        if len(predictions) == 0:
            return {"success": False, "error": "历史数据不足"}
        
        return {
            "success": True,
            "gas_type": gas_type.value,
            "predictions": predictions.tolist(),
            "confidence_lower": lower.tolist(),
            "confidence_upper": upper.tolist(),
            "horizon": horizon
        }
    
    def _get_critical_threshold(self, gas_type: GasType) -> float:
        """获取临界阈值"""
        thresholds = {
            GasType.SF6: 2000,
            GasType.H2: 500,
            GasType.CO: 200,
            GasType.C2H2: 20
        }
        return thresholds.get(gas_type, 1000)
    
    def _alarm_to_dict(self, alarm: GasAlarm) -> Dict[str, Any]:
        """告警转字典"""
        return {
            "gas_type": alarm.gas_type.value,
            "alarm_level": alarm.alarm_level.value,
            "current_value": alarm.current_value,
            "threshold": alarm.threshold,
            "trend": alarm.trend,
            "timestamp": alarm.timestamp,
            "description": alarm.description,
            "recommendations": alarm.recommendations
        }
    
    def inspect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行巡检(插件统一接口)
        
        Args:
            data: 包含气体数据的字典
        
        Returns:
            巡检结果
        """
        # 支持单个读数或批量读数
        if "readings" in data:
            readings = []
            for r in data["readings"]:
                readings.append(GasReading(
                    gas_type=GasType(r.get("gas_type", "sf6")),
                    concentration=r.get("concentration", 0),
                    timestamp=r.get("timestamp", time.time()),
                    temperature=r.get("temperature"),
                    humidity=r.get("humidity"),
                    sensor_id=r.get("sensor_id")
                ))
            return self.process_batch(readings)
        else:
            reading = GasReading(
                gas_type=GasType(data.get("gas_type", "sf6")),
                concentration=data.get("concentration", 0),
                timestamp=data.get("timestamp", time.time()),
                temperature=data.get("temperature"),
                humidity=data.get("humidity"),
                sensor_id=data.get("sensor_id")
            )
            return self.process_reading(reading)


# =============================================================================
# 检测器增强版
# =============================================================================
class GasDetectorEnhanced:
    """气体检测器增强版"""
    
    def __init__(self, config: Dict[str, Any] = None, model_registry=None):
        self.config = config or {}
        self._model_registry = model_registry
        
        # 核心组件
        self.plugin = GasDetectionPlugin(config)
        
        if model_registry:
            self.plugin.set_model_registry(model_registry)
    
    def set_model_registry(self, registry) -> None:
        """设置模型注册中心"""
        self._model_registry = registry
        self.plugin.set_model_registry(registry)
    
    def detect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行检测"""
        return self.plugin.inspect(data)
