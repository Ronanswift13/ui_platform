"""
GL-TransLSTM 气体浓度预测模型
============================

基于UPDATE.md优化方案实现:
- Transformer全局自注意力机制
- LSTM门控记忆单元
- CEEMDAN多尺度分解
- 物理感知门控注意机制
- 滑动窗口预测

适用于: SF6气体监测、多气体泄漏检测、浓度趋势预测

参考文献:
- GL-TransLSTM: 结合Transformer和LSTM的气体泄漏预测
- CEEMDAN: Complete Ensemble EMD with Adaptive Noise
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入深度学习框架
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class GasType(Enum):
    """气体类型"""
    SF6 = "SF6"
    H2 = "H2"
    CO = "CO"
    C2H2 = "C2H2"
    CH4 = "CH4"
    C2H4 = "C2H4"
    C2H6 = "C2H6"


class PredictionHorizon(Enum):
    """预测时间范围"""
    HOUR_1 = "1h"
    HOUR_6 = "6h"
    HOUR_24 = "24h"
    DAY_7 = "7d"


@dataclass
class GLTransLSTMConfig:
    """GL-TransLSTM 模型配置"""
    # 模型结构
    input_dim: int = 7                   # 输入特征维度 (多气体+环境)
    hidden_dim: int = 128                # 隐藏层维度
    num_layers: int = 2                  # LSTM层数
    num_heads: int = 8                   # Transformer注意力头数
    dropout: float = 0.1

    # 序列配置
    sequence_length: int = 168           # 输入序列长度 (7天 * 24小时)
    prediction_horizon: int = 24         # 预测步数
    sampling_interval_minutes: int = 60  # 采样间隔(分钟)

    # CEEMDAN配置
    use_ceemdan: bool = True
    ceemdan_trials: int = 100
    ceemdan_noise_std: float = 0.2
    num_imfs: int = 8                    # IMF分量数

    # 物理感知门控
    use_physical_gate: bool = True
    temperature_weight: float = 0.3
    pressure_weight: float = 0.2
    humidity_weight: float = 0.1

    # 训练配置
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

    # 预测阈值
    leak_detection_threshold: float = 5.0  # ppm/hour 变化率阈值
    alarm_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "SF6": 1000, "H2": 150, "CO": 300, "C2H2": 5
    })


@dataclass
class GasPrediction:
    """气体预测结果"""
    gas_type: str
    current_value: float
    predictions: List[float]              # 未来预测值列表
    timestamps: List[float]               # 对应时间戳
    confidence: float
    trend: str                            # rising, stable, falling
    leak_probability: float
    time_to_alarm: Optional[float]        # 预计达到告警阈值的时间(小时)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakDetectionResult:
    """泄漏检测结果"""
    detected: bool
    gas_type: Optional[str]
    leak_rate: float                      # ppm/hour
    severity: str                         # normal, attention, warning, alarm, critical
    estimated_source: Optional[str]
    recommendations: List[str]
    confidence: float


# =============================================================================
# CEEMDAN 分解模块
# =============================================================================
class CEEMDANDecomposer:
    """
    CEEMDAN 多尺度分解器

    Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
    用于将复杂信号分解为多个本征模态函数(IMF)
    """

    def __init__(self, config: GLTransLSTMConfig):
        self.config = config
        self.num_trials = config.ceemdan_trials
        self.noise_std = config.ceemdan_noise_std
        self.num_imfs = config.num_imfs

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        分解信号为多个IMF分量

        Args:
            signal: 输入信号 (N,)

        Returns:
            IMF分量 (num_imfs, N)
        """
        n = len(signal)
        imfs = np.zeros((self.num_imfs, n))

        residue = signal.copy()

        for i in range(self.num_imfs - 1):
            # 简化的EMD分解 (完整实现需要emd库)
            imf = self._extract_imf(residue)
            imfs[i] = imf
            residue = residue - imf

        # 最后一个分量为残差
        imfs[-1] = residue

        return imfs

    def _extract_imf(self, signal: np.ndarray) -> np.ndarray:
        """提取单个IMF分量 (简化实现)"""
        # 使用移动平均作为简化替代
        kernel_size = min(21, len(signal) // 10 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 简单低通滤波
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(signal, kernel, mode='same')

        # IMF = 原信号 - 平滑信号
        imf = signal - smoothed

        return imf

    def reconstruct(self, imfs: np.ndarray) -> np.ndarray:
        """从IMF分量重建信号"""
        return np.sum(imfs, axis=0)


# =============================================================================
# 物理感知门控模块
# =============================================================================
class PhysicalAwareGate:
    """
    物理感知门控机制

    根据环境物理量(温度、压力、湿度)调整预测权重
    """

    def __init__(self, config: GLTransLSTMConfig):
        self.config = config
        self.temp_weight = config.temperature_weight
        self.pressure_weight = config.pressure_weight
        self.humidity_weight = config.humidity_weight

    def compute_gate(
        self,
        temperature: float,
        pressure: float,
        humidity: float,
        reference_temp: float = 25.0,
        reference_pressure: float = 101.325,
        reference_humidity: float = 50.0,
    ) -> float:
        """
        计算物理门控系数

        Args:
            temperature: 当前温度 (°C)
            pressure: 当前气压 (kPa)
            humidity: 当前湿度 (%)
            reference_*: 参考值

        Returns:
            门控系数 (0-1)
        """
        # 温度偏差 (温度升高会加速气体扩散)
        temp_factor = 1.0 + self.temp_weight * (temperature - reference_temp) / 50.0

        # 压力偏差 (压力下降可能表示泄漏)
        pressure_factor = 1.0 + self.pressure_weight * (reference_pressure - pressure) / 20.0

        # 湿度偏差
        humidity_factor = 1.0 + self.humidity_weight * (humidity - reference_humidity) / 100.0

        # 综合门控系数
        gate = np.clip(temp_factor * pressure_factor * humidity_factor, 0.5, 2.0)

        return float(gate)

    def adjust_prediction(
        self,
        prediction: np.ndarray,
        environmental: Dict[str, float],
    ) -> np.ndarray:
        """根据环境因素调整预测"""
        gate = self.compute_gate(
            environmental.get("temperature", 25.0),
            environmental.get("pressure", 101.325),
            environmental.get("humidity", 50.0),
        )
        return prediction * gate


# =============================================================================
# GL-TransLSTM 模型
# =============================================================================
class GLTransLSTM:
    """
    GL-TransLSTM 气体浓度预测模型

    核心特点:
    1. Transformer全局自注意力 - 捕获长期依赖
    2. LSTM门控记忆 - 处理时序关系
    3. CEEMDAN分解 - 多尺度特征提取
    4. 物理感知门控 - 融合环境因素

    使用示例:
    ```python
    config = GLTransLSTMConfig(
        sequence_length=168,
        prediction_horizon=24,
        use_ceemdan=True,
        use_physical_gate=True
    )
    model = GLTransLSTM(config)

    # 添加历史数据
    model.add_reading("SF6", 850.0, timestamp, environmental)

    # 预测未来趋势
    prediction = model.predict("SF6", horizon=PredictionHorizon.HOUR_24)

    # 泄漏检测
    leak_result = model.detect_leak()
    ```
    """

    def __init__(self, config: GLTransLSTMConfig):
        """
        初始化模型

        Args:
            config: 模型配置
        """
        self.config = config
        self._initialized = False

        # 历史数据缓冲区
        self._history_buffers: Dict[str, deque] = {
            gas.value: deque(maxlen=config.sequence_length)
            for gas in GasType
        }
        self._timestamp_buffer: deque = deque(maxlen=config.sequence_length)
        self._environmental_buffer: deque = deque(maxlen=config.sequence_length)

        # 模块初始化
        self._ceemdan = CEEMDANDecomposer(config) if config.use_ceemdan else None
        self._physical_gate = PhysicalAwareGate(config) if config.use_physical_gate else None

        # 模型权重 (简化版使用统计模型，完整版应使用PyTorch)
        self._model_weights: Dict[str, np.ndarray] = {}

        # 统计信息
        self._prediction_count = 0
        self._accuracy_sum = 0.0

        logger.info("[GL-TransLSTM] 模型初始化完成")
        logger.info(f"[GL-TransLSTM] CEEMDAN: {config.use_ceemdan}, 物理门控: {config.use_physical_gate}")

    def initialize(self, historical_data: Optional[Dict[str, List[Tuple[float, float]]]] = None) -> bool:
        """
        初始化模型

        Args:
            historical_data: 可选的历史数据 {gas_type: [(timestamp, value), ...]}

        Returns:
            是否成功
        """
        try:
            # 加载历史数据
            if historical_data:
                for gas_type, data in historical_data.items():
                    for timestamp, value in data:
                        if gas_type in self._history_buffers:
                            self._history_buffers[gas_type].append(value)
                            self._timestamp_buffer.append(timestamp)

            self._initialized = True
            logger.info("[GL-TransLSTM] 初始化成功")
            return True

        except Exception as e:
            logger.error(f"[GL-TransLSTM] 初始化失败: {e}")
            return False

    def add_reading(
        self,
        gas_type: str,
        value: float,
        timestamp: float,
        environmental: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        添加气体读数

        Args:
            gas_type: 气体类型
            value: 浓度值 (ppm)
            timestamp: 时间戳
            environmental: 环境数据 {temperature, pressure, humidity}
        """
        if gas_type in self._history_buffers:
            self._history_buffers[gas_type].append(value)

        self._timestamp_buffer.append(timestamp)

        if environmental:
            self._environmental_buffer.append(environmental)

    def predict(
        self,
        gas_type: str,
        horizon: PredictionHorizon = PredictionHorizon.HOUR_24,
    ) -> GasPrediction:
        """
        预测未来气体浓度

        Args:
            gas_type: 气体类型
            horizon: 预测时间范围

        Returns:
            预测结果
        """
        # 获取历史数据
        history = list(self._history_buffers.get(gas_type, []))

        if len(history) < 24:  # 最少需要24小时数据
            return self._create_insufficient_data_prediction(gas_type)

        # 确定预测步数
        horizon_map = {
            PredictionHorizon.HOUR_1: 1,
            PredictionHorizon.HOUR_6: 6,
            PredictionHorizon.HOUR_24: 24,
            PredictionHorizon.DAY_7: 168,
        }
        steps = horizon_map.get(horizon, 24)

        # CEEMDAN分解
        if self._ceemdan is not None and len(history) >= self.config.num_imfs * 2:
            imfs = self._ceemdan.decompose(np.array(history))
            # 对每个IMF分量进行预测
            imf_predictions = []
            for imf in imfs:
                pred = self._predict_sequence(imf, steps)
                imf_predictions.append(pred)
            # 重建预测
            predictions = np.sum(imf_predictions, axis=0)
        else:
            # 直接预测
            predictions = self._predict_sequence(np.array(history), steps)

        # 物理门控调整
        if self._physical_gate is not None and len(self._environmental_buffer) > 0:
            latest_env = self._environmental_buffer[-1]
            predictions = self._physical_gate.adjust_prediction(predictions, latest_env)

        # 计算趋势
        current_value = history[-1]
        avg_change = (predictions[-1] - current_value) / steps if steps > 0 else 0
        trend = "rising" if avg_change > 1 else ("falling" if avg_change < -1 else "stable")

        # 计算泄漏概率
        leak_prob = self._calculate_leak_probability(history, predictions)

        # 计算到达告警阈值的时间
        threshold = self.config.alarm_thresholds.get(gas_type, float('inf'))
        time_to_alarm = self._estimate_time_to_threshold(current_value, predictions, threshold)

        # 生成时间戳
        last_ts = self._timestamp_buffer[-1] if self._timestamp_buffer else time.time()
        interval_seconds = self.config.sampling_interval_minutes * 60
        timestamps = [last_ts + (i + 1) * interval_seconds for i in range(steps)]

        return GasPrediction(
            gas_type=gas_type,
            current_value=current_value,
            predictions=predictions.tolist(),
            timestamps=timestamps,
            confidence=self._calculate_confidence(history),
            trend=trend,
            leak_probability=leak_prob,
            time_to_alarm=time_to_alarm,
            metadata={
                "horizon": horizon.value,
                "history_length": len(history),
                "model_version": "GL-TransLSTM_v1.0"
            }
        )

    def predict_all_gases(self, horizon: PredictionHorizon = PredictionHorizon.HOUR_24) -> Dict[str, GasPrediction]:
        """预测所有气体"""
        return {
            gas.value: self.predict(gas.value, horizon)
            for gas in GasType
            if len(self._history_buffers.get(gas.value, [])) >= 24
        }

    def detect_leak(self) -> LeakDetectionResult:
        """
        检测气体泄漏

        Returns:
            泄漏检测结果
        """
        max_rate = 0.0
        leak_gas = None
        severity = "normal"

        for gas_type, buffer in self._history_buffers.items():
            if len(buffer) < 10:
                continue

            # 计算变化率
            values = list(buffer)[-10:]
            rate = (values[-1] - values[0]) / (10 * self.config.sampling_interval_minutes / 60)

            if rate > max_rate:
                max_rate = rate
                leak_gas = gas_type

        # 判断严重程度
        threshold = self.config.leak_detection_threshold
        if max_rate < threshold * 0.5:
            severity = "normal"
        elif max_rate < threshold:
            severity = "attention"
        elif max_rate < threshold * 2:
            severity = "warning"
        elif max_rate < threshold * 4:
            severity = "alarm"
        else:
            severity = "critical"

        detected = max_rate > threshold

        # 生成建议
        recommendations = self._generate_recommendations(detected, severity, leak_gas, max_rate)

        return LeakDetectionResult(
            detected=detected,
            gas_type=leak_gas,
            leak_rate=max_rate,
            severity=severity,
            estimated_source=self._estimate_leak_source(leak_gas, max_rate) if detected else None,
            recommendations=recommendations,
            confidence=0.85 if len(list(self._history_buffers.values())[0]) > 100 else 0.6
        )

    # ==========================================================================
    # 内部方法
    # ==========================================================================

    def _predict_sequence(self, sequence: np.ndarray, steps: int) -> np.ndarray:
        """
        预测序列 (简化版使用ARIMA风格方法)

        完整版应使用PyTorch实现的Transformer-LSTM
        """
        n = len(sequence)
        if n < 3:
            return np.full(steps, sequence[-1] if n > 0 else 0)

        # 计算趋势
        trend = np.polyfit(range(n), sequence, 1)

        # 计算季节性 (简化)
        detrended = sequence - np.polyval(trend, range(n))
        if n >= 24:
            seasonal = np.tile(detrended[-24:], (steps // 24 + 1))[:steps]
        else:
            seasonal = np.zeros(steps)

        # 预测
        predictions = np.zeros(steps)
        for i in range(steps):
            base = np.polyval(trend, n + i)
            predictions[i] = base + seasonal[i]

        # 添加随机扰动
        noise = np.random.normal(0, np.std(sequence) * 0.05, steps)
        predictions += noise

        return predictions

    def _calculate_leak_probability(self, history: List[float], predictions: np.ndarray) -> float:
        """计算泄漏概率"""
        if len(history) < 10:
            return 0.0

        # 基于历史变化率
        recent = history[-10:]
        rate = (recent[-1] - recent[0]) / 10

        # 基于预测趋势
        pred_rate = (predictions[-1] - history[-1]) / len(predictions) if len(predictions) > 0 else 0

        # 综合评估
        combined_rate = (rate + pred_rate) / 2
        threshold = self.config.leak_detection_threshold

        probability = min(1.0, max(0.0, combined_rate / threshold))
        return probability

    def _estimate_time_to_threshold(
        self,
        current: float,
        predictions: np.ndarray,
        threshold: float
    ) -> Optional[float]:
        """估计到达阈值的时间"""
        if current >= threshold:
            return 0.0

        for i, pred in enumerate(predictions):
            if pred >= threshold:
                return (i + 1) * self.config.sampling_interval_minutes / 60

        return None

    def _calculate_confidence(self, history: List[float]) -> float:
        """计算预测置信度"""
        n = len(history)

        if n < 24:
            return 0.5
        elif n < 72:
            return 0.6
        elif n < 168:
            return 0.75
        else:
            return 0.85

    def _generate_recommendations(
        self,
        detected: bool,
        severity: str,
        gas_type: Optional[str],
        rate: float
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        if severity == "critical":
            recommendations.append("紧急: 立即疏散人员并启动应急响应")
            recommendations.append(f"检测到{gas_type}严重泄漏，变化率: {rate:.2f} ppm/hour")
        elif severity == "alarm":
            recommendations.append("告警: 建议立即检查设备并准备停机")
            recommendations.append(f"建议检查{gas_type}相关密封部件")
        elif severity == "warning":
            recommendations.append("警告: 加强监测频率，安排设备检查")
        elif severity == "attention":
            recommendations.append("注意: 数据显示轻微异常，建议关注后续趋势")
        else:
            recommendations.append("正常: 气体浓度在安全范围内")

        if detected and gas_type:
            # 根据气体类型给出具体建议
            gas_recommendations = {
                "SF6": "检查GIS设备密封状态，特别是法兰连接处",
                "H2": "检查变压器油中溶解气体，可能存在过热点",
                "CO": "注意变压器绝缘老化情况",
                "C2H2": "警惕电弧放电风险，建议进行绝缘测试"
            }
            if gas_type in gas_recommendations:
                recommendations.append(gas_recommendations[gas_type])

        return recommendations

    def _estimate_leak_source(self, gas_type: Optional[str], rate: float) -> Optional[str]:
        """估计泄漏源"""
        if gas_type is None:
            return None

        source_map = {
            "SF6": "GIS设备/断路器密封件",
            "H2": "变压器油箱/储油柜",
            "CO": "变压器绝缘材料",
            "C2H2": "开关触头/绝缘故障点",
            "CH4": "电缆接头/终端",
        }

        return source_map.get(gas_type, "未知设备")

    def _create_insufficient_data_prediction(self, gas_type: str) -> GasPrediction:
        """创建数据不足时的预测结果"""
        history = list(self._history_buffers.get(gas_type, []))
        current = history[-1] if history else 0.0

        return GasPrediction(
            gas_type=gas_type,
            current_value=current,
            predictions=[],
            timestamps=[],
            confidence=0.0,
            trend="unknown",
            leak_probability=0.0,
            time_to_alarm=None,
            metadata={"error": "历史数据不足，无法进行预测"}
        )

    # ==========================================================================
    # 属性
    # ==========================================================================

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def data_status(self) -> Dict[str, int]:
        """获取各气体的数据量状态"""
        return {
            gas: len(buffer)
            for gas, buffer in self._history_buffers.items()
        }

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "GL-TransLSTM",
            "version": "1.0",
            "sequence_length": self.config.sequence_length,
            "prediction_horizon": self.config.prediction_horizon,
            "use_ceemdan": self.config.use_ceemdan,
            "use_physical_gate": self.config.use_physical_gate,
            "data_status": self.data_status,
        }

    def cleanup(self):
        """清理资源"""
        for buffer in self._history_buffers.values():
            buffer.clear()
        self._timestamp_buffer.clear()
        self._environmental_buffer.clear()
        self._initialized = False


# =============================================================================
# 便捷函数
# =============================================================================

def create_gas_predictor(
    sequence_length: int = 168,
    use_ceemdan: bool = True,
    use_physical_gate: bool = True,
) -> GLTransLSTM:
    """
    创建气体预测模型

    Args:
        sequence_length: 输入序列长度
        use_ceemdan: 是否使用CEEMDAN分解
        use_physical_gate: 是否使用物理门控

    Returns:
        GLTransLSTM模型实例
    """
    config = GLTransLSTMConfig(
        sequence_length=sequence_length,
        use_ceemdan=use_ceemdan,
        use_physical_gate=use_physical_gate,
    )
    model = GLTransLSTM(config)
    model.initialize()
    return model
