#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气体浓度预测器模块 V3.0
========================

基于GL-TransLSTM的时序预测 (UPDATE.md优化方案)

核心特性:
1. GL-TransLSTM深度学习预测 (优先)
   - Transformer全局自注意力机制
   - LSTM门控记忆单元
   - CEEMDAN多尺度分解
   - 物理感知门控注意机制
2. 多气体综合预测
3. 泄漏检测与预警
4. 预测区间估计 (不确定性量化)
5. 阈值超越预警

版本历史:
- V1.0: 基础LSTM预测
- V2.0: 添加Transformer支持
- V3.0: 集成GL-TransLSTM深度学习模型
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入GL-TransLSTM深度学习模型
try:
    from ai_models.deep_learning.gl_translstm import (
        GLTransLSTM,
        GLTransLSTMConfig,
        GasPrediction,
        LeakDetectionResult,
        GasType,
        PredictionHorizon,
    )
    GL_TRANSLSTM_AVAILABLE = True
except ImportError:
    GL_TRANSLSTM_AVAILABLE = False
    GLTransLSTM = None
    GLTransLSTMConfig = None
    GasPrediction = None
    LeakDetectionResult = None


class GasConcentrationPredictor:
    """
    气体浓度预测器 V3.0

    支持:
    1. GL-TransLSTM深度学习预测 (优先)
    2. LSTM单变量预测 (后备)
    3. Transformer多变量预测 (后备)
    4. 概率预测 (输出均值和方差)
    5. 泄漏检测与预警
    """

    def __init__(self, config):
        self.config = config
        self._model_registry = None
        self._use_deep_learning = False

        # V3.0: GL-TransLSTM深度学习模型
        self._gl_translstm: Optional[GLTransLSTM] = None
        self._dl_initialized = False
        self._model_version = "gas_predictor_v3.0"

        # 数据归一化参数
        self.normalization_params = {
            "SF6": {"mean": 100.0, "std": 20.0},
            "H2": {"mean": 50.0, "std": 30.0},
            "CO": {"mean": 100.0, "std": 50.0},
            "C2H2": {"mean": 5.0, "std": 3.0},
            "temperature": {"mean": 25.0, "std": 10.0},
            "humidity": {"mean": 50.0, "std": 15.0},
            "pressure": {"mean": 101.3, "std": 5.0}
        }
    
    def initialize(self) -> bool:
        """
        初始化预测器 (V3.0)

        Returns:
            是否成功初始化
        """
        # 初始化GL-TransLSTM
        self._init_gl_translstm()

        logger.info(f"[GasPredictor] V3.0 初始化完成")
        logger.info(f"[GasPredictor] GL-TransLSTM可用: {self._dl_initialized}")
        return True

    def _init_gl_translstm(self) -> None:
        """初始化GL-TransLSTM深度学习模型"""
        if not GL_TRANSLSTM_AVAILABLE:
            logger.warning("[GasPredictor] GL-TransLSTM模块不可用")
            return

        try:
            # 创建配置
            config = GLTransLSTMConfig(
                sequence_length=getattr(self.config, 'history_length', 168),
                prediction_horizon=getattr(self.config, 'prediction_horizon', 24),
                use_ceemdan=True,           # 启用CEEMDAN多尺度分解
                use_physical_gate=True,     # 启用物理感知门控
                num_imfs=8,                 # IMF分量数
                sampling_interval_minutes=60,  # 采样间隔(分钟)
            )

            # 设置告警阈值
            if hasattr(self.config, 'thresholds'):
                config.alarm_thresholds = {
                    gas: getattr(th, 'alarm', 1000)
                    for gas, th in self.config.thresholds.items()
                }

            # 创建模型
            self._gl_translstm = GLTransLSTM(config)
            self._gl_translstm.initialize()
            self._dl_initialized = True

            logger.info("[GasPredictor] GL-TransLSTM初始化成功")
            logger.info(f"[GasPredictor] 序列长度: {config.sequence_length}, "
                       f"预测步数: {config.prediction_horizon}")

        except Exception as e:
            logger.warning(f"[GasPredictor] GL-TransLSTM初始化失败: {e}")
            self._gl_translstm = None
            self._dl_initialized = False

    def set_model_registry(self, model_registry):
        """设置模型注册中心 (兼容旧版)"""
        self._model_registry = model_registry

        # 检查模型可用性
        try:
            lstm_id = self.config.model_ids.get("lstm")
            transformer_id = self.config.model_ids.get("transformer")

            if model_registry:
                if lstm_id and model_registry.is_model_loaded(lstm_id):
                    self._use_deep_learning = True
                    logger.info("气体预测器: LSTM模型已启用")
                elif transformer_id and model_registry.is_model_loaded(transformer_id):
                    self._use_deep_learning = True
                    logger.info("气体预测器: Transformer模型已启用")
        except Exception as e:
            logger.warning(f"深度学习模型不可用: {e}")
            self._use_deep_learning = False

        # 如果尚未初始化GL-TransLSTM，尝试初始化
        if not self._dl_initialized:
            self._init_gl_translstm()
    
    def predict(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测气体浓度趋势 (V3.0)

        Args:
            history: {
                "timestamps": List[float],
                "gas_data": {gas_type: List[float]},
                "environmental": {param: List[float]}
            }

        Returns:
            预测结果
        """
        if not history.get("timestamps"):
            return {
                "available": False,
                "reason": "无历史数据"
            }

        # V3.0: 优先使用GL-TransLSTM
        if self._dl_initialized and self._gl_translstm is not None:
            gl_result = self._predict_by_gl_translstm(history)
            if gl_result.get("success"):
                return gl_result

        # 回退: 尝试model_registry深度学习预测
        if self._use_deep_learning and self._model_registry:
            dl_result = self._predict_by_deep_learning(history)
            if dl_result.get("success"):
                return dl_result

        # 最终回退: 传统方法
        return self._predict_by_traditional(history)

    def _predict_by_gl_translstm(self, history: Dict) -> Dict[str, Any]:
        """
        使用GL-TransLSTM深度学习模型预测 (V3.0)

        特性:
        - CEEMDAN多尺度分解
        - Transformer全局自注意力
        - LSTM门控记忆
        - 物理感知门控
        """
        try:
            # 更新模型历史数据
            self._update_gl_translstm_history(history)

            predictions = {
                "available": True,
                "success": True,
                "method": "GL-TransLSTM",
                "model_version": self._model_version,
                "prediction_horizon": self.config.prediction_horizon,
                "gas_predictions": {},
                "predicted_alarms": [],
                "confidence_intervals": {},
                "leak_detection": None
            }

            # 预测每种气体
            for gas, values in history["gas_data"].items():
                if len(values) < 24:
                    continue

                # 使用GL-TransLSTM预测
                gas_prediction = self._gl_translstm.predict(
                    gas,
                    PredictionHorizon.HOUR_24 if self.config.prediction_horizon >= 24
                    else PredictionHorizon.HOUR_6
                )

                if gas_prediction.predictions:
                    predictions["gas_predictions"][gas] = {
                        "values": gas_prediction.predictions,
                        "time_steps": list(range(1, len(gas_prediction.predictions) + 1)),
                        "unit": "ppm",
                        "current_value": gas_prediction.current_value,
                        "trend": gas_prediction.trend,
                        "confidence": gas_prediction.confidence,
                        "leak_probability": gas_prediction.leak_probability
                    }

                    # 检查是否超阈值
                    if gas_prediction.time_to_alarm is not None:
                        predictions["predicted_alarms"].append({
                            "gas": gas,
                            "predicted_value": gas_prediction.predictions[-1] if gas_prediction.predictions else 0,
                            "threshold": self.config.thresholds.get(gas, {}).get('alarm', 1000),
                            "hours_until": gas_prediction.time_to_alarm,
                            "predicted_time": f"+{gas_prediction.time_to_alarm:.1f}h",
                            "severity": "alarm" if gas_prediction.time_to_alarm < 6 else "warning"
                        })

            # 泄漏检测
            leak_result = self._gl_translstm.detect_leak()
            predictions["leak_detection"] = {
                "detected": leak_result.detected,
                "gas_type": leak_result.gas_type,
                "leak_rate": leak_result.leak_rate,
                "severity": leak_result.severity,
                "estimated_source": leak_result.estimated_source,
                "recommendations": leak_result.recommendations,
                "confidence": leak_result.confidence
            }

            logger.debug(f"[GasPredictor] GL-TransLSTM预测完成: {len(predictions['gas_predictions'])}种气体")

            return predictions

        except Exception as e:
            logger.warning(f"[GasPredictor] GL-TransLSTM预测失败: {e}")
            return {"success": False, "reason": str(e)}

    def _update_gl_translstm_history(self, history: Dict) -> None:
        """更新GL-TransLSTM模型的历史数据"""
        if self._gl_translstm is None:
            return

        timestamps = history.get("timestamps", [])
        gas_data = history.get("gas_data", {})
        environmental = history.get("environmental", {})

        # 添加最新数据点
        if timestamps:
            latest_ts = timestamps[-1]

            # 获取最新环境数据
            env_data = {}
            for param in ["temperature", "humidity", "pressure"]:
                values = environmental.get(param, [])
                if values:
                    env_data[param] = values[-1]

            # 更新各气体数据
            for gas, values in gas_data.items():
                if values:
                    self._gl_translstm.add_reading(
                        gas_type=gas,
                        value=values[-1],
                        timestamp=latest_ts,
                        environmental=env_data if env_data else None
                    )
    
    def _predict_by_deep_learning(self, history: Dict) -> Dict[str, Any]:
        """使用深度学习模型预测"""
        try:
            # 准备输入数据
            input_sequence = self._prepare_input_sequence(history)
            
            if input_sequence is None:
                return {"success": False, "reason": "数据准备失败"}
            
            # 选择模型
            model_id = self.config.model_ids.get("transformer", 
                        self.config.model_ids.get("lstm"))
            
            # 推理
            result = self._model_registry.infer(model_id, {"input": input_sequence})
            
            if not result.get("success"):
                return {"success": False, "reason": "模型推理失败"}
            
            # 解析输出
            outputs = result.get("outputs", {})
            predictions = self._parse_predictions(outputs, history)
            
            return {
                "available": True,
                "success": True,
                "method": "deep_learning",
                "model_id": model_id,
                **predictions
            }
            
        except Exception as e:
            logger.warning(f"深度学习预测失败: {e}")
            return {"success": False, "reason": str(e)}
    
    def _prepare_input_sequence(self, history: Dict) -> Optional[np.ndarray]:
        """准备模型输入序列"""
        try:
            seq_len = self.config.history_length
            num_features = 8  # 4气体 + 4环境参数
            
            # 收集所有特征
            features = []
            
            # 气体数据
            for gas in ["SF6", "H2", "CO", "C2H2"]:
                values = history["gas_data"].get(gas, [])
                if values:
                    normalized = self._normalize(values, gas)
                    features.append(normalized)
                else:
                    features.append([0] * len(history["timestamps"]))
            
            # 环境参数
            for param in ["temperature", "humidity", "pressure"]:
                values = history["environmental"].get(param, [])
                if values:
                    normalized = self._normalize(values, param)
                    features.append(normalized)
                else:
                    features.append([0] * len(history["timestamps"]))
            
            # 负载 (假设值)
            features.append([0.5] * len(history["timestamps"]))
            
            # 转换为数组
            data = np.array(features).T  # (time, features)
            
            # 填充或截断到目标长度
            if len(data) > seq_len:
                data = data[-seq_len:]
            elif len(data) < seq_len:
                padding = np.zeros((seq_len - len(data), num_features))
                data = np.vstack([padding, data])
            
            # 添加批次维度
            return data[np.newaxis, ...].astype(np.float32)
            
        except Exception as e:
            logger.error(f"准备输入序列失败: {e}")
            return None
    
    def _normalize(self, values: List[float], param_name: str) -> List[float]:
        """Z-score归一化"""
        params = self.normalization_params.get(param_name, {"mean": 0, "std": 1})
        return [(v - params["mean"]) / (params["std"] + 1e-8) for v in values]
    
    def _denormalize(self, values: np.ndarray, param_name: str) -> np.ndarray:
        """反归一化"""
        params = self.normalization_params.get(param_name, {"mean": 0, "std": 1})
        return values * params["std"] + params["mean"]
    
    def _parse_predictions(self, outputs: Dict, history: Dict) -> Dict[str, Any]:
        """解析模型输出"""
        predictions = {
            "prediction_horizon": self.config.prediction_horizon,
            "gas_predictions": {},
            "predicted_alarms": [],
            "confidence_intervals": {}
        }
        
        # 获取预测输出
        mu = outputs.get("mu", outputs.get("prediction"))
        sigma = outputs.get("sigma")
        
        if mu is None:
            return predictions
        
        mu = np.array(mu).squeeze()
        
        # 解析各气体预测
        gases = ["SF6", "H2", "CO", "C2H2"]
        
        for i, gas in enumerate(gases):
            if mu.ndim > 1 and i < mu.shape[-1]:
                gas_pred = self._denormalize(mu[:, i], gas)
            elif mu.ndim == 1 and i == 0:
                gas_pred = self._denormalize(mu, gas)
            else:
                continue
            
            predictions["gas_predictions"][gas] = {
                "values": gas_pred.tolist(),
                "time_steps": list(range(1, len(gas_pred) + 1)),
                "unit": "ppm"
            }
            
            # 置信区间
            if sigma is not None:
                sigma_arr = np.array(sigma).squeeze()
                if sigma_arr.ndim > 1 and i < sigma_arr.shape[-1]:
                    gas_sigma = self._denormalize(sigma_arr[:, i], gas)
                    predictions["confidence_intervals"][gas] = {
                        "lower_95": (gas_pred - 1.96 * gas_sigma).tolist(),
                        "upper_95": (gas_pred + 1.96 * gas_sigma).tolist()
                    }
            
            # 检查是否超阈值
            threshold = self.config.thresholds.get(gas)
            if threshold:
                for t, value in enumerate(gas_pred):
                    if value > threshold.warning:
                        predictions["predicted_alarms"].append({
                            "gas": gas,
                            "predicted_value": float(value),
                            "threshold": threshold.warning,
                            "hours_until": t + 1,
                            "predicted_time": f"+{t+1}h",
                            "severity": "warning" if value < threshold.alarm else "alarm"
                        })
                        break  # 只记录首次超阈值
        
        return predictions
    
    def _predict_by_traditional(self, history: Dict) -> Dict[str, Any]:
        """使用传统方法预测 (线性外推 + 移动平均)"""
        predictions = {
            "available": True,
            "method": "traditional",
            "prediction_horizon": self.config.prediction_horizon,
            "gas_predictions": {},
            "predicted_alarms": []
        }
        
        for gas, values in history["gas_data"].items():
            if len(values) < 24:
                continue
            
            # 移动平均平滑
            window_size = min(24, len(values))
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            
            if len(smoothed) < 2:
                continue
            
            # 线性趋势
            x = np.arange(len(smoothed))
            slope = (smoothed[-1] - smoothed[0]) / (len(smoothed) - 1)
            
            # 外推预测
            last_value = values[-1]
            pred_values = [
                last_value + slope * (t + 1)
                for t in range(self.config.prediction_horizon)
            ]
            
            # 限制预测值非负
            pred_values = [max(0, v) for v in pred_values]
            
            predictions["gas_predictions"][gas] = {
                "values": pred_values,
                "time_steps": list(range(1, self.config.prediction_horizon + 1)),
                "unit": "ppm",
                "trend_slope": float(slope)
            }
            
            # 检查阈值
            threshold = self.config.thresholds.get(gas)
            if threshold:
                for t, value in enumerate(pred_values):
                    if value > threshold.warning:
                        predictions["predicted_alarms"].append({
                            "gas": gas,
                            "predicted_value": float(value),
                            "threshold": threshold.warning,
                            "hours_until": t + 1,
                            "predicted_time": f"+{t+1}h",
                            "severity": "warning" if value < threshold.alarm else "alarm"
                        })
                        break
        
        return predictions
    
    def predict_single_gas(self, gas: str, history: List[float],
                           horizon: int = 24) -> Dict[str, Any]:
        """
        单一气体预测 (简化接口) V3.0

        Args:
            gas: 气体类型
            history: 历史浓度数据列表
            horizon: 预测步数

        Returns:
            预测结果
        """
        if len(history) < 24:
            return {"success": False, "reason": "数据不足"}

        # V3.0: 优先使用GL-TransLSTM
        if self._dl_initialized and self._gl_translstm is not None:
            try:
                # 添加历史数据到模型
                current_time = time.time()
                for i, value in enumerate(history):
                    self._gl_translstm.add_reading(
                        gas_type=gas,
                        value=value,
                        timestamp=current_time - (len(history) - i) * 3600
                    )

                # 选择预测范围
                if horizon <= 1:
                    pred_horizon = PredictionHorizon.HOUR_1
                elif horizon <= 6:
                    pred_horizon = PredictionHorizon.HOUR_6
                elif horizon <= 24:
                    pred_horizon = PredictionHorizon.HOUR_24
                else:
                    pred_horizon = PredictionHorizon.DAY_7

                # 执行预测
                result = self._gl_translstm.predict(gas, pred_horizon)

                if result.predictions:
                    return {
                        "success": True,
                        "gas": gas,
                        "predictions": result.predictions[:horizon],
                        "trend": result.trend,
                        "confidence": result.confidence,
                        "leak_probability": result.leak_probability,
                        "time_to_alarm": result.time_to_alarm,
                        "method": "GL-TransLSTM"
                    }

            except Exception as e:
                logger.warning(f"[GasPredictor] GL-TransLSTM单气体预测失败: {e}")

        # 回退: 移动平均 + 线性外推
        window_size = min(24, len(history))
        smoothed = np.convolve(history, np.ones(window_size)/window_size, mode='valid')

        slope = (smoothed[-1] - smoothed[0]) / (len(smoothed) - 1 + 1e-8)
        last_value = history[-1]

        predictions = [
            max(0, last_value + slope * (t + 1))
            for t in range(horizon)
        ]

        return {
            "success": True,
            "gas": gas,
            "predictions": predictions,
            "trend": "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable",
            "slope": float(slope),
            "method": "traditional"
        }

    def detect_leak(self) -> Dict[str, Any]:
        """
        泄漏检测 (V3.0新增)

        Returns:
            泄漏检测结果
        """
        if self._dl_initialized and self._gl_translstm is not None:
            try:
                result = self._gl_translstm.detect_leak()
                return {
                    "success": True,
                    "detected": result.detected,
                    "gas_type": result.gas_type,
                    "leak_rate": result.leak_rate,
                    "severity": result.severity,
                    "estimated_source": result.estimated_source,
                    "recommendations": result.recommendations,
                    "confidence": result.confidence,
                    "method": "GL-TransLSTM"
                }
            except Exception as e:
                logger.warning(f"[GasPredictor] 泄漏检测失败: {e}")

        return {
            "success": False,
            "detected": False,
            "reason": "GL-TransLSTM模型不可用"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息 (V3.0新增)"""
        info = {
            "version": self._model_version,
            "gl_translstm_available": GL_TRANSLSTM_AVAILABLE,
            "dl_initialized": self._dl_initialized,
            "model_registry_available": self._model_registry is not None,
            "use_deep_learning": self._use_deep_learning
        }

        if self._gl_translstm is not None:
            info["gl_translstm_info"] = self._gl_translstm.model_info
            info["data_status"] = self._gl_translstm.data_status

        return info

    def cleanup(self) -> None:
        """清理资源 (V3.0新增)"""
        if self._gl_translstm is not None:
            self._gl_translstm.cleanup()
            self._gl_translstm = None
        self._dl_initialized = False
        logger.info("[GasPredictor] 资源已清理")
