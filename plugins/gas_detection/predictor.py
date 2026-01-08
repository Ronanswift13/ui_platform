#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气体浓度预测器模块
基于LSTM/Transformer的时序预测

功能:
1. 单气体浓度预测 (LSTM)
2. 多气体综合预测 (Transformer)
3. 预测区间估计 (不确定性量化)
4. 阈值超越预警
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class GasConcentrationPredictor:
    """
    气体浓度预测器
    
    支持:
    1. LSTM单变量预测
    2. Transformer多变量预测
    3. 概率预测 (输出均值和方差)
    """
    
    def __init__(self, config):
        self.config = config
        self._model_registry = None
        self._use_deep_learning = False
        
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
    
    def set_model_registry(self, model_registry):
        """设置模型注册中心"""
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
    
    def predict(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测气体浓度趋势
        
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
        
        # 尝试深度学习预测
        if self._use_deep_learning and self._model_registry:
            dl_result = self._predict_by_deep_learning(history)
            if dl_result.get("success"):
                return dl_result
        
        # 回退到传统方法
        return self._predict_by_traditional(history)
    
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
        """单一气体预测 (简化接口)"""
        if len(history) < 24:
            return {"success": False, "reason": "数据不足"}
        
        # 移动平均 + 线性外推
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
            "slope": float(slope)
        }
