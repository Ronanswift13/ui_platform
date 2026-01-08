#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气体数据分析器模块
提供趋势分析、异常检测、关联分析功能

功能:
1. 趋势分析 (上升/下降/稳定)
2. 异常点检测
3. 多气体关联分析
4. 油中溶解气体分析 (DGA)
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class GasDataAnalyzer:
    """
    气体数据分析器
    
    提供:
    1. 趋势分析
    2. 异常检测
    3. DGA故障诊断
    """
    
    def __init__(self, config):
        self.config = config
        
        # DGA比值法参考值 (IEC 60599)
        self.dga_ratios = {
            "C2H2/C2H4": {"low": 0.1, "mid": 1.0, "high": 3.0},
            "CH4/H2": {"low": 0.1, "mid": 1.0, "high": 3.0},
            "C2H4/C2H6": {"low": 1.0, "mid": 3.0, "high": 10.0}
        }
        
        # 故障类型编码
        self.fault_codes = {
            (0, 0, 0): "正常老化",
            (0, 1, 0): "低温过热 (150-300°C)",
            (0, 2, 0): "中温过热 (300-700°C)",
            (0, 2, 1): "高温过热 (>700°C)",
            (1, 0, 0): "低能放电 (局部放电)",
            (1, 0, 1): "低能放电 (局部放电)",
            (1, 0, 2): "低能放电 (局部放电)",
            (2, 0, 1): "高能放电 (电弧)",
            (2, 0, 2): "高能放电 (电弧)",
        }
    
    def analyze_trends(self, history: Dict[str, Any],
                       current_readings: Dict[str, float]) -> Dict[str, Any]:
        """
        分析气体趋势
        
        Args:
            history: 历史数据
            current_readings: 当前读数
        
        Returns:
            趋势分析结果
        """
        results = {
            "gas_trends": {},
            "abnormal_trends": [],
            "rate_of_change": {},
            "statistical_summary": {}
        }
        
        gas_data = history.get("gas_data", {})
        
        for gas, values in gas_data.items():
            if len(values) < 24:
                continue
            
            values_arr = np.array(values)
            
            # 趋势分析
            trend_info = self._analyze_single_trend(values_arr)
            results["gas_trends"][gas] = trend_info
            
            # 变化率
            roc = self._calculate_rate_of_change(values_arr)
            results["rate_of_change"][gas] = roc
            
            # 统计摘要
            results["statistical_summary"][gas] = {
                "current": float(current_readings.get(gas, values_arr[-1])),
                "mean_24h": float(np.mean(values_arr[-24:])),
                "std_24h": float(np.std(values_arr[-24:])),
                "min_24h": float(np.min(values_arr[-24:])),
                "max_24h": float(np.max(values_arr[-24:])),
                "mean_7d": float(np.mean(values_arr)) if len(values_arr) >= 168 else None,
            }
            
            # 异常趋势检测
            if trend_info["is_abnormal"]:
                results["abnormal_trends"].append({
                    "gas": gas,
                    "pattern": trend_info["pattern"],
                    "severity": trend_info["severity"],
                    "description": trend_info["description"]
                })
        
        # 添加DGA分析 (如果有油中溶解气体)
        if all(g in gas_data for g in ["H2", "CH4", "C2H2", "C2H4", "C2H6"]):
            results["dga_analysis"] = self._dga_analysis(current_readings)
        
        return results
    
    def _analyze_single_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """分析单一气体趋势"""
        n = len(values)
        
        # 线性回归
        x = np.arange(n)
        slope, intercept = np.polyfit(x, values, 1)
        
        # R²值
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        # 判断趋势
        mean_value = np.mean(values)
        relative_slope = slope / (mean_value + 1e-8) * 100  # %/hour
        
        if relative_slope > 1:
            pattern = "rapid_increase"
            description = f"快速上升 ({relative_slope:.2f}%/h)"
            is_abnormal = True
            severity = "warning" if relative_slope < 5 else "alarm"
        elif relative_slope > 0.1:
            pattern = "slow_increase"
            description = f"缓慢上升 ({relative_slope:.2f}%/h)"
            is_abnormal = relative_slope > 0.5
            severity = "attention"
        elif relative_slope < -1:
            pattern = "rapid_decrease"
            description = f"快速下降 ({relative_slope:.2f}%/h)"
            is_abnormal = True
            severity = "attention"
        elif relative_slope < -0.1:
            pattern = "slow_decrease"
            description = f"缓慢下降 ({relative_slope:.2f}%/h)"
            is_abnormal = False
            severity = "normal"
        else:
            pattern = "stable"
            description = "稳定"
            is_abnormal = False
            severity = "normal"
        
        # 检测波动
        std = np.std(values)
        cv = std / (mean_value + 1e-8)  # 变异系数
        
        if cv > 0.3:
            pattern = "fluctuating"
            description = f"波动较大 (CV={cv:.2f})"
            is_abnormal = True
            severity = "attention"
        
        return {
            "pattern": pattern,
            "description": description,
            "slope": float(slope),
            "relative_slope_percent_per_hour": float(relative_slope),
            "r_squared": float(r_squared),
            "coefficient_of_variation": float(cv),
            "is_abnormal": is_abnormal,
            "severity": severity
        }
    
    def _calculate_rate_of_change(self, values: np.ndarray) -> Dict[str, float]:
        """计算变化率"""
        if len(values) < 2:
            return {}
        
        # 小时变化率
        hourly_change = values[-1] - values[-2] if len(values) >= 2 else 0
        
        # 日变化率
        daily_change = values[-1] - values[-24] if len(values) >= 24 else None
        
        # 周变化率
        weekly_change = values[-1] - values[-168] if len(values) >= 168 else None
        
        return {
            "hourly": float(hourly_change),
            "daily": float(daily_change) if daily_change else None,
            "weekly": float(weekly_change) if weekly_change else None
        }
    
    def _dga_analysis(self, readings: Dict[str, float]) -> Dict[str, Any]:
        """油中溶解气体分析 (DGA)"""
        result = {
            "method": "IEC_60599_ratio",
            "ratios": {},
            "fault_type": None,
            "fault_description": None,
            "recommendations": []
        }
        
        # 计算关键比值
        h2 = readings.get("H2", 0) + 1e-8
        ch4 = readings.get("CH4", 0) + 1e-8
        c2h2 = readings.get("C2H2", 0) + 1e-8
        c2h4 = readings.get("C2H4", 0) + 1e-8
        c2h6 = readings.get("C2H6", 0) + 1e-8
        
        ratios = {
            "C2H2/C2H4": c2h2 / c2h4,
            "CH4/H2": ch4 / h2,
            "C2H4/C2H6": c2h4 / c2h6
        }
        result["ratios"] = {k: float(v) for k, v in ratios.items()}
        
        # 编码
        def encode_ratio(value, thresholds):
            if value < thresholds["low"]:
                return 0
            elif value < thresholds["mid"]:
                return 1
            else:
                return 2
        
        code = tuple(
            encode_ratio(ratios[r], self.dga_ratios[r])
            for r in ["C2H2/C2H4", "CH4/H2", "C2H4/C2H6"]
        )
        
        result["fault_code"] = code
        result["fault_type"] = self.fault_codes.get(code, "未知故障类型")
        
        # 生成建议
        if "放电" in result["fault_type"]:
            result["recommendations"].append("建议进行超声波局放检测")
            result["recommendations"].append("检查绝缘状况和接地")
        
        if "过热" in result["fault_type"]:
            result["recommendations"].append("建议进行红外热成像检测")
            result["recommendations"].append("检查接触电阻和负荷情况")
        
        if result["fault_type"] == "正常老化":
            result["recommendations"].append("设备状态良好，保持监测")
        
        return result
    
    def detect_anomalies(self, values: np.ndarray,
                         method: str = "zscore") -> Dict[str, Any]:
        """
        异常点检测
        
        Args:
            values: 数据序列
            method: 检测方法 (zscore, iqr, isolation)
        
        Returns:
            异常检测结果
        """
        if len(values) < 10:
            return {"anomalies": [], "method": method}
        
        if method == "zscore":
            return self._detect_anomalies_zscore(values)
        elif method == "iqr":
            return self._detect_anomalies_iqr(values)
        else:
            return self._detect_anomalies_zscore(values)
    
    def _detect_anomalies_zscore(self, values: np.ndarray,
                                  threshold: float = 3.0) -> Dict[str, Any]:
        """Z-score异常检测"""
        mean = np.mean(values)
        std = np.std(values)
        
        z_scores = np.abs((values - mean) / (std + 1e-8))
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        anomalies = [
            {
                "index": int(idx),
                "value": float(values[idx]),
                "z_score": float(z_scores[idx])
            }
            for idx in anomaly_indices
        ]
        
        return {
            "method": "zscore",
            "threshold": threshold,
            "mean": float(mean),
            "std": float(std),
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "anomaly_ratio": len(anomalies) / len(values)
        }
    
    def _detect_anomalies_iqr(self, values: np.ndarray,
                              k: float = 1.5) -> Dict[str, Any]:
        """IQR异常检测"""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
        
        anomalies = [
            {
                "index": int(idx),
                "value": float(values[idx]),
                "type": "low" if values[idx] < lower_bound else "high"
            }
            for idx in anomaly_indices
        ]
        
        return {
            "method": "iqr",
            "k": k,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "anomalies": anomalies,
            "anomaly_count": len(anomalies)
        }
    
    def analyze_correlation(self, gas_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """分析多气体相关性"""
        gases = list(gas_data.keys())
        n = len(gases)
        
        if n < 2:
            return {"available": False, "reason": "气体种类不足"}
        
        # 找到最小长度
        min_len = min(len(v) for v in gas_data.values())
        if min_len < 10:
            return {"available": False, "reason": "数据长度不足"}
        
        # 计算相关矩阵
        correlation_matrix = np.zeros((n, n))
        
        for i, gas1 in enumerate(gases):
            for j, gas2 in enumerate(gases):
                vals1 = np.array(gas_data[gas1][-min_len:])
                vals2 = np.array(gas_data[gas2][-min_len:])
                
                # Pearson相关系数
                corr = np.corrcoef(vals1, vals2)[0, 1]
                correlation_matrix[i, j] = corr
        
        # 找出强相关对
        strong_correlations = []
        for i in range(n):
            for j in range(i+1, n):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.7:
                    strong_correlations.append({
                        "gas1": gases[i],
                        "gas2": gases[j],
                        "correlation": float(corr),
                        "type": "positive" if corr > 0 else "negative"
                    })
        
        return {
            "available": True,
            "gases": gases,
            "correlation_matrix": correlation_matrix.tolist(),
            "strong_correlations": strong_correlations
        }
