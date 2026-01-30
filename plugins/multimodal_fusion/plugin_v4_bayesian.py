#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合插件 - MultiModalFusionPlugin
========================================

根据《室外监测平台迭代方案》实现:
1. 因果贝叶斯网络构建
2. 多传感器证据融合
3. 实时故障推断
4. 可解释性输出

版本: 1.0.0
"""

from __future__ import annotations
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FaultType(str, Enum):
    """故障类型"""
    NORMAL = "normal"
    OVERHEATING = "overheating"
    PARTIAL_DISCHARGE = "partial_discharge"
    GAS_LEAK = "gas_leak"
    CONTACT_FAULT = "contact_fault"
    INSULATION_DEFECT = "insulation_defect"
    MECHANICAL_FAULT = "mechanical_fault"
    OIL_LEAK = "oil_leak"


class SensorType(str, Enum):
    """传感器类型"""
    THERMAL = "thermal"
    ACOUSTIC = "acoustic"
    GAS = "gas"
    VISUAL = "visual"
    HYPERSPECTRAL = "hyperspectral"
    VIBRATION = "vibration"


@dataclass
class SensorEvidence:
    """传感器证据"""
    sensor_type: SensorType
    value: float  # 归一化值 [0, 1]
    confidence: float
    health: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def effective_confidence(self) -> float:
        return self.confidence * self.health


@dataclass
class DiagnosisResult:
    """诊断结果"""
    fault_type: FaultType
    probability: float
    confidence: float
    contributing_factors: List[Dict[str, Any]]
    recommendation: str
    explanation: str


class BayesianNetwork:
    """
    简化的贝叶斯网络
    
    实现故障类型与传感器观测之间的因果推断
    """
    
    def __init__(self):
        # 先验概率 P(Fault)
        self.prior_probs = {
            FaultType.NORMAL: 0.85,
            FaultType.OVERHEATING: 0.03,
            FaultType.PARTIAL_DISCHARGE: 0.02,
            FaultType.GAS_LEAK: 0.02,
            FaultType.CONTACT_FAULT: 0.03,
            FaultType.INSULATION_DEFECT: 0.02,
            FaultType.MECHANICAL_FAULT: 0.02,
            FaultType.OIL_LEAK: 0.01,
        }
        
        # 条件概率 P(Sensor | Fault) - 传感器在各故障下的响应概率
        # sensor_type -> fault_type -> (mean, std) for abnormal reading
        self.likelihood = {
            SensorType.THERMAL: {
                FaultType.NORMAL: (0.1, 0.1),
                FaultType.OVERHEATING: (0.9, 0.1),
                FaultType.PARTIAL_DISCHARGE: (0.5, 0.2),
                FaultType.GAS_LEAK: (0.2, 0.15),
                FaultType.CONTACT_FAULT: (0.8, 0.15),
                FaultType.INSULATION_DEFECT: (0.4, 0.2),
                FaultType.MECHANICAL_FAULT: (0.3, 0.2),
                FaultType.OIL_LEAK: (0.2, 0.15),
            },
            SensorType.ACOUSTIC: {
                FaultType.NORMAL: (0.05, 0.05),
                FaultType.OVERHEATING: (0.2, 0.15),
                FaultType.PARTIAL_DISCHARGE: (0.95, 0.05),
                FaultType.GAS_LEAK: (0.3, 0.2),
                FaultType.CONTACT_FAULT: (0.4, 0.2),
                FaultType.INSULATION_DEFECT: (0.6, 0.2),
                FaultType.MECHANICAL_FAULT: (0.8, 0.15),
                FaultType.OIL_LEAK: (0.1, 0.1),
            },
            SensorType.GAS: {
                FaultType.NORMAL: (0.05, 0.05),
                FaultType.OVERHEATING: (0.3, 0.2),
                FaultType.PARTIAL_DISCHARGE: (0.4, 0.2),
                FaultType.GAS_LEAK: (0.95, 0.05),
                FaultType.CONTACT_FAULT: (0.2, 0.15),
                FaultType.INSULATION_DEFECT: (0.5, 0.2),
                FaultType.MECHANICAL_FAULT: (0.1, 0.1),
                FaultType.OIL_LEAK: (0.2, 0.15),
            },
            SensorType.VISUAL: {
                FaultType.NORMAL: (0.05, 0.05),
                FaultType.OVERHEATING: (0.3, 0.2),
                FaultType.PARTIAL_DISCHARGE: (0.3, 0.2),
                FaultType.GAS_LEAK: (0.2, 0.15),
                FaultType.CONTACT_FAULT: (0.4, 0.2),
                FaultType.INSULATION_DEFECT: (0.5, 0.2),
                FaultType.MECHANICAL_FAULT: (0.3, 0.2),
                FaultType.OIL_LEAK: (0.9, 0.1),
            },
            SensorType.HYPERSPECTRAL: {
                FaultType.NORMAL: (0.05, 0.05),
                FaultType.OVERHEATING: (0.4, 0.2),
                FaultType.PARTIAL_DISCHARGE: (0.3, 0.2),
                FaultType.GAS_LEAK: (0.2, 0.15),
                FaultType.CONTACT_FAULT: (0.3, 0.2),
                FaultType.INSULATION_DEFECT: (0.85, 0.1),
                FaultType.MECHANICAL_FAULT: (0.2, 0.15),
                FaultType.OIL_LEAK: (0.7, 0.15),
            },
            SensorType.VIBRATION: {
                FaultType.NORMAL: (0.1, 0.1),
                FaultType.OVERHEATING: (0.2, 0.15),
                FaultType.PARTIAL_DISCHARGE: (0.2, 0.15),
                FaultType.GAS_LEAK: (0.1, 0.1),
                FaultType.CONTACT_FAULT: (0.3, 0.2),
                FaultType.INSULATION_DEFECT: (0.2, 0.15),
                FaultType.MECHANICAL_FAULT: (0.9, 0.1),
                FaultType.OIL_LEAK: (0.1, 0.1),
            },
        }
    
    def compute_likelihood(
        self, 
        sensor_type: SensorType, 
        fault_type: FaultType, 
        observed_value: float
    ) -> float:
        """计算似然度 P(观测值 | 故障类型)"""
        if sensor_type not in self.likelihood:
            return 1.0
        
        if fault_type not in self.likelihood[sensor_type]:
            return 1.0
        
        mean, std = self.likelihood[sensor_type][fault_type]
        
        # 高斯似然
        likelihood = np.exp(-0.5 * ((observed_value - mean) / (std + 1e-6)) ** 2)
        return float(likelihood)
    
    def infer(self, evidences: List[SensorEvidence]) -> Dict[FaultType, float]:
        """
        贝叶斯推断
        
        计算 P(Fault | Evidences) ∝ P(Evidences | Fault) * P(Fault)
        """
        posteriors = {}
        
        for fault_type in FaultType:
            # 先验
            prior = self.prior_probs.get(fault_type, 0.01)
            
            # 似然乘积
            likelihood = 1.0
            for evidence in evidences:
                sensor_likelihood = self.compute_likelihood(
                    evidence.sensor_type,
                    fault_type,
                    evidence.value
                )
                # 考虑传感器健康度
                weighted_likelihood = (
                    sensor_likelihood * evidence.health + 
                    (1 - evidence.health) * 0.5  # 不健康时趋向均匀
                )
                likelihood *= weighted_likelihood
            
            posteriors[fault_type] = prior * likelihood
        
        # 归一化
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        
        return posteriors


class MultiModalFusionPlugin:
    """
    多模态融合插件
    
    实现基于贝叶斯网络的多传感器融合诊断
    """
    
    PLUGIN_ID = "multimodal_fusion"
    PLUGIN_NAME = "多模态融合"
    VERSION = "1.0.0"
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._initialized = False
        
        # 贝叶斯网络
        self.bayesian_network = BayesianNetwork()
        
        # 传感器权重 (可通过在线学习更新)
        self.sensor_weights = {
            SensorType.THERMAL: 1.0,
            SensorType.ACOUSTIC: 1.0,
            SensorType.GAS: 1.0,
            SensorType.VISUAL: 0.8,
            SensorType.HYPERSPECTRAL: 0.9,
            SensorType.VIBRATION: 0.7,
        }
        
        # 故障处置建议
        self.recommendations = {
            FaultType.NORMAL: "设备状态正常，继续常规巡检",
            FaultType.OVERHEATING: "发现过热，建议降低负荷并安排热成像详查",
            FaultType.PARTIAL_DISCHARGE: "检测到局放，建议停电检修绝缘系统",
            FaultType.GAS_LEAK: "发现气体泄漏，立即检查密封并补气",
            FaultType.CONTACT_FAULT: "接触不良，建议紧固导电接头",
            FaultType.INSULATION_DEFECT: "绝缘缺陷，安排绝缘测试和检修",
            FaultType.MECHANICAL_FAULT: "机械故障，检查机构润滑和紧固",
            FaultType.OIL_LEAK: "渗漏油，检查密封件并补充油位",
        }
        
        logger.info(f"[{self.PLUGIN_ID}] 初始化完成 v{self.VERSION}")
    
    def init(self, context: Optional[Dict] = None) -> bool:
        self._initialized = True
        return True
    
    def process(
        self,
        sensor_data: Dict[str, Dict[str, Any]],
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        处理多传感器数据，执行融合诊断
        
        Args:
            sensor_data: 各传感器数据，格式为 {sensor_type: {value, confidence, health, ...}}
            context: 运行上下文
            
        Returns:
            融合诊断结果
        """
        start_time = time.time()
        
        # 构建证据列表
        evidences = []
        sensor_status = {}
        
        for sensor_name, data in sensor_data.items():
            try:
                sensor_type = SensorType(sensor_name)
            except ValueError:
                continue
            
            evidence = SensorEvidence(
                sensor_type=sensor_type,
                value=data.get("value", 0.0),
                confidence=data.get("confidence", 0.8),
                health=data.get("health", 1.0),
                raw_data=data
            )
            evidences.append(evidence)
            
            sensor_status[sensor_name] = {
                "value": evidence.value,
                "confidence": evidence.confidence,
                "health": evidence.health,
                "effective": evidence.effective_confidence
            }
        
        # 贝叶斯推断
        posteriors = self.bayesian_network.infer(evidences)
        
        # 找出最可能的故障
        sorted_faults = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        primary_fault = sorted_faults[0][0]
        primary_prob = sorted_faults[0][1]
        
        # 如果正常概率最高且大于阈值，判定为正常
        normal_threshold = self.config.get("normal_threshold", 0.6)
        if primary_fault == FaultType.NORMAL and primary_prob > normal_threshold:
            diagnosis_status = "normal"
        elif primary_prob < 0.3:
            # 不确定
            diagnosis_status = "uncertain"
        else:
            diagnosis_status = "warning" if primary_prob < 0.7 else "alarm"
        
        # 计算综合健康指数
        health_index = posteriors[FaultType.NORMAL]
        
        # 生成解释
        explanation = self._generate_explanation(evidences, posteriors, primary_fault)
        
        # 构建诊断结果
        detections = [
            {
                "id": "fusion_diagnosis",
                "label": "综合诊断",
                "confidence": primary_prob,
                "status": diagnosis_status,
                "details": f"诊断结果: {primary_fault.value}",
                "metadata": {
                    "fault_type": primary_fault.value,
                    "probability": primary_prob,
                    "all_posteriors": {k.value: v for k, v in posteriors.items()}
                }
            },
            {
                "id": "health_index",
                "label": "健康指数",
                "confidence": health_index,
                "status": "normal" if health_index > 0.7 else "warning" if health_index > 0.4 else "alarm",
                "details": f"健康度: {health_index*100:.1f}%"
            }
        ]
        
        # 添加各传感器状态
        for sensor_name, status in sensor_status.items():
            sensor_st = "normal" if status["value"] < 0.3 else "warning" if status["value"] < 0.7 else "alarm"
            detections.append({
                "id": f"sensor_{sensor_name}",
                "label": f"传感器-{sensor_name}",
                "confidence": status["confidence"],
                "status": sensor_st,
                "details": f"值: {status['value']:.2f}, 健康度: {status['health']:.2f}"
            })
        
        # 告警
        alarms = []
        if diagnosis_status in ["warning", "alarm"]:
            alarms.append({
                "id": f"alm_fusion_{int(time.time()*1000) % 10000}",
                "type": "fusion_diagnosis",
                "level": "error" if diagnosis_status == "alarm" else "warning",
                "message": f"融合诊断: {primary_fault.value} (概率: {primary_prob*100:.1f}%)",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "fault_type": primary_fault.value,
                    "recommendation": self.recommendations.get(primary_fault, ""),
                    "explanation": explanation
                }
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "module_id": self.PLUGIN_ID,
            "module_name": self.PLUGIN_NAME,
            "status": diagnosis_status,
            "timestamp": int(time.time() * 1000),
            "detections": detections,
            "alarms": alarms,
            "metrics": {
                "processing_time_ms": processing_time,
                "health_index": health_index,
                "sensor_count": len(evidences),
                "primary_fault": primary_fault.value,
                "primary_probability": primary_prob
            },
            "metadata": {
                "posteriors": {k.value: v for k, v in posteriors.items()},
                "sensor_status": sensor_status,
                "explanation": explanation,
                "recommendation": self.recommendations.get(primary_fault, "")
            }
        }
    
    def _generate_explanation(
        self,
        evidences: List[SensorEvidence],
        posteriors: Dict[FaultType, float],
        primary_fault: FaultType
    ) -> str:
        """生成可解释性说明"""
        lines = []
        
        # 主要诊断
        lines.append(f"【诊断结果】{primary_fault.value} (概率: {posteriors[primary_fault]*100:.1f}%)")
        
        # 证据分析
        lines.append("\n【证据分析】")
        for ev in evidences:
            status = "异常" if ev.value > 0.5 else "正常"
            lines.append(f"  - {ev.sensor_type.value}: {status} (值={ev.value:.2f}, 置信度={ev.confidence:.2f})")
        
        # 推理过程
        lines.append("\n【推理过程】")
        if primary_fault == FaultType.NORMAL:
            lines.append("  所有传感器读数在正常范围内，设备运行正常。")
        else:
            # 找出支持该诊断的证据
            supporting = []
            for ev in evidences:
                mean, std = self.bayesian_network.likelihood.get(ev.sensor_type, {}).get(primary_fault, (0.5, 0.2))
                if ev.value > mean - std:
                    supporting.append(ev.sensor_type.value)
            
            if supporting:
                lines.append(f"  支持该诊断的传感器: {', '.join(supporting)}")
        
        return "\n".join(lines)
    
    def update_sensor_weight(self, sensor_type: SensorType, feedback: float):
        """
        根据反馈更新传感器权重 (在线学习)
        
        Args:
            sensor_type: 传感器类型
            feedback: 反馈值 (正为有效，负为无效)
        """
        if sensor_type in self.sensor_weights:
            # 简单的学习率更新
            learning_rate = 0.1
            self.sensor_weights[sensor_type] += learning_rate * feedback
            self.sensor_weights[sensor_type] = max(0.1, min(2.0, self.sensor_weights[sensor_type]))
    
    def shutdown(self):
        self._initialized = False
        logger.info(f"[{self.PLUGIN_ID}] 已关闭")


if __name__ == "__main__":
    print("=== 多模态融合插件测试 ===")
    
    plugin = MultiModalFusionPlugin()
    plugin.init()
    
    # 模拟传感器数据 - 过热场景
    sensor_data = {
        "thermal": {"value": 0.8, "confidence": 0.9, "health": 1.0},
        "acoustic": {"value": 0.3, "confidence": 0.85, "health": 0.95},
        "gas": {"value": 0.1, "confidence": 0.88, "health": 1.0},
        "visual": {"value": 0.2, "confidence": 0.92, "health": 1.0},
    }
    
    result = plugin.process(sensor_data)
    
    print(f"状态: {result['status']}")
    print(f"健康指数: {result['metrics']['health_index']*100:.1f}%")
    print(f"主要故障: {result['metrics']['primary_fault']}")
    print(f"故障概率: {result['metrics']['primary_probability']*100:.1f}%")
    print(f"\n{result['metadata']['explanation']}")
    print(f"\n建议: {result['metadata']['recommendation']}")
    
    print("\n测试完成!")
