#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型融合引擎 - FusionEngine
==============================

根据《室外监测平台迭代方案》核心基础设施要求实现

功能:
1. 传感器健康度加权融合
2. Dempster-Shafer证据理论
3. 物理逻辑约束检查
4. 多模态数据融合

版本: 2.0.0
作者: 输变电监测平台开发组
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """传感器类型"""
    VISUAL = "visual"           # 可见光相机
    THERMAL = "thermal"         # 红外热成像
    ACOUSTIC = "acoustic"       # 声学传感器
    GAS = "gas"                 # 气体传感器
    HYPERSPECTRAL = "hyperspectral"  # 高光谱
    LIDAR = "lidar"             # 激光雷达
    VIBRATION = "vibration"     # 振动传感器


class FaultType(str, Enum):
    """故障类型"""
    OVERHEATING = "overheating"         # 过热
    PARTIAL_DISCHARGE = "partial_discharge"  # 局部放电
    GAS_LEAK = "gas_leak"               # 气体泄漏
    MECHANICAL_FAULT = "mechanical_fault"   # 机械故障
    INSULATION_DEFECT = "insulation_defect"  # 绝缘缺陷
    CONTACT_FAULT = "contact_fault"     # 接触不良
    OIL_LEAK = "oil_leak"               # 漏油
    CORROSION = "corrosion"             # 腐蚀
    NORMAL = "normal"                   # 正常


class LogicConflictError(Exception):
    """逻辑冲突异常"""
    def __init__(self, message: str, conflicts: List[Dict]):
        super().__init__(message)
        self.conflicts = conflicts


@dataclass
class SensorReading:
    """传感器读数"""
    sensor_id: str
    sensor_type: SensorType
    value: Any                      # 原始值
    confidence: float               # 置信度 [0, 1]
    health: float                   # 传感器健康度 [0, 1]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def effective_confidence(self) -> float:
        """有效置信度 = 置信度 * 健康度"""
        return self.confidence * self.health


@dataclass
class FusionResult:
    """融合结果"""
    fault_type: FaultType
    confidence: float
    contributing_sensors: List[str]
    evidence_breakdown: Dict[str, float]
    physical_checks_passed: bool
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DempsterShafer:
    """
    Dempster-Shafer证据理论实现
    
    用于多传感器证据融合
    """
    
    def __init__(self, hypotheses: List[str]):
        """
        Args:
            hypotheses: 假设空间（故障类型列表）
        """
        self.hypotheses = hypotheses
        self.power_set = self._generate_power_set()
    
    def _generate_power_set(self) -> List[frozenset]:
        """生成幂集"""
        power_set = [frozenset()]
        for h in self.hypotheses:
            power_set.extend([s | {h} for s in power_set])
        return power_set
    
    def create_mass_function(
        self, 
        belief_assignments: Dict[str, float],
        uncertainty: float = 0.0
    ) -> Dict[frozenset, float]:
        """
        创建质量函数
        
        Args:
            belief_assignments: {hypothesis: belief_value} 假设到信念值的映射
            uncertainty: 不确定性（分配给全集）
            
        Returns:
            质量函数
        """
        mass = {s: 0.0 for s in self.power_set}
        
        for h, m in belief_assignments.items():
            mass[frozenset([h])] = m
        
        # 分配不确定性给全集
        theta = frozenset(self.hypotheses)
        mass[theta] = uncertainty
        
        # 归一化
        total = sum(mass.values())
        if total > 0:
            mass = {k: v / total for k, v in mass.items()}
        
        return mass
    
    def combine(
        self, 
        m1: Dict[frozenset, float], 
        m2: Dict[frozenset, float]
    ) -> Dict[frozenset, float]:
        """
        Dempster组合规则
        
        Args:
            m1, m2: 两个质量函数
            
        Returns:
            组合后的质量函数
        """
        combined = {s: 0.0 for s in self.power_set}
        
        # 计算冲突质量
        conflict = 0.0
        
        for s1, mass1 in m1.items():
            for s2, mass2 in m2.items():
                intersection = s1 & s2
                product = mass1 * mass2
                
                if len(intersection) == 0 and len(s1) > 0 and len(s2) > 0:
                    conflict += product
                else:
                    combined[intersection] += product
        
        # 归一化 (除以1-conflict)
        if conflict >= 1.0 - 1e-10:
            logger.warning("证据完全冲突，使用均匀分布")
            n = len(self.hypotheses)
            return {frozenset([h]): 1.0/n for h in self.hypotheses}
        
        normalizer = 1.0 - conflict
        combined = {k: v / normalizer for k, v in combined.items()}
        
        return combined
    
    def combine_multiple(self, mass_functions: List[Dict[frozenset, float]]) -> Dict[frozenset, float]:
        """组合多个质量函数"""
        if len(mass_functions) == 0:
            return {}
        
        result = mass_functions[0]
        for mf in mass_functions[1:]:
            result = self.combine(result, mf)
        
        return result
    
    def get_belief(self, mass: Dict[frozenset, float], hypothesis: str) -> float:
        """计算信念函数 Bel(A)"""
        h_set = frozenset([hypothesis])
        bel = 0.0
        for s, m in mass.items():
            if s <= h_set and len(s) > 0:
                bel += m
        return bel
    
    def get_plausibility(self, mass: Dict[frozenset, float], hypothesis: str) -> float:
        """计算似然函数 Pl(A)"""
        h_set = frozenset([hypothesis])
        pl = 0.0
        for s, m in mass.items():
            if len(s & h_set) > 0:
                pl += m
        return pl


class PhysicalLogicChecker:
    """
    物理逻辑约束检查器
    
    检查传感器读数之间的物理一致性
    """
    
    def __init__(self):
        self.rules = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认规则"""
        # 规则1: 高温应伴随红外异常
        self.rules.append({
            "name": "temperature_thermal_consistency",
            "description": "高温应该在热成像中可见",
            "check": self._check_temp_thermal_consistency
        })
        
        # 规则2: 局放应伴随超声波异常
        self.rules.append({
            "name": "pd_acoustic_consistency",
            "description": "局部放电应伴随超声波信号",
            "check": self._check_pd_acoustic_consistency
        })
        
        # 规则3: SF6泄漏不应伴随持续高温
        self.rules.append({
            "name": "gas_thermal_consistency",
            "description": "SF6泄漏区域不应持续高温",
            "check": self._check_gas_thermal_consistency
        })
    
    def _check_temp_thermal_consistency(
        self, 
        readings: Dict[SensorType, SensorReading]
    ) -> Tuple[bool, Optional[str]]:
        """检查温度与热成像一致性"""
        thermal = readings.get(SensorType.THERMAL)
        
        if thermal is None:
            return True, None
        
        temp_value = thermal.value if isinstance(thermal.value, (int, float)) else \
                     thermal.value.get('max_temp', 0) if isinstance(thermal.value, dict) else 0
        
        # 如果检测到高温（>80°C），置信度应该较高
        if temp_value > 80 and thermal.confidence < 0.5:
            return False, f"热成像显示高温{temp_value}°C但置信度过低{thermal.confidence}"
        
        return True, None
    
    def _check_pd_acoustic_consistency(
        self, 
        readings: Dict[SensorType, SensorReading]
    ) -> Tuple[bool, Optional[str]]:
        """检查局放与声学信号一致性"""
        acoustic = readings.get(SensorType.ACOUSTIC)
        visual = readings.get(SensorType.VISUAL)
        
        if acoustic is None or visual is None:
            return True, None
        
        # 如果视觉检测到电弧但声学无异常
        visual_val = visual.value if isinstance(visual.value, dict) else {}
        acoustic_val = acoustic.value if isinstance(acoustic.value, dict) else {}
        
        has_arc = visual_val.get('arc_detected', False)
        has_pd_sound = acoustic_val.get('pd_level', 0) > 0.3
        
        if has_arc and not has_pd_sound and acoustic.health > 0.8:
            return False, "检测到电弧但声学传感器无局放信号"
        
        return True, None
    
    def _check_gas_thermal_consistency(
        self, 
        readings: Dict[SensorType, SensorReading]
    ) -> Tuple[bool, Optional[str]]:
        """检查气体与热成像一致性"""
        gas = readings.get(SensorType.GAS)
        thermal = readings.get(SensorType.THERMAL)
        
        if gas is None or thermal is None:
            return True, None
        
        gas_val = gas.value if isinstance(gas.value, dict) else {}
        thermal_val = thermal.value if isinstance(thermal.value, dict) else {}
        
        sf6_leak = gas_val.get('sf6_concentration', 0) > 1000  # ppm
        high_temp = thermal_val.get('max_temp', 0) > 100
        
        # SF6泄漏通常不会伴随高温（SF6是绝缘气体）
        if sf6_leak and high_temp:
            return False, "SF6泄漏与高温同时出现，可能存在多重故障"
        
        return True, None
    
    def check_all(
        self, 
        readings: Dict[SensorType, SensorReading]
    ) -> Tuple[bool, List[str]]:
        """
        执行所有逻辑检查
        
        Returns:
            (是否全部通过, 警告列表)
        """
        all_passed = True
        warnings = []
        
        for rule in self.rules:
            passed, msg = rule["check"](readings)
            if not passed:
                all_passed = False
                warnings.append(f"[{rule['name']}] {msg}")
                logger.warning(f"逻辑检查失败: {rule['name']} - {msg}")
        
        return all_passed, warnings


class FusionEngine:
    """
    增强型融合引擎
    
    功能:
    1. 传感器健康度加权融合
    2. Dempster-Shafer证据理论融合
    3. 物理逻辑约束检查
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 初始化D-S推理器
        fault_types = [ft.value for ft in FaultType]
        self.ds = DempsterShafer(fault_types)
        
        # 物理逻辑检查器
        self.logic_checker = PhysicalLogicChecker()
        
        # 传感器到故障类型的映射关系
        self.sensor_fault_mapping = {
            SensorType.THERMAL: [FaultType.OVERHEATING, FaultType.CONTACT_FAULT],
            SensorType.ACOUSTIC: [FaultType.PARTIAL_DISCHARGE, FaultType.MECHANICAL_FAULT],
            SensorType.GAS: [FaultType.GAS_LEAK, FaultType.INSULATION_DEFECT],
            SensorType.VISUAL: [FaultType.OIL_LEAK, FaultType.CORROSION],
            SensorType.HYPERSPECTRAL: [FaultType.CORROSION, FaultType.INSULATION_DEFECT],
            SensorType.VIBRATION: [FaultType.MECHANICAL_FAULT],
        }
        
        # 健康度阈值
        self.health_threshold = self.config.get('health_threshold', 0.5)
    
    def fuse(
        self, 
        readings: List[SensorReading],
        strict_logic_check: bool = False
    ) -> FusionResult:
        """
        融合多传感器读数
        
        Args:
            readings: 传感器读数列表
            strict_logic_check: 是否严格逻辑检查（失败则抛异常）
            
        Returns:
            融合结果
        """
        if not readings:
            return FusionResult(
                fault_type=FaultType.NORMAL,
                confidence=1.0,
                contributing_sensors=[],
                evidence_breakdown={},
                physical_checks_passed=True
            )
        
        # 1. 按传感器类型组织读数
        readings_by_type: Dict[SensorType, SensorReading] = {}
        for r in readings:
            # 选择健康度最高的读数
            if r.sensor_type not in readings_by_type or \
               r.health > readings_by_type[r.sensor_type].health:
                readings_by_type[r.sensor_type] = r
        
        # 2. 过滤低健康度传感器
        valid_readings = {
            k: v for k, v in readings_by_type.items() 
            if v.health >= self.health_threshold
        }
        
        if not valid_readings:
            logger.warning("所有传感器健康度过低，使用原始读数")
            valid_readings = readings_by_type
        
        # 3. 物理逻辑检查
        logic_passed, logic_warnings = self.logic_checker.check_all(valid_readings)
        
        if not logic_passed and strict_logic_check:
            raise LogicConflictError(
                "传感器读数存在逻辑冲突",
                [{"warning": w} for w in logic_warnings]
            )
        
        # 4. 构建质量函数
        mass_functions = []
        contributing_sensors = []
        evidence_breakdown = {}
        
        for sensor_type, reading in valid_readings.items():
            # 根据传感器类型和读数构建质量函数
            mf = self._build_mass_function(sensor_type, reading)
            mass_functions.append(mf)
            contributing_sensors.append(reading.sensor_id)
            
            # 记录每个传感器的贡献
            for h, m in mf.items():
                if len(h) == 1:
                    fault = list(h)[0]
                    if fault not in evidence_breakdown:
                        evidence_breakdown[fault] = 0.0
                    evidence_breakdown[fault] += m * reading.effective_confidence
        
        # 5. D-S证据融合
        if len(mass_functions) > 1:
            combined_mass = self.ds.combine_multiple(mass_functions)
        elif len(mass_functions) == 1:
            combined_mass = mass_functions[0]
        else:
            combined_mass = {}
        
        # 6. 提取最可能的故障类型
        best_fault = FaultType.NORMAL
        best_confidence = 0.0
        
        for fault_type in FaultType:
            if fault_type == FaultType.NORMAL:
                continue
            
            bel = self.ds.get_belief(combined_mass, fault_type.value)
            pl = self.ds.get_plausibility(combined_mass, fault_type.value)
            
            # 使用信念和似然的加权平均
            confidence = 0.7 * bel + 0.3 * pl
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_fault = fault_type
        
        # 如果最高置信度很低，判定为正常
        if best_confidence < 0.3:
            best_fault = FaultType.NORMAL
            best_confidence = 1.0 - best_confidence
        
        return FusionResult(
            fault_type=best_fault,
            confidence=best_confidence,
            contributing_sensors=contributing_sensors,
            evidence_breakdown=evidence_breakdown,
            physical_checks_passed=logic_passed,
            warnings=logic_warnings,
            metadata={
                "combined_mass": {str(k): v for k, v in combined_mass.items()},
                "sensor_count": len(valid_readings),
                "filtered_sensors": len(readings) - len(valid_readings)
            }
        )
    
    def _build_mass_function(
        self, 
        sensor_type: SensorType, 
        reading: SensorReading
    ) -> Dict[frozenset, float]:
        """根据传感器读数构建质量函数"""
        
        # 获取该传感器可能检测到的故障类型
        possible_faults = self.sensor_fault_mapping.get(sensor_type, [])
        
        # 解析传感器值
        value = reading.value
        severity = self._extract_severity(value)
        
        # 分配质量
        belief_assignments = {}
        
        if severity > 0.7:  # 严重异常
            for fault in possible_faults:
                belief_assignments[fault.value] = severity / len(possible_faults)
        elif severity > 0.3:  # 中等异常
            for fault in possible_faults:
                belief_assignments[fault.value] = 0.5 * severity / len(possible_faults)
            belief_assignments[FaultType.NORMAL.value] = 0.5 * (1 - severity)
        else:  # 轻微或正常
            belief_assignments[FaultType.NORMAL.value] = 1 - severity
        
        # 应用健康度权重
        uncertainty = 1 - reading.health  # 不确定性与健康度成反比
        
        return self.ds.create_mass_function(belief_assignments, uncertainty)
    
    def _extract_severity(self, value: Any) -> float:
        """从传感器值中提取严重程度 [0, 1]"""
        if isinstance(value, (int, float)):
            return min(1.0, max(0.0, value))
        
        if isinstance(value, dict):
            # 尝试从字典中提取严重程度
            for key in ['severity', 'level', 'alarm_level', 'anomaly_score']:
                if key in value:
                    v = value[key]
                    if isinstance(v, (int, float)):
                        return min(1.0, max(0.0, v))
            
            # 根据状态判断
            status = value.get('status', '')
            if status in ['critical', 'alarm', 'danger']:
                return 0.9
            elif status in ['warning', 'attention']:
                return 0.6
            elif status in ['normal', 'ok', 'good']:
                return 0.1
        
        return 0.5  # 默认中等


# ==================== 便捷函数 ====================

def create_fusion_engine(config: Optional[Dict] = None) -> FusionEngine:
    """创建融合引擎实例"""
    return FusionEngine(config)


if __name__ == "__main__":
    # 测试代码
    print("=== 融合引擎测试 ===")
    
    engine = FusionEngine()
    
    # 创建测试读数
    readings = [
        SensorReading(
            sensor_id="thermal_001",
            sensor_type=SensorType.THERMAL,
            value={"max_temp": 95, "status": "warning"},
            confidence=0.85,
            health=0.95
        ),
        SensorReading(
            sensor_id="acoustic_001",
            sensor_type=SensorType.ACOUSTIC,
            value={"pd_level": 0.6, "status": "warning"},
            confidence=0.78,
            health=0.90
        ),
        SensorReading(
            sensor_id="visual_001",
            sensor_type=SensorType.VISUAL,
            value={"oil_leak": False, "status": "normal"},
            confidence=0.92,
            health=0.98
        ),
    ]
    
    result = engine.fuse(readings)
    
    print(f"故障类型: {result.fault_type.value}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"贡献传感器: {result.contributing_sensors}")
    print(f"物理检查: {'通过' if result.physical_checks_passed else '失败'}")
    print(f"警告: {result.warnings}")
    
    print("\n测试完成!")
