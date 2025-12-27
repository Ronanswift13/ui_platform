"""
多证据融合引擎
输变电激光监测平台 - 全自动AI巡检增强

实现功能:
- 多源证据融合(OCR/颜色/角度/深度学习)
- 可配置的融合权重
- 置信度校准
- 冲突检测和解决
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class EvidenceType(Enum):
    """证据类型"""
    OCR_TEXT = "ocr_text"               # OCR文字识别
    COLOR_DETECTION = "color_detection" # 颜色检测
    ANGLE_DETECTION = "angle_detection" # 角度检测
    DEEP_LEARNING = "deep_learning"     # 深度学习
    TEMPLATE_MATCH = "template_match"   # 模板匹配
    RULE_BASED = "rule_based"           # 规则推理
    THERMAL = "thermal"                 # 热成像
    HISTORICAL = "historical"           # 历史数据


class ConflictStrategy(Enum):
    """冲突解决策略"""
    WEIGHTED_VOTE = "weighted_vote"     # 加权投票
    MAX_CONFIDENCE = "max_confidence"   # 最大置信度
    PRIORITY_ORDER = "priority_order"   # 优先级顺序
    CONSENSUS = "consensus"             # 共识(全部一致)
    MAJORITY = "majority"               # 多数决


@dataclass
class Evidence:
    """证据定义"""
    evidence_id: str
    evidence_type: EvidenceType
    source: str                         # 来源模块/算法
    value: Any                          # 证据值
    confidence: float                   # 置信度 [0, 1]
    weight: float = 1.0                 # 权重
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def weighted_confidence(self) -> float:
        """加权置信度"""
        return self.confidence * self.weight


@dataclass
class FusionResult:
    """融合结果"""
    final_value: Any                    # 最终值
    final_confidence: float             # 最终置信度
    evidences: List[Evidence]           # 参与融合的证据
    fusion_method: str                  # 融合方法
    conflict_detected: bool = False     # 是否检测到冲突
    conflict_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FusionRule(ABC):
    """融合规则基类"""
    
    @abstractmethod
    def apply(self, evidences: List[Evidence]) -> FusionResult:
        """应用融合规则"""
        pass


class WeightedVoteFusion(FusionRule):
    """加权投票融合"""
    
    def __init__(self, value_key: str = "state"):
        self.value_key = value_key
    
    def apply(self, evidences: List[Evidence]) -> FusionResult:
        if not evidences:
            return FusionResult(
                final_value=None,
                final_confidence=0.0,
                evidences=[],
                fusion_method="weighted_vote"
            )
        
        # 按值分组
        vote_scores: Dict[Any, float] = {}
        for ev in evidences:
            value = ev.value if not isinstance(ev.value, dict) else ev.value.get(self.value_key)
            score = ev.weighted_confidence()
            vote_scores[value] = vote_scores.get(value, 0) + score
        
        # 找最高得分
        best_value = max(vote_scores, key=lambda x: vote_scores.get(x, 0.0))
        total_score = sum(vote_scores.values())
        final_confidence = vote_scores[best_value] / total_score if total_score > 0 else 0
        
        # 检测冲突
        conflict_detected = len(vote_scores) > 1
        conflict_details = None
        if conflict_detected:
            conflict_details = f"多个候选值: {list(vote_scores.keys())}"
        
        return FusionResult(
            final_value=best_value,
            final_confidence=final_confidence,
            evidences=evidences,
            fusion_method="weighted_vote",
            conflict_detected=conflict_detected,
            conflict_details=conflict_details,
            metadata={"vote_scores": vote_scores}
        )


class MaxConfidenceFusion(FusionRule):
    """最大置信度融合"""
    
    def apply(self, evidences: List[Evidence]) -> FusionResult:
        if not evidences:
            return FusionResult(
                final_value=None,
                final_confidence=0.0,
                evidences=[],
                fusion_method="max_confidence"
            )
        
        best_evidence = max(evidences, key=lambda e: e.weighted_confidence())
        
        return FusionResult(
            final_value=best_evidence.value,
            final_confidence=best_evidence.confidence,
            evidences=evidences,
            fusion_method="max_confidence",
            conflict_detected=False,
            metadata={"selected_source": best_evidence.source}
        )


class BayesianFusion(FusionRule):
    """贝叶斯融合"""
    
    def __init__(self, prior: float = 0.5):
        self.prior = prior
    
    def apply(self, evidences: List[Evidence]) -> FusionResult:
        if not evidences:
            return FusionResult(
                final_value=None,
                final_confidence=0.0,
                evidences=[],
                fusion_method="bayesian"
            )
        
        # 简化的贝叶斯融合
        # P(H|E1,E2,...) ∝ P(H) * ∏P(Ei|H)
        
        # 假设二值分类场景
        log_odds = np.log(self.prior / (1 - self.prior + 1e-10))
        
        for ev in evidences:
            # 将置信度转换为似然比
            conf = np.clip(ev.confidence, 0.01, 0.99)
            likelihood_ratio = conf / (1 - conf)
            log_odds += ev.weight * np.log(likelihood_ratio)
        
        # 转换回概率
        final_confidence = 1 / (1 + np.exp(-log_odds))
        
        # 多数值作为最终值
        value_counts: Dict[Any, int] = {}
        for ev in evidences:
            v = ev.value
            value_counts[v] = value_counts.get(v, 0) + 1
        final_value = max(value_counts, key=lambda x: value_counts.get(x, 0))
        
        return FusionResult(
            final_value=final_value,
            final_confidence=float(final_confidence),
            evidences=evidences,
            fusion_method="bayesian",
            metadata={"log_odds": float(log_odds)}
        )


class DempsterShaferFusion(FusionRule):
    """Dempster-Shafer证据理论融合"""
    
    def apply(self, evidences: List[Evidence]) -> FusionResult:
        if not evidences:
            return FusionResult(
                final_value=None,
                final_confidence=0.0,
                evidences=[],
                fusion_method="dempster_shafer"
            )
        
        # 简化实现: 使用加权平均
        # 完整DS理论需要定义信度函数和组合规则
        
        total_weight = sum(e.weight for e in evidences)
        if total_weight == 0:
            total_weight = 1
        
        # 按值累加信度
        belief_mass: Dict[Any, float] = {}
        for ev in evidences:
            v = ev.value
            mass = ev.confidence * ev.weight / total_weight
            belief_mass[v] = belief_mass.get(v, 0) + mass
        
        # 归一化
        total_mass = sum(belief_mass.values())
        if total_mass > 0:
            belief_mass = {k: v / total_mass for k, v in belief_mass.items()}
        
        # 选择最高信度值
        final_value = max(belief_mass, key=lambda x: belief_mass.get(x, 0.0))
        final_confidence = belief_mass[final_value]
        
        return FusionResult(
            final_value=final_value,
            final_confidence=final_confidence,
            evidences=evidences,
            fusion_method="dempster_shafer",
            metadata={"belief_mass": belief_mass}
        )


class EvidenceFusionEngine:
    """
    证据融合引擎
    
    支持多种融合策略和自定义规则
    """
    
    # 预定义的融合规则
    FUSION_RULES = {
        "weighted_vote": WeightedVoteFusion,
        "max_confidence": MaxConfidenceFusion,
        "bayesian": BayesianFusion,
        "dempster_shafer": DempsterShaferFusion,
    }
    
    # 默认证据类型权重
    DEFAULT_WEIGHTS = {
        EvidenceType.DEEP_LEARNING: 0.5,
        EvidenceType.OCR_TEXT: 0.3,
        EvidenceType.COLOR_DETECTION: 0.2,
        EvidenceType.ANGLE_DETECTION: 0.2,
        EvidenceType.TEMPLATE_MATCH: 0.3,
        EvidenceType.RULE_BASED: 0.4,
        EvidenceType.THERMAL: 0.3,
        EvidenceType.HISTORICAL: 0.1,
    }
    
    def __init__(
        self,
        default_method: str = "weighted_vote",
        weights: Optional[Dict[EvidenceType, float]] = None,
        conflict_strategy: ConflictStrategy = ConflictStrategy.WEIGHTED_VOTE,
    ):
        self.default_method = default_method
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.conflict_strategy = conflict_strategy
        self._custom_rules: Dict[str, FusionRule] = {}
    
    def register_rule(self, name: str, rule: FusionRule) -> None:
        """注册自定义融合规则"""
        self._custom_rules[name] = rule
    
    def fuse(
        self,
        evidences: List[Evidence],
        method: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> FusionResult:
        """
        融合多个证据
        
        Args:
            evidences: 证据列表
            method: 融合方法名称
            min_confidence: 最小置信度过滤
            
        Returns:
            融合结果
        """
        # 过滤低置信度证据
        filtered_evidences = [e for e in evidences if e.confidence >= min_confidence]
        
        if not filtered_evidences:
            return FusionResult(
                final_value=None,
                final_confidence=0.0,
                evidences=evidences,
                fusion_method=method or self.default_method,
                metadata={"reason": "all_evidences_filtered"}
            )
        
        # 应用默认权重
        for ev in filtered_evidences:
            if ev.weight == 1.0:  # 未设置权重
                ev.weight = self.weights.get(ev.evidence_type, 1.0)
        
        # 获取融合规则
        method = method or self.default_method
        if method in self._custom_rules:
            rule = self._custom_rules[method]
        elif method in self.FUSION_RULES:
            rule = self.FUSION_RULES[method]()
        else:
            raise ValueError(f"未知的融合方法: {method}")
        
        # 执行融合
        result = rule.apply(filtered_evidences)
        
        # 处理冲突
        if result.conflict_detected:
            result = self._resolve_conflict(result)
        
        return result
    
    def _resolve_conflict(self, result: FusionResult) -> FusionResult:
        """解决冲突"""
        if self.conflict_strategy == ConflictStrategy.MAX_CONFIDENCE:
            # 选择最高置信度
            if result.evidences:
                best = max(result.evidences, key=lambda e: e.weighted_confidence())
                result.final_value = best.value
                result.final_confidence = best.confidence
                result.metadata["conflict_resolved_by"] = "max_confidence"
        
        elif self.conflict_strategy == ConflictStrategy.CONSENSUS:
            # 要求全部一致
            values = set(e.value for e in result.evidences)
            if len(values) > 1:
                result.final_confidence *= 0.5  # 降低置信度
                result.metadata["conflict_resolved_by"] = "consensus_penalty"
        
        return result
    
    def calibrate_confidence(
        self,
        evidence: Evidence,
        calibration_map: Optional[Dict[float, float]] = None,
    ) -> Evidence:
        """校准置信度"""
        if calibration_map is None:
            # 默认校准: 压缩极端值
            conf = evidence.confidence
            calibrated = 0.5 + 0.4 * (2 * conf - 1)  # 压缩到 [0.1, 0.9]
            evidence.confidence = max(0.1, min(0.9, calibrated))
        else:
            # 使用校准映射
            for threshold, target in sorted(calibration_map.items()):
                if evidence.confidence <= threshold:
                    evidence.confidence = target
                    break
        
        return evidence


# ==================== 开关状态融合示例 ====================

class SwitchStateFusionEngine(EvidenceFusionEngine):
    """
    开关状态专用融合引擎
    
    实现文档中描述的多证据融合公式:
    state = argmax( w_text*P_text + w_color*P_color + w_angle*P_angle )
    """
    
    def __init__(self):
        super().__init__(
            default_method="weighted_vote",
            weights={
                EvidenceType.OCR_TEXT: 0.5,
                EvidenceType.COLOR_DETECTION: 0.3,
                EvidenceType.ANGLE_DETECTION: 0.2,
                EvidenceType.DEEP_LEARNING: 0.6,
            }
        )
    
    def fuse_switch_state(
        self,
        ocr_result: Optional[Dict] = None,
        color_result: Optional[Dict] = None,
        angle_result: Optional[Dict] = None,
        dl_result: Optional[Dict] = None,
    ) -> FusionResult:
        """
        融合开关状态识别结果
        
        Args:
            ocr_result: OCR识别结果 {"state": "open"|"closed", "confidence": float}
            color_result: 颜色检测结果
            angle_result: 角度检测结果
            dl_result: 深度学习结果
            
        Returns:
            融合结果
        """
        evidences = []
        
        if ocr_result:
            evidences.append(Evidence(
                evidence_id="ocr",
                evidence_type=EvidenceType.OCR_TEXT,
                source="ocr_engine",
                value=ocr_result.get("state"),
                confidence=ocr_result.get("confidence", 0.5),
            ))
        
        if color_result:
            evidences.append(Evidence(
                evidence_id="color",
                evidence_type=EvidenceType.COLOR_DETECTION,
                source="color_detector",
                value=color_result.get("state"),
                confidence=color_result.get("confidence", 0.5),
            ))
        
        if angle_result:
            evidences.append(Evidence(
                evidence_id="angle",
                evidence_type=EvidenceType.ANGLE_DETECTION,
                source="angle_detector",
                value=angle_result.get("state"),
                confidence=angle_result.get("confidence", 0.5),
            ))
        
        if dl_result:
            evidences.append(Evidence(
                evidence_id="deep_learning",
                evidence_type=EvidenceType.DEEP_LEARNING,
                source="switch_classifier",
                value=dl_result.get("state"),
                confidence=dl_result.get("confidence", 0.5),
            ))
        
        return self.fuse(evidences)


# ==================== 便捷函数 ====================

_fusion_engine: Optional[EvidenceFusionEngine] = None

def get_fusion_engine() -> EvidenceFusionEngine:
    """获取融合引擎实例"""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = EvidenceFusionEngine()
    return _fusion_engine


def fuse_evidences(evidences: List[Evidence], method: str = "weighted_vote") -> FusionResult:
    """融合证据的便捷函数"""
    return get_fusion_engine().fuse(evidences, method)
