# -*- coding: utf-8 -*-
"""
动作 + 告警 + 一次设备证据 联合根因分析服务
============================================

证据受限的工程判断模型
---------------------
不再是简单的"先验×乘子→归一化", 而是:
1. 证据清点 → 判定证据充分性
2. 链证据加权(乘子 × 链置信度) → 初步后验
3. 外部证据逐项更新 → 调整后验
4. 结论门禁(cap/floor) → 防止无证据时概率被无限拉高
5. 结论组装(反证 + 缺口 + 人工复核项 + 下一步动作)

每个结论必须回答:
- 为什么是这个分类 (evidence_chain + root_cause_reason)
- 为什么不是其他分类 (counter_evidence)
- 还缺什么 (evidence_gaps)
- 人工该查什么 (manual_review_items)
- 下一步做什么 (next_actions)

复用: multimodal_fusion/fusion_engine_enhanced.py 的贝叶斯推理思路
复用: alarm_manager.py 的告警数据
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from platform_core.action_event_schema import (
    ActionEvent, ActionType, RootCauseResult, RootCauseCategory, EvidenceNode,
    TripScopeResult, ChainType, FaultArchive, SourceSystem,
)
from platform_core.action_sequence_analyzer import ChainAnalysisResult
from platform_core.device_correlation import DeviceCorrelationService

logger = logging.getLogger(__name__)


# =============================================================================
# 辅助证据输入
# =============================================================================

@dataclass
class PrimaryDeviceEvidence:
    """一次设备巡视/检测证据"""
    device_id: str = ""
    device_type: str = ""
    source_plugin: str = ""         # transformer_inspection / switch_inspection / busbar_inspection / capacitor_inspection
    has_defect: bool = False
    defect_type: str = ""
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecondaryDeviceEvidence:
    """二次设备健康证据"""
    device_id: str = ""
    device_type: str = ""
    source_plugin: str = ""         # device_monitoring
    health_index: float = 1.0       # 0-1, 1=健康
    anomaly_score: float = 0.0      # 0-1, 1=严重异常
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuxiliaryEvidence:
    """辅助证据(声学/气体/录波分析等)"""
    evidence_type: str = ""         # acoustic / gas / thermal / wave_analysis ...
    source_plugin: str = ""
    is_abnormal: bool = False
    value: float = 0.0
    threshold: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 根因分析服务
# =============================================================================

class RootCauseService:
    """
    证据受限的工程判断根因分析服务

    核心设计:
    1. 证据清点 → evidence_sufficiency
    2. 链加权后验 = 先验 × (base_multiplier × chain.confidence)
    3. 外部证据逐项更新
    4. 结论门禁: 无一次证据→PRIMARY cap; 信号抖动→SECONDARY cap; etc.
    5. 组装: 反证 + 缺口 + 复核项 + 下一步
    """

    # 先验概率(可从配置/历史数据调整)
    PRIORS = {
        RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value: 0.30,
        RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value: 0.25,
        RootCauseCategory.CONTROL_LOOP_ISSUE.value: 0.20,
        RootCauseCategory.SIGNAL_ANOMALY.value: 0.15,
        RootCauseCategory.EXTERNAL_CAUSE.value: 0.10,
    }

    # 链类型 → 目标根因 + 基础乘子(不含链置信度加权)
    # 乘子含义: 该链类型对目标类别的似然放大倍数
    _CHAIN_LIKELIHOOD = {
        ChainType.NORMAL_TRIP.value: [
            (RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value, 1.8),
        ],
        ChainType.RECLOSER_FAIL.value: [
            (RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value, 2.2),
        ],
        ChainType.REFUSE_ACTION.value: [
            (RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value, 1.8),
            (RootCauseCategory.CONTROL_LOOP_ISSUE.value, 1.4),
        ],
        ChainType.FALSE_ACTION.value: [
            (RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value, 1.6),
            (RootCauseCategory.SIGNAL_ANOMALY.value, 1.3),
        ],
        ChainType.CONTROL_LOOP_ABNORMAL.value: [
            (RootCauseCategory.CONTROL_LOOP_ISSUE.value, 2.8),
        ],
        ChainType.MECHANISM_ABNORMAL.value: [
            (RootCauseCategory.CONTROL_LOOP_ISSUE.value, 2.2),
        ],
        ChainType.SIGNAL_JITTER.value: [
            (RootCauseCategory.SIGNAL_ANOMALY.value, 2.8),
        ],
    }

    # 断路器状态变化类型(用于证据清点)
    _BREAKER_STATE_TYPES = frozenset({
        ActionType.BREAKER_OPEN.value, ActionType.BREAKER_CLOSE.value, ActionType.TRIP_OPEN.value,
    })

    # 结论门禁: 各类别在缺乏特定证据时的后验上限(归一化前)
    # 超过此值的概率会被 clamp, 防止无证据时概率被无限拉高
    _PRIMARY_CAP_NO_EVIDENCE = 0.55     # 无一次侧证据 → PRIMARY 归一化后 ≤ 55%
    _SECONDARY_CAP_JITTER_ONLY = 0.25   # 仅信号抖动链 → SECONDARY 不超过先验

    def __init__(
        self,
        correlation_service: Optional[DeviceCorrelationService] = None,
        alarm_manager=None,
    ):
        self.correlation = correlation_service
        self.alarm_manager = alarm_manager
        logger.info("RootCauseService 初始化完成(证据受限工程判断模型)")

    def analyze(
        self,
        chain_results: List[ChainAnalysisResult],
        action_events: List[ActionEvent],
        primary_evidence: Optional[List[PrimaryDeviceEvidence]] = None,
        secondary_evidence: Optional[List[SecondaryDeviceEvidence]] = None,
        auxiliary_evidence: Optional[List[AuxiliaryEvidence]] = None,
        historical_faults: Optional[List[Dict]] = None,
    ) -> RootCauseResult:
        """
        执行根因分析(证据受限工程判断模型)

        返回 RootCauseResult，包含概率分布、证据链、反证、缺口、建议。
        """
        primary_evidence = primary_evidence or []
        secondary_evidence = secondary_evidence or []
        auxiliary_evidence = auxiliary_evidence or []
        historical_faults = historical_faults or []

        # =====================================================================
        # Phase 1: 证据清点
        # =====================================================================
        inventory = self._inventory_evidence(
            chain_results, action_events, primary_evidence,
            secondary_evidence, auxiliary_evidence)

        evidence_chain: List[EvidenceNode] = []
        evidence_gaps: List[str] = list(inventory["gaps"])
        counter_evidence: List[str] = []
        manual_review_items: List[str] = []
        next_actions: List[str] = []

        # =====================================================================
        # Phase 2: 链证据加权更新
        # =====================================================================
        posteriors = dict(self.PRIORS)
        best_chain = chain_results[0] if chain_results else None

        if best_chain:
            posteriors = self._update_from_chain_weighted(posteriors, best_chain)
            evidence_chain.extend(best_chain.evidence_nodes)

        # =====================================================================
        # Phase 3: 外部证据更新
        # =====================================================================

        # 3a: 一次设备证据
        for pe in primary_evidence:
            posteriors = self._update_from_primary(posteriors, pe, inventory)
            evidence_chain.append(EvidenceNode(
                node_type="primary_evidence",
                source=pe.source_plugin,
                timestamp="",
                description=f"一次设备{pe.device_id}: {'有缺陷(' + pe.defect_type + ')' if pe.has_defect else '巡视正常'}"
                            f" 置信度={pe.confidence:.2f}",
                confidence=pe.confidence,
                data={"device_id": pe.device_id, "has_defect": pe.has_defect,
                      "defect_type": pe.defect_type},
            ))

        # 3b: 二次设备证据
        for se in secondary_evidence:
            posteriors = self._update_from_secondary(posteriors, se, inventory)
            evidence_chain.append(EvidenceNode(
                node_type="secondary_health",
                source=se.source_plugin,
                timestamp="",
                description=f"二次设备{se.device_id}: 健康指数={se.health_index:.2f},"
                            f" 异常分数={se.anomaly_score:.2f}",
                confidence=1.0 - se.anomaly_score,
                data={"device_id": se.device_id, "health_index": se.health_index,
                      "anomaly_score": se.anomaly_score},
            ))

        # 3c: 辅助证据(录波分析/声学/气体)
        for ae in auxiliary_evidence:
            posteriors = self._update_from_auxiliary(posteriors, ae)
            evidence_chain.append(EvidenceNode(
                node_type=ae.evidence_type,
                source=ae.source_plugin,
                timestamp="",
                description=f"辅助证据({ae.evidence_type}): "
                            f"{'异常' if ae.is_abnormal else '正常'} 值={ae.value}",
                confidence=0.8 if ae.is_abnormal else 0.5,
                data={"type": ae.evidence_type, "value": ae.value,
                      "threshold": ae.threshold},
            ))

        # 3d: 历史故障 (扩展点 — 清晰接口, 当前不阻塞)
        if historical_faults:
            posteriors = self._update_from_history(posteriors, historical_faults)
        else:
            if inventory["sufficiency"] != "insufficient":
                evidence_gaps.append("缺少历史故障记录(暂不影响结论, 接入后可提升准确性)")

        # =====================================================================
        # Phase 4: 结论门禁
        # =====================================================================
        posteriors, gate_notes = self._apply_conclusion_gates(
            posteriors, inventory, best_chain)
        counter_evidence.extend(gate_notes)

        # 归一化
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}

        # =====================================================================
        # Phase 5: 结论组装
        # =====================================================================
        root_cause_cat = max(posteriors, key=posteriors.get)

        # 如果证据严重不足, 强制 unknown
        if inventory["sufficiency"] == "insufficient":
            root_cause_cat = RootCauseCategory.UNKNOWN.value

        top_confidence = posteriors.get(root_cause_cat, 0.0)

        # 生成结构化输出
        root_cause_reason = self._generate_reason(
            root_cause_cat, best_chain, primary_evidence, secondary_evidence,
            inventory, posteriors)
        counter_evidence.extend(self._generate_counter_evidence(
            root_cause_cat, posteriors, inventory))
        manual_review_items = self._generate_manual_review_items(
            root_cause_cat, inventory, evidence_gaps)
        next_actions = self._generate_next_actions(
            root_cause_cat, best_chain, inventory)
        recommendations = self._generate_recommendations(
            root_cause_cat, best_chain, posteriors)

        result = RootCauseResult(
            trace_id=action_events[0].trace_id if action_events else "",
            root_cause_category=root_cause_cat,
            root_cause_reason=root_cause_reason,
            confidence=top_confidence,
            probabilities=dict(posteriors),
            primary_equipment_fault_probability=posteriors.get(
                RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value, 0.0),
            secondary_equipment_fault_probability=posteriors.get(
                RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value, 0.0),
            control_loop_fault_probability=posteriors.get(
                RootCauseCategory.CONTROL_LOOP_ISSUE.value, 0.0),
            signal_anomaly_probability=posteriors.get(
                RootCauseCategory.SIGNAL_ANOMALY.value, 0.0),
            external_cause_probability=posteriors.get(
                RootCauseCategory.EXTERNAL_CAUSE.value, 0.0),
            evidence_chain=evidence_chain,
            counter_evidence=counter_evidence,
            evidence_gaps=evidence_gaps,
            evidence_sufficiency=inventory["sufficiency"],
            manual_review_items=manual_review_items,
            next_actions=next_actions,
            recommendations=recommendations,
            chain_type=best_chain.chain_type if best_chain else ChainType.UNKNOWN.value,
            trip_scope=best_chain.trip_scope if best_chain else None,
        )

        # 推送告警
        if self.alarm_manager and top_confidence > 0.5:
            self._emit_alarm(result)

        return result

    # =========================================================================
    # Phase 1: 证据清点
    # =========================================================================

    def _inventory_evidence(
        self,
        chain_results: List[ChainAnalysisResult],
        events: List[ActionEvent],
        primary_ev: List[PrimaryDeviceEvidence],
        secondary_ev: List[SecondaryDeviceEvidence],
        auxiliary_ev: List[AuxiliaryEvidence],
    ) -> Dict[str, Any]:
        """扫描所有输入, 生成证据清单和充分性判定"""
        flags: Dict[str, bool] = {
            "has_chain": bool(chain_results),
            "has_primary_evidence": bool(primary_ev),
            "has_primary_defect": any(pe.has_defect for pe in primary_ev),
            "has_secondary_evidence": bool(secondary_ev),
            "has_secondary_anomaly": any(se.anomaly_score > 0.5 for se in secondary_ev),
            "has_auxiliary": bool(auxiliary_ev),
            "has_breaker_state_change": any(
                e.action_type in self._BREAKER_STATE_TYPES for e in events),
            "has_protection_trip": any(
                e.action_type in (ActionType.PROTECTION_TRIP.value,
                                  ActionType.BUS_DIFF_TRIP.value,
                                  ActionType.BREAKER_FAIL_TRIP.value)
                for e in events),
            "has_wave_record": any(
                getattr(e, 'wave_record_id', None) for e in events),
            "has_fault_current": any(
                getattr(e, 'fault_current_ka', None) is not None for e in events),
            "has_protection_device_source": any(
                e.source_system in (SourceSystem.PROTECTION_DEVICE.value,
                                    SourceSystem.PROTECTION_INFO.value)
                for e in events),
        }

        # 判定充分性
        if not flags["has_chain"] and not flags["has_protection_trip"]:
            sufficiency = "insufficient"
        elif flags["has_chain"] and flags["has_breaker_state_change"] and (
                flags["has_primary_evidence"] or flags["has_wave_record"]):
            sufficiency = "sufficient"
        elif flags["has_chain"]:
            sufficiency = "partial"
        else:
            sufficiency = "insufficient"

        # 证据缺口
        gaps: List[str] = []
        if not flags["has_primary_evidence"]:
            gaps.append("缺少一次设备巡视/检测证据")
        if not flags["has_secondary_evidence"]:
            gaps.append("缺少二次设备健康监测证据")
        if not flags["has_wave_record"]:
            gaps.append("缺少故障录波数据")
        if not flags["has_fault_current"]:
            gaps.append("缺少故障电流数据")
        if not flags["has_breaker_state_change"]:
            gaps.append("无断路器状态变化信号(无法确认实际动作)")
        if not flags["has_protection_device_source"]:
            gaps.append("事件来源非保护装置/保信子站(可信度受限)")

        return {"flags": flags, "sufficiency": sufficiency, "gaps": gaps}

    # =========================================================================
    # Phase 2: 链证据加权更新
    # =========================================================================

    def _update_from_chain_weighted(
        self, posteriors: Dict[str, float], chain: ChainAnalysisResult
    ) -> Dict[str, float]:
        """
        根据动作链类型+链置信度加权更新后验

        乘子 = base_multiplier × chain.confidence
        (链置信度越高, 似然放大越大; 低置信度链影响较弱)
        """
        p = dict(posteriors)
        ct = chain.chain_type
        chain_conf = max(chain.confidence, 0.3)  # 下限 0.3 避免完全失效

        likelihood_entries = self._CHAIN_LIKELIHOOD.get(ct, [])
        for category, base_mult in likelihood_entries:
            # 实际乘子 = 1.0 + (base_mult - 1.0) × chain_confidence
            # 这样 chain_confidence=1.0 时得到完整 base_mult,
            # chain_confidence=0.3 时仅得到 1.0 + (base-1.0)*0.3
            actual_mult = 1.0 + (base_mult - 1.0) * chain_conf
            p[category] *= actual_mult

        return p

    # =========================================================================
    # Phase 3: 外部证据更新
    # =========================================================================

    def _update_from_primary(
        self, posteriors: Dict[str, float],
        evidence: PrimaryDeviceEvidence,
        inventory: Dict[str, Any],
    ) -> Dict[str, float]:
        """根据一次设备证据更新(有缺陷→升, 无缺陷→降)"""
        p = dict(posteriors)
        if evidence.has_defect and evidence.confidence > 0.5:
            p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] *= (1.0 + evidence.confidence)
        else:
            # 巡视正常 → 降低一次故障概率, 但不低于 0.5×
            p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] *= max(0.5, 1.0 - evidence.confidence * 0.3)
        return p

    def _update_from_secondary(
        self, posteriors: Dict[str, float],
        evidence: SecondaryDeviceEvidence,
        inventory: Dict[str, Any],
    ) -> Dict[str, float]:
        """根据二次设备健康证据更新"""
        p = dict(posteriors)
        if evidence.anomaly_score > 0.5:
            p[RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value] *= (1.0 + evidence.anomaly_score)
        elif evidence.health_index < 0.5:
            p[RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value] *= 1.5
        # 二次设备健康且无异常 → 轻微降低二次故障概率
        elif evidence.health_index > 0.8 and evidence.anomaly_score < 0.2:
            p[RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value] *= 0.8
        return p

    def _update_from_auxiliary(
        self, posteriors: Dict[str, float], evidence: AuxiliaryEvidence
    ) -> Dict[str, float]:
        """根据辅助证据更新(区分证据类型)"""
        p = dict(posteriors)
        if evidence.is_abnormal:
            if evidence.evidence_type in ("acoustic", "gas", "thermal"):
                # 声学/气体/热成像异常 → 强烈指向一次设备
                p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] *= 1.5
            elif evidence.evidence_type == "wave_analysis":
                # 录波分析异常 → 一次设备 + 外部原因
                p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] *= 1.3
                p[RootCauseCategory.EXTERNAL_CAUSE.value] *= 1.2
            else:
                p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] *= 1.2
                p[RootCauseCategory.EXTERNAL_CAUSE.value] *= 1.1
        return p

    def _update_from_history(
        self, posteriors: Dict[str, float], historical_faults: List[Dict]
    ) -> Dict[str, float]:
        """
        根据历史故障记录更新后验 (扩展点)

        接口约定:
        historical_faults = [
            {
                "fault_type": "primary_equipment_fault" | "secondary_equipment_fault" | ...,
                "bay_id": "BAY-001",
                "time": "2026-01-15T...",
                "similarity": 0.8,  # 与当前故障的相似度(0-1)
            }, ...
        ]

        当前实现: 按类别计数加权. 同间隔同类故障出现多次 → 轻微提升该类概率.
        正式部署时可接入故障知识库做更精确匹配.
        """
        p = dict(posteriors)
        category_boost: Dict[str, float] = {}
        for hf in historical_faults:
            ft = hf.get("fault_type", "")
            sim = hf.get("similarity", 0.5)
            if ft in p:
                category_boost[ft] = category_boost.get(ft, 0.0) + sim * 0.1

        for cat, boost in category_boost.items():
            p[cat] *= (1.0 + min(boost, 0.3))  # 历史证据上限 +30%

        return p

    # =========================================================================
    # Phase 4: 结论门禁
    # =========================================================================

    def _apply_conclusion_gates(
        self,
        posteriors: Dict[str, float],
        inventory: Dict[str, Any],
        best_chain: Optional[ChainAnalysisResult],
    ) -> tuple:
        """
        应用结论门禁条件, 防止证据不足时概率失控

        返回: (adjusted_posteriors, gate_notes)
        """
        p = dict(posteriors)
        flags = inventory["flags"]
        notes: List[str] = []

        # Gate A: 无一次侧证据 → PRIMARY 后验 cap
        # "没有一次侧证据时，一次设备故障概率不能被无限抬高"
        if not flags["has_primary_evidence"]:
            total = sum(p.values())
            if total > 0:
                primary_ratio = p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] / total
                if primary_ratio > self._PRIMARY_CAP_NO_EVIDENCE:
                    # 按比例压缩: 将 PRIMARY 设为 cap×total, 差值分摊给其他类
                    target = self._PRIMARY_CAP_NO_EVIDENCE * total
                    excess = p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] - target
                    p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] = target
                    # 差值按现有比例分给其他类别
                    others = {k: v for k, v in p.items()
                              if k != RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value}
                    other_total = sum(others.values())
                    if other_total > 0:
                        for k in others:
                            p[k] += excess * (others[k] / other_total)
                    notes.append(
                        f"[门禁A] 无一次设备巡视证据, 一次故障概率已从{primary_ratio:.0%}"
                        f"限制至{self._PRIMARY_CAP_NO_EVIDENCE:.0%}")

        # Gate B: 链=SIGNAL_JITTER → SECONDARY 不超过先验(除非有二次异常证据)
        # "仅凭信号反复变化，不应直接落到 secondary_equipment_fault"
        if best_chain and best_chain.chain_type == ChainType.SIGNAL_JITTER.value:
            if not flags["has_secondary_anomaly"]:
                prior_sec = self.PRIORS[RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value]
                if p[RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value] > prior_sec * sum(p.values()) / 1.0:
                    p[RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value] = prior_sec
                    notes.append("[门禁B] 仅有信号抖动且无二次设备异常证据, SECONDARY概率回落至先验")

        # Gate C: 链=CONTROL_LOOP/MECHANISM → PRIMARY 不超过先验
        # "仅凭控制回路断线和弹簧未储能，应优先落到 control_loop_issue"
        if best_chain and best_chain.chain_type in (
                ChainType.CONTROL_LOOP_ABNORMAL.value, ChainType.MECHANISM_ABNORMAL.value):
            prior_pri = self.PRIORS[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value]
            if p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] > prior_pri:
                p[RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value] = prior_pri
                notes.append("[门禁C] 控制回路/机构类异常链, PRIMARY概率回落至先验")

        # Gate D: 无断路器状态变化 → 降级 sufficiency (已在 Phase 1 处理)
        if not flags["has_breaker_state_change"]:
            notes.append("[门禁D] 无断路器状态变化信号, 结论充分性受限")

        return p, notes

    # =========================================================================
    # Phase 5: 结论组装 — 原因描述
    # =========================================================================

    def _generate_reason(
        self,
        category: str,
        chain: Optional[ChainAnalysisResult],
        primary_ev: List[PrimaryDeviceEvidence],
        secondary_ev: List[SecondaryDeviceEvidence],
        inventory: Dict[str, Any],
        posteriors: Dict[str, float],
    ) -> str:
        """生成引用实际证据的根因描述(非模板)"""
        parts = []
        chain_desc = chain.description if chain else "无匹配动作链"
        sufficiency = inventory["sufficiency"]

        # 分类标签
        cat_label = {
            RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value: "一次设备故障",
            RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value: "二次设备故障",
            RootCauseCategory.CONTROL_LOOP_ISSUE.value: "控制回路/机构问题",
            RootCauseCategory.SIGNAL_ANOMALY.value: "信号异常/监测点异常",
            RootCauseCategory.EXTERNAL_CAUSE.value: "外部原因",
            RootCauseCategory.UNKNOWN.value: "根因未明确",
        }.get(category, "未知")

        parts.append(f"判定更偏{cat_label}")

        # 证据引用
        if chain:
            parts.append(f"动作链依据: {chain_desc}")
        if primary_ev:
            defects = [pe for pe in primary_ev if pe.has_defect]
            normals = [pe for pe in primary_ev if not pe.has_defect]
            if defects:
                parts.append(f"一次设备巡视发现{len(defects)}处缺陷"
                             f"({', '.join(d.defect_type or d.device_id for d in defects)})")
            if normals:
                parts.append(f"一次设备巡视{len(normals)}台正常")
        if secondary_ev:
            anomalies = [se for se in secondary_ev if se.anomaly_score > 0.5]
            if anomalies:
                parts.append(f"二次设备异常{len(anomalies)}台")

        # 充分性标注
        if sufficiency == "insufficient":
            parts.append("⚠ 证据严重不足, 此结论仅供参考")
        elif sufficiency == "partial":
            parts.append("⚠ 证据部分缺失, 建议补充后复核")

        # 概率
        prob = posteriors.get(category, 0)
        parts.append(f"后验概率={prob:.1%}")

        return "。".join(parts) + "。"

    # =========================================================================
    # Phase 5: 反证生成
    # =========================================================================

    def _generate_counter_evidence(
        self,
        chosen_category: str,
        posteriors: Dict[str, float],
        inventory: Dict[str, Any],
    ) -> List[str]:
        """生成反证: 为什么不是其他类别"""
        flags = inventory["flags"]
        counter: List[str] = []

        cat_label = {
            RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value: "一次设备故障",
            RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value: "二次设备故障",
            RootCauseCategory.CONTROL_LOOP_ISSUE.value: "控制回路/机构问题",
            RootCauseCategory.SIGNAL_ANOMALY.value: "信号异常",
            RootCauseCategory.EXTERNAL_CAUSE.value: "外部原因",
        }

        # 对非选中类别, 说明为什么排除
        for cat, label in cat_label.items():
            if cat == chosen_category:
                continue
            prob = posteriors.get(cat, 0)
            reasons = []

            if cat == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value:
                if not flags["has_primary_defect"]:
                    reasons.append("无一次侧缺陷证据")
                if not flags["has_wave_record"]:
                    reasons.append("无录波佐证")
            elif cat == RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value:
                if not flags["has_secondary_anomaly"]:
                    reasons.append("无二次设备异常监测")
            elif cat == RootCauseCategory.CONTROL_LOOP_ISSUE.value:
                if flags["has_breaker_state_change"]:
                    reasons.append("断路器已正常响应(回路完好)")
            elif cat == RootCauseCategory.SIGNAL_ANOMALY.value:
                if flags["has_protection_trip"] and flags["has_breaker_state_change"]:
                    reasons.append("有保护出口+断路器响应(非孤立信号)")
            elif cat == RootCauseCategory.EXTERNAL_CAUSE.value:
                reasons.append("无外部环境异常信息")

            if reasons:
                counter.append(f"排除{label}(概率{prob:.1%}): {'; '.join(reasons)}")

        return counter

    # =========================================================================
    # Phase 5: 人工复核项
    # =========================================================================

    def _generate_manual_review_items(
        self,
        category: str,
        inventory: Dict[str, Any],
        evidence_gaps: List[str],
    ) -> List[str]:
        """基于证据缺口生成具体的人工复核要求"""
        items: List[str] = []
        flags = inventory["flags"]

        # 通用: 每个证据缺口对应一个复核项
        if not flags["has_wave_record"]:
            items.append("请调取故障录波文件, 确认故障类型和相别")
        if not flags["has_primary_evidence"]:
            items.append("请安排一次设备现场巡视, 确认设备状态")
        if not flags["has_secondary_evidence"]:
            items.append("请检查二次设备运行状态和健康指数")
        if not flags["has_breaker_state_change"]:
            items.append("请核实断路器实际位置(现场确认分/合位)")

        # 分类相关
        if category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value:
            if not flags["has_primary_defect"]:
                items.append("一次故障判定缺乏巡视缺陷佐证, 请重点检查故障点设备")
        elif category == RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value:
            items.append("请检查保护装置定值和压板状态")
            items.append("请核查二次回路接线完整性")
        elif category == RootCauseCategory.CONTROL_LOOP_ISSUE.value:
            items.append("请检查控制回路直流电压和回路导通性")
            items.append("请检查端子箱内接线和航空插头")
        elif category == RootCauseCategory.SIGNAL_ANOMALY.value:
            items.append("请检查信号回路屏蔽和接地")
            items.append("请排查周围电磁干扰源")
        elif category == RootCauseCategory.UNKNOWN.value:
            items.append("证据不足, 请补充录波/巡视/二次设备监测数据后重新分析")

        return items

    # =========================================================================
    # Phase 5: 下一步动作
    # =========================================================================

    def _generate_next_actions(
        self,
        category: str,
        chain: Optional[ChainAnalysisResult],
        inventory: Dict[str, Any],
    ) -> List[str]:
        """推荐下一步动作(与证据缺口强相关)"""
        actions: List[str] = []
        flags = inventory["flags"]

        # 最紧迫的证据补充
        if not flags["has_wave_record"] and flags["has_protection_trip"]:
            actions.append("[紧急] 调取故障录波文件")
        if not flags["has_primary_evidence"] and category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value:
            actions.append("[紧急] 安排故障设备现场巡视")

        # 分类相关
        if category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value:
            actions.append("检查故障录波确认故障类型和距离")
            if chain and chain.chain_type == ChainType.RECLOSER_FAIL.value:
                actions.append("重合闸失败→检查线路永久性故障(可能需要登杆巡线)")
        elif category == RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value:
            actions.append("调取保护装置动作报告")
            actions.append("核查保护定值清单")
        elif category == RootCauseCategory.CONTROL_LOOP_ISSUE.value:
            actions.append("测量控制回路绝缘和导通")
            actions.append("检查操作机构储能状态")
        elif category == RootCauseCategory.SIGNAL_ANOMALY.value:
            actions.append("用万用表逐点核查信号回路")
            actions.append("观察信号是否在特定时段重复出现")
        elif category == RootCauseCategory.UNKNOWN.value:
            actions.append("[高优] 补充缺失证据后重新触发分析")

        return actions

    # =========================================================================
    # Phase 5: 通用建议(向后兼容 recommendations)
    # =========================================================================

    def _generate_recommendations(
        self,
        category: str,
        chain: Optional[ChainAnalysisResult],
        posteriors: Dict[str, float],
    ) -> List[str]:
        """生成通用建议(保留向后兼容)"""
        recs = []

        if category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value:
            recs.append("建议安排一次设备现场检查")
            recs.append("检查故障录波数据确认故障类型")
            if chain and chain.chain_type == ChainType.RECLOSER_FAIL.value:
                recs.append("重合闸失败，建议检查线路永久性故障")
        elif category == RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value:
            recs.append("建议检查保护装置运行状态")
            recs.append("核查保护定值是否正确")
            if chain and chain.chain_type == ChainType.REFUSE_ACTION.value:
                recs.append("断路器拒动，建议检查二次回路接线")
        elif category == RootCauseCategory.CONTROL_LOOP_ISSUE.value:
            recs.append("建议检查控制回路完整性")
            recs.append("检查断路器操作机构(弹簧储能/液压等)")
            recs.append("核查端子箱接线")
        elif category == RootCauseCategory.SIGNAL_ANOMALY.value:
            recs.append("建议检查相关信号采集装置")
            recs.append("核查信号回路接线与屏蔽")
            recs.append("排查电磁干扰源")
        elif category == RootCauseCategory.EXTERNAL_CAUSE.value:
            recs.append("建议调取气象和环境监测数据")
            recs.append("检查线路通道(雷击/鸟害/树障等)")
        else:
            recs.append("建议综合排查一次设备和二次设备")
            recs.append("补充录波/巡视证据后重新分析")

        return recs

    # =========================================================================
    # 告警发送
    # =========================================================================

    def _emit_alarm(self, result: RootCauseResult) -> None:
        """将根因分析结果推送到 alarm_manager"""
        if not self.alarm_manager:
            return

        severity_map = {
            RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value: "alarm",
            RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value: "alarm",
            RootCauseCategory.CONTROL_LOOP_ISSUE.value: "warning",
            RootCauseCategory.SIGNAL_ANOMALY.value: "attention",
            RootCauseCategory.EXTERNAL_CAUSE.value: "warning",
        }

        level = severity_map.get(result.root_cause_category, "warning")

        self.alarm_manager.add_alarm(
            level=level,
            source_plugin="root_cause_service",
            device_id=result.trace_id,
            alarm_type=f"root_cause_{result.root_cause_category}",
            message=result.root_cause_reason[:200],
            confidence=result.primary_equipment_fault_probability,
            details={
                "root_cause_category": result.root_cause_category,
                "chain_type": result.chain_type,
                "evidence_sufficiency": result.evidence_sufficiency,
                "probabilities": {
                    "primary": result.primary_equipment_fault_probability,
                    "secondary": result.secondary_equipment_fault_probability,
                    "control_loop": result.control_loop_fault_probability,
                    "signal": result.signal_anomaly_probability,
                    "external": result.external_cause_probability,
                },
            },
            recommendations=result.recommendations,
        )

    # =========================================================================
    # 故障归档生成
    # =========================================================================

    def generate_fault_archive(
        self,
        root_cause: RootCauseResult,
        action_events: List[ActionEvent],
    ) -> FaultArchive:
        """根据根因分析结果生成故障归档记录"""
        trip_scope = root_cause.trip_scope

        # 从事件提取基本信息
        station_id = ""
        station_name = ""
        voltage_level = ""
        for e in action_events:
            if e.station_id:
                station_id = e.station_id
                station_name = e.station_name
            vl = getattr(e, 'voltage_level', '')
            if vl and not voltage_level:
                voltage_level = vl
            if station_id and voltage_level:
                break

        # 从 trip_scope 补充电压等级
        if not voltage_level and trip_scope:
            voltage_level = trip_scope.voltage_level or ""

        # 汇总动作描述
        action_descs = []
        for e in action_events:
            if e.action_desc:
                action_descs.append(e.action_desc)
        action_description = " → ".join(action_descs) if action_descs else ""

        # 确定告警组名(取最高优先级的信号组)
        SIGNAL_GROUP_PRIORITY = {
            "protection": 0, "breaker": 1, "recloser": 2,
            "control_loop": 3, "mechanism": 4, "backup_auto": 5,
            "dc_system": 6, "alarm": 7, "abnormal": 8,
            "soe": 9, "telemetry": 10, "other": 11,
        }
        alarm_group = ""
        best_priority = 999
        for e in action_events:
            sg = e.signal_group
            p = SIGNAL_GROUP_PRIORITY.get(sg, 999)
            if p < best_priority:
                best_priority = p
                alarm_group = sg

        archive = FaultArchive(
            trace_id=root_cause.trace_id,
            station_id=station_id,
            station_name=station_name,
            fault_time=action_events[0].source_ts if action_events else "",
            fault_line=trip_scope.fault_line if trip_scope else "",
            voltage_level=voltage_level,
            fault_type=trip_scope.fault_type if trip_scope else "",
            fault_phase=trip_scope.fault_phase if trip_scope else "",
            primary_repair_scope=trip_scope.primary_scope if trip_scope else [],
            secondary_repair_scope=trip_scope.secondary_scope if trip_scope else [],
            outage_scope=trip_scope.outage_scope if trip_scope else [],
            action_description=action_description,
            alarm_group=alarm_group,
            recloser_enabled=trip_scope.recloser_enabled if trip_scope else None,
            recloser_success=trip_scope.recloser_success if trip_scope else None,
            fault_current=trip_scope.fault_current if trip_scope else None,
            wave_record_status=trip_scope.wave_record_status if trip_scope else "",
            root_cause=root_cause,
            trip_scope=trip_scope,
        )

        return archive
