# -*- coding: utf-8 -*-
"""
动作链/时序异常分析器
=====================

首批覆盖7类动作链规则:
1. 正常跳闸链
2. 重合闸失败链
3. 拒动链
4. 误动链
5. 控制回路异常链
6. 机构异常链
7. 信号抖动/重复动作链

规则写在 service/analyzer 中，支持配置化阈值与动作词映射。

复用: multimodal_fusion/fusion_engine_enhanced.py 的故障链(fault_chain)思路
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from platform_core.action_event_schema import (
    ActionEvent, ActionType, ChainType, TripScopeResult, EvidenceNode, SourceSystem
)
from platform_core.device_correlation import DeviceCorrelationService

logger = logging.getLogger(__name__)


# =============================================================================
# 配置化阈值
# =============================================================================

@dataclass
class AnalyzerConfig:
    """分析器配置(可从YAML加载)"""
    # 时间窗口(秒)
    trip_chain_window_s: float = 10.0           # 正常跳闸链: 保护出口→断路器分闸 最大允许间隔
    recloser_window_s: float = 30.0             # 重合闸: 从首次跳闸到重合闸失败 最大窗口
    refuse_action_timeout_s: float = 5.0        # 拒动: 保护出口后等待断路器响应的超时
    false_action_window_s: float = 10.0         # 误动: 断路器动作前后查找保护前兆的窗口
    jitter_window_s: float = 3.0                # 信号抖动: 同一信号反复变位的窗口
    jitter_min_count: int = 3                   # 信号抖动: 最小变位次数
    control_loop_window_s: float = 60.0         # 控制回路: 断线信号关联事件的窗口
    mechanism_window_s: float = 60.0            # 机构异常: 弹簧未储能关联事件的窗口


# =============================================================================
# 分析结果
# =============================================================================

@dataclass
class ChainAnalysisResult:
    """单条动作链分析结果"""
    chain_type: str = ChainType.UNKNOWN.value
    confidence: float = 0.0
    description: str = ""
    matched_events: List[ActionEvent] = field(default_factory=list)
    evidence_nodes: List[EvidenceNode] = field(default_factory=list)
    trip_scope: Optional[TripScopeResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    """metadata 记录规则判定的中间过程, 供调试和可解释性使用"""


# =============================================================================
# 动作链分析器
# =============================================================================

class ActionSequenceAnalyzer:
    """
    动作链/时序异常分析器

    输入: 一组按时间排序的 ActionEvent (同一 trace_id 或同一时间窗口)
    输出: List[ChainAnalysisResult] (按置信度降序)

    规则设计原则:
    - 每条规则有明确的 前置条件 / 排除条件 / 时间窗口
    - 不仅看 action_type, 还校验 signal_group / bay_id / device_id / source_system
    - 置信度由实际证据强度逐项计算, 不使用固定常量
    - 规则之间存在互斥关系(正常跳闸 与 拒动/误动 互斥)
    """

    # 保护类 action_type 集合
    _PROTECTION_TYPES = frozenset({
        ActionType.PROTECTION_START.value,
        ActionType.PROTECTION_TRIP.value,
        ActionType.BUS_DIFF_TRIP.value,
        ActionType.BREAKER_FAIL_TRIP.value,
    })
    # 断路器状态变化 action_type 集合
    _BREAKER_STATE_TYPES = frozenset({
        ActionType.BREAKER_OPEN.value,
        ActionType.BREAKER_CLOSE.value,
        ActionType.TRIP_OPEN.value,
    })
    # 断路器跳开(分闸) action_type 集合
    _BREAKER_OPEN_TYPES = frozenset({
        ActionType.BREAKER_OPEN.value,
        ActionType.TRIP_OPEN.value,
    })
    # 信号抖动排除集: 保护/断路器类信号不算抖动
    _JITTER_EXCLUDE_GROUPS = frozenset({
        "protection", "breaker", "recloser",
    })

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        correlation_service: Optional[DeviceCorrelationService] = None,
    ):
        self.config = config or AnalyzerConfig()
        self.correlation = correlation_service
        logger.info("ActionSequenceAnalyzer 初始化完成")

    # =========================================================================
    # 主入口
    # =========================================================================

    def analyze(self, events: List[ActionEvent]) -> List[ChainAnalysisResult]:
        """分析一组动作事件, 返回所有匹配的动作链(按置信度降序)"""
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.source_ts or e.receive_ts)

        results: List[ChainAnalysisResult] = []

        # 先跑所有规则
        for rule_fn in [
            self._check_normal_trip,
            self._check_recloser_fail,
            self._check_refuse_action,
            self._check_false_action,
            self._check_control_loop_abnormal,
            self._check_mechanism_abnormal,
            self._check_signal_jitter,
        ]:
            result = rule_fn(sorted_events)
            if result and result.confidence > 0:
                results.append(result)

        # 互斥门禁: 如果匹配了正常跳闸链, 排除拒动和误动
        has_normal_trip = any(r.chain_type == ChainType.NORMAL_TRIP.value for r in results)
        if has_normal_trip:
            results = [r for r in results if r.chain_type not in (
                ChainType.REFUSE_ACTION.value, ChainType.FALSE_ACTION.value)]

        # 互斥门禁: 拒动和误动不能同时存在(拒动=有保护无断路器, 误动=有断路器无保护)
        has_refuse = any(r.chain_type == ChainType.REFUSE_ACTION.value for r in results)
        has_false = any(r.chain_type == ChainType.FALSE_ACTION.value for r in results)
        if has_refuse and has_false:
            # 保留置信度更高的
            refuse_conf = next(r.confidence for r in results if r.chain_type == ChainType.REFUSE_ACTION.value)
            false_conf = next(r.confidence for r in results if r.chain_type == ChainType.FALSE_ACTION.value)
            drop = ChainType.FALSE_ACTION.value if refuse_conf >= false_conf else ChainType.REFUSE_ACTION.value
            results = [r for r in results if r.chain_type != drop]

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    # =========================================================================
    # 规则1: 正常跳闸链
    #
    # 前置条件:
    #   - 必须有 protection_trip (保护出口)
    #   - 必须有 breaker_open/trip_open (断路器分闸)
    #   - 保护出口 与 断路器分闸 必须在同一间隔(bay_id)或关联同一断路器
    #   - 严格时序: 保护出口时间 ≤ 断路器分闸时间
    #   - 时间差 ≤ trip_chain_window_s
    # 排除条件:
    #   - 告知信号(alarm_signal) 不算保护出口
    # 置信度:
    #   - 基准 0.7
    #   - 有 protection_start 在 trip 之前: +0.1
    #   - source_system 非 simulated/manual: +0.05
    #   - signal_group 校验通过: +0.05
    # =========================================================================

    def _check_normal_trip(self, events: List[ActionEvent]) -> Optional[ChainAnalysisResult]:
        protection_trip = self._find_events_by_type_and_group(
            events, [ActionType.PROTECTION_TRIP.value, ActionType.BUS_DIFF_TRIP.value],
            required_groups=["protection"])
        breaker_open = self._find_events_by_type_and_group(
            events, list(self._BREAKER_OPEN_TYPES),
            required_groups=["breaker"])

        if not protection_trip or not breaker_open:
            return None

        # 找同间隔的配对
        paired_trip, paired_breaker = self._find_same_bay_pair(protection_trip, breaker_open)
        if not paired_trip:
            # 回退: 仅当双方都没有bay_id时才按时序配对
            # 如果双方都有bay_id但不匹配, 说明不是同一间隔, 不应配对
            all_have_bay = all(e.bay_id for e in protection_trip) and all(e.bay_id for e in breaker_open)
            if all_have_bay:
                return None  # 所有事件都有bay_id但无匹配 → 不是同间隔因果链
            paired_trip = protection_trip[0]
            paired_breaker = breaker_open[0]

        trip_ts = self._parse_ts(paired_trip.source_ts)
        breaker_ts = self._parse_ts(paired_breaker.source_ts)

        if not trip_ts or not breaker_ts:
            return None

        # 严格时序: 保护出口 必须 ≤ 断路器分闸 (允许极小反向误差 0.01s 兼容采集精度)
        delta = (breaker_ts - trip_ts).total_seconds()
        if delta < -0.01:
            return None  # 断路器比保护出口更早 → 不是正常因果链
        if delta > self.config.trip_chain_window_s:
            return None

        # 收集所有匹配事件
        protection_start = self._find_events_by_type_and_group(
            events, [ActionType.PROTECTION_START.value], required_groups=["protection"])
        # 仅保留同间隔的 protection_start
        bay_id = paired_trip.bay_id
        protection_start = [e for e in protection_start if not bay_id or e.bay_id == bay_id]
        # 仅保留在 protection_trip 之前的 start
        protection_start = [e for e in protection_start
                            if self._is_before_or_equal(e.source_ts, paired_trip.source_ts)]

        matched = protection_start + [paired_trip, paired_breaker]
        # 加入同间隔的其他断路器事件
        for e in breaker_open:
            if e.event_id != paired_breaker.event_id and (not bay_id or e.bay_id == bay_id):
                matched.append(e)

        # 置信度计算
        conf = 0.70
        reasons = ["有保护出口+断路器分闸"]
        if protection_start:
            conf += 0.10
            reasons.append("有保护启动前兆")
        if paired_trip.source_system not in (SourceSystem.SIMULATED.value, SourceSystem.MANUAL.value):
            conf += 0.05
            reasons.append(f"来源可信({paired_trip.source_system})")
        if paired_trip.signal_group == "protection" and paired_breaker.signal_group == "breaker":
            conf += 0.05
            reasons.append("signal_group校验通过")
        conf = min(conf, 0.95)

        evidence = self._build_evidence_nodes(matched)
        trip_scope = self._build_trip_scope(matched, is_real_trip=True)

        return ChainAnalysisResult(
            chain_type=ChainType.NORMAL_TRIP.value,
            confidence=round(conf, 3),
            description=f"正常跳闸链: {', '.join(reasons)}，时差{delta:.3f}s",
            matched_events=matched,
            evidence_nodes=evidence,
            trip_scope=trip_scope,
            metadata={"delta_s": round(delta, 4), "bay_id": bay_id, "reasons": reasons},
        )

    # =========================================================================
    # 规则2: 重合闸失败链
    #
    # 前置条件:
    #   - 必须有 recloser_action 或 recloser_charge (重合闸动作/充电)
    #   - 必须满足以下之一:
    #     a) 有 recloser_fail 信号
    #     b) 在 recloser_action 之后 有第2次 protection_trip
    #   - 所有事件在 recloser_window_s 内
    #   - recloser 事件与 trip 事件在同一间隔
    # 排除条件:
    #   - 仅有 recloser_charge 但无 fail 且无二次 trip → 不算失败
    # =========================================================================

    def _check_recloser_fail(self, events: List[ActionEvent]) -> Optional[ChainAnalysisResult]:
        recloser_events = self._find_events(events, [
            ActionType.RECLOSER_ACTION.value,
            ActionType.RECLOSER_CHARGE.value,
        ])
        recloser_fail = self._find_events(events, [ActionType.RECLOSER_FAIL.value])
        all_trips = self._find_events(events, [ActionType.PROTECTION_TRIP.value])

        if not recloser_events:
            return None

        # 确定重合闸所在间隔
        rc_bay = recloser_events[0].bay_id
        # 过滤同间隔的 trip
        bay_trips = [e for e in all_trips if not rc_bay or e.bay_id == rc_bay]
        bay_fail = [e for e in recloser_fail if not rc_bay or e.bay_id == rc_bay]

        is_fail = bool(bay_fail)

        # 检查"重合闸动作之后"是否有二次trip
        if not is_fail and len(bay_trips) >= 2:
            rc_action = [e for e in recloser_events if e.action_type == ActionType.RECLOSER_ACTION.value]
            if rc_action:
                rc_ts = self._parse_ts(rc_action[0].source_ts)
                if rc_ts:
                    trips_after_rc = [e for e in bay_trips
                                      if self._is_after(e.source_ts, rc_action[0].source_ts)]
                    if trips_after_rc:
                        is_fail = True

        if not is_fail:
            return None

        # 时间窗口校验
        all_relevant = recloser_events + bay_fail + bay_trips
        if not self._within_window(all_relevant, self.config.recloser_window_s):
            return None

        matched = recloser_events + bay_fail + bay_trips
        # 去重
        matched = list({e.event_id: e for e in matched}.values())
        matched.sort(key=lambda e: e.source_ts or "")

        conf = 0.75
        reasons = ["有重合闸动作"]
        if bay_fail:
            conf += 0.10
            reasons.append("有明确重合闸失败信号")
        if len(bay_trips) >= 2:
            conf += 0.05
            reasons.append(f"二次跳闸({len(bay_trips)}次)")
        conf = min(conf, 0.95)

        evidence = self._build_evidence_nodes(matched)
        trip_scope = self._build_trip_scope(matched, is_real_trip=True)
        if trip_scope:
            trip_scope.recloser_enabled = True
            trip_scope.recloser_success = False

        return ChainAnalysisResult(
            chain_type=ChainType.RECLOSER_FAIL.value,
            confidence=round(conf, 3),
            description=f"重合闸失败链: {', '.join(reasons)}",
            matched_events=matched,
            evidence_nodes=evidence,
            trip_scope=trip_scope,
            metadata={"bay_id": rc_bay, "trip_count": len(bay_trips), "reasons": reasons},
        )

    # =========================================================================
    # 规则3: 拒动链
    #
    # 前置条件:
    #   - 必须有 protection_trip (保护出口, 不是仅 protection_start)
    #   - 在 protection_trip 之后 refuse_action_timeout_s 秒内,
    #     该保护出口对应的断路器(同bay)无 breaker_open/trip_open
    #   - source_system 非 simulated
    # 排除条件:
    #   - 仅有 protection_start 无 protection_trip → 不判拒动
    #   - 同间隔有 breaker_open → 不是拒动
    #   - 有 breaker_fail_trip (失灵跳闸) → 说明已有后备处理, 仍记拒动但标注
    # =========================================================================

    def _check_refuse_action(self, events: List[ActionEvent]) -> Optional[ChainAnalysisResult]:
        protection_trip = self._find_events_by_type_and_group(
            events, [ActionType.PROTECTION_TRIP.value, ActionType.BUS_DIFF_TRIP.value],
            required_groups=["protection"])

        if not protection_trip:
            return None  # 仅有 start 无 trip 不判拒动

        # 检查每个 protection_trip 是否在超时窗口内无对应断路器响应
        all_breaker = self._find_events(events, list(self._BREAKER_OPEN_TYPES))
        breaker_fail_trip = self._find_events(events, [ActionType.BREAKER_FAIL_TRIP.value])

        unmatched_trips = []
        for pt in protection_trip:
            pt_ts = self._parse_ts(pt.source_ts)
            if not pt_ts:
                continue
            # 在同间隔寻找断路器响应
            found_breaker = False
            for be in all_breaker:
                if pt.bay_id and be.bay_id and pt.bay_id != be.bay_id:
                    continue  # 不同间隔
                be_ts = self._parse_ts(be.source_ts)
                if be_ts and 0 <= (be_ts - pt_ts).total_seconds() <= self.config.refuse_action_timeout_s:
                    found_breaker = True
                    break
            if not found_breaker:
                unmatched_trips.append(pt)

        if not unmatched_trips:
            return None

        matched = unmatched_trips + breaker_fail_trip
        conf = 0.60
        reasons = [f"保护出口{len(unmatched_trips)}次无断路器响应"]
        if all(e.source_system != SourceSystem.SIMULATED.value for e in unmatched_trips):
            conf += 0.10
            reasons.append("来源非模拟")
        if breaker_fail_trip:
            conf += 0.05
            reasons.append("已触发失灵保护")
        # 有保护启动在前进一步增加可信度
        protection_start = self._find_events(events, [ActionType.PROTECTION_START.value])
        start_in_bay = [e for e in protection_start
                        if not unmatched_trips[0].bay_id or e.bay_id == unmatched_trips[0].bay_id]
        if start_in_bay:
            conf += 0.10
            reasons.append("有保护启动前兆")
        conf = min(conf, 0.90)

        evidence = self._build_evidence_nodes(matched)
        trip_scope = self._build_trip_scope(matched, is_real_trip=False)

        return ChainAnalysisResult(
            chain_type=ChainType.REFUSE_ACTION.value,
            confidence=round(conf, 3),
            description=f"拒动链: {', '.join(reasons)}",
            matched_events=matched,
            evidence_nodes=evidence,
            trip_scope=trip_scope,
            metadata={"unmatched_count": len(unmatched_trips),
                       "timeout_s": self.config.refuse_action_timeout_s, "reasons": reasons},
        )

    # =========================================================================
    # 规则4: 误动链
    #
    # 前置条件:
    #   - 必须有 breaker_open/trip_open (断路器实际分闸)
    #   - 在 false_action_window_s 窗口内, 无 protection_start 也无 protection_trip
    #   - signal_group 为 breaker (不接受 protection 信号被错误归类)
    # 排除条件:
    #   - 有任何 protection_start 或 protection_trip → 不是误动(是正常跳闸或拒动)
    #   - 仅有 alarm_signal/abnormal_signal 无断路器状态变化 → 不算误动
    #   - 信号抖动类事件(同信号多次变位) → 走信号抖动规则, 不走误动
    # 置信度:
    #   - 基准 0.40 (误动判定天然低信心, 需外部证据佐证)
    #   - 来源系统可信 +0.10
    #   - 有异常/告知信号伴随 +0.10
    # =========================================================================

    def _check_false_action(self, events: List[ActionEvent]) -> Optional[ChainAnalysisResult]:
        # 严格: 只看断路器状态变化, 不混入 protection_trip
        breaker_open = self._find_events_by_type_and_group(
            events, list(self._BREAKER_OPEN_TYPES),
            required_groups=["breaker"])

        if not breaker_open:
            return None

        # 排除: 有任何保护事件 → 不是误动
        any_protection = self._find_events(events, [
            ActionType.PROTECTION_START.value,
            ActionType.PROTECTION_TRIP.value,
            ActionType.BUS_DIFF_TRIP.value,
            ActionType.BREAKER_FAIL_TRIP.value,
        ])
        if any_protection:
            return None

        # 排除: 控制回路断线伴随 → 走控制回路规则
        loop_break = self._find_events(events, [ActionType.CONTROL_LOOP_BREAK.value])
        if loop_break:
            return None

        alarm_signals = self._find_events(events, [
            ActionType.ALARM_SIGNAL.value,
            ActionType.ABNORMAL_SIGNAL.value,
        ])

        matched = breaker_open + alarm_signals
        conf = 0.40
        reasons = [f"断路器分闸{len(breaker_open)}次无保护前兆"]
        if all(e.source_system not in (SourceSystem.SIMULATED.value, SourceSystem.MANUAL.value)
               for e in breaker_open):
            conf += 0.10
            reasons.append("来源可信")
        if alarm_signals:
            conf += 0.10
            reasons.append(f"有{len(alarm_signals)}条异常/告知信号")
        conf = min(conf, 0.70)  # 误动判定上限, 需外部证据才能更高

        evidence = self._build_evidence_nodes(matched)
        # 误动: 断路器确实动作了, 但不应该动作 → is_real_trip=False(不是正当跳闸)
        trip_scope = self._build_trip_scope(matched, is_real_trip=False)

        return ChainAnalysisResult(
            chain_type=ChainType.FALSE_ACTION.value,
            confidence=round(conf, 3),
            description=f"疑似误动链: {', '.join(reasons)}",
            matched_events=matched,
            evidence_nodes=evidence,
            trip_scope=trip_scope,
            metadata={"breaker_count": len(breaker_open), "reasons": reasons},
        )

    # =========================================================================
    # 规则5: 控制回路异常链
    #
    # 前置条件:
    #   - 必须有 control_loop_break (控制回路断线)
    #   - signal_group 应为 control_loop 或 mechanism
    # 排除条件: (无)
    # 置信度:
    #   - 基准 0.70 (控制回路断线本身是明确信号)
    #   - 同间隔伴随断路器异常 +0.10 (影响更大)
    #   - 同间隔伴随保护事件 +0.05 (可能因回路断线导致后续问题)
    #   - 来源为 protection_device/protection_info +0.05
    # =========================================================================

    def _check_control_loop_abnormal(self, events: List[ActionEvent]) -> Optional[ChainAnalysisResult]:
        loop_break = self._find_events(events, [ActionType.CONTROL_LOOP_BREAK.value])
        if not loop_break:
            return None

        bay_id = loop_break[0].bay_id

        # 同间隔在时间窗口内的伴随事件
        breaker_events = [e for e in self._find_events(events, list(self._BREAKER_STATE_TYPES))
                          if not bay_id or e.bay_id == bay_id]
        if breaker_events:
            breaker_events = [e for e in breaker_events
                              if self._events_within_window([loop_break[0], e], self.config.control_loop_window_s)]
        protection_events = [e for e in self._find_events(events, list(self._PROTECTION_TYPES))
                             if not bay_id or e.bay_id == bay_id]

        matched = loop_break + breaker_events
        conf = 0.70
        reasons = ["控制回路断线"]
        if breaker_events:
            conf += 0.10
            reasons.append(f"伴随断路器异常{len(breaker_events)}条")
        if protection_events:
            conf += 0.05
            reasons.append("伴随保护事件")
        if loop_break[0].source_system in (SourceSystem.PROTECTION_DEVICE.value, SourceSystem.PROTECTION_INFO.value):
            conf += 0.05
            reasons.append("来源为保护装置/保信子站")
        conf = min(conf, 0.95)

        evidence = self._build_evidence_nodes(matched)
        trip_scope = self._build_trip_scope(matched, is_real_trip=False)

        return ChainAnalysisResult(
            chain_type=ChainType.CONTROL_LOOP_ABNORMAL.value,
            confidence=round(conf, 3),
            description=f"控制回路异常链: {', '.join(reasons)}",
            matched_events=matched,
            evidence_nodes=evidence,
            trip_scope=trip_scope,
            metadata={"bay_id": bay_id, "breaker_count": len(breaker_events), "reasons": reasons},
        )

    # =========================================================================
    # 规则6: 机构异常链
    #
    # 前置条件:
    #   - 必须有 spring_not_charged (弹簧未储能)
    #   - signal_group 应为 mechanism 或 breaker
    # 排除条件:
    #   - 同间隔有正常的 protection_trip + breaker_open 链 → 机构未必异常
    # 置信度:
    #   - 基准 0.60
    #   - 同间隔伴随断路器动作(可能因机构问题导致慢分/拒分) +0.15
    #   - 来源为 protection_device +0.05
    # =========================================================================

    def _check_mechanism_abnormal(self, events: List[ActionEvent]) -> Optional[ChainAnalysisResult]:
        mechanism_events = self._find_events(events, [ActionType.SPRING_NOT_CHARGED.value])
        if not mechanism_events:
            return None

        bay_id = mechanism_events[0].bay_id

        # 排除: 同间隔已有正常跳闸链
        trip_in_bay = [e for e in self._find_events(events, [ActionType.PROTECTION_TRIP.value])
                       if not bay_id or e.bay_id == bay_id]
        breaker_in_bay = [e for e in self._find_events(events, list(self._BREAKER_OPEN_TYPES))
                          if not bay_id or e.bay_id == bay_id]
        if trip_in_bay and breaker_in_bay:
            # 正常跳闸链已形成, 弹簧未储能可能是跳闸后的正常状态
            return None

        # 伴随断路器事件
        breaker_events = [e for e in self._find_events(events, list(self._BREAKER_STATE_TYPES))
                          if not bay_id or e.bay_id == bay_id]
        if breaker_events:
            breaker_events = [e for e in breaker_events
                              if self._events_within_window([mechanism_events[0], e], self.config.mechanism_window_s)]

        matched = mechanism_events + breaker_events
        conf = 0.60
        reasons = ["弹簧未储能"]
        if breaker_events:
            conf += 0.15
            reasons.append(f"伴随断路器事件{len(breaker_events)}条")
        if mechanism_events[0].source_system == SourceSystem.PROTECTION_DEVICE.value:
            conf += 0.05
            reasons.append("来源为保护装置")
        conf = min(conf, 0.85)

        evidence = self._build_evidence_nodes(matched)
        trip_scope = self._build_trip_scope(matched, is_real_trip=False)

        return ChainAnalysisResult(
            chain_type=ChainType.MECHANISM_ABNORMAL.value,
            confidence=round(conf, 3),
            description=f"机构异常链: {', '.join(reasons)}",
            matched_events=matched,
            evidence_nodes=evidence,
            trip_scope=trip_scope,
            metadata={"bay_id": bay_id, "reasons": reasons},
        )

    # =========================================================================
    # 规则7: 信号抖动/重复动作链
    #
    # 前置条件:
    #   - 同一 signal_id 在 jitter_window_s 内出现 ≥ jitter_min_count 次
    #   - signal_group 不是 protection / breaker / recloser
    #     (保护/断路器连续动作可能是级联故障, 不是抖动)
    # 排除条件:
    #   - 如果该信号同时伴随正常跳闸链事件, 不判为抖动
    # 置信度:
    #   - 基准 0.50
    #   - 变位次数每多1次 +0.05 (上限 +0.25)
    #   - value_before/value_after 交替翻转(0→1→0→1) +0.10
    # =========================================================================

    def _check_signal_jitter(self, events: List[ActionEvent]) -> Optional[ChainAnalysisResult]:
        # 按 signal_id 分组, 排除保护/断路器/重合闸类信号
        signal_groups: Dict[str, List[ActionEvent]] = defaultdict(list)
        for e in events:
            if e.signal_id and e.signal_group not in self._JITTER_EXCLUDE_GROUPS:
                signal_groups[e.signal_id].append(e)

        jitter_sets: List[Tuple[str, List[ActionEvent]]] = []
        for sig_id, sig_events in signal_groups.items():
            if len(sig_events) < self.config.jitter_min_count:
                continue
            timestamps = [self._parse_ts(e.source_ts) for e in sig_events]
            timestamps = [t for t in timestamps if t]
            if len(timestamps) < self.config.jitter_min_count:
                continue
            timestamps.sort()
            window = (timestamps[-1] - timestamps[0]).total_seconds()
            if window <= self.config.jitter_window_s:
                jitter_sets.append((sig_id, sig_events))

        if not jitter_sets:
            return None

        all_jitter = []
        for _, evts in jitter_sets:
            all_jitter.extend(evts)

        # 置信度
        total_count = len(all_jitter)
        conf = 0.50
        extra = min((total_count - self.config.jitter_min_count) * 0.05, 0.25)
        conf += extra
        reasons = [f"{len(jitter_sets)}个信号在{self.config.jitter_window_s}s内变位{total_count}次"]

        # 检查是否交替翻转
        for _, evts in jitter_sets:
            if len(evts) >= 3:
                vals = [e.value_after for e in evts if e.value_after is not None]
                if len(vals) >= 3:
                    is_alternating = all(vals[i] != vals[i+1] for i in range(len(vals)-1))
                    if is_alternating:
                        conf += 0.10
                        reasons.append("值交替翻转")
                        break
        conf = min(conf, 0.85)

        evidence = self._build_evidence_nodes(all_jitter)

        return ChainAnalysisResult(
            chain_type=ChainType.SIGNAL_JITTER.value,
            confidence=round(conf, 3),
            description=f"信号抖动链: {', '.join(reasons)}",
            matched_events=all_jitter,
            evidence_nodes=evidence,
            metadata={"signal_count": len(jitter_sets), "event_count": total_count, "reasons": reasons},
        )

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _find_events(self, events: List[ActionEvent], action_types: List[str]) -> List[ActionEvent]:
        """从事件列表中筛选指定动作类型"""
        type_set = frozenset(action_types)
        return [e for e in events if e.action_type in type_set]

    def _find_events_by_type_and_group(
        self, events: List[ActionEvent],
        action_types: List[str],
        required_groups: Optional[List[str]] = None,
    ) -> List[ActionEvent]:
        """按 action_type + signal_group 双重过滤"""
        type_set = frozenset(action_types)
        group_set = frozenset(required_groups) if required_groups else None
        result = []
        for e in events:
            if e.action_type not in type_set:
                continue
            if group_set and e.signal_group not in group_set:
                continue
            result.append(e)
        return result

    def _find_same_bay_pair(
        self,
        group_a: List[ActionEvent],
        group_b: List[ActionEvent],
    ) -> Tuple[Optional[ActionEvent], Optional[ActionEvent]]:
        """在两组事件中找到 bay_id 相同的第一对"""
        for a in group_a:
            if not a.bay_id:
                continue
            for b in group_b:
                if a.bay_id == b.bay_id:
                    return a, b
        return None, None

    def _is_before_or_equal(self, ts_a: str, ts_b: str) -> bool:
        """判断 ts_a <= ts_b"""
        a = self._parse_ts(ts_a)
        b = self._parse_ts(ts_b)
        if not a or not b:
            return True  # 无法解析时不拦截
        return a <= b

    def _is_after(self, ts_a: str, ts_b: str) -> bool:
        """判断 ts_a > ts_b"""
        a = self._parse_ts(ts_a)
        b = self._parse_ts(ts_b)
        if not a or not b:
            return False
        return a > b

    def _within_window(self, events: List[ActionEvent], window_s: float) -> bool:
        """检查一组事件是否在指定时间窗口内"""
        if len(events) < 2:
            return True
        timestamps = [self._parse_ts(e.source_ts) for e in events]
        timestamps = [t for t in timestamps if t]
        if len(timestamps) < 2:
            return True
        timestamps.sort()
        return (timestamps[-1] - timestamps[0]).total_seconds() <= window_s

    def _events_within_window(self, events: List[ActionEvent], window_s: float) -> bool:
        """_within_window 的别名, 用于两事件间距检查"""
        return self._within_window(events, window_s)

    def _build_evidence_nodes(self, events: List[ActionEvent]) -> List[EvidenceNode]:
        """从匹配事件构建标准化的证据链节点"""
        evidence = []
        for e in events:
            # node_type 基于 signal_group 和 action_type 确定
            if e.action_type in self._PROTECTION_TYPES or e.signal_group == "protection":
                node_type = "protection"
            elif e.action_type in self._BREAKER_STATE_TYPES or e.signal_group == "breaker":
                node_type = "breaker"
            elif e.signal_group == "recloser" or "recloser" in e.action_type:
                node_type = "recloser"
            elif e.signal_group == "control_loop":
                node_type = "control_loop"
            elif e.signal_group == "mechanism":
                node_type = "mechanism"
            else:
                node_type = "signal_change"

            # confidence 基于来源系统
            if e.source_system in (SourceSystem.PROTECTION_DEVICE.value, SourceSystem.PROTECTION_INFO.value,
                                   SourceSystem.FAULT_RECORDER.value):
                node_conf = 0.90
            elif e.source_system in (SourceSystem.SIMULATED.value, SourceSystem.MANUAL.value):
                node_conf = 0.50
            else:
                node_conf = 0.75

            evidence.append(EvidenceNode(
                node_type=node_type,
                source=e.source_system,
                timestamp=e.source_ts,
                description=e.action_desc,
                confidence=node_conf,
                data={
                    "event_id": e.event_id,
                    "action_type": e.action_type,
                    "signal_group": e.signal_group,
                    "bay_id": e.bay_id,
                    "device_id": e.secondary_device_id or e.primary_device_id,
                },
            ))
        return evidence

    def _parse_ts(self, ts_str: str) -> Optional[datetime]:
        """解析ISO8601时间戳"""
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

    def _build_trip_scope(
        self,
        events: List[ActionEvent],
        is_real_trip: bool
    ) -> TripScopeResult:
        """根据匹配事件构建跳闸范围"""
        scope = TripScopeResult(
            trace_id=events[0].trace_id if events else "",
            is_real_trip=is_real_trip,
            action_events=[e.event_id for e in events],
        )

        # 收集范围信息
        primary_set = set()
        secondary_set = set()
        impact_set = set()
        protection_acted = []
        trip_breakers = []
        phases = set()
        protection_types = set()
        max_fault_current = None
        wave_record_ids = []

        for e in events:
            if e.primary_device_id:
                primary_set.add((e.primary_device_id, e.primary_device_type,
                                 getattr(e, 'bay_id', ''), getattr(e, 'bay_name', '')))
            if e.secondary_device_id:
                secondary_set.add((e.secondary_device_id, e.secondary_device_type,
                                   getattr(e, 'bay_id', ''), getattr(e, 'cabinet_id', '')))
            if e.station_id and e.bay_id:
                vl = getattr(e, 'voltage_level', '') or ''
                impact_set.add((e.station_id, e.bay_id, e.bay_name, vl))

            # 收集保护动作信息
            if e.action_type in (ActionType.PROTECTION_START.value, ActionType.PROTECTION_TRIP.value,
                                 ActionType.BUS_DIFF_TRIP.value, ActionType.BREAKER_FAIL_TRIP.value):
                ptype = getattr(e, 'protection_type', '') or ''
                protection_acted.append({
                    "device_id": e.secondary_device_id,
                    "device_name": e.signal_name or e.action_desc,
                    "protection_type": ptype,
                    "action_time": e.source_ts,
                })
                if ptype:
                    protection_types.add(ptype)

            # 收集断路器跳闸信息
            if e.action_type in (ActionType.BREAKER_OPEN.value, ActionType.TRIP_OPEN.value):
                trip_breakers.append({
                    "device_id": e.primary_device_id,
                    "device_name": e.signal_name or e.action_desc,
                    "bay_id": e.bay_id,
                    "trip_time": e.source_ts,
                })

            # 收集相别
            phase = getattr(e, 'phase', '') or ''
            if phase:
                phases.add(phase)

            # 收集故障电流(取最大值)
            fc = getattr(e, 'fault_current_ka', None)
            if fc is not None:
                if max_fault_current is None or fc > max_fault_current:
                    max_fault_current = fc

            # 收集录波ID
            wrid = getattr(e, 'wave_record_id', None)
            if wrid:
                wave_record_ids.append(wrid)

        # 电压等级(取事件中出现的最高)
        voltage_levels = [getattr(e, 'voltage_level', '') for e in events if getattr(e, 'voltage_level', '')]
        if voltage_levels:
            scope.voltage_level = voltage_levels[0]

        # 故障相别汇总
        if phases:
            scope.fault_phase = "/".join(sorted(phases))
            scope.fault_type = self._infer_fault_type(phases)

        # 设备名通过 correlation 补充
        scope.primary_scope = []
        for d in primary_set:
            entry = {"device_id": d[0], "device_type": d[1], "bay_id": d[2], "bay_name": d[3], "device_name": ""}
            if self.correlation:
                pd = self.correlation.get_primary_device(d[0])
                if pd:
                    entry["device_name"] = pd.device_name
            scope.primary_scope.append(entry)

        scope.secondary_scope = []
        for d in secondary_set:
            entry = {"device_id": d[0], "device_type": d[1], "bay_id": d[2], "cabinet_id": d[3], "device_name": ""}
            if self.correlation:
                sd = self.correlation.get_secondary_device(d[0])
                if sd:
                    entry["device_name"] = sd.device_name
            scope.secondary_scope.append(entry)

        scope.impact_scope = [
            {"station_id": d[0], "bay_id": d[1], "bay_name": d[2], "voltage_level": d[3]}
            for d in impact_set
        ]

        scope.protection_acted = protection_acted
        scope.trip_breakers = trip_breakers

        # 故障电流 & 录波
        scope.fault_current = max_fault_current
        if wave_record_ids:
            scope.wave_record_status = f"有录波({len(wave_record_ids)}份)"
        else:
            scope.wave_record_status = "无录波"

        # 通过关系服务补充故障线路和停电范围
        if self.correlation and events:
            for e in events:
                if e.secondary_device_id and e.signal_id:
                    cr = self.correlation.correlate_by_signal(e.secondary_device_id, e.signal_id)
                    if cr.station and cr.bay:
                        scope.fault_line = f"{cr.station.station_name}-{cr.bay.bay_name}"
                        if not scope.voltage_level and cr.bay.voltage_level:
                            scope.voltage_level = cr.bay.voltage_level
                    break

            # 推导停电范围(需要母线拓扑)
            tripped_ids = [tb["device_id"] for tb in trip_breakers if tb["device_id"]]
            if tripped_ids:
                scope.outage_scope = self.correlation.get_outage_scope(tripped_ids)

        scope.reason = f"{'真正跳闸' if is_real_trip else '动作异常(未形成跳闸)'}"
        scope.confidence = 0.9 if is_real_trip else 0.6
        return scope

    @staticmethod
    def _infer_fault_type(phases: set) -> str:
        """根据相别集合推导故障类型"""
        from platform_core.action_event_schema import FaultType
        all_phases = set()
        for p in phases:
            for ch in p:
                if ch in ('A', 'B', 'C', 'N'):
                    all_phases.add(ch)
        phase_count = len(all_phases - {'N'})
        has_ground = 'N' in all_phases
        if phase_count >= 3:
            return FaultType.THREE_PHASE.value
        elif phase_count == 2:
            return FaultType.TWO_PHASE_GROUND.value if has_ground else FaultType.PHASE_TO_PHASE.value
        elif phase_count == 1:
            return FaultType.SINGLE_PHASE_GROUND.value if has_ground else FaultType.SINGLE_PHASE_GROUND.value
        return FaultType.UNKNOWN.value
