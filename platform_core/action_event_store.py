# -*- coding: utf-8 -*-
"""
动作事件存储与查询
==================

提供动作事件的内存存储、查询、时间窗口聚合等能力。
生产环境可扩展为数据库后端(SQLite/PostgreSQL/InfluxDB)。

复用: alarm_manager.py 的线程安全存储模式
"""

from __future__ import annotations
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from platform_core.action_event_schema import (
    ActionEvent, TripScopeResult, RootCauseResult, FaultArchive,
    CandidateEvent, ManualReviewStatus,
)

logger = logging.getLogger(__name__)


class ActionEventStore:
    """
    动作事件存储引擎

    功能:
    1. 动作事件CRUD
    2. 按trace_id聚合
    3. 按时间窗口查询
    4. 按厂站/间隔/设备筛选
    5. 关联TripScopeResult和RootCauseResult
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._lock = threading.Lock()

        # 主存储
        self._events: Dict[str, ActionEvent] = {}           # event_id -> ActionEvent
        self._events_by_trace: Dict[str, List[str]] = defaultdict(list)  # trace_id -> [event_id]
        self._trip_scopes: Dict[str, TripScopeResult] = {}  # trace_id -> TripScopeResult
        self._root_causes: Dict[str, RootCauseResult] = {}  # trace_id -> RootCauseResult
        self._fault_archives: Dict[str, FaultArchive] = {}  # archive_id -> FaultArchive
        self._candidate_events: Dict[str, CandidateEvent] = {}  # event_id -> CandidateEvent
        self._candidates_by_trace: Dict[str, List[str]] = defaultdict(list)  # trace_id -> [event_id]

        # 时间索引(按小时桶)
        self._hourly_index: Dict[str, List[str]] = defaultdict(list)

        # 配置
        self.max_events = self.config.get('max_events', 100000)
        self.max_archives = self.config.get('max_archives', 10000)

        # 事件回调
        self._on_event_callbacks: List[Callable[[ActionEvent], None]] = []

        logger.info("ActionEventStore 初始化完成")

    # -------------------------------------------------------------------------
    # 事件写入
    # -------------------------------------------------------------------------

    def add_event(self, event: ActionEvent) -> str:
        """写入一条动作事件"""
        with self._lock:
            self._events[event.event_id] = event
            self._events_by_trace[event.trace_id].append(event.event_id)

            # 时间索引
            hour_key = self._hour_key(event.source_ts or event.receive_ts)
            self._hourly_index[hour_key].append(event.event_id)

            # 容量限制
            if len(self._events) > self.max_events:
                self._evict_oldest()

        # 触发回调
        for cb in self._on_event_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"事件回调异常: {e}")

        logger.debug(f"动作事件已存储: {event.event_id} trace={event.trace_id}")
        return event.event_id

    def add_events(self, events: List[ActionEvent]) -> List[str]:
        """批量写入动作事件"""
        return [self.add_event(e) for e in events]

    # -------------------------------------------------------------------------
    # 事件查询
    # -------------------------------------------------------------------------

    def get_event(self, event_id: str) -> Optional[ActionEvent]:
        """根据event_id获取"""
        return self._events.get(event_id)

    def get_events_by_trace(self, trace_id: str) -> List[ActionEvent]:
        """根据trace_id获取同一追踪链的所有事件"""
        event_ids = self._events_by_trace.get(trace_id, [])
        events = [self._events[eid] for eid in event_ids if eid in self._events]
        events.sort(key=lambda e: e.source_ts or e.receive_ts)
        return events

    def query_events(
        self,
        station_id: Optional[str] = None,
        bay_id: Optional[str] = None,
        device_id: Optional[str] = None,
        action_type: Optional[str] = None,
        signal_group: Optional[str] = None,
        source_system: Optional[str] = None,
        severity_hint: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[ActionEvent]:
        """多条件查询动作事件"""
        results = list(self._events.values())

        if station_id:
            results = [e for e in results if e.station_id == station_id]
        if bay_id:
            results = [e for e in results if e.bay_id == bay_id]
        if device_id:
            results = [e for e in results
                       if e.primary_device_id == device_id or e.secondary_device_id == device_id]
        if action_type:
            results = [e for e in results if e.action_type == action_type]
        if signal_group:
            results = [e for e in results if e.signal_group == signal_group]
        if source_system:
            results = [e for e in results if e.source_system == source_system]
        if severity_hint:
            results = [e for e in results if e.severity_hint == severity_hint]
        if start_time:
            results = [e for e in results if (e.source_ts or e.receive_ts) >= start_time]
        if end_time:
            results = [e for e in results if (e.source_ts or e.receive_ts) <= end_time]

        # 按时间倒序
        results.sort(key=lambda e: e.source_ts or e.receive_ts, reverse=True)
        return results[offset:offset + limit]

    def count_events(self, **kwargs) -> int:
        """统计事件数量"""
        return len(self.query_events(**kwargs, limit=self.max_events))

    # -------------------------------------------------------------------------
    # 跳闸范围 & 根因存储
    # -------------------------------------------------------------------------

    def save_trip_scope(self, result: TripScopeResult) -> None:
        with self._lock:
            self._trip_scopes[result.trace_id] = result
        logger.info(f"跳闸范围判定已存储: trace={result.trace_id}")

    def get_trip_scope(self, trace_id: str) -> Optional[TripScopeResult]:
        return self._trip_scopes.get(trace_id)

    def save_root_cause(self, result: RootCauseResult) -> None:
        with self._lock:
            self._root_causes[result.trace_id] = result
        logger.info(f"根因分析结果已存储: trace={result.trace_id}")

    def get_root_cause(self, trace_id: str) -> Optional[RootCauseResult]:
        return self._root_causes.get(trace_id)

    # -------------------------------------------------------------------------
    # 故障归档
    # -------------------------------------------------------------------------

    def save_fault_archive(self, archive: FaultArchive) -> str:
        with self._lock:
            self._fault_archives[archive.archive_id] = archive
            if len(self._fault_archives) > self.max_archives:
                oldest_key = next(iter(self._fault_archives))
                del self._fault_archives[oldest_key]
        logger.info(f"故障归档已存储: {archive.archive_id}")
        return archive.archive_id

    def get_fault_archive(self, archive_id: str) -> Optional[FaultArchive]:
        return self._fault_archives.get(archive_id)

    def query_fault_archives(
        self,
        station_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[FaultArchive]:
        results = list(self._fault_archives.values())
        if station_id:
            results = [a for a in results if a.station_id == station_id]
        if status:
            results = [a for a in results if a.status == status]
        results.sort(key=lambda a: a.created_at, reverse=True)
        return results[:limit]

    # -------------------------------------------------------------------------
    # 时间线聚合
    # -------------------------------------------------------------------------

    def get_timeline(
        self,
        trace_id: Optional[str] = None,
        station_id: Optional[str] = None,
        bay_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        获取时间线数据(供前端展示)

        返回:
        {
            "events": [...],
            "trip_scope": {...} or null,
            "root_cause": {...} or null,
            "summary": {...}
        }
        """
        if trace_id:
            events = self.get_events_by_trace(trace_id)
        else:
            events = self.query_events(
                station_id=station_id, bay_id=bay_id,
                start_time=start_time, end_time=end_time, limit=limit
            )

        # 找到关联的分析结果
        trace_ids = list(set(e.trace_id for e in events))

        trip_scopes = {}
        root_causes = {}
        for tid in trace_ids:
            ts = self.get_trip_scope(tid)
            if ts:
                trip_scopes[tid] = ts.to_dict()
            rc = self.get_root_cause(tid)
            if rc:
                root_causes[tid] = rc.to_dict()

        # 统计
        action_type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        for e in events:
            action_type_counts[e.action_type] += 1
            severity_counts[e.severity_hint] += 1

        # 候选事件
        candidate_events = {}
        for tid in trace_ids:
            cands = self.get_candidates_by_trace(tid)
            if cands:
                candidate_events[tid] = [c.to_dict() for c in cands]

        return {
            "events": [e.to_dict() for e in events],
            "trip_scopes": trip_scopes,
            "root_causes": root_causes,
            "candidate_events": candidate_events,
            "summary": {
                "total_events": len(events),
                "trace_count": len(trace_ids),
                "action_type_distribution": dict(action_type_counts),
                "severity_distribution": dict(severity_counts),
                "candidate_count": sum(len(v) for v in candidate_events.values()),
            }
        }

    # -------------------------------------------------------------------------
    # 回调注册
    # -------------------------------------------------------------------------

    def on_event(self, callback: Callable[[ActionEvent], None]) -> None:
        """注册新事件回调(供alarm_manager等消费)"""
        self._on_event_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------

    def _hour_key(self, ts: str) -> str:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.strftime('%Y%m%d%H')
        except (ValueError, AttributeError):
            return datetime.now().strftime('%Y%m%d%H')

    def _evict_oldest(self):
        """淘汰最旧的事件"""
        if not self._events:
            return
        sorted_ids = sorted(self._events.keys(),
                            key=lambda eid: self._events[eid].source_ts or self._events[eid].receive_ts)
        to_remove = sorted_ids[:len(sorted_ids) // 10]  # 淘汰10%
        for eid in to_remove:
            ev = self._events.pop(eid, None)
            if ev and ev.trace_id in self._events_by_trace:
                trace_list = self._events_by_trace[ev.trace_id]
                if eid in trace_list:
                    trace_list.remove(eid)
        logger.info(f"已淘汰 {len(to_remove)} 条旧事件")

    # -------------------------------------------------------------------------
    # 候选事件
    # -------------------------------------------------------------------------

    def save_candidate_event(self, candidate: CandidateEvent) -> str:
        """保存候选事件"""
        with self._lock:
            self._candidate_events[candidate.event_id] = candidate
            if candidate.trace_id:
                if candidate.event_id not in self._candidates_by_trace[candidate.trace_id]:
                    self._candidates_by_trace[candidate.trace_id].append(candidate.event_id)
        logger.info(f"候选事件已存储: {candidate.event_id} trace={candidate.trace_id}")
        return candidate.event_id

    def get_candidate_event(self, event_id: str) -> Optional[CandidateEvent]:
        """获取候选事件"""
        return self._candidate_events.get(event_id)

    def get_candidates_by_trace(self, trace_id: str) -> List[CandidateEvent]:
        """根据trace_id获取候选事件"""
        event_ids = self._candidates_by_trace.get(trace_id, [])
        candidates = [self._candidate_events[eid] for eid in event_ids if eid in self._candidate_events]
        candidates.sort(key=lambda c: c.start_time or c.created_at)
        return candidates

    def query_candidate_events(
        self,
        station_id: Optional[str] = None,
        device_id: Optional[str] = None,
        event_type: Optional[str] = None,
        review_status: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CandidateEvent]:
        """多条件查询候选事件"""
        results = list(self._candidate_events.values())
        if station_id:
            results = [c for c in results if c.station_id == station_id]
        if device_id:
            results = [c for c in results if c.device_id == device_id]
        if event_type:
            results = [c for c in results if c.event_type == event_type]
        if review_status:
            results = [c for c in results if c.manual_review_status == review_status]
        if severity:
            results = [c for c in results if c.severity == severity]
        if start_time:
            results = [c for c in results if (c.start_time or c.created_at) >= start_time]
        if end_time:
            results = [c for c in results if (c.start_time or c.created_at) <= end_time]
        results.sort(key=lambda c: c.start_time or c.created_at, reverse=True)
        return results[offset:offset + limit]

    def update_review_status(
        self,
        event_id: str,
        status: str,
        comment: str = "",
        reviewed_by: str = "",
    ) -> Optional[CandidateEvent]:
        """
        更新候选事件的人工复核状态

        支持的状态流转:
        - pending/required -> confirmed/rejected/false_positive/handled/field_check
        - confirmed/rejected/false_positive/handled/field_check -> 任意状态(允许修正)
        """
        with self._lock:
            candidate = self._candidate_events.get(event_id)
            if not candidate:
                return None
            candidate.manual_review_status = status
            if comment:
                candidate.review_comment = comment
            if reviewed_by:
                candidate.reviewed_by = reviewed_by
            candidate.reviewed_at = datetime.now().isoformat()
            candidate.updated_at = candidate.reviewed_at
        logger.info(f"候选事件复核状态更新: {event_id} -> {status}")
        return candidate

    # -------------------------------------------------------------------------
    # 跨插件查询接口(为 switch_inspection 等提供统一查询)
    # -------------------------------------------------------------------------

    def query_events_for_device(
        self,
        device_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        按设备/时间窗查询动作事件，返回统一JSON结构。
        为 switch_inspection 等跨插件状态一致性校核提供复用接口。

        返回:
        {
            "device_id": str,
            "events": [...],
            "candidate_events": [...],
            "data_status": "sufficient" | "partial" | "no_data",
            "event_count": int,
            "candidate_count": int,
            "time_range": {"start": str, "end": str},
        }
        """
        events = self.query_events(
            device_id=device_id, start_time=start_time, end_time=end_time, limit=limit)
        candidates = [c for c in self._candidate_events.values()
                      if c.device_id == device_id]
        if start_time:
            candidates = [c for c in candidates if (c.start_time or c.created_at) >= start_time]
        if end_time:
            candidates = [c for c in candidates if (c.start_time or c.created_at) <= end_time]

        if not events and not candidates:
            data_status = "no_data"
        elif len(events) < 3:
            data_status = "partial"
        else:
            data_status = "sufficient"

        actual_start = ""
        actual_end = ""
        if events:
            sorted_ev = sorted(events, key=lambda e: e.source_ts or e.receive_ts)
            actual_start = sorted_ev[0].source_ts or sorted_ev[0].receive_ts
            actual_end = sorted_ev[-1].source_ts or sorted_ev[-1].receive_ts

        return {
            "device_id": device_id,
            "events": [e.to_dict() for e in events],
            "candidate_events": [c.to_dict() for c in candidates],
            "data_status": data_status,
            "event_count": len(events),
            "candidate_count": len(candidates),
            "time_range": {"start": actual_start, "end": actual_end},
        }

    # -------------------------------------------------------------------------
    # 统计
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        review_counts: Dict[str, int] = defaultdict(int)
        for c in self._candidate_events.values():
            review_counts[c.manual_review_status] += 1

        return {
            "total_events": len(self._events),
            "total_traces": len(self._events_by_trace),
            "total_trip_scopes": len(self._trip_scopes),
            "total_root_causes": len(self._root_causes),
            "total_archives": len(self._fault_archives),
            "total_candidates": len(self._candidate_events),
            "review_status_distribution": dict(review_counts),
        }
