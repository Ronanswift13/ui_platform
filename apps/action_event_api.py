# -*- coding: utf-8 -*-
"""
动作事件监测 API 路由
=====================

提供以下 API:
- POST /api/action-events          写入动作事件
- GET  /api/action-events          查询动作事件
- POST /api/action-analysis/analyze  触发动作链分析
- GET  /api/device-correlation/{device_id}  查询设备关系
- POST /api/root-cause/analyze     触发根因分析
- GET  /api/timeline               获取时间线数据
- POST /api/fault-archive          创建/更新故障归档

遵循现有 FastAPI 路由风格 (参考 apps/api_server.py)
"""

from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from platform_core.action_event_schema import (
    ActionEvent, TripScopeResult, RootCauseResult, FaultArchive,
    CandidateEvent, ManualReviewStatus, SequenceCompleteness,
    ChainType, SeverityHint, RootCauseCategory,
    normalize_action_type,
)
from platform_core.action_event_store import ActionEventStore
from platform_core.action_sequence_analyzer import ActionSequenceAnalyzer, AnalyzerConfig
from platform_core.root_cause_service import RootCauseService
from platform_core.device_correlation import DeviceCorrelationService

# =============================================================================
# 全局服务实例(惰性初始化)
# =============================================================================

_event_store: Optional[ActionEventStore] = None
_analyzer: Optional[ActionSequenceAnalyzer] = None
_root_cause_service: Optional[RootCauseService] = None
_correlation_service: Optional[DeviceCorrelationService] = None


def get_event_store() -> ActionEventStore:
    global _event_store
    if _event_store is None:
        _event_store = ActionEventStore()
    return _event_store


def get_correlation_service() -> DeviceCorrelationService:
    global _correlation_service
    if _correlation_service is None:
        _correlation_service = DeviceCorrelationService()
    return _correlation_service


def get_analyzer() -> ActionSequenceAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ActionSequenceAnalyzer(
            config=AnalyzerConfig(),
            correlation_service=get_correlation_service(),
        )
    return _analyzer


def get_root_cause_service() -> RootCauseService:
    global _root_cause_service
    if _root_cause_service is None:
        _root_cause_service = RootCauseService(
            correlation_service=get_correlation_service(),
        )
    return _root_cause_service


# =============================================================================
# Pydantic 请求/响应模型
# =============================================================================

class ActionEventRequest(BaseModel):
    """写入动作事件请求"""
    trace_id: str = ""
    station_id: str = ""
    station_name: str = ""
    bay_id: str = ""
    bay_name: str = ""
    primary_device_id: str = ""
    primary_device_type: str = ""
    secondary_device_id: str = ""
    secondary_device_type: str = ""
    signal_id: str = ""
    signal_name: str = ""
    signal_group: str = "other"
    action_type: str = ""
    action_desc: str = ""
    value_before: Optional[str] = None
    value_after: Optional[str] = None
    source_ts: str = ""
    source_system: str = "simulated"
    seq_no: int = 0
    severity_hint: str = "info"
    raw_payload: Optional[dict] = None


class BatchActionEventRequest(BaseModel):
    """批量写入"""
    events: List[ActionEventRequest]


class AnalysisRequest(BaseModel):
    """分析请求"""
    trace_id: str = ""
    event_ids: List[str] = Field(default_factory=list)


class RootCauseRequest(BaseModel):
    """根因分析请求"""
    trace_id: str = ""
    event_ids: List[str] = Field(default_factory=list)
    primary_evidence: List[dict] = Field(default_factory=list)
    secondary_evidence: List[dict] = Field(default_factory=list)
    auxiliary_evidence: List[dict] = Field(default_factory=list)


class FaultArchiveRequest(BaseModel):
    """故障归档请求"""
    trace_id: str = ""
    station_id: str = ""
    station_name: str = ""
    fault_time: str = ""
    fault_line: str = ""
    voltage_level: str = ""
    primary_repair_scope: str = ""
    secondary_repair_scope: str = ""
    recloser_enabled: Optional[bool] = None
    recloser_success: Optional[bool] = None
    fault_current: Optional[float] = None
    wave_record_status: str = ""
    status: str = "draft"


class ReviewRequest(BaseModel):
    """人工复核请求"""
    event_id: str
    status: str  # confirmed/rejected/false_positive/handled/field_check
    comment: str = ""
    reviewed_by: str = ""


class ReviewCommentRequest(BaseModel):
    """追加备注请求"""
    event_id: str
    comment: str
    reviewed_by: str = ""


class ApiResponse(BaseModel):
    """通用响应"""
    success: bool = True
    data: Any = None
    message: str = ""


# =============================================================================
# API Router
# =============================================================================

router = APIRouter(prefix="/api", tags=["action-events"])


# ---- POST /api/action-events ----
@router.post("/action-events", response_model=ApiResponse)
async def create_action_events(request: ActionEventRequest):
    """写入单条动作事件"""
    store = get_event_store()
    data = request.model_dump()

    # 自动归一化 action_type
    if not data.get("action_type") and data.get("action_desc"):
        data["action_type"] = normalize_action_type(data["action_desc"])

    event = ActionEvent.from_dict(data)
    event_id = store.add_event(event)

    return ApiResponse(data={"event_id": event_id, "trace_id": event.trace_id})


@router.post("/action-events/batch", response_model=ApiResponse)
async def create_action_events_batch(request: BatchActionEventRequest):
    """批量写入动作事件"""
    store = get_event_store()
    event_ids = []
    for req in request.events:
        data = req.model_dump()
        if not data.get("action_type") and data.get("action_desc"):
            data["action_type"] = normalize_action_type(data["action_desc"])
        event = ActionEvent.from_dict(data)
        eid = store.add_event(event)
        event_ids.append(eid)

    return ApiResponse(data={"event_ids": event_ids, "count": len(event_ids)})


# ---- GET /api/action-events ----
@router.get("/action-events", response_model=ApiResponse)
async def query_action_events(
    station_id: Optional[str] = None,
    bay_id: Optional[str] = None,
    device_id: Optional[str] = None,
    action_type: Optional[str] = None,
    signal_group: Optional[str] = None,
    source_system: Optional[str] = None,
    severity_hint: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(default=200, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """查询动作事件"""
    store = get_event_store()
    events = store.query_events(
        station_id=station_id, bay_id=bay_id, device_id=device_id,
        action_type=action_type, signal_group=signal_group,
        source_system=source_system, severity_hint=severity_hint,
        start_time=start_time, end_time=end_time,
        limit=limit, offset=offset,
    )
    return ApiResponse(data={
        "events": [e.to_dict() for e in events],
        "total": len(events),
    })


# ---- POST /api/action-analysis/analyze ----
@router.post("/action-analysis/analyze", response_model=ApiResponse)
async def analyze_action_sequence(request: AnalysisRequest):
    """触发动作链分析"""
    store = get_event_store()
    analyzer = get_analyzer()

    if request.trace_id:
        events = store.get_events_by_trace(request.trace_id)
    elif request.event_ids:
        events = [store.get_event(eid) for eid in request.event_ids]
        events = [e for e in events if e is not None]
    else:
        raise HTTPException(status_code=400, detail="需要提供 trace_id 或 event_ids")

    if not events:
        raise HTTPException(status_code=404, detail="未找到匹配的动作事件")

    results = analyzer.analyze(events)

    # 保存跳闸范围
    for r in results:
        if r.trip_scope:
            store.save_trip_scope(r.trip_scope)

    return ApiResponse(data={
        "chain_results": [
            {
                "chain_type": r.chain_type,
                "confidence": r.confidence,
                "description": r.description,
                "matched_event_count": len(r.matched_events),
                "trip_scope": r.trip_scope.to_dict() if r.trip_scope else None,
                "evidence_nodes": [n.to_dict() for n in r.evidence_nodes],
            }
            for r in results
        ],
        "total_chains": len(results),
    })


# ---- GET /api/device-correlation/{device_id} ----
@router.get("/device-correlation/{device_id}", response_model=ApiResponse)
async def get_device_correlation(device_id: str):
    """查询设备关系"""
    service = get_correlation_service()
    result = service.correlate_by_device(device_id)
    return ApiResponse(data=result.to_dict())


# ---- POST /api/root-cause/analyze ----
@router.post("/root-cause/analyze", response_model=ApiResponse)
async def analyze_root_cause(request: RootCauseRequest):
    """触发根因分析"""
    store = get_event_store()
    analyzer = get_analyzer()
    rcs = get_root_cause_service()

    if request.trace_id:
        events = store.get_events_by_trace(request.trace_id)
    elif request.event_ids:
        events = [store.get_event(eid) for eid in request.event_ids]
        events = [e for e in events if e is not None]
    else:
        raise HTTPException(status_code=400, detail="需要提供 trace_id 或 event_ids")

    if not events:
        raise HTTPException(status_code=404, detail="未找到匹配的动作事件")

    # 先做动作链分析
    chain_results = analyzer.analyze(events)

    # 解析外部证据
    from platform_core.root_cause_service import (
        PrimaryDeviceEvidence, SecondaryDeviceEvidence, AuxiliaryEvidence
    )
    primary_ev = [PrimaryDeviceEvidence(**e) for e in request.primary_evidence]
    secondary_ev = [SecondaryDeviceEvidence(**e) for e in request.secondary_evidence]
    auxiliary_ev = [AuxiliaryEvidence(**e) for e in request.auxiliary_evidence]

    # 根因分析
    result = rcs.analyze(
        chain_results=chain_results,
        action_events=events,
        primary_evidence=primary_ev,
        secondary_evidence=secondary_ev,
        auxiliary_evidence=auxiliary_ev,
    )

    # 保存
    store.save_root_cause(result)

    return ApiResponse(data=result.to_dict())


# ---- GET /api/timeline ----
@router.get("/timeline", response_model=ApiResponse)
async def get_timeline(
    trace_id: Optional[str] = None,
    station_id: Optional[str] = None,
    bay_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(default=100, le=500),
):
    """获取时间线数据(供前端展示)"""
    store = get_event_store()
    timeline = store.get_timeline(
        trace_id=trace_id, station_id=station_id, bay_id=bay_id,
        start_time=start_time, end_time=end_time, limit=limit,
    )
    return ApiResponse(data=timeline)


# ---- POST /api/fault-archive ----
@router.post("/fault-archive", response_model=ApiResponse)
async def create_or_update_fault_archive(request: FaultArchiveRequest):
    """创建或更新故障归档"""
    store = get_event_store()
    rcs = get_root_cause_service()

    data = request.model_dump()
    trace_id = data.get("trace_id", "")

    # 如果有 trace_id，尝试关联已有分析结果
    root_cause = store.get_root_cause(trace_id) if trace_id else None
    trip_scope = store.get_trip_scope(trace_id) if trace_id else None

    archive = FaultArchive(
        trace_id=trace_id,
        station_id=data.get("station_id", ""),
        station_name=data.get("station_name", ""),
        fault_time=data.get("fault_time", ""),
        fault_line=data.get("fault_line", ""),
        voltage_level=data.get("voltage_level", ""),
        primary_repair_scope=data.get("primary_repair_scope", ""),
        secondary_repair_scope=data.get("secondary_repair_scope", ""),
        recloser_enabled=data.get("recloser_enabled"),
        recloser_success=data.get("recloser_success"),
        fault_current=data.get("fault_current"),
        wave_record_status=data.get("wave_record_status", ""),
        root_cause=root_cause,
        trip_scope=trip_scope,
        status=data.get("status", "draft"),
    )

    archive_id = store.save_fault_archive(archive)
    return ApiResponse(data={"archive_id": archive_id, "status": archive.status})


# ---- GET /api/fault-archive ----
@router.get("/fault-archive", response_model=ApiResponse)
async def query_fault_archives(
    station_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
):
    """查询故障归档"""
    store = get_event_store()
    archives = store.query_fault_archives(station_id=station_id, status=status, limit=limit)
    return ApiResponse(data={
        "archives": [a.to_dict() for a in archives],
        "total": len(archives),
    })


# ---- GET /api/action-events/statistics ----
@router.get("/action-events/statistics", response_model=ApiResponse)
async def get_statistics():
    """获取统计信息"""
    store = get_event_store()
    return ApiResponse(data=store.get_statistics())


# =============================================================================
# 候选事件 API
# =============================================================================

def _build_candidate_from_analysis(
    trace_id: str,
    events: List[ActionEvent],
    chain_results,
    root_cause_result=None,
) -> CandidateEvent:
    """从分析结果构建候选事件对象"""
    from platform_core.action_sequence_analyzer import ChainAnalysisResult

    best_chain: Optional[ChainAnalysisResult] = chain_results[0] if chain_results else None

    # 确定时间范围
    sorted_events = sorted(events, key=lambda e: e.source_ts or e.receive_ts)
    start_time = sorted_events[0].source_ts or sorted_events[0].receive_ts if sorted_events else ""
    end_time = sorted_events[-1].source_ts or sorted_events[-1].receive_ts if sorted_events else ""

    # 确定设备/厂站
    station_id = ""
    station_name = ""
    device_id = ""
    bay_id = ""
    bay_name = ""
    for e in sorted_events:
        if e.station_id and not station_id:
            station_id = e.station_id
            station_name = e.station_name
        if e.bay_id and not bay_id:
            bay_id = e.bay_id
            bay_name = e.bay_name
        if (e.secondary_device_id or e.primary_device_id) and not device_id:
            device_id = e.secondary_device_id or e.primary_device_id

    # 事件类型与严重程度
    event_type = best_chain.chain_type if best_chain else ChainType.UNKNOWN.value
    severity = SeverityHint.INFO.value
    confidence = best_chain.confidence if best_chain else 0.0

    # 严重程度由链类型决定
    severity_map = {
        ChainType.NORMAL_TRIP.value: SeverityHint.ALARM.value,
        ChainType.RECLOSER_FAIL.value: SeverityHint.CRITICAL.value,
        ChainType.REFUSE_ACTION.value: SeverityHint.CRITICAL.value,
        ChainType.FALSE_ACTION.value: SeverityHint.ALARM.value,
        ChainType.CONTROL_LOOP_ABNORMAL.value: SeverityHint.WARNING.value,
        ChainType.MECHANISM_ABNORMAL.value: SeverityHint.WARNING.value,
        ChainType.SIGNAL_JITTER.value: SeverityHint.WARNING.value,
    }
    severity = severity_map.get(event_type, SeverityHint.INFO.value)

    # 构建关键动作序列
    action_sequence = [
        {
            "event_id": e.event_id,
            "action_type": e.action_type,
            "action_desc": e.action_desc,
            "source_ts": e.source_ts,
            "signal_group": e.signal_group,
        }
        for e in sorted_events
    ]

    # 判定序列完整性
    has_start = any(e.action_type == "protection_start" for e in events)
    has_trip = any(e.action_type in ("protection_trip", "bus_diff_trip") for e in events)
    has_breaker = any(e.action_type in ("breaker_open", "trip_open") for e in events)
    if has_start and has_trip and has_breaker:
        seq_completeness = SequenceCompleteness.COMPLETE.value
    elif has_trip or has_breaker:
        seq_completeness = SequenceCompleteness.PARTIAL.value
    elif events:
        seq_completeness = SequenceCompleteness.INCOMPLETE.value
    else:
        seq_completeness = SequenceCompleteness.UNKNOWN.value

    # 异常标记
    chain_types_found = {r.chain_type for r in chain_results} if chain_results else set()
    has_refuse = ChainType.REFUSE_ACTION.value in chain_types_found
    has_false = ChainType.FALSE_ACTION.value in chain_types_found
    has_jitter = ChainType.SIGNAL_JITTER.value in chain_types_found

    # 检查保护先后异常: 后备保护先于主保护
    has_abnormal_seq = False
    prot_events = [e for e in sorted_events if e.action_type in ("protection_start", "protection_trip")]
    if len(prot_events) >= 2:
        for i in range(len(prot_events) - 1):
            pt_i = prot_events[i].protection_type or ""
            pt_j = prot_events[i + 1].protection_type or ""
            if "backup" in pt_i and "main" in pt_j:
                has_abnormal_seq = True
                break

    # 检查重复动作
    from collections import Counter
    action_counter = Counter((e.action_type, e.bay_id) for e in sorted_events)
    has_repeated = any(count > 2 for count in action_counter.values())

    # 证据
    evidence = []
    if best_chain:
        evidence = [n.to_dict() for n in best_chain.evidence_nodes]

    # 根因候选
    root_cause_candidates = []
    if root_cause_result:
        probs = root_cause_result.probabilities or {}
        cat_labels = {
            RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value: "一次设备故障",
            RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value: "二次设备故障",
            RootCauseCategory.CONTROL_LOOP_ISSUE.value: "控制回路/机构问题",
            RootCauseCategory.SIGNAL_ANOMALY.value: "信号异常",
            RootCauseCategory.EXTERNAL_CAUSE.value: "外部原因",
        }
        for cat, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            root_cause_candidates.append({
                "category": cat,
                "probability": round(prob, 4),
                "reason": cat_labels.get(cat, cat),
            })

    # 历史相似度(接口占位，返回明确状态)
    historical_similarity = {
        "status": "no_history_data",
        "similarity_score": 0.0,
        "matched_archive_id": "",
        "matched_event_type": "",
        "match_basis": "历史故障库尚未接入，接入后将自动匹配相似故障案例",
    }

    # 推断依据
    inference_parts = []
    if best_chain:
        inference_parts.append(f"动作链分析识别为{best_chain.description}")
    if root_cause_result:
        inference_parts.append(f"根因分析结论: {root_cause_result.root_cause_reason}")
    if has_abnormal_seq:
        inference_parts.append("检测到保护动作顺序异常(后备保护先于主保护)")
    if has_repeated:
        inference_parts.append("检测到重复动作")
    inference_basis = "；".join(inference_parts) if inference_parts else "数据不足，无法给出推断"

    # 人工复核状态
    review_status = ManualReviewStatus.PENDING.value
    boundary_note = "本结果仅为辅助决策建议，不构成自动控制指令，最终判定需人工确认"

    # 证据不足 → 强制要求人工复核
    if root_cause_result and root_cause_result.evidence_sufficiency in ("insufficient", "partial"):
        review_status = ManualReviewStatus.REQUIRED.value
        confidence = min(confidence, 0.5)
        boundary_note = (
            f"证据充分性: {root_cause_result.evidence_sufficiency}。"
            f"{'、'.join(root_cause_result.evidence_gaps[:3]) if root_cause_result.evidence_gaps else '无具体缺口说明'}。"
            "因证据不足，置信度已降级，结论仅供参考，需人工复核确认"
        )
    elif confidence < 0.6:
        review_status = ManualReviewStatus.REQUIRED.value
        boundary_note = f"置信度 {confidence:.1%} 低于判定门限(60%)，不能自动给出强结论，需人工复核"

    candidate = CandidateEvent(
        trace_id=trace_id,
        device_id=device_id,
        station_id=station_id,
        station_name=station_name,
        bay_id=bay_id,
        bay_name=bay_name,
        start_time=start_time,
        end_time=end_time,
        event_type=event_type,
        severity=severity,
        confidence=round(confidence, 4),
        evidence=evidence,
        action_sequence=action_sequence,
        sequence_completeness=seq_completeness,
        has_refuse_action=has_refuse,
        has_false_action=has_false,
        has_jitter=has_jitter,
        has_repeated_action=has_repeated,
        has_abnormal_sequence=has_abnormal_seq,
        root_cause_candidates=root_cause_candidates,
        historical_similarity=historical_similarity,
        inference_basis=inference_basis,
        manual_review_status=review_status,
        boundary_note=boundary_note,
        related_event_ids=[e.event_id for e in events],
        trip_scope=best_chain.trip_scope if best_chain else None,
        root_cause_result=root_cause_result,
    )
    return candidate


# ---- POST /api/candidate-events/generate ----
@router.post("/candidate-events/generate", response_model=ApiResponse)
async def generate_candidate_event(request: AnalysisRequest):
    """
    从动作事件生成候选事件对象(含动作链分析+根因分析+候选事件聚合)

    【边界说明】本接口输出仅用于辅助决策和人工复核，不得作为自动控制指令的依据。
    """
    store = get_event_store()
    analyzer = get_analyzer()
    rcs = get_root_cause_service()

    if request.trace_id:
        events = store.get_events_by_trace(request.trace_id)
    elif request.event_ids:
        events = [store.get_event(eid) for eid in request.event_ids]
        events = [e for e in events if e is not None]
    else:
        raise HTTPException(status_code=400, detail="需要提供 trace_id 或 event_ids")

    if not events:
        raise HTTPException(status_code=404, detail="未找到匹配的动作事件")

    trace_id = request.trace_id or events[0].trace_id

    # 动作链分析
    chain_results = analyzer.analyze(events)

    # 保存跳闸范围
    for r in chain_results:
        if r.trip_scope:
            store.save_trip_scope(r.trip_scope)

    # 根因分析
    from platform_core.root_cause_service import (
        PrimaryDeviceEvidence, SecondaryDeviceEvidence, AuxiliaryEvidence
    )
    root_cause_result = rcs.analyze(
        chain_results=chain_results,
        action_events=events,
    )
    store.save_root_cause(root_cause_result)

    # 构建候选事件
    candidate = _build_candidate_from_analysis(
        trace_id=trace_id,
        events=events,
        chain_results=chain_results,
        root_cause_result=root_cause_result,
    )
    store.save_candidate_event(candidate)

    return ApiResponse(data=candidate.to_dict())


# ---- GET /api/candidate-events ----
@router.get("/candidate-events", response_model=ApiResponse)
async def query_candidate_events(
    station_id: Optional[str] = None,
    device_id: Optional[str] = None,
    event_type: Optional[str] = None,
    review_status: Optional[str] = None,
    severity: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
):
    """查询候选事件列表"""
    store = get_event_store()
    candidates = store.query_candidate_events(
        station_id=station_id, device_id=device_id,
        event_type=event_type, review_status=review_status,
        severity=severity, start_time=start_time, end_time=end_time,
        limit=limit, offset=offset,
    )
    return ApiResponse(data={
        "candidates": [c.to_dict() for c in candidates],
        "total": len(candidates),
    })


# ---- GET /api/candidate-events/{event_id} ----
@router.get("/candidate-events/{event_id}", response_model=ApiResponse)
async def get_candidate_event(event_id: str):
    """获取单个候选事件详情"""
    store = get_event_store()
    candidate = store.get_candidate_event(event_id)
    if not candidate:
        raise HTTPException(status_code=404, detail=f"候选事件 {event_id} 不存在")
    return ApiResponse(data=candidate.to_dict())


# =============================================================================
# 人工复核 API
# =============================================================================

# ---- POST /api/candidate-events/review ----
@router.post("/candidate-events/review", response_model=ApiResponse)
async def review_candidate_event(request: ReviewRequest):
    """
    确认/驳回/标记候选事件

    支持状态:
    - confirmed: 确认事件结论正确
    - rejected: 驳回事件结论
    - false_positive: 标记为误报
    - handled: 标记为已处置
    - field_check: 标记为需现场复核
    """
    valid_statuses = {s.value for s in ManualReviewStatus} - {
        ManualReviewStatus.PENDING.value, ManualReviewStatus.REQUIRED.value}
    if request.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"无效的复核状态: {request.status}，允许值: {', '.join(sorted(valid_statuses))}"
        )

    store = get_event_store()
    candidate = store.update_review_status(
        event_id=request.event_id,
        status=request.status,
        comment=request.comment,
        reviewed_by=request.reviewed_by,
    )
    if not candidate:
        raise HTTPException(status_code=404, detail=f"候选事件 {request.event_id} 不存在")

    return ApiResponse(
        data=candidate.to_dict(),
        message=f"候选事件 {request.event_id} 已标记为 {request.status}",
    )


# ---- POST /api/candidate-events/comment ----
@router.post("/candidate-events/comment", response_model=ApiResponse)
async def add_review_comment(request: ReviewCommentRequest):
    """追加复核备注(不改变状态)"""
    store = get_event_store()
    candidate = store.get_candidate_event(request.event_id)
    if not candidate:
        raise HTTPException(status_code=404, detail=f"候选事件 {request.event_id} 不存在")

    existing = candidate.review_comment
    separator = "\n---\n" if existing else ""
    timestamp = datetime.now().isoformat()
    reviewer = request.reviewed_by or "unknown"
    new_comment = f"{existing}{separator}[{timestamp}] {reviewer}: {request.comment}"

    store.update_review_status(
        event_id=request.event_id,
        status=candidate.manual_review_status,
        comment=new_comment,
        reviewed_by=request.reviewed_by,
    )
    return ApiResponse(
        data={"event_id": request.event_id, "comment": new_comment},
        message="备注已追加",
    )


# =============================================================================
# 跨插件查询接口(为 switch_inspection 状态一致性校核提供)
# =============================================================================

# ---- GET /api/device-events/{device_id} ----
@router.get("/device-events/{device_id}", response_model=ApiResponse)
async def get_device_events(
    device_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(default=100, le=500),
):
    """
    按设备/时间窗查询动作事件，返回统一JSON结构。

    为 switch_inspection 等插件提供一致性校核接口。
    返回结构中包含 data_status 字段标识数据完整程度。
    """
    store = get_event_store()
    result = store.query_events_for_device(
        device_id=device_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    return ApiResponse(data=result)
