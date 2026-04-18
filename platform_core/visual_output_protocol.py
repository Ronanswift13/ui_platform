"""
A 类视觉插件统一输出协议
========================

为 11 个视觉/状态识别插件提供统一的 metadata 构建工具，
确保 infer() 返回的 RecognitionResult.metadata 包含完整的：
- 运行时状态字段
- 质量门控状态
- 复核与决策字段 (review_required / reason_codes / recommended_actions)
- 模型信息与升级占位
- 第二阶段接口预留 (embedding / feature / lineage)

使用方式:
    from platform_core.visual_output_protocol import build_visual_meta, TASK_TYPES

版本: 1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ─── 任务类型常量 ────────────────────────────────────────────────
TASK_TYPES = {
    "detection": "detection",
    "classification": "classification",
    "inspection": "inspection",
    "ocr": "ocr",
    "thermal_analysis": "thermal_analysis",
    "spectral_analysis": "spectral_analysis",
}

# ─── 质量门状态常量 ──────────────────────────────────────────────
QUALITY_GATE_PASS = "pass"
QUALITY_GATE_SOFT_FAIL = "soft_fail"
QUALITY_GATE_HARD_FAIL = "hard_fail"
QUALITY_GATE_NOT_APPLICABLE = "not_applicable"

# ─── 算法阶段常量 ───────────────────────────────────────────────
ALGORITHM_STAGE_BASELINE = "baseline"
ALGORITHM_STAGE_FALLBACK = "fallback"
ALGORITHM_STAGE_PLACEHOLDER = "placeholder"
ALGORITHM_STAGE_REAL_MODEL = "real_model"

# ─── 模态常量 ───────────────────────────────────────────────────
MODALITY_VISUAL = "visual"
MODALITY_THERMAL = "thermal"
MODALITY_HYPERSPECTRAL = "hyperspectral"
MODALITY_METER = "meter"
MODALITY_FIRE = "fire"
MODALITY_ANIMAL = "animal"
MODALITY_BIRD = "bird"


def build_visual_meta(
    *,
    plugin_name: str,
    task_type: str = "detection",
    modality: str = MODALITY_VISUAL,
    # 运行时状态
    runtime_mode: str = "simulation",
    algorithm_stage: str = ALGORITHM_STAGE_BASELINE,
    model_status: str = "placeholder",
    fallback_level: int = 0,
    # 质量门控
    quality_gate_status: str = QUALITY_GATE_PASS,
    # 复核与决策
    review_required: bool = False,
    reason_codes: Optional[List[str]] = None,
    recommended_actions: Optional[List[str]] = None,
    # 证据
    evidence_hint: str = "visual_frame",
    # 模型信息
    model_info: Optional[Dict[str, Any]] = None,
    # 额外的插件特有字段
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建 A 类视觉插件统一 metadata 字典。

    返回值可直接赋给 RecognitionResult(metadata=...) 或合并到现有 metadata。

    Args:
        plugin_name: 插件标识 (e.g. "busbar_inspection")
        task_type: 任务类型，参见 TASK_TYPES
        modality: 输入模态
        runtime_mode: 运行模式 (simulation / real_dl / traditional_fallback / standalone_simulation)
        algorithm_stage: 算法阶段
        model_status: 模型状态 (placeholder / loaded / verified / production)
        fallback_level: 降级层数 (0=主链, 1=一级降级, ...)
        quality_gate_status: 质量门状态
        review_required: 是否需要人工复核
        reason_codes: 原因码列表
        recommended_actions: 建议操作列表
        evidence_hint: 证据类型提示
        model_info: 模型详细信息 (可覆盖默认值)
        extra: 插件特有的附加字段，会 merge 到顶层

    Returns:
        统一 metadata dict
    """

    meta: Dict[str, Any] = {
        # ── 插件标识 ──
        "plugin_name": plugin_name,
        "task_type": task_type,
        # ── 输入模态 ──
        "modality": modality,
        # ── 运行时状态 ──
        "runtime_mode": runtime_mode,
        "algorithm_stage": algorithm_stage,
        "model_status": model_status,
        "fallback_level": fallback_level,
        # ── 质量门控 ──
        "quality_gate_status": quality_gate_status,
        # ── 复核与决策 ──
        "review_required": review_required,
        "reason_codes": reason_codes or [],
        "recommended_actions": recommended_actions or [],
        # ── 证据 ──
        "evidence_hint": evidence_hint,
        # ── 模型信息 ──
        "model_info": _build_model_info(model_info),
        # ── 第二阶段升级占位 ──
        "placeholders": _build_placeholders(),
        # ── 训练预留 ──
        "training_placeholders": _build_training_placeholders(),
    }

    if extra:
        meta.update(extra)

    return meta


def enrich_review(
    meta: Dict[str, Any],
    *,
    review_required: bool,
    reason_codes: Optional[List[str]] = None,
    recommended_actions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """对已有 metadata 追加/更新复核字段。"""
    meta["review_required"] = review_required
    if reason_codes:
        existing = meta.get("reason_codes", [])
        meta["reason_codes"] = list(dict.fromkeys(existing + reason_codes))
    if recommended_actions:
        existing = meta.get("recommended_actions", [])
        meta["recommended_actions"] = list(dict.fromkeys(existing + recommended_actions))
    return meta


# ─── 内部构建函数 ───────────────────────────────────────────────

def _build_model_info(override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构建模型信息段。"""
    info = {
        "model_version": "placeholder",
        "model_type": "rule_based",
        "training_date": None,
        "dataset_version": None,
        # 第二阶段预留
        "model_registry_id": None,
        "model_asset_status": "not_available",
        "calibration_required": False,
        "dataset_requirements": None,
        "annotation_status": "not_started",
    }
    if override:
        info.update(override)
    return info


def _build_placeholders() -> Dict[str, Any]:
    """构建第二阶段视觉/多模态升级占位字段。"""
    return {
        "visual_embedding_placeholder": None,
        "thermal_feature_placeholder": None,
        "spectral_signature_placeholder": None,
        "model_upgrade_candidate_placeholder": None,
        "annotation_feedback_placeholder": None,
    }


def _build_training_placeholders() -> Dict[str, Any]:
    """构建训练反馈占位字段。"""
    return {
        "hard_negative_candidate": False,
        "hard_positive_candidate": False,
        "suggested_label_for_dataset": None,
        "annotation_status": "not_started",
        "model_placeholder": True,
    }


# ─── 校验函数（用于合同测试） ────────────────────────────────────

# 统一 metadata 必须包含的顶层 key
REQUIRED_META_KEYS = frozenset({
    "plugin_name",
    "task_type",
    "modality",
    "runtime_mode",
    "algorithm_stage",
    "model_status",
    "fallback_level",
    "quality_gate_status",
    "review_required",
    "reason_codes",
    "recommended_actions",
    "evidence_hint",
    "model_info",
    "placeholders",
    "training_placeholders",
})

REQUIRED_PLACEHOLDER_KEYS = frozenset({
    "visual_embedding_placeholder",
    "thermal_feature_placeholder",
    "spectral_signature_placeholder",
    "model_upgrade_candidate_placeholder",
    "annotation_feedback_placeholder",
})

REQUIRED_TRAINING_KEYS = frozenset({
    "hard_negative_candidate",
    "hard_positive_candidate",
    "suggested_label_for_dataset",
    "annotation_status",
    "model_placeholder",
})

VALID_QUALITY_GATE_STATUSES = frozenset({
    QUALITY_GATE_PASS,
    QUALITY_GATE_SOFT_FAIL,
    QUALITY_GATE_HARD_FAIL,
    QUALITY_GATE_NOT_APPLICABLE,
})

VALID_ALGORITHM_STAGES = frozenset({
    ALGORITHM_STAGE_BASELINE,
    ALGORITHM_STAGE_FALLBACK,
    ALGORITHM_STAGE_PLACEHOLDER,
    ALGORITHM_STAGE_REAL_MODEL,
})


def validate_visual_meta(meta: Dict[str, Any]) -> List[str]:
    """校验 metadata 是否符合 A 类统一输出协议。

    Returns:
        错误列表；空列表表示通过。
    """
    errors: List[str] = []

    # 检查顶层字段
    missing_top = REQUIRED_META_KEYS - meta.keys()
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")

    # 检查 quality_gate_status
    qgs = meta.get("quality_gate_status")
    if qgs and qgs not in VALID_QUALITY_GATE_STATUSES:
        errors.append(f"invalid quality_gate_status: {qgs}")

    # 检查 algorithm_stage
    alg = meta.get("algorithm_stage")
    if alg and alg not in VALID_ALGORITHM_STAGES:
        errors.append(f"invalid algorithm_stage: {alg}")

    # 检查 confidence 范围 (如果存在顶层 confidence)
    conf = meta.get("confidence")
    if conf is not None and not (0 <= conf <= 1):
        errors.append(f"confidence out of range: {conf}")

    # 检查 review_required 类型
    rr = meta.get("review_required")
    if rr is not None and not isinstance(rr, bool):
        errors.append(f"review_required must be bool, got {type(rr).__name__}")

    # 检查 reason_codes 类型
    rc = meta.get("reason_codes")
    if rc is not None and not isinstance(rc, list):
        errors.append(f"reason_codes must be list, got {type(rc).__name__}")

    # 检查 recommended_actions 类型
    ra = meta.get("recommended_actions")
    if ra is not None and not isinstance(ra, list):
        errors.append(f"recommended_actions must be list, got {type(ra).__name__}")

    # 检查 placeholders 子字段
    placeholders = meta.get("placeholders")
    if isinstance(placeholders, dict):
        missing_ph = REQUIRED_PLACEHOLDER_KEYS - placeholders.keys()
        if missing_ph:
            errors.append(f"missing placeholder keys: {sorted(missing_ph)}")

    # 检查 training_placeholders 子字段
    training = meta.get("training_placeholders")
    if isinstance(training, dict):
        missing_tr = REQUIRED_TRAINING_KEYS - training.keys()
        if missing_tr:
            errors.append(f"missing training_placeholder keys: {sorted(missing_tr)}")

    return errors
