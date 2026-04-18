"""Authoritative label contract for ``busbar_inspection``.

This module is the single source of truth for:
- runtime supported labels
- planned/blocked labels
- legacy alias normalization
"""

from __future__ import annotations

from typing import Any

RUNTIME_SUPPORTED_LABELS = (
    "pin_missing",
    "crack",
    "foreign_object",
    "quality_failed",
)

RUNTIME_SUPPORTED_DEFECT_LABELS = (
    "pin_missing",
    "crack",
    "foreign_object",
)

LABEL_ALIASES = {
    "loose_fitting": "fitting_loose",
}

PLANNED_LABEL_STATUSES = {
    "broken_part": "blocked",
    "fitting_loose": "blocked",
}

LABEL_DISPLAY_NAMES = {
    "pin_missing": "销钉缺失",
    "crack": "裂纹",
    "foreign_object": "异物",
    "quality_failed": "质量门禁未通过",
    "broken_part": "部件破损",
    "fitting_loose": "金具松动",
}


def canonicalize_label(label: Any) -> str:
    """Normalize legacy label aliases to canonical contract labels."""
    normalized = str(label or "").strip().lower()
    if not normalized:
        return ""
    return LABEL_ALIASES.get(normalized, normalized)


def is_runtime_supported_label(label: Any) -> bool:
    """Whether the label is currently in the frozen runtime contract."""
    return canonicalize_label(label) in RUNTIME_SUPPORTED_LABELS


def is_runtime_supported_defect_label(label: Any) -> bool:
    """Whether the label is a currently supported defect output label."""
    return canonicalize_label(label) in RUNTIME_SUPPORTED_DEFECT_LABELS


def get_label_contract_snapshot() -> dict[str, Any]:
    """Return a serializable snapshot for docs/tests/healthcheck."""
    return {
        "runtime_supported": list(RUNTIME_SUPPORTED_LABELS),
        "runtime_supported_defects": list(RUNTIME_SUPPORTED_DEFECT_LABELS),
        "aliases": dict(LABEL_ALIASES),
        "planned": dict(PLANNED_LABEL_STATUSES),
    }
