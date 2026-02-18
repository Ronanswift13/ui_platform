"""
Alarm schemas.

Contains Alarm, AlarmLevel, AlarmRule, and AlarmStatus models - extracted from
platform_core/schema/models.py with identical fields and validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from darkbreaker_sdk.schemas.detection import BoundingBox  # noqa: F401 - needed for model references


def _generate_id() -> str:
    """Generate a unique ID."""
    from uuid import uuid4
    return str(uuid4())


class AlarmLevel(str, Enum):
    """Alarm severity level."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlarmStatus(str, Enum):
    """Alarm status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class AlarmRule(BaseModel):
    """Alarm rule definition."""
    id: str = Field(default_factory=_generate_id)
    name: str
    condition: str
    level: str = "warning"
    message_template: str = ""
    enabled: bool = True


class Alarm(BaseModel):
    """Alarm model."""
    id: str = Field(default_factory=_generate_id)
    task_id: str
    result_id: Optional[str] = None
    rule_id: Optional[str] = None
    level: AlarmLevel = AlarmLevel.WARNING
    status: AlarmStatus = AlarmStatus.ACTIVE
    title: str
    message: str
    site_id: str
    device_id: str
    component_id: str = ""
    evidence_path: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: str = ""
    resolved_by: str = ""
    notes: str = ""
