"""
Common schemas.

Contains BaseEntity, ROI, ROIType, Evidence, EvidenceType, and generate_id -
extracted from platform_core/schema/models.py with identical fields and validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from darkbreaker_sdk.schemas.detection import BoundingBox
from darkbreaker_sdk.schemas.alarm import AlarmRule


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid4())


class BaseEntity(BaseModel):
    """Base entity model."""
    id: str = Field(default_factory=generate_id)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ROIType(str, Enum):
    """ROI recognition type."""
    DEFECT = "defect"
    STATE = "state"
    METER = "meter"
    THERMAL = "thermal"
    INTRUSION = "intrusion"


class ROI(BaseEntity):
    """Region of interest model."""
    component_id: str
    roi_type: ROIType
    bbox: BoundingBox
    recognition_types: list[str] = Field(default_factory=list)
    rules: list[AlarmRule] = Field(default_factory=list)


class EvidenceType(str, Enum):
    """Evidence type."""
    RAW_IMAGE = "raw_image"
    ANNOTATED_IMAGE = "annotated_image"
    VIDEO_CLIP = "video_clip"
    THERMAL_IMAGE = "thermal_image"
    LOG = "log"
    RESULT_JSON = "result_json"


class Evidence(BaseModel):
    """Evidence record."""
    id: str = Field(default_factory=generate_id)
    run_id: str
    task_id: str
    evidence_type: EvidenceType
    file_path: str
    file_size: int = 0
    checksum: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
