"""
Detection schemas.

Contains BoundingBox and RecognitionResult models - extracted from
platform_core/schema/models.py with identical fields and validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Bounding box with normalized coordinates (0-1)."""
    x: float  # Top-left x (normalized 0-1)
    y: float  # Top-left y (normalized 0-1)
    width: float
    height: float

    @field_validator("x", "y", "width", "height")
    @classmethod
    def validate_normalized(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Coordinate values must be between 0 and 1")
        return v


class RecognitionResult(BaseModel):
    """Single recognition result - minimum output unit of a plugin."""
    task_id: str
    site_id: str
    device_id: str
    component_id: str
    roi_id: str
    bbox: BoundingBox
    label: str
    value: Optional[Any] = None
    confidence: float = Field(ge=0, le=1)
    evidence_path: str = ""
    model_version: str = ""
    code_version: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    failure_reason: Optional[str] = None
