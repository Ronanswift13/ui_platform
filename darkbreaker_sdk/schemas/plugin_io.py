"""
Plugin I/O schema.

Contains PluginOutput model - extracted from platform_core/schema/models.py
with identical fields and validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from darkbreaker_sdk.schemas.alarm import Alarm
from darkbreaker_sdk.schemas.detection import RecognitionResult


class PluginOutput(BaseModel):
    """Plugin standard output format."""
    task_id: str
    plugin_id: str
    plugin_version: str
    code_hash: str
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    results: list[RecognitionResult] = Field(default_factory=list)
    alarms: list[Alarm] = Field(default_factory=list)
    error_message: str = ""
    error_code: Optional[str] = None
    processing_time_ms: float = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
