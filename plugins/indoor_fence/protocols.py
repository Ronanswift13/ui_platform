"""
Indoor Fence V3.0 - Data Protocols
===================================

All inter-layer communication uses these Pydantic models.
Ensures type safety, serialization, and clear API contracts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# --- Enums ---

class SensorType(str, Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    UWB = "uwb"
    IMU = "imu"
    BLE = "ble"


class AdapterMode(str, Enum):
    LIVE = "live"
    SIMULATED = "simulated"
    REPLAY = "replay"


class PersonStateV3(str, Enum):
    NORMAL = "normal"
    ON_LINE = "on_line"
    CROSS_LINE = "cross_line"
    MISPLACED = "misplaced"
    HIGH_RISK = "high_risk"
    CLIMBING = "climbing"
    PROLONGED_STAY = "prolonged_stay"
    FALLEN = "fallen"
    MULTI_PERSON = "multi_person"


class GlobalAlarmLevelV3(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    CRITICAL = "critical"


# --- Sensor Data ---

class SensorData(BaseModel):
    """Unified sensor output format."""
    sensor_type: SensorType
    timestamp: float
    data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    is_simulated: bool = False


# --- Detection ---

class PoseKeypoint(BaseModel):
    """Single body keypoint."""
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)
    name: str = ""


class DetectionResult(BaseModel):
    """Unified detection output from Layer 2."""
    track_id: int
    position_3d: Tuple[float, float, float]
    velocity_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: Optional[Tuple[int, int, int, int]] = None
    pose_keypoints: Optional[List[PoseKeypoint]] = None
    behavior: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    fusion_sources: List[str] = Field(default_factory=list)
    class_label: str = "person"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- Fusion ---

class FusionInput(BaseModel):
    """Input to fusion layer from a single sensor."""
    sensor_type: SensorType
    timestamp: float
    position_2d: Optional[Tuple[float, float]] = None
    position_3d: Optional[Tuple[float, float, float]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class FusionOutput(BaseModel):
    """Output from fusion layer per tracked target."""
    track_id: int
    position_3d: Tuple[float, float, float]
    velocity_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[SensorType] = Field(default_factory=list)
    covariance_diag: Optional[Tuple[float, float, float]] = None


# --- Risk Assessment ---

class RiskAssessment(BaseModel):
    """Output from state machine + rule engine."""
    person_state: PersonStateV3
    risk_score: float = Field(ge=0.0, le=1.0)
    zone_id: Optional[str] = None
    violations: List[str] = Field(default_factory=list)
    recommended_action: str = "none"
    metadata: Dict[str, Any] = Field(default_factory=dict)
