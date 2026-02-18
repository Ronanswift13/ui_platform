# DarkBreaker Plugin SDK Architecture Refactoring - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract a lightweight `darkbreaker_sdk` package from `platform_core`, migrate all 17 plugins to import from SDK only, and give every plugin standalone running capability (FastAPI server + Bootstrap UI + demo scripts + tests).

**Architecture:** Three-layer separation: `darkbreaker_sdk` (interfaces + schemas + standalone runner) -> `plugins/*` (each self-contained with standalone/) -> `platform_core` (orchestration only, depends on SDK). All algorithms in plugin `core/` directories are preserved unchanged.

**Tech Stack:** Python 3.10+, FastAPI, Jinja2, Bootstrap 5, Pydantic v2, ONNX Runtime, pytest

---

## Phase 1: Create darkbreaker_sdk Package (Foundation)

### Task 1: SDK Package Skeleton

**Files:**
- Create: `darkbreaker_sdk/__init__.py`
- Create: `darkbreaker_sdk/interfaces/__init__.py`
- Create: `darkbreaker_sdk/schemas/__init__.py`
- Create: `darkbreaker_sdk/standalone/__init__.py`
- Create: `darkbreaker_sdk/utils/__init__.py`

**Step 1: Create the SDK directory structure**

```bash
mkdir -p darkbreaker_sdk/interfaces
mkdir -p darkbreaker_sdk/schemas
mkdir -p darkbreaker_sdk/standalone/templates/components
mkdir -p darkbreaker_sdk/standalone/static/css
mkdir -p darkbreaker_sdk/standalone/static/js
mkdir -p darkbreaker_sdk/utils
```

**Step 2: Create `darkbreaker_sdk/__init__.py`**

```python
"""
DarkBreaker SDK - Plugin Development Kit
=========================================

Lightweight SDK for building standalone DarkBreaker plugins.
Plugins depend ONLY on this package, not on platform_core.

Usage:
    from darkbreaker_sdk.interfaces import BasePlugin, PluginManifest, PluginContext
    from darkbreaker_sdk.schemas import RecognitionResult, Alarm, BoundingBox
    from darkbreaker_sdk.standalone import StandalonePluginRunner
"""

__version__ = "1.0.0"
__sdk_name__ = "darkbreaker-sdk"
```

**Step 3: Create `darkbreaker_sdk/interfaces/__init__.py`**

```python
"""Plugin interfaces - abstract base classes and data contracts."""

from darkbreaker_sdk.interfaces.lifecycle import (
    PluginCapability,
    PluginStatus,
    HealthStatus,
)
from darkbreaker_sdk.interfaces.base_plugin import (
    BasePlugin,
    PluginManifest,
    PluginContext,
)
from darkbreaker_sdk.interfaces.base_adapter import BaseAdapter, AdapterStatus

__all__ = [
    "BasePlugin",
    "PluginManifest",
    "PluginContext",
    "PluginCapability",
    "PluginStatus",
    "HealthStatus",
    "BaseAdapter",
    "AdapterStatus",
]
```

**Step 4: Create `darkbreaker_sdk/schemas/__init__.py`**

```python
"""Data models for plugin input/output."""

from darkbreaker_sdk.schemas.detection import BoundingBox, RecognitionResult
from darkbreaker_sdk.schemas.alarm import Alarm, AlarmLevel, AlarmRule, AlarmStatus
from darkbreaker_sdk.schemas.plugin_io import PluginOutput
from darkbreaker_sdk.schemas.common import (
    ROI,
    ROIType,
    BaseEntity,
    Evidence,
    EvidenceType,
    generate_id,
)

__all__ = [
    "BoundingBox",
    "RecognitionResult",
    "Alarm",
    "AlarmLevel",
    "AlarmRule",
    "AlarmStatus",
    "PluginOutput",
    "ROI",
    "ROIType",
    "BaseEntity",
    "Evidence",
    "EvidenceType",
    "generate_id",
]
```

**Step 5: Create `darkbreaker_sdk/standalone/__init__.py`**

```python
"""Standalone plugin runner - run any plugin as independent FastAPI app."""

from darkbreaker_sdk.standalone.runner import StandalonePluginRunner

__all__ = ["StandalonePluginRunner"]
```

**Step 6: Create `darkbreaker_sdk/utils/__init__.py`**

```python
"""Utility functions for plugin development."""

from darkbreaker_sdk.utils.logging import setup_plugin_logger
from darkbreaker_sdk.utils.config import load_plugin_config
from darkbreaker_sdk.utils.model_loader import load_onnx_model

__all__ = ["setup_plugin_logger", "load_plugin_config", "load_onnx_model"]
```

**Step 7: Commit**

```bash
git add darkbreaker_sdk/
git commit -m "feat(sdk): create darkbreaker_sdk package skeleton"
```

---

### Task 2: SDK Lifecycle Interfaces

**Files:**
- Create: `darkbreaker_sdk/interfaces/lifecycle.py`
- Source: Extract from `platform_core/plugin_manager/base.py` lines 32-173

**Step 1: Write test**

```python
# tests/sdk/test_lifecycle.py
"""Tests for SDK lifecycle interfaces."""
import pytest
from darkbreaker_sdk.interfaces.lifecycle import (
    PluginCapability, PluginStatus, HealthStatus,
)


def test_plugin_status_values():
    assert PluginStatus.UNLOADED == "unloaded"
    assert PluginStatus.READY == "ready"
    assert PluginStatus.RUNNING == "running"
    assert PluginStatus.ERROR == "error"


def test_health_status_creation():
    hs = HealthStatus(healthy=True, message="OK")
    assert hs.healthy is True
    assert hs.message == "OK"
    assert hs.details == {}


def test_health_status_unhealthy():
    hs = HealthStatus(healthy=False, message="Model not loaded", details={"model": "missing"})
    assert hs.healthy is False
    assert hs.details["model"] == "missing"


def test_plugin_capability_enum():
    assert PluginCapability.DEFECT_DETECTION == "defect_detection"
    assert PluginCapability.ANIMAL_DETECTION == "animal_detection"
    assert PluginCapability.FIRE_DETECTION == "fire_detection"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sdk/test_lifecycle.py -v`
Expected: FAIL (module not found)

**Step 3: Create `darkbreaker_sdk/interfaces/lifecycle.py`**

Copy the `PluginCapability`, `PluginStatus`, and `HealthStatus` classes from `platform_core/plugin_manager/base.py` (lines 32-173). These are self-contained with only stdlib dependencies.

```python
"""
Plugin lifecycle types - status, capabilities, health.

Extracted from platform_core/plugin_manager/base.py.
Zero external dependencies beyond stdlib.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PluginCapability(str, Enum):
    """Plugin capability enumeration."""
    DEFECT_DETECTION = "defect_detection"
    STATE_RECOGNITION = "state_recognition"
    METER_READING = "meter_reading"
    THERMAL_ANALYSIS = "thermal_analysis"
    INTRUSION_DETECTION = "intrusion_detection"
    IMAGE_QUALITY = "image_quality"
    FOCUS_SUGGESTION = "focus_suggestion"
    BIRD_DETECTION = "bird_detection"
    SPECIES_IDENTIFICATION = "species_identification"
    RISK_ASSESSMENT = "risk_assessment"
    DETERRENT_CONTROL = "deterrent_control"
    PERSON_DETECTION = "person_detection"
    MULTI_TARGET_TRACKING = "multi_target_tracking"
    ZONE_INTRUSION = "zone_intrusion"
    AUTHORIZATION_CHECK = "authorization_check"
    LIDAR_FENCE = "lidar_fence"
    PARTIAL_DISCHARGE_DETECTION = "partial_discharge_detection"
    ACOUSTIC_MONITORING = "acoustic_monitoring"
    GAS_CONCENTRATION_MONITORING = "gas_concentration_monitoring"
    LEAKAGE_DETECTION = "leakage_detection"
    HYPERSPECTRAL_ANALYSIS = "hyperspectral_analysis"
    POINT_CLOUD_PROCESSING = "point_cloud_processing"
    PATH_PLANNING = "path_planning"
    MULTIMODAL_DATA_FUSION = "multimodal_data_fusion"
    ANIMAL_DETECTION = "animal_detection"
    SPECIES_CLASSIFICATION = "species_classification"
    THERMAL_FUSION_DETECTION = "thermal_fusion_detection"
    BEHAVIOR_TRACKING = "behavior_tracking"
    INTRUSION_STATISTICS = "intrusion_statistics"
    THERMAL_IMAGING = "thermal_imaging"
    HOTSPOT_DETECTION = "hotspot_detection"
    TEMPERATURE_TREND_ANALYSIS = "temperature_trend_analysis"
    HEATMAP_GENERATION = "heatmap_generation"
    TEMPERATURE_PREDICTION = "temperature_prediction"
    CROSS_MODULE_LINKAGE = "cross_module_linkage"
    DATA_ARCHIVING = "data_archiving"
    DEVICE_STATUS_MONITORING = "device_status_monitoring"
    HEALTH_INDEX_CALCULATION = "health_index_calculation"
    FAULT_PREDICTION = "fault_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    STATISTICS_REPORTING = "statistics_reporting"
    PROTOCOL_INTEGRATION = "protocol_integration"
    FIRE_DETECTION = "fire_detection"
    SMOKE_DETECTION = "smoke_detection"
    THERMAL_ANOMALY_DETECTION = "thermal_anomaly_detection"
    MULTI_SENSOR_FUSION = "multi_sensor_fusion"
    ACTIVE_SUPPRESSION_CONTROL = "active_suppression_control"
    EVACUATION_GUIDANCE = "evacuation_guidance"
    DRILL_SIMULATION = "drill_simulation"


class PluginStatus(str, Enum):
    """Plugin lifecycle status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class HealthStatus:
    """Plugin health check response."""
    healthy: bool
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/sdk/test_lifecycle.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add darkbreaker_sdk/interfaces/lifecycle.py tests/sdk/
git commit -m "feat(sdk): add lifecycle interfaces - PluginStatus, PluginCapability, HealthStatus"
```

---

### Task 3: SDK Schema Models

**Files:**
- Create: `darkbreaker_sdk/schemas/detection.py`
- Create: `darkbreaker_sdk/schemas/alarm.py`
- Create: `darkbreaker_sdk/schemas/plugin_io.py`
- Create: `darkbreaker_sdk/schemas/common.py`
- Source: Extract from `platform_core/schema/models.py`

**Step 1: Write tests**

```python
# tests/sdk/test_schemas.py
"""Tests for SDK schema models."""
import pytest
from darkbreaker_sdk.schemas import (
    BoundingBox, RecognitionResult, Alarm, AlarmLevel, AlarmRule,
    PluginOutput, ROI, ROIType, BaseEntity, generate_id,
)


def test_bounding_box_creation():
    bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
    assert bbox.x == 0.1
    assert bbox.width == 0.3


def test_bounding_box_validation():
    with pytest.raises(ValueError):
        BoundingBox(x=1.5, y=0.2, width=0.3, height=0.4)


def test_recognition_result():
    result = RecognitionResult(
        task_id="t1", site_id="s1", device_id="d1",
        component_id="c1", roi_id="r1",
        bbox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
        label="defect", confidence=0.95,
    )
    assert result.label == "defect"
    assert result.confidence == 0.95


def test_alarm_creation():
    alarm = Alarm(
        task_id="t1", title="Test", message="Test alarm",
        site_id="s1", device_id="d1",
        level=AlarmLevel.WARNING,
    )
    assert alarm.level == AlarmLevel.WARNING
    assert alarm.id  # auto-generated


def test_alarm_rule():
    rule = AlarmRule(name="temp_high", condition="value > 80")
    assert rule.enabled is True


def test_plugin_output():
    output = PluginOutput(
        task_id="t1", plugin_id="test", plugin_version="1.0.0", code_hash="abc123",
    )
    assert output.success is True
    assert output.results == []
    assert output.alarms == []


def test_roi_type():
    assert ROIType.DEFECT == "defect"
    assert ROIType.THERMAL == "thermal"


def test_generate_id():
    id1 = generate_id()
    id2 = generate_id()
    assert id1 != id2
    assert len(id1) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sdk/test_schemas.py -v`
Expected: FAIL

**Step 3: Create `darkbreaker_sdk/schemas/common.py`**

Extract `BaseEntity`, `ROI`, `ROIType`, `Evidence`, `EvidenceType`, `generate_id` from `platform_core/schema/models.py`. Keep Pydantic v2 models exactly as-is.

```python
"""Common schema models shared across all plugins."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def generate_id() -> str:
    """Generate unique ID."""
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


# NOTE: ROI depends on BoundingBox and AlarmRule, imported at module level
# to avoid circular imports. Full ROI definition is here but uses forward refs.
```

**Step 4: Create `darkbreaker_sdk/schemas/detection.py`**

```python
"""Detection result models."""

from __future__ import annotations
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Normalized bounding box (0-1 coordinates)."""
    x: float
    y: float
    width: float
    height: float

    @field_validator("x", "y", "width", "height")
    @classmethod
    def validate_normalized(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Coordinate values must be between 0 and 1")
        return v


class RecognitionResult(BaseModel):
    """Single recognition result - minimum output unit from a plugin."""
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
```

**Step 5: Create `darkbreaker_sdk/schemas/alarm.py`**

```python
"""Alarm and alert models."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from darkbreaker_sdk.schemas.common import generate_id


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
    """Alarm triggering rule."""
    id: str = Field(default_factory=generate_id)
    name: str
    condition: str
    level: str = "warning"
    message_template: str = ""
    enabled: bool = True


class Alarm(BaseModel):
    """Alarm event."""
    id: str = Field(default_factory=generate_id)
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
```

**Step 6: Create `darkbreaker_sdk/schemas/plugin_io.py`**

```python
"""Plugin input/output models."""

from __future__ import annotations
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from darkbreaker_sdk.schemas.detection import RecognitionResult
from darkbreaker_sdk.schemas.alarm import Alarm


class PluginOutput(BaseModel):
    """Standard plugin output format."""
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
```

**Step 7: Finish `darkbreaker_sdk/schemas/common.py` - add ROI**

Append to `common.py`:

```python
from darkbreaker_sdk.schemas.detection import BoundingBox
from darkbreaker_sdk.schemas.alarm import AlarmRule


class ROI(BaseEntity):
    """Region of Interest."""
    component_id: str
    roi_type: ROIType
    bbox: BoundingBox
    recognition_types: list[str] = Field(default_factory=list)
    rules: list[AlarmRule] = Field(default_factory=list)
```

**Step 8: Run tests**

Run: `pytest tests/sdk/test_schemas.py -v`
Expected: PASS

**Step 9: Commit**

```bash
git add darkbreaker_sdk/schemas/ tests/sdk/test_schemas.py
git commit -m "feat(sdk): add schema models - BoundingBox, RecognitionResult, Alarm, ROI, PluginOutput"
```

---

### Task 4: SDK Base Plugin Interface

**Files:**
- Create: `darkbreaker_sdk/interfaces/base_plugin.py`
- Create: `darkbreaker_sdk/interfaces/base_adapter.py`
- Source: Extract from `platform_core/plugin_manager/base.py` lines 105-343

**Step 1: Write test**

```python
# tests/sdk/test_base_plugin.py
"""Tests for SDK BasePlugin interface."""
import pytest
import numpy as np
from pathlib import Path
from darkbreaker_sdk.interfaces import (
    BasePlugin, PluginManifest, PluginContext, HealthStatus, PluginStatus,
    BaseAdapter, AdapterStatus,
)
from darkbreaker_sdk.schemas import RecognitionResult, Alarm, AlarmRule, ROI


class MockPlugin(BasePlugin):
    """Minimal plugin implementation for testing."""

    def init(self, config):
        self._config = config
        return True

    def infer(self, frame, rois, context):
        return []

    def postprocess(self, results, rules):
        return []

    def healthcheck(self):
        return HealthStatus(healthy=True, message="OK")


def test_plugin_manifest_from_dict():
    data = {
        "id": "test_plugin",
        "name": "Test Plugin",
        "version": "1.0.0",
        "capabilities": ["defect_detection"],
    }
    manifest = PluginManifest.from_dict(data)
    assert manifest.id == "test_plugin"
    assert manifest.version == "1.0.0"


def test_plugin_manifest_from_file(tmp_path):
    import json
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "id": "file_plugin", "name": "File Plugin", "version": "2.0.0",
    }))
    manifest = PluginManifest.from_file(manifest_file)
    assert manifest.id == "file_plugin"


def test_plugin_context():
    ctx = PluginContext(task_id="t1", site_id="s1", device_id="d1")
    d = ctx.to_dict()
    assert d["task_id"] == "t1"


def test_mock_plugin_lifecycle(tmp_path):
    manifest = PluginManifest.from_dict({
        "id": "mock", "name": "Mock", "version": "1.0.0",
    })
    plugin = MockPlugin(manifest, tmp_path)
    assert plugin.status == PluginStatus.UNLOADED

    assert plugin.init({}) is True
    assert plugin.id == "mock"
    assert plugin.version == "1.0.0"


def test_plugin_create_standalone(tmp_path):
    """Test standalone factory method."""
    import json
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "id": "standalone_test", "name": "Standalone", "version": "1.0.0",
    }))
    # create_standalone is tested via subclass in real plugins


def test_plugin_create_output(tmp_path):
    manifest = PluginManifest.from_dict({
        "id": "mock", "name": "Mock", "version": "1.0.0",
    })
    plugin = MockPlugin(manifest, tmp_path)
    output = plugin.create_output("task1", [], [], processing_time_ms=42.0)
    assert output.task_id == "task1"
    assert output.plugin_id == "mock"
    assert output.processing_time_ms == 42.0


def test_adapter_status():
    assert AdapterStatus.DISCONNECTED == "disconnected"
    assert AdapterStatus.CONNECTED == "connected"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sdk/test_base_plugin.py -v`
Expected: FAIL

**Step 3: Create `darkbreaker_sdk/interfaces/base_plugin.py`**

Extract from `platform_core/plugin_manager/base.py`, replacing all `platform_core` imports with `darkbreaker_sdk` imports. Add `create_standalone()` classmethod and `get_standalone_routes()` method.

```python
"""
Base plugin interface.

All plugins must inherit BasePlugin and implement:
1. init(config) -> bool
2. infer(frame, rois, context) -> results
3. postprocess(results, rules) -> alarms
4. healthcheck() -> status
"""

from __future__ import annotations
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from darkbreaker_sdk.interfaces.lifecycle import (
    PluginCapability,
    PluginStatus,
    HealthStatus,
)
from darkbreaker_sdk.schemas import (
    Alarm,
    AlarmRule,
    PluginOutput,
    RecognitionResult,
    ROI,
)


@dataclass
class PluginManifest:
    """Plugin manifest - loaded from manifest.json."""
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    entrypoint: str = "plugin.py"
    plugin_class: str = "Plugin"
    capabilities: list[PluginCapability] = field(default_factory=list)
    device_types: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    min_platform_version: str = "1.0.0"
    config_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginManifest":
        """Create manifest from dictionary."""
        capabilities = []
        for c in data.get("capabilities", []):
            try:
                capabilities.append(
                    PluginCapability(c) if isinstance(c, str) else c
                )
            except ValueError:
                pass  # Skip unknown capabilities for forward compatibility
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            entrypoint=data.get("entrypoint", "plugin.py"),
            plugin_class=data.get("plugin_class", "Plugin"),
            capabilities=capabilities,
            device_types=data.get("device_types", []),
            dependencies=data.get("dependencies", []),
            min_platform_version=data.get("min_platform_version", "1.0.0"),
            config_schema=data.get("config_schema", {}),
        )

    @classmethod
    def from_file(cls, path: Path) -> "PluginManifest":
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class PluginContext:
    """Plugin runtime context."""
    task_id: str
    site_id: str
    device_id: str
    component_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "site_id": self.site_id,
            "device_id": self.device_id,
            "component_id": self.component_id,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config,
            "metadata": self.metadata,
        }


class BasePlugin(ABC):
    """
    Base class for all DarkBreaker plugins.

    All plugins must inherit this class and implement:
    - init(config): Initialize the plugin
    - infer(frame, rois, context): Run inference
    - postprocess(results, rules): Post-process and generate alarms
    - healthcheck(): Health check
    """

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        self.manifest = manifest
        self.plugin_dir = plugin_dir
        self.status = PluginStatus.UNLOADED
        self._config: dict[str, Any] = {}
        self._code_hash: str = ""
        self._last_error: str = ""

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def code_hash(self) -> str:
        if not self._code_hash:
            self._code_hash = self._calculate_code_hash()
        return self._code_hash

    def _calculate_code_hash(self) -> str:
        hasher = hashlib.sha256()
        if self.plugin_dir.exists():
            for py_file in sorted(self.plugin_dir.rglob("*.py")):
                hasher.update(py_file.read_bytes())
        return hasher.hexdigest()[:12]

    # ============== Required interface ==============

    @abstractmethod
    def init(self, config: dict[str, Any]) -> bool:
        """Initialize the plugin."""
        pass

    @abstractmethod
    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """Run inference on a frame."""
        pass

    @abstractmethod
    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """Post-process results and generate alarms."""
        pass

    @abstractmethod
    def healthcheck(self) -> HealthStatus:
        """Health check."""
        pass

    # ============== Optional interface ==============

    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def on_config_update(self, new_config: dict[str, Any]) -> None:
        """Config update callback."""
        self._config = new_config

    def analyze_thermal(
        self, frame: np.ndarray, config: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Thermal image analysis (optional)."""
        return None

    def get_model_version(self) -> str:
        """Get model version."""
        return self.version

    def get_ui_config(self) -> dict[str, Any] | None:
        """Get plugin UI configuration."""
        return None

    def get_standalone_routes(self) -> list:
        """
        Return plugin-specific API routes for standalone mode.

        Override in subclass to add custom endpoints.
        Returns list of (method, path, handler) tuples.
        """
        return []

    # ============== Helper methods ==============

    @classmethod
    def create_standalone(cls, config: dict[str, Any] | None = None) -> "BasePlugin":
        """
        Factory method to create plugin instance without PluginManager.

        Usage:
            plugin = MyPlugin.create_standalone({"threshold": 0.5})
        """
        plugin_dir = Path(__file__).parent
        # Look for manifest.json in the plugin directory
        # Subclasses should override to point to their own directory
        manifest_path = plugin_dir / "manifest.json"
        if manifest_path.exists():
            manifest = PluginManifest.from_file(manifest_path)
        else:
            manifest = PluginManifest(
                id=cls.__module__.split(".")[-2] if "." in cls.__module__ else "unknown",
                name=cls.__name__,
                version="0.0.0",
            )
        instance = cls(manifest, plugin_dir)
        instance.init(config or {})
        return instance

    def create_output(
        self,
        task_id: str,
        results: list[RecognitionResult],
        alarms: list[Alarm],
        processing_time_ms: float = 0,
        success: bool = True,
        error_message: str = "",
    ) -> PluginOutput:
        """Create standard plugin output."""
        return PluginOutput(
            task_id=task_id,
            plugin_id=self.id,
            plugin_version=self.version,
            code_hash=self.code_hash,
            success=success,
            results=results,
            alarms=alarms,
            error_message=error_message,
            processing_time_ms=processing_time_ms,
        )

    def set_status(self, status: PluginStatus, error: str = "") -> None:
        """Set plugin status."""
        self.status = status
        if error:
            self._last_error = error

    def __repr__(self) -> str:
        return f"<Plugin {self.id} v{self.version} [{self.status.value}]>"
```

**Step 4: Create `darkbreaker_sdk/interfaces/base_adapter.py`**

```python
"""
Base adapter interface for hardware devices.

Formalizes the adapter pattern used by indoor_fence plugin.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class AdapterStatus(str, Enum):
    """Device adapter status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    SIMULATED = "simulated"


class BaseAdapter(ABC):
    """Abstract base class for device adapters (camera, lidar, etc.)."""

    def __init__(self):
        self._status = AdapterStatus.DISCONNECTED
        self._is_simulated = False

    @property
    def status(self) -> AdapterStatus:
        return self._status

    @property
    def is_simulated(self) -> bool:
        return self._is_simulated

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the device."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the device."""
        pass

    def get_info(self) -> dict[str, Any]:
        """Get device information."""
        return {
            "status": self._status.value,
            "simulated": self._is_simulated,
        }
```

**Step 5: Run tests**

Run: `pytest tests/sdk/test_base_plugin.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add darkbreaker_sdk/interfaces/ tests/sdk/test_base_plugin.py
git commit -m "feat(sdk): add BasePlugin, PluginManifest, PluginContext, BaseAdapter interfaces"
```

---

### Task 5: SDK Utilities

**Files:**
- Create: `darkbreaker_sdk/utils/logging.py`
- Create: `darkbreaker_sdk/utils/config.py`
- Create: `darkbreaker_sdk/utils/model_loader.py`

**Step 1: Write tests**

```python
# tests/sdk/test_utils.py
"""Tests for SDK utility functions."""
import pytest
import logging
from pathlib import Path
from darkbreaker_sdk.utils import setup_plugin_logger, load_plugin_config


def test_setup_plugin_logger():
    logger = setup_plugin_logger("test_plugin")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "darkbreaker.test_plugin"


def test_load_plugin_config_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("threshold: 0.5\nmax_detections: 10\n")
    config = load_plugin_config(config_file)
    assert config["threshold"] == 0.5
    assert config["max_detections"] == 10


def test_load_plugin_config_json(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"threshold": 0.8}')
    config = load_plugin_config(config_file)
    assert config["threshold"] == 0.8


def test_load_plugin_config_missing():
    config = load_plugin_config(Path("/nonexistent/config.yaml"))
    assert config == {}
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/sdk/test_utils.py -v`
Expected: FAIL

**Step 3: Create utility files**

`darkbreaker_sdk/utils/logging.py`:
```python
"""Standard Python logging setup for plugins."""
import logging
import sys


def setup_plugin_logger(
    plugin_id: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
) -> logging.Logger:
    """
    Set up a standard Python logger for a plugin.

    Args:
        plugin_id: Plugin identifier (used as logger name suffix)
        level: Logging level
        fmt: Log format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"darkbreaker.{plugin_id}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
```

`darkbreaker_sdk/utils/config.py`:
```python
"""Configuration loading utilities."""
import json
from pathlib import Path
from typing import Any


def load_plugin_config(path: Path) -> dict[str, Any]:
    """
    Load plugin configuration from YAML or JSON file.

    Returns empty dict if file doesn't exist or can't be parsed.
    """
    if not path.exists():
        return {}

    try:
        content = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(content) or {}
        elif path.suffix == ".json":
            return json.loads(content)
        else:
            return {}
    except Exception:
        return {}
```

`darkbreaker_sdk/utils/model_loader.py`:
```python
"""ONNX model loading utilities."""
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_onnx_model(
    model_path: Path,
    providers: list[str] | None = None,
) -> Optional[Any]:
    """
    Load an ONNX model with optional GPU acceleration.

    Args:
        model_path: Path to .onnx model file
        providers: ONNX Runtime execution providers

    Returns:
        ONNX InferenceSession or None if loading fails
    """
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None

    try:
        import onnxruntime as ort
        if providers is None:
            providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=providers)
        logger.info(f"Loaded model: {model_path.name}")
        return session
    except ImportError:
        logger.warning("onnxruntime not installed, model loading skipped")
        return None
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None
```

**Step 4: Run tests**

Run: `pytest tests/sdk/test_utils.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add darkbreaker_sdk/utils/ tests/sdk/test_utils.py
git commit -m "feat(sdk): add utility modules - logging, config, model_loader"
```

---

### Task 6: SDK Standalone Runner

**Files:**
- Create: `darkbreaker_sdk/standalone/runner.py`
- Create: `darkbreaker_sdk/standalone/templates/base_standalone.html`
- Create: `darkbreaker_sdk/standalone/templates/plugin_dashboard.html`
- Create: `darkbreaker_sdk/standalone/templates/components/video_panel.html`
- Create: `darkbreaker_sdk/standalone/templates/components/detection_list.html`
- Create: `darkbreaker_sdk/standalone/templates/components/alarm_panel.html`
- Create: `darkbreaker_sdk/standalone/templates/components/config_editor.html`
- Create: `darkbreaker_sdk/standalone/templates/components/stats_panel.html`
- Create: `darkbreaker_sdk/standalone/static/css/standalone.css`
- Create: `darkbreaker_sdk/standalone/static/js/standalone.js`

**Step 1: Write test**

```python
# tests/sdk/test_standalone_runner.py
"""Tests for standalone plugin runner."""
import pytest
from unittest.mock import MagicMock
from darkbreaker_sdk.standalone.runner import StandalonePluginRunner
from darkbreaker_sdk.interfaces import BasePlugin, PluginManifest, HealthStatus


class FakePlugin(BasePlugin):
    def init(self, config): return True
    def infer(self, frame, rois, context): return []
    def postprocess(self, results, rules): return []
    def healthcheck(self): return HealthStatus(healthy=True)
    def get_ui_config(self):
        return {"detection_types": ["test"]}


def test_runner_creation(tmp_path):
    manifest = PluginManifest.from_dict({"id": "fake", "name": "Fake", "version": "1.0.0"})
    plugin = FakePlugin(manifest, tmp_path)
    plugin.init({})
    runner = StandalonePluginRunner(plugin)
    assert runner.app is not None


def test_runner_has_routes(tmp_path):
    manifest = PluginManifest.from_dict({"id": "fake", "name": "Fake", "version": "1.0.0"})
    plugin = FakePlugin(manifest, tmp_path)
    plugin.init({})
    runner = StandalonePluginRunner(plugin)
    routes = [r.path for r in runner.app.routes]
    assert "/api/status" in routes
    assert "/api/config" in routes
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sdk/test_standalone_runner.py -v`
Expected: FAIL

**Step 3: Create `darkbreaker_sdk/standalone/runner.py`**

```python
"""
Standalone Plugin Runner

Generic FastAPI server that can host ANY DarkBreaker plugin independently.

Usage:
    from darkbreaker_sdk.standalone import StandalonePluginRunner

    plugin = MyPlugin.create_standalone(config)
    runner = StandalonePluginRunner(plugin)
    runner.run(port=8080)
"""

from __future__ import annotations
import asyncio
import base64
import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from darkbreaker_sdk.interfaces.base_plugin import BasePlugin

logger = logging.getLogger(__name__)

# SDK template/static directories
_SDK_DIR = Path(__file__).parent
_SDK_TEMPLATES = _SDK_DIR / "templates"
_SDK_STATIC = _SDK_DIR / "static"


class StandalonePluginRunner:
    """
    Run any DarkBreaker plugin as a standalone FastAPI application.

    Provides:
    - Web dashboard UI (Bootstrap + Jinja2)
    - REST API for detection, config, status
    - WebSocket for real-time streaming
    - Plugin-specific routes via get_standalone_routes()
    """

    def __init__(
        self,
        plugin: BasePlugin,
        plugin_templates_dir: Path | None = None,
        plugin_static_dir: Path | None = None,
        title: str | None = None,
    ):
        self.plugin = plugin
        self.title = title or f"{plugin.name} - Standalone"

        self.app = FastAPI(title=self.title, version=plugin.version)

        # Template directories: plugin-specific first, then SDK defaults
        template_dirs = []
        if plugin_templates_dir and plugin_templates_dir.exists():
            template_dirs.append(str(plugin_templates_dir))
        template_dirs.append(str(_SDK_TEMPLATES))
        self.templates = Jinja2Templates(directory=template_dirs)

        # Static files
        if plugin_static_dir and plugin_static_dir.exists():
            self.app.mount(
                "/static/plugin", StaticFiles(directory=str(plugin_static_dir)),
                name="plugin_static",
            )
        if _SDK_STATIC.exists():
            self.app.mount(
                "/static/sdk", StaticFiles(directory=str(_SDK_STATIC)),
                name="sdk_static",
            )

        # Detection statistics
        self._stats = {
            "total_detections": 0,
            "total_alarms": 0,
            "total_frames": 0,
            "start_time": datetime.now().isoformat(),
            "last_detection_time": None,
            "avg_inference_ms": 0,
        }

        # WebSocket connections
        self._ws_clients: list[WebSocket] = []

        self._register_routes()
        self._register_plugin_routes()

    def _register_routes(self):
        """Register standard API routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            plugin_template = f"{self.plugin.id}.html"
            # Try plugin-specific template first, fall back to generic
            try:
                return self.templates.TemplateResponse(
                    plugin_template,
                    {
                        "request": request,
                        "plugin_id": self.plugin.id,
                        "plugin_name": self.plugin.name,
                        "plugin_version": self.plugin.version,
                        "ui_config": self.plugin.get_ui_config() or {},
                    },
                )
            except Exception:
                return self.templates.TemplateResponse(
                    "plugin_dashboard.html",
                    {
                        "request": request,
                        "plugin_id": self.plugin.id,
                        "plugin_name": self.plugin.name,
                        "plugin_version": self.plugin.version,
                        "ui_config": self.plugin.get_ui_config() or {},
                    },
                )

        @self.app.get("/api/status")
        async def get_status():
            health = self.plugin.healthcheck()
            return {
                "plugin_id": self.plugin.id,
                "plugin_name": self.plugin.name,
                "version": self.plugin.version,
                "status": self.plugin.status.value if hasattr(self.plugin.status, 'value') else str(self.plugin.status),
                "healthy": health.healthy,
                "message": health.message,
                "stats": self._stats,
            }

        @self.app.get("/api/config")
        async def get_config():
            return {
                "plugin_id": self.plugin.id,
                "config": self.plugin._config,
                "ui_config": self.plugin.get_ui_config(),
            }

        @self.app.put("/api/config")
        async def update_config(new_config: dict):
            self.plugin.on_config_update(new_config)
            return {"success": True, "config": self.plugin._config}

        @self.app.post("/api/detect")
        async def detect(file: UploadFile = File(...)):
            """Run detection on an uploaded image."""
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid image file"},
                )

            from darkbreaker_sdk.interfaces.base_plugin import PluginContext
            context = PluginContext(
                task_id=f"standalone_{int(time.time())}",
                site_id="standalone",
                device_id="upload",
            )

            start = time.time()
            results = self.plugin.infer(frame, [], context)
            elapsed = (time.time() - start) * 1000

            alarms = self.plugin.postprocess(results, [])

            # Update stats
            self._stats["total_frames"] += 1
            self._stats["total_detections"] += len(results)
            self._stats["total_alarms"] += len(alarms)
            self._stats["last_detection_time"] = datetime.now().isoformat()

            # Calculate running average
            n = self._stats["total_frames"]
            self._stats["avg_inference_ms"] = (
                (self._stats["avg_inference_ms"] * (n - 1) + elapsed) / n
            )

            return {
                "success": True,
                "results": [r.model_dump() if hasattr(r, 'model_dump') else r for r in results],
                "alarms": [a.model_dump() if hasattr(a, 'model_dump') else a for a in alarms],
                "inference_time_ms": round(elapsed, 2),
                "frame_count": self._stats["total_frames"],
            }

        @self.app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            await websocket.accept()
            self._ws_clients.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Client can send commands via WebSocket
                    try:
                        cmd = json.loads(data)
                        if cmd.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                self._ws_clients.remove(websocket)

    def _register_plugin_routes(self):
        """Register plugin-specific routes from get_standalone_routes()."""
        for route_info in self.plugin.get_standalone_routes():
            if isinstance(route_info, dict):
                method = route_info.get("method", "GET").upper()
                path = route_info["path"]
                handler = route_info["handler"]
            elif isinstance(route_info, (list, tuple)) and len(route_info) == 3:
                method, path, handler = route_info
            else:
                continue

            if method == "GET":
                self.app.get(path)(handler)
            elif method == "POST":
                self.app.post(path)(handler)
            elif method == "PUT":
                self.app.put(path)(handler)
            elif method == "DELETE":
                self.app.delete(path)(handler)

    async def broadcast_ws(self, data: dict):
        """Broadcast data to all connected WebSocket clients."""
        disconnected = []
        for ws in self._ws_clients:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._ws_clients.remove(ws)

    def run(self, host: str = "0.0.0.0", port: int = 8080, **kwargs):
        """Start the standalone server."""
        import uvicorn
        logger.info(f"Starting {self.title} on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)
```

**Step 4: Create HTML templates**

Create `base_standalone.html`, `plugin_dashboard.html`, and all component templates. These use Bootstrap 5 with the same dark theme as the main platform.

> **Note to implementer:** The base template should:
> - Use the same CSS variables as `indoor_center.html` (--panel-bg: #1e293b, etc.)
> - Include Bootstrap 5 CDN, Chart.js CDN
> - Have a header showing plugin name, version, status badge
> - Three-column layout: left (controls), center (visualization), right (results)
> - Bottom bar with stats (FPS, detection count, connection status)
> - Component templates are Jinja2 includes for video panel, detection list, alarm panel, config editor, stats panel

Create `standalone.css` with the dark theme styling.
Create `standalone.js` with WebSocket connection, image upload, and result rendering logic.

**Step 5: Run tests**

Run: `pytest tests/sdk/test_standalone_runner.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add darkbreaker_sdk/standalone/ tests/sdk/test_standalone_runner.py
git commit -m "feat(sdk): add StandalonePluginRunner with FastAPI server and Bootstrap UI templates"
```

---

## Phase 2: Platform Core Compatibility Layer

### Task 7: Create Compatibility Re-exports in platform_core

**Files:**
- Modify: `platform_core/schema/models.py` - add re-exports from SDK
- Modify: `platform_core/plugin_manager/base.py` - add re-exports from SDK
- Modify: `platform_core/plugin_manager/__init__.py`

**Step 1: Write test**

```python
# tests/sdk/test_compatibility.py
"""Tests that platform_core still exports everything via SDK re-exports."""
import pytest


def test_schema_models_backward_compatible():
    """All existing imports from platform_core.schema.models must still work."""
    from platform_core.schema.models import (
        BoundingBox, RecognitionResult, Alarm, AlarmLevel, AlarmRule,
        AlarmStatus, PluginOutput, ROI, ROIType, BaseEntity,
        Evidence, EvidenceType, generate_id,
    )
    bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
    assert bbox.x == 0.1


def test_base_plugin_backward_compatible():
    """All existing imports from platform_core.plugin_manager.base must still work."""
    from platform_core.plugin_manager.base import (
        BasePlugin, HealthStatus, PluginContext, PluginManifest,
        PluginStatus, PluginCapability,
    )
    assert PluginStatus.READY == "ready"


def test_sdk_and_platform_share_identity():
    """SDK classes and platform re-exports must be the same objects."""
    from darkbreaker_sdk.schemas import BoundingBox as SDK_BB
    from platform_core.schema.models import BoundingBox as Platform_BB
    assert SDK_BB is Platform_BB

    from darkbreaker_sdk.interfaces import BasePlugin as SDK_BP
    from platform_core.plugin_manager.base import BasePlugin as Platform_BP
    assert SDK_BP is Platform_BP
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sdk/test_compatibility.py -v`
Expected: FAIL (platform_core still has its own definitions)

**Step 3: Modify `platform_core/schema/models.py`**

Replace the existing content with re-exports from SDK, while preserving ALL classes that are NOT in the SDK (Site, Position, Device, Component, Task, etc.):

```python
"""
Unified data model definitions.

Schema models are now defined in darkbreaker_sdk.schemas.
This module re-exports them for backward compatibility,
and adds platform-specific models (Site, Position, Device, etc.)
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ===== Re-exports from SDK (backward compatibility) =====
from darkbreaker_sdk.schemas.common import (
    BaseEntity,
    ROI,
    ROIType,
    Evidence,
    EvidenceType,
    generate_id,
)
from darkbreaker_sdk.schemas.detection import (
    BoundingBox,
    RecognitionResult,
)
from darkbreaker_sdk.schemas.alarm import (
    Alarm,
    AlarmLevel,
    AlarmRule,
    AlarmStatus,
)
from darkbreaker_sdk.schemas.plugin_io import PluginOutput

# ===== Platform-specific models (NOT in SDK) =====
# These stay here because they're platform orchestration concerns.

class Site(BaseEntity):
    """Site model."""
    code: str
    location: str = ""
    voltage_level: str = ""
    positions: list["Position"] = Field(default_factory=list)

class Position(BaseEntity):
    """Position model."""
    site_id: str
    camera_id: str = ""
    ptz_preset: dict[str, float] = Field(default_factory=dict)
    devices: list["Device"] = Field(default_factory=list)

class DeviceType(str, Enum):
    """Device type enumeration."""
    TRANSFORMER = "transformer"
    SWITCH = "switch"
    BUSBAR = "busbar"
    CAPACITOR = "capacitor"
    METER = "meter"
    OTHER = "other"

class Device(BaseEntity):
    """Device model."""
    position_id: str
    device_type: DeviceType
    model: str = ""
    components: list["Component"] = Field(default_factory=list)

class Component(BaseEntity):
    """Component model."""
    device_id: str
    component_type: str
    rois: list["ROI"] = Field(default_factory=list)

class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskTemplate(BaseEntity):
    """Task template."""
    plugin_id: str
    device_type: DeviceType
    default_config: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)

class Task(BaseEntity):
    """Task instance."""
    template_id: str
    site_id: str
    position_id: str
    device_id: str
    plugin_id: str
    roi_ids: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    config: dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""
    result_id: Optional[str] = None

# Update forward references
Site.model_rebuild()
Position.model_rebuild()
Device.model_rebuild()
Component.model_rebuild()
```

**Step 4: Modify `platform_core/plugin_manager/base.py`**

Replace with re-exports from SDK:

```python
"""
Plugin base class definitions.

Plugin interfaces are now defined in darkbreaker_sdk.interfaces.
This module re-exports them for backward compatibility.
"""

# ===== Re-exports from SDK =====
from darkbreaker_sdk.interfaces.lifecycle import (
    PluginCapability,
    PluginStatus,
    HealthStatus,
)
from darkbreaker_sdk.interfaces.base_plugin import (
    BasePlugin,
    PluginManifest,
    PluginContext,
)

# Re-export schema models that were previously imported here
from darkbreaker_sdk.schemas import (
    Alarm,
    AlarmRule,
    PluginOutput,
    RecognitionResult,
    ROI,
)

__all__ = [
    "BasePlugin",
    "PluginManifest",
    "PluginContext",
    "PluginCapability",
    "PluginStatus",
    "HealthStatus",
    "Alarm",
    "AlarmRule",
    "PluginOutput",
    "RecognitionResult",
    "ROI",
]
```

**Step 5: Run tests**

Run: `pytest tests/sdk/test_compatibility.py -v`
Expected: PASS

**Step 6: Run ALL existing tests to verify no regression**

Run: `pytest tests/ -v --tb=short 2>&1 | head -100`
Expected: All existing tests still pass

**Step 7: Commit**

```bash
git add platform_core/schema/models.py platform_core/plugin_manager/base.py tests/sdk/test_compatibility.py
git commit -m "refactor(platform_core): replace definitions with re-exports from darkbreaker_sdk"
```

---

## Phase 3: Migrate Plugins (17 plugins, one at a time)

The pattern is identical for each plugin. I'll detail the first 3 fully, then provide the template.

### Task 8: Migrate indoor_fence Plugin (Template Plugin)

**Files:**
- Modify: `plugins/indoor_fence/plugin.py` - update imports
- Create: `plugins/indoor_fence/requirements.txt`
- Create: `plugins/indoor_fence/standalone/__init__.py`
- Create: `plugins/indoor_fence/standalone/app.py`
- Create: `plugins/indoor_fence/standalone/templates/indoor_fence.html`
- Create: `plugins/indoor_fence/standalone/static/indoor_fence.js`
- Enhance: `plugins/indoor_fence/demo/` (restructure from existing demo.py)
- Create: `plugins/indoor_fence/scripts/benchmark.py`
- Create: `plugins/indoor_fence/tests/test_standalone.py`

**Step 1: Update imports in plugin.py**

Replace:
```python
from platform_core.plugin_manager.base import (
    BasePlugin, HealthStatus, PluginContext, PluginManifest, PluginStatus,
)
from platform_core.schema.models import (
    Alarm, AlarmLevel, AlarmRule, RecognitionResult, ROI, BoundingBox,
)
```

With:
```python
from darkbreaker_sdk.interfaces import (
    BasePlugin, HealthStatus, PluginContext, PluginManifest, PluginStatus,
)
from darkbreaker_sdk.schemas import (
    Alarm, AlarmLevel, AlarmRule, RecognitionResult, ROI, BoundingBox,
)
```

**Step 2: Add `create_standalone()` to the Plugin class**

Add classmethod to the Plugin class in `plugin.py`:
```python
@classmethod
def create_standalone(cls, config=None):
    """Create plugin instance for standalone operation."""
    plugin_dir = Path(__file__).resolve().parent
    manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
    instance = cls(manifest, plugin_dir)
    if config is None:
        from darkbreaker_sdk.utils import load_plugin_config
        config = load_plugin_config(plugin_dir / "configs" / "default.yaml")
    instance.init(config)
    return instance
```

**Step 3: Create `requirements.txt`**

```
# plugins/indoor_fence/requirements.txt
darkbreaker-sdk>=1.0.0
numpy>=1.24.0
opencv-python>=4.8.0
pydantic>=2.0.0
pyyaml>=6.0
```

**Step 4: Create `standalone/app.py`**

```python
#!/usr/bin/env python3
"""
Indoor Fence Plugin - Standalone Application

Run:  python -m plugins.indoor_fence.standalone.app
Web:  http://localhost:8081
"""
import sys
from pathlib import Path

# Ensure project root is in path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from darkbreaker_sdk.standalone import StandalonePluginRunner
from plugins.indoor_fence.plugin import Plugin


def main():
    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent / "templates",
        plugin_static_dir=Path(__file__).parent / "static",
    )
    runner.run(host="0.0.0.0", port=8081)


if __name__ == "__main__":
    main()
```

**Step 5: Create standalone HTML template**

Create `standalone/templates/indoor_fence.html` extending `base_standalone.html` with:
- Video feed panel with person detection overlay
- Zone configuration visualization (polygon zones, cabinets)
- Person tracking status table
- Authorization management panel
- Lidar data visualization (if enabled)
- Alarm log with state machine outputs
- Light control panel (Green/Yellow/Red)

**Step 6: Restructure demo scripts**

Move existing `demo.py` contents into `demo/` directory:
```
plugins/indoor_fence/demo/
├── __init__.py
├── run_demo.py          # Main entry: runs all demos
├── demo_geometry.py     # From demo.py::demo_geometry()
├── demo_tracking.py     # From demo.py::demo_tracking()
├── demo_zone.py         # From demo.py::demo_zone_config()
├── demo_state_machine.py  # From demo.py::demo_state_machine()
├── demo_adapters.py     # From demo.py::demo_adapters()
├── demo_fusion.py       # From demo.py::demo_fusion()
└── sample_data/         # Sample images for demo
```

Each demo file is self-contained and can be run individually:
```bash
python -m plugins.indoor_fence.demo.demo_geometry
python -m plugins.indoor_fence.demo.run_demo  # runs all
```

**Step 7: Create benchmark script**

```python
# plugins/indoor_fence/scripts/benchmark.py
"""Performance benchmark for indoor fence plugin."""
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from plugins.indoor_fence.plugin import Plugin


def main():
    plugin = Plugin.create_standalone()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        plugin.infer(frame, [], None)

    # Benchmark
    times = []
    for i in range(100):
        start = time.perf_counter()
        plugin.infer(frame, [], None)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    print(f"Indoor Fence Benchmark (100 iterations)")
    print(f"  Mean: {sum(times)/len(times):.2f} ms")
    print(f"  Min:  {min(times):.2f} ms")
    print(f"  Max:  {max(times):.2f} ms")
    print(f"  P95:  {sorted(times)[94]:.2f} ms")


if __name__ == "__main__":
    main()
```

**Step 8: Create standalone test**

```python
# plugins/indoor_fence/tests/test_standalone.py
"""Tests for indoor_fence standalone operation."""
import pytest
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def test_create_standalone():
    from plugins.indoor_fence.plugin import Plugin
    plugin = Plugin.create_standalone()
    assert plugin.id == "indoor_fence"
    assert plugin.status.value == "ready" or plugin.status == "ready"


def test_standalone_healthcheck():
    from plugins.indoor_fence.plugin import Plugin
    plugin = Plugin.create_standalone()
    health = plugin.healthcheck()
    assert health.healthy is True


def test_standalone_infer():
    import numpy as np
    from plugins.indoor_fence.plugin import Plugin
    plugin = Plugin.create_standalone()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = plugin.infer(frame, [], None)
    assert isinstance(results, list)


def test_standalone_runner_creation():
    from plugins.indoor_fence.plugin import Plugin
    from darkbreaker_sdk.standalone import StandalonePluginRunner
    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(plugin)
    assert runner.app is not None
```

**Step 9: Run all indoor_fence tests**

Run: `pytest plugins/indoor_fence/tests/ -v`
Expected: PASS

**Step 10: Verify standalone launch**

Run: `timeout 5 python -m plugins.indoor_fence.standalone.app || true`
Expected: Server starts, prints "Starting Indoor Fence - Standalone on http://0.0.0.0:8081"

**Step 11: Commit**

```bash
git add plugins/indoor_fence/
git commit -m "feat(indoor_fence): migrate to SDK, add standalone app, demos, benchmarks, tests"
```

---

### Task 9: Migrate animal_detection Plugin

Same pattern as Task 8. Key differences:
- **Port**: 8082
- **Existing fallback**: Remove the `try/except PLATFORM_AVAILABLE` pattern, replace with direct SDK imports
- **Standalone UI**: Video feed + species detection boxes + thermal view + deterrent controls + statistics
- **Demo scripts**: Species detection demo, thermal validation demo, tracking demo, deterrent control demo
- **Has existing tests**: Enhance `tests/test_plugin.py` and `tests/test_onnx_inference.py`, add `test_standalone.py`

**Step 1:** Update imports (replace `platform_core` imports AND remove the `PLATFORM_AVAILABLE` fallback pattern)
**Step 2:** Add `create_standalone()` classmethod
**Step 3:** Create `requirements.txt` (add onnxruntime to deps)
**Step 4:** Create `standalone/app.py` (port 8082)
**Step 5:** Create `standalone/templates/animal_detection.html`
**Step 6:** Create `demo/run_demo.py` with individual demos
**Step 7:** Create `scripts/benchmark.py`
**Step 8:** Create `tests/test_standalone.py`
**Step 9:** Run tests: `pytest plugins/animal_detection/tests/ -v`
**Step 10:** Verify standalone: `timeout 5 python -m plugins.animal_detection.standalone.app || true`
**Step 11:** Commit

---

### Task 10: Migrate fire_detection Plugin

Same pattern. Key differences:
- **Port**: 8083
- **Existing fallback**: Has 3 scattered try/except blocks for platform imports - replace all with clean SDK imports
- **Standalone UI**: Video feed + flame/smoke detection overlay + thermal view + suppression controls
- **No existing tests or core/ directory**: This is a single-file plugin; still create tests/ and demo/

**Steps 1-11:** Same pattern as Tasks 8-9

---

### Task 11: Migrate slam_mapping Plugin

- **Port**: 8084
- **Special**: Uses `platform_core.plugin_mixin.PluginStatusMixin` - must replace with SDK interface
- **Standalone UI**: 3D point cloud viewer (use Three.js), occupancy grid, loop closure status

---

### Task 12: Migrate temperature_monitoring Plugin

- **Port**: 8085
- **Standalone UI**: Thermal heatmap, hotspot list, trend charts

---

### Task 13: Migrate device_monitoring Plugin

- **Port**: 8086
- **Standalone UI**: Equipment status cards, health gauges

---

### Task 14: Migrate transformer_inspection Plugin

- **Port**: 8087
- **Standalone UI**: Image upload, defect detection overlay, thermal analysis

---

### Task 15: Migrate switch_inspection Plugin

- **Port**: 8088
- **Special**: detector_enhanced.py imports from `platform_core.fusion_engine` (Evidence, EvidenceType) - replace with SDK import
- **Standalone UI**: State recognition, position detection, logic verification

---

### Task 16: Migrate busbar_inspection Plugin

- **Port**: 8089
- **Standalone UI**: Insulator detection, hardware fitting inspection

---

### Task 17: Migrate capacitor_inspection Plugin

- **Port**: 8090
- **Standalone UI**: Structural inspection, bulging/leakage detection

---

### Task 18: Migrate meter_reading Plugin

- **Port**: 8091
- **Standalone UI**: Meter image upload, reading display, perspective correction

---

### Task 19: Migrate bird_monitoring Plugin

- **Port**: 8092
- **Special**: Has no `__init__.py` - create one. Has `main.py` with additional platform_core imports
- **Standalone UI**: Video feed, species ID, risk zones, deterrent controls

---

### Task 20: Migrate acoustic_monitoring Plugin

- **Port**: 8093
- **Standalone UI**: Waveform display, spectrogram, PD events

---

### Task 21: Migrate gas_detection Plugin

- **Port**: 8094
- **Standalone UI**: Concentration gauges, trend prediction

---

### Task 22: Migrate hyperspectral_detection Plugin

- **Port**: 8095
- **Standalone UI**: Spectral bands viewer, analysis results

---

### Task 23: Migrate multimodal_fusion Plugin

- **Port**: 8096
- **Standalone UI**: Multi-stream view, fusion confidence

---

## Phase 4: Plugin Selector UI & Platform Integration

### Task 24: Create Plugin Installer Controller

**Files:**
- Create: `platform_core/plugin_manager/installer.py`
- Test: `tests/sdk/test_installer.py`

**Step 1: Write test**

```python
# tests/sdk/test_installer.py
def test_list_available_plugins():
    from platform_core.plugin_manager.installer import PluginInstaller
    installer = PluginInstaller(plugins_dir="plugins/")
    available = installer.list_available()
    assert len(available) >= 17

def test_enable_disable_plugin():
    from platform_core.plugin_manager.installer import PluginInstaller
    installer = PluginInstaller(plugins_dir="plugins/")
    installer.enable("indoor_fence")
    assert "indoor_fence" in installer.get_enabled()
    installer.disable("indoor_fence")
    assert "indoor_fence" not in installer.get_enabled()
```

**Step 2: Implement PluginInstaller**

```python
class PluginInstaller:
    """Plugin enable/disable controller (MATLAB Add-On Explorer style)."""

    def __init__(self, plugins_dir, config_path=None):
        self.plugins_dir = Path(plugins_dir)
        self.config_path = config_path or self.plugins_dir / ".enabled_plugins.json"
        self._enabled = self._load_enabled()

    def list_available(self) -> list[dict]:
        """List all available plugins with their enable status."""
        ...

    def enable(self, plugin_id: str) -> bool:
        """Enable a plugin."""
        ...

    def disable(self, plugin_id: str) -> bool:
        """Disable a plugin."""
        ...

    def get_enabled(self) -> list[str]:
        """Get list of enabled plugin IDs."""
        ...

    def check_dependencies(self, plugin_id: str) -> list[str]:
        """Check if plugin dependencies are satisfied."""
        ...
```

**Step 3-5: Run tests, commit**

---

### Task 25: Create Plugin Selector UI Page

**Files:**
- Create: `ui/templates/pages/plugin_manager.html`
- Create: `ui/static/js/plugin_manager.js`
- Modify: `apps/ui_server.py` - add route for plugin manager page
- Modify: `apps/api_server.py` - add API endpoints for plugin enable/disable

The plugin manager page shows a grid of plugin cards (similar to MATLAB's Add-On Explorer):
- Category tabs: All / Indoor / Outdoor
- Each card: Plugin icon, name, version, description, status badge, enable/disable toggle
- Dependency warnings if disabling a required plugin

**Step 1:** Create HTML template
**Step 2:** Create JavaScript for toggle logic
**Step 3:** Add FastAPI route in `ui_server.py`
**Step 4:** Add API endpoints in `api_server.py`
**Step 5:** Test via browser
**Step 6:** Commit

---

### Task 26: Dynamic Dashboard Plugin Loading

**Files:**
- Modify: `ui/templates/pages/indoor_center.html` - load only enabled indoor plugins
- Modify: `ui/templates/pages/outdoor_center_v4.html` - load only enabled outdoor plugins
- Modify: `ui/static/js/indoor_center.js` - dynamic component loading
- Modify: `ui/static/js/outdoor_center_v4.js` - dynamic component loading

Add JavaScript that queries `/api/plugins/enabled` and dynamically loads only enabled plugin components.

---

## Phase 5: Integration Testing & Verification

### Task 27: Full Integration Test Suite

**Files:**
- Create: `tests/integration/test_plugin_standalone.py`
- Create: `tests/integration/test_sdk_migration.py`

**Step 1: Write integration tests**

```python
# tests/integration/test_plugin_standalone.py
"""Verify ALL 17 plugins can run standalone."""
import pytest
import subprocess
import sys

PLUGINS = [
    "indoor_fence", "animal_detection", "fire_detection", "slam_mapping",
    "temperature_monitoring", "device_monitoring", "transformer_inspection",
    "switch_inspection", "busbar_inspection", "capacitor_inspection",
    "meter_reading", "bird_monitoring", "acoustic_monitoring",
    "gas_detection", "hyperspectral_detection", "multimodal_fusion",
]

@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_create_standalone(plugin_id):
    """Each plugin must instantiate via create_standalone()."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"from plugins.{plugin_id}.plugin import Plugin; "
         f"p = Plugin.create_standalone(); "
         f"print(f'{{p.id}} v{{p.version}} OK')"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"{plugin_id} failed: {result.stderr}"
    assert "OK" in result.stdout

@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_has_requirements(plugin_id):
    """Each plugin must have a requirements.txt."""
    from pathlib import Path
    req = Path(f"plugins/{plugin_id}/requirements.txt")
    assert req.exists(), f"{plugin_id} missing requirements.txt"

@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_has_standalone_app(plugin_id):
    """Each plugin must have standalone/app.py."""
    from pathlib import Path
    app = Path(f"plugins/{plugin_id}/standalone/app.py")
    assert app.exists(), f"{plugin_id} missing standalone/app.py"

@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_has_demo(plugin_id):
    """Each plugin must have demo/run_demo.py."""
    from pathlib import Path
    demo = Path(f"plugins/{plugin_id}/demo/run_demo.py")
    assert demo.exists(), f"{plugin_id} missing demo/run_demo.py"
```

**Step 2: Run integration tests**

Run: `pytest tests/integration/test_plugin_standalone.py -v`
Expected: ALL 17 plugins pass ALL 4 checks

**Step 3: Run full test suite for regression**

Run: `pytest tests/ -v --tb=short`
Expected: All existing tests still pass

**Step 4: Commit**

```bash
git add tests/integration/
git commit -m "test: add integration tests verifying all 17 plugins run standalone"
```

---

## Execution Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| **Phase 1** | Tasks 1-6 | Create `darkbreaker_sdk` package (interfaces, schemas, runner, utils) |
| **Phase 2** | Task 7 | Platform core compatibility layer (re-exports) |
| **Phase 3** | Tasks 8-23 | Migrate all 17 plugins (one per task) |
| **Phase 4** | Tasks 24-26 | Plugin selector UI & dynamic dashboard |
| **Phase 5** | Task 27 | Integration testing & verification |

**Total Tasks:** 27
**Estimated execution:** Each plugin migration (Tasks 8-23) follows identical pattern.

**Verification commands per plugin:**
```bash
# 1. Standalone creation
python -c "from plugins.{id}.plugin import Plugin; p = Plugin.create_standalone(); print(p.id, 'OK')"

# 2. Standalone server
python -m plugins.{id}.standalone.app  # Visit http://localhost:{port}

# 3. Demo scripts
python -m plugins.{id}.demo.run_demo

# 4. Unit tests
pytest plugins/{id}/tests/ -v

# 5. Benchmark
python -m plugins.{id}.scripts.benchmark
```

**Final verification:**
```bash
# All 17 plugins standalone
pytest tests/integration/test_plugin_standalone.py -v

# Full regression
pytest tests/ -v

# Platform still works
python -m apps.main
```
