# Indoor Fence V3.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the indoor_fence plugin from V2.1 to V3.0 — fix Python 3.9 startup crash, restructure into 7-layer modular architecture, add UWB/IMU adapters, EKF/UKF fusion, deep learning detection pipeline, behavior recognition, rule engine, simulator, data recording/training pipeline, and ensure full standalone operation without UI.

**Architecture:** Seven-layer modular architecture (Sensor Adaptation → Detection & Recognition → Fusion & Tracking → State Machine → Rule Engine → Output & Control → Standalone Service) with cross-cutting concerns (config, data recording, training, health). All layers communicate via Pydantic models. Every adapter has built-in simulation mode for hardware-free operation.

**Tech Stack:** Python 3.9+, FastAPI, NumPy, OpenCV, ONNX Runtime, Pydantic v2, PyYAML, scipy (EKF/UKF), pytest

---

## Phase 1: Fix Startup & Foundation (Tasks 1-6)

### Task 1: Fix Python 3.9 Compatibility in SDK Runner

**Files:**
- Modify: `/Users/ronan/Desktop/DarkBreaker/darkbreaker_sdk/standalone/runner.py`

**Step 1: Add `from __future__ import annotations` at line 1 of runner.py**

The file already has `from __future__ import annotations` at line 22. But the issue is that FastAPI evaluates type annotations at runtime for its endpoint signatures. The `from __future__ import annotations` makes all annotations strings (deferred), but Pydantic/FastAPI then evaluates them — and `str | None` fails on Python 3.9.

The actual fix: replace all `str | None` and `X | None` with `Optional[X]` in function signatures that FastAPI inspects, and `dict[str, Any]` with `Dict[str, Any]`.

```python
# Line 71-77: __init__ signature
def __init__(
    self,
    plugin: BasePlugin,
    title: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    plugin_templates_dir: Optional[Union[str, Path]] = None,
    plugin_static_dir: Optional[Union[str, Path]] = None,
) -> None:

# Line 89: _stats type hint
self._stats: Dict[str, Any] = {

# Line 229: _get_template_context return
def _get_template_context(self, request: Request) -> Dict[str, Any]:

# Line 520: _broadcast parameter
async def _broadcast(self, data: Dict[str, Any]) -> None:

# Line 631: _get_alarms parameter
async def _get_alarms(self, level: Optional[str] = None) -> JSONResponse:

# Line 654: run method
def run(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
```

**Step 2: Run standalone to verify startup**

Run: `cd /Users/ronan/Desktop/DarkBreaker && timeout 5 python3 -m plugins.indoor_fence 2>&1 || true`
Expected: Server starts (no TypeError), then times out. Should see the banner with port 8081.

**Step 3: Commit**

```bash
git add darkbreaker_sdk/standalone/runner.py
git commit -m "fix: Python 3.9 compatibility for SDK standalone runner type annotations"
```

---

### Task 2: Create V3.0 Data Protocols Module

**Files:**
- Create: `plugins/indoor_fence/protocols.py`
- Test: `plugins/indoor_fence/tests/test_protocols.py`

**Step 1: Write the failing test**

```python
"""Tests for V3.0 data protocol models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.protocols import (
    SensorData, SensorType, AdapterMode,
    DetectionResult, PoseKeypoint,
    RiskAssessment, PersonStateV3,
    FusionInput, FusionOutput,
)


def test_sensor_data_creation():
    sd = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=1000.0,
        data={"bbox": [0.1, 0.2, 0.3, 0.4]},
        confidence=0.95,
    )
    assert sd.sensor_type == SensorType.CAMERA
    assert sd.is_simulated is False


def test_sensor_data_simulated():
    sd = SensorData(
        sensor_type=SensorType.UWB,
        timestamp=1000.0,
        data={"x": 1.0, "y": 2.0, "z": 0.5},
        confidence=0.8,
        is_simulated=True,
    )
    assert sd.is_simulated is True


def test_detection_result():
    dr = DetectionResult(
        track_id=1,
        position_3d=(1.0, 2.0, 0.0),
        velocity_3d=(0.1, 0.0, 0.0),
        confidence=0.9,
        fusion_sources=["camera", "lidar"],
    )
    assert dr.position_3d == (1.0, 2.0, 0.0)
    assert dr.bbox is None
    assert dr.behavior is None


def test_detection_result_with_pose():
    kp = PoseKeypoint(x=0.5, y=0.3, confidence=0.9, name="nose")
    dr = DetectionResult(
        track_id=2,
        position_3d=(3.0, 4.0, 0.0),
        velocity_3d=(0.0, 0.0, 0.0),
        confidence=0.85,
        fusion_sources=["camera"],
        pose_keypoints=[kp],
        behavior="normal",
    )
    assert len(dr.pose_keypoints) == 1
    assert dr.behavior == "normal"


def test_risk_assessment():
    ra = RiskAssessment(
        person_state=PersonStateV3.CLIMBING,
        risk_score=0.8,
        zone_id="cabinet_1",
        violations=["unauthorized_zone", "climbing"],
        recommended_action="alarm_red",
    )
    assert ra.risk_score == 0.8
    assert PersonStateV3.CLIMBING.value == "climbing"


def test_person_state_v3_has_all_states():
    states = [s.value for s in PersonStateV3]
    assert "normal" in states
    assert "climbing" in states
    assert "prolonged_stay" in states
    assert "fallen" in states
    assert "multi_person" in states


def test_adapter_mode_enum():
    assert AdapterMode.LIVE.value == "live"
    assert AdapterMode.SIMULATED.value == "simulated"
    assert AdapterMode.REPLAY.value == "replay"


def test_fusion_input_output():
    fi = FusionInput(
        sensor_type=SensorType.LIDAR,
        timestamp=1000.0,
        position_2d=(2.0, 3.0),
        confidence=0.9,
    )
    assert fi.position_3d is None

    fo = FusionOutput(
        track_id=1,
        position_3d=(2.0, 3.0, 0.0),
        velocity_3d=(0.0, 0.0, 0.0),
        confidence=0.85,
        sources=[SensorType.LIDAR, SensorType.CAMERA],
        covariance_diag=(0.01, 0.01, 0.05),
    )
    assert len(fo.sources) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_protocols.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.indoor_fence.protocols'`

**Step 3: Write the protocols module**

```python
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


# ─── Enums ──────────────────────────────────────────────

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


# ─── Sensor Data ────────────────────────────────────────

class SensorData(BaseModel):
    """Unified sensor output format."""
    sensor_type: SensorType
    timestamp: float
    data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    is_simulated: bool = False


# ─── Detection ──────────────────────────────────────────

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


# ─── Fusion ─────────────────────────────────────────────

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


# ─── Risk Assessment ────────────────────────────────────

class RiskAssessment(BaseModel):
    """Output from state machine + rule engine."""
    person_state: PersonStateV3
    risk_score: float = Field(ge=0.0, le=1.0)
    zone_id: Optional[str] = None
    violations: List[str] = Field(default_factory=list)
    recommended_action: str = "none"
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_protocols.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add plugins/indoor_fence/protocols.py plugins/indoor_fence/tests/test_protocols.py
git commit -m "feat(indoor_fence): add V3.0 data protocol models"
```

---

### Task 3: Refactor BaseAdapter Interface

**Files:**
- Modify: `plugins/indoor_fence/adapters/base_adapter.py`
- Test: `plugins/indoor_fence/tests/test_adapters_base.py`

**Step 1: Write the failing test**

```python
"""Tests for enhanced BaseAdapter interface."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import time
from plugins.indoor_fence.adapters.base_adapter import (
    BaseAdapter, AdapterConfig, AdapterStatus, AdapterHealth, AdapterMode,
)
from plugins.indoor_fence.protocols import SensorData, SensorType


class MockAdapter(BaseAdapter):
    """Concrete adapter for testing."""

    def __init__(self):
        super().__init__(AdapterConfig(adapter_type="mock"))
        self._connected = False

    @property
    def sensor_type(self) -> SensorType:
        return SensorType.CAMERA

    def connect(self) -> bool:
        self._connected = True
        self._status = AdapterStatus.CONNECTED
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def _read_live(self) -> SensorData:
        return SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=time.time(),
            data={"test": True},
            confidence=1.0,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected


def test_adapter_lifecycle():
    adapter = MockAdapter()
    assert adapter.status == AdapterStatus.DISCONNECTED
    assert adapter.mode == AdapterMode.LIVE

    adapter.connect()
    assert adapter.status == AdapterStatus.CONNECTED
    assert adapter.is_connected is True

    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_adapter_get_data_live():
    adapter = MockAdapter()
    adapter.connect()
    data = adapter.get_data()
    assert data is not None
    assert data.sensor_type == SensorType.CAMERA
    assert data.is_simulated is False


def test_adapter_simulation_mode():
    adapter = MockAdapter()
    adapter.set_mode(AdapterMode.SIMULATED)
    assert adapter.mode == AdapterMode.SIMULATED
    data = adapter.get_data()
    assert data is not None
    assert data.is_simulated is True


def test_adapter_health():
    adapter = MockAdapter()
    adapter.connect()
    # Read a few times to build stats
    for _ in range(3):
        adapter.get_data()
    health = adapter.get_health()
    assert isinstance(health, AdapterHealth)
    assert health.frames_read >= 3
    assert health.error_count == 0


def test_adapter_recording(tmp_path):
    adapter = MockAdapter()
    adapter.connect()
    rec_path = str(tmp_path / "recording.jsonl")
    adapter.start_recording(rec_path)
    assert adapter.is_recording is True

    for _ in range(3):
        adapter.get_data()

    adapter.stop_recording()
    assert adapter.is_recording is False

    # Verify file was written
    import json
    with open(rec_path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    entry = json.loads(lines[0])
    assert "sensor_type" in entry


def test_adapter_replay(tmp_path):
    import json
    # Create a replay file
    rec_path = str(tmp_path / "replay.jsonl")
    with open(rec_path, "w") as f:
        for i in range(3):
            entry = SensorData(
                sensor_type=SensorType.CAMERA,
                timestamp=1000.0 + i,
                data={"frame": i},
                confidence=0.9,
                is_simulated=False,
            )
            f.write(entry.model_dump_json() + "\n")

    adapter = MockAdapter()
    adapter.load_replay(rec_path)
    assert adapter.mode == AdapterMode.REPLAY

    data = adapter.get_data()
    assert data is not None
    assert data.data["frame"] == 0

    data = adapter.get_data()
    assert data.data["frame"] == 1


def test_adapter_weight():
    adapter = MockAdapter()
    assert 0.0 <= adapter.get_weight() <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_adapters_base.py -v`
Expected: FAIL

**Step 3: Rewrite base_adapter.py with enhanced interface**

Read existing file first, then rewrite preserving backward compat. New `BaseAdapter` should be an ABC with:
- `connect()`, `disconnect()`, `is_connected` (existing)
- `get_data() -> Optional[SensorData]` — routes to `_read_live()`, `_read_simulated()`, or `_read_replay()` based on mode
- `_read_live() -> SensorData` — abstract, each subclass implements
- `_read_simulated() -> SensorData` — default returns mock data, subclasses override
- `_read_replay() -> Optional[SensorData]` — reads from JSONL file
- `get_health() -> AdapterHealth` — built-in stats tracking
- `start_recording(path)` / `stop_recording()` — JSONL recording
- `load_replay(path)` — switch to replay mode
- `set_mode(mode)` — switch adapter mode
- `get_weight() -> float` — default 1.0, subclasses can override
- `is_recording` property
- `mode` property
- `sensor_type` abstract property

Import `AdapterMode` from protocols (re-export from base_adapter for compat).

**Step 4: Run test to verify it passes**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_adapters_base.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add plugins/indoor_fence/adapters/base_adapter.py plugins/indoor_fence/tests/test_adapters_base.py
git commit -m "feat(indoor_fence): enhance BaseAdapter with recording, replay, simulation modes"
```

---

### Task 4: Create Unified Simulator

**Files:**
- Create: `plugins/indoor_fence/adapters/simulator.py`
- Create: `plugins/indoor_fence/configs/scenarios/basic_patrol.json`
- Test: `plugins/indoor_fence/tests/test_simulator.py`

**Step 1: Write the failing test**

```python
"""Tests for unified simulator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.adapters.simulator import (
    Simulator, SimulatorConfig, PersonPath, PathType,
)
from plugins.indoor_fence.protocols import SensorType


def test_simulator_creation():
    config = SimulatorConfig(
        num_persons=2,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
    )
    sim = Simulator(config)
    assert sim.num_persons == 2


def test_simulator_step():
    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
    )
    sim = Simulator(config)
    data = sim.step()
    # Should return data for each sensor type
    assert SensorType.CAMERA in data
    assert SensorType.LIDAR in data
    for sd in data.values():
        assert sd.is_simulated is True


def test_simulator_with_uwb():
    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR, SensorType.UWB],
    )
    sim = Simulator(config)
    data = sim.step()
    assert SensorType.UWB in data
    uwb_data = data[SensorType.UWB]
    # UWB should have 3D position
    assert "positions" in uwb_data.data


def test_simulator_random_path():
    config = SimulatorConfig(
        num_persons=3,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA],
        path_type=PathType.RANDOM,
    )
    sim = Simulator(config)
    # Step multiple times, positions should change
    data1 = sim.step()
    for _ in range(10):
        sim.step()
    data2 = sim.step()
    # Positions should differ after 10 steps
    pos1 = data1[SensorType.CAMERA].data["detections"][0]["position"]
    pos2 = data2[SensorType.CAMERA].data["detections"][0]["position"]
    assert pos1 != pos2


def test_simulator_cross_line_scenario():
    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA],
        path_type=PathType.CROSS_LINE,
        yellow_line_y=2.5,
    )
    sim = Simulator(config)
    crossed = False
    for _ in range(100):
        data = sim.step()
        pos = data[SensorType.CAMERA].data["detections"][0]["position"]
        if pos[1] > 2.5:
            crossed = True
            break
    assert crossed, "Person should cross yellow line in CROSS_LINE scenario"


def test_simulator_fault_injection():
    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
    )
    sim = Simulator(config)
    # Inject camera failure
    sim.inject_fault(SensorType.CAMERA, "offline")
    data = sim.step()
    assert data[SensorType.CAMERA].confidence == 0.0
    assert SensorType.LIDAR in data  # LiDAR still works

    # Remove fault
    sim.clear_fault(SensorType.CAMERA)
    data = sim.step()
    assert data[SensorType.CAMERA].confidence > 0.0


def test_load_scenario_file():
    scenario_dir = Path(__file__).parent.parent / "configs" / "scenarios"
    scenario_file = scenario_dir / "basic_patrol.json"
    if not scenario_file.exists():
        pytest.skip("Scenario file not created yet")

    sim = Simulator.from_scenario(str(scenario_file))
    assert sim.num_persons >= 1
    data = sim.step()
    assert len(data) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_simulator.py -v`
Expected: FAIL

**Step 3: Implement simulator.py**

Create `Simulator` class that:
- Manages N simulated persons with configurable paths
- Each `step()` generates synchronized sensor data for all configured sensor types
- Camera data: bounding boxes + foot points from projected 3D position
- LiDAR data: range-bearing measurements from virtual scan
- UWB data: 3D coordinates with Gaussian noise (sigma ~0.1m)
- IMU data: acceleration vectors from velocity differentiation
- Path types: RANDOM (random walk), FIXED (waypoint), CROSS_LINE (approaches then crosses yellow line), CLIMBING (elevation changes)
- Fault injection: set any sensor to offline/degraded
- Scenario loading from JSON files

Also create `configs/scenarios/basic_patrol.json`:
```json
{
  "name": "basic_patrol",
  "description": "Single person patrolling along the main passage",
  "num_persons": 1,
  "scene_bounds": [0.0, 0.0, 10.0, 6.0],
  "sensor_types": ["camera", "lidar"],
  "path_type": "fixed",
  "waypoints": [[1.0, 1.0], [5.0, 1.0], [9.0, 1.0], [5.0, 1.0]],
  "speed_m_per_step": 0.1,
  "loop": true
}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_simulator.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add plugins/indoor_fence/adapters/simulator.py plugins/indoor_fence/configs/scenarios/ plugins/indoor_fence/tests/test_simulator.py
git commit -m "feat(indoor_fence): add unified multi-sensor simulator with scenario support"
```

---

### Task 5: Create UWB Adapter

**Files:**
- Create: `plugins/indoor_fence/adapters/uwb_adapter.py`
- Test: `plugins/indoor_fence/tests/test_uwb_adapter.py`

**Step 1: Write the failing test**

```python
"""Tests for UWB adapter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.adapters.uwb_adapter import UWBAdapter, UWBConfig
from plugins.indoor_fence.adapters.base_adapter import AdapterMode, AdapterStatus
from plugins.indoor_fence.protocols import SensorType


def test_uwb_adapter_creation():
    config = UWBConfig(protocol="udp", host="127.0.0.1", port=5000)
    adapter = UWBAdapter(config)
    assert adapter.sensor_type == SensorType.UWB
    assert adapter.mode == AdapterMode.LIVE


def test_uwb_adapter_simulation():
    config = UWBConfig(protocol="simulate")
    adapter = UWBAdapter(config)
    adapter.connect()
    assert adapter.is_connected
    data = adapter.get_data()
    assert data is not None
    assert data.sensor_type == SensorType.UWB
    assert "positions" in data.data
    # Should have 3D positions
    positions = data.data["positions"]
    assert len(positions) > 0
    pos = positions[0]
    assert "x" in pos and "y" in pos and "z" in pos


def test_uwb_adapter_nlos_indicator():
    config = UWBConfig(protocol="simulate", simulate_nlos=True)
    adapter = UWBAdapter(config)
    adapter.connect()
    data = adapter.get_data()
    positions = data.data["positions"]
    # Each position should have nlos_probability
    for pos in positions:
        assert "nlos_probability" in pos
        assert 0.0 <= pos["nlos_probability"] <= 1.0


def test_uwb_adapter_disconnect():
    config = UWBConfig(protocol="simulate")
    adapter = UWBAdapter(config)
    adapter.connect()
    adapter.disconnect()
    assert not adapter.is_connected
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_uwb_adapter_weight_with_nlos():
    config = UWBConfig(protocol="simulate", simulate_nlos=True, nlos_ratio=0.8)
    adapter = UWBAdapter(config)
    adapter.connect()
    # With high NLOS ratio, weight should decrease
    for _ in range(10):
        adapter.get_data()
    weight = adapter.get_weight()
    assert weight < 1.0  # Should be degraded due to NLOS
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_uwb_adapter.py -v`
Expected: FAIL

**Step 3: Implement UWBAdapter**

UWBAdapter extends BaseAdapter:
- `UWBConfig(AdapterConfig)`: protocol (udp/serial/dw3000/simulate), host, port, tag_ids, simulate_nlos, nlos_ratio
- `_read_live()`: reads from UDP/serial socket, parses DW1000/3000 protocol frames
- `_read_simulated()`: generates 3D positions with Gaussian noise (sigma=0.1m), optional NLOS with larger noise (sigma=0.5m)
- `get_weight()`: returns 1.0 minus average NLOS probability of recent readings
- Each position includes: `{x, y, z, tag_id, nlos_probability, timestamp}`

**Step 4: Run test to verify it passes**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_uwb_adapter.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add plugins/indoor_fence/adapters/uwb_adapter.py plugins/indoor_fence/tests/test_uwb_adapter.py
git commit -m "feat(indoor_fence): add UWB adapter with NLOS detection and simulation"
```

---

### Task 6: Create IMU Adapter

**Files:**
- Create: `plugins/indoor_fence/adapters/imu_adapter.py`
- Test: `plugins/indoor_fence/tests/test_imu_adapter.py`

**Step 1: Write the failing test**

```python
"""Tests for IMU adapter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.adapters.imu_adapter import IMUAdapter, IMUConfig
from plugins.indoor_fence.protocols import SensorType


def test_imu_adapter_creation():
    config = IMUConfig(protocol="simulate")
    adapter = IMUAdapter(config)
    assert adapter.sensor_type == SensorType.IMU


def test_imu_adapter_simulation():
    config = IMUConfig(protocol="simulate")
    adapter = IMUAdapter(config)
    adapter.connect()
    data = adapter.get_data()
    assert data is not None
    assert "acceleration" in data.data
    assert "gyroscope" in data.data
    acc = data.data["acceleration"]
    assert "x" in acc and "y" in acc and "z" in acc
```

**Step 2: Run test, verify fail**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_imu_adapter.py -v`

**Step 3: Implement IMUAdapter**

Lightweight adapter:
- `IMUConfig`: protocol (serial/i2c/simulate), device_path, baud_rate
- `_read_live()`: reads from serial/I2C, parses 6-axis data
- `_read_simulated()`: generates acceleration + gyroscope with noise
- Output: `{acceleration: {x,y,z}, gyroscope: {x,y,z}, timestamp}`

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add plugins/indoor_fence/adapters/imu_adapter.py plugins/indoor_fence/tests/test_imu_adapter.py
git commit -m "feat(indoor_fence): add IMU adapter with simulation mode"
```

---

## Phase 2: Detection & Recognition Layer (Tasks 7-11)

### Task 7: Extract Object Detector from Camera Adapter

**Files:**
- Create: `plugins/indoor_fence/detection/__init__.py`
- Create: `plugins/indoor_fence/detection/object_detector.py`
- Create: `plugins/indoor_fence/detection/yolo_detector.py`
- Test: `plugins/indoor_fence/tests/test_detection.py`

**Step 1: Write the failing test**

```python
"""Tests for object detection module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from plugins.indoor_fence.detection.object_detector import ObjectDetector, DetectionBox
from plugins.indoor_fence.detection.yolo_detector import YOLODetector


def test_detection_box():
    box = DetectionBox(
        x1=100, y1=200, x2=200, y2=400,
        confidence=0.85,
        class_id=0,
        class_name="person",
    )
    assert box.width == 100
    assert box.height == 200
    assert box.center == (150, 300)
    assert box.foot_point == (150, 400)


def test_yolo_detector_simulation():
    """YOLO detector in simulation mode (no model file)."""
    detector = YOLODetector(model_path=None, device="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    # Simulation mode should return at least one detection
    assert len(detections) >= 1
    for d in detections:
        assert isinstance(d, DetectionBox)
        assert d.class_name == "person"
        assert 0.0 <= d.confidence <= 1.0


def test_object_detector_interface():
    """Verify ObjectDetector is an ABC with detect method."""
    with pytest.raises(TypeError):
        ObjectDetector()  # Cannot instantiate abstract class
```

**Step 2: Run test, verify fail**

**Step 3: Implement**

- `ObjectDetector`: ABC with `detect(frame: np.ndarray) -> List[DetectionBox]`
- `DetectionBox`: dataclass with x1,y1,x2,y2, confidence, class_id, class_name, plus computed properties (width, height, center, foot_point)
- `YOLODetector(ObjectDetector)`: extracts detection logic from existing `camera_adapter.py` and `detector.py`. In simulation mode (no model file), generates random detections.

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add plugins/indoor_fence/detection/
git commit -m "feat(indoor_fence): extract object detection into standalone detection module"
```

---

### Task 8: Add Pose Estimator

**Files:**
- Create: `plugins/indoor_fence/detection/pose_estimator.py`
- Test: `plugins/indoor_fence/tests/test_pose.py`

**Step 1: Write the failing test**

```python
"""Tests for pose estimation module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from plugins.indoor_fence.detection.pose_estimator import (
    PoseEstimatorV3, PoseResult, PostureType, KEYPOINT_NAMES,
)


def test_pose_result():
    result = PoseResult(
        keypoints=[(0.5, 0.3, 0.9)] * 17,
        posture=PostureType.STANDING,
        confidence=0.85,
    )
    assert len(result.keypoints) == 17
    assert result.posture == PostureType.STANDING


def test_pose_estimator_simulation():
    estimator = PoseEstimatorV3(model_path=None)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (100, 100, 200, 400)
    result = estimator.estimate(frame, bbox)
    assert result is not None
    assert len(result.keypoints) == 17
    assert result.posture in PostureType


def test_posture_types():
    assert PostureType.STANDING.value == "standing"
    assert PostureType.BENDING.value == "bending"
    assert PostureType.CLIMBING.value == "climbing"
    assert PostureType.FALLEN.value == "fallen"
    assert PostureType.CROUCHING.value == "crouching"


def test_keypoint_names():
    assert len(KEYPOINT_NAMES) == 17
    assert "nose" in KEYPOINT_NAMES
    assert "left_ankle" in KEYPOINT_NAMES
```

**Step 2: Run test, verify fail**

**Step 3: Implement PoseEstimatorV3**

- 17 COCO keypoints (MoveNet format)
- `PostureType` enum: STANDING, BENDING, CLIMBING, CROUCHING, FALLEN, UNKNOWN
- `estimate(frame, bbox) -> PoseResult`: runs model or simulation
- Posture classification from keypoint geometry (shoulder-hip angle, knee angles, etc.)
- Simulation mode: generates plausible keypoints for standing person

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add plugins/indoor_fence/detection/pose_estimator.py plugins/indoor_fence/tests/test_pose.py
git commit -m "feat(indoor_fence): add pose estimator with posture classification"
```

---

### Task 9: Add Behavior Recognizer

**Files:**
- Create: `plugins/indoor_fence/detection/behavior_recognizer.py`
- Test: `plugins/indoor_fence/tests/test_behavior.py`

**Step 1: Write the failing test**

```python
"""Tests for behavior recognition module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.detection.behavior_recognizer import (
    BehaviorRecognizerV3, BehaviorType, BehaviorResult,
)
from plugins.indoor_fence.detection.pose_estimator import PoseResult, PostureType


def test_behavior_types():
    assert BehaviorType.NORMAL_WALK.value == "normal_walk"
    assert BehaviorType.PROLONGED_STAY.value == "prolonged_stay"
    assert BehaviorType.CLIMBING.value == "climbing"
    assert BehaviorType.CROSSING.value == "crossing"
    assert BehaviorType.FALLEN.value == "fallen"


def test_behavior_recognizer_creation():
    rec = BehaviorRecognizerV3(
        window_size=30,
        prolonged_stay_threshold_s=30.0,
    )
    assert rec.window_size == 30


def test_behavior_normal_walk():
    rec = BehaviorRecognizerV3(window_size=5)
    # Simulate walking person - different positions each frame
    for i in range(5):
        pose = PoseResult(
            keypoints=[(0.3 + i * 0.01, 0.5, 0.9)] * 17,
            posture=PostureType.STANDING,
            confidence=0.9,
        )
        result = rec.update("person_1", pose, position=(float(i), 1.0))

    assert result is not None
    assert result.behavior in (BehaviorType.NORMAL_WALK, BehaviorType.UNKNOWN)


def test_behavior_prolonged_stay():
    rec = BehaviorRecognizerV3(
        window_size=5,
        prolonged_stay_threshold_s=0.1,  # Very short for testing
    )
    import time
    for i in range(10):
        pose = PoseResult(
            keypoints=[(0.5, 0.5, 0.9)] * 17,
            posture=PostureType.STANDING,
            confidence=0.9,
        )
        result = rec.update("person_1", pose, position=(5.0, 3.0))
        time.sleep(0.02)

    assert result.behavior == BehaviorType.PROLONGED_STAY


def test_behavior_fallen():
    rec = BehaviorRecognizerV3(window_size=3)
    for _ in range(5):
        pose = PoseResult(
            keypoints=[(0.5, 0.5, 0.9)] * 17,
            posture=PostureType.FALLEN,
            confidence=0.9,
        )
        result = rec.update("person_1", pose, position=(5.0, 3.0))

    assert result.behavior == BehaviorType.FALLEN


def test_behavior_clear_track():
    rec = BehaviorRecognizerV3(window_size=5)
    pose = PoseResult(
        keypoints=[(0.5, 0.5, 0.9)] * 17,
        posture=PostureType.STANDING,
        confidence=0.9,
    )
    rec.update("person_1", pose, position=(5.0, 3.0))
    rec.clear_track("person_1")
    # After clearing, should start fresh
    result = rec.update("person_1", pose, position=(5.0, 3.0))
    assert result is not None
```

**Step 2: Run test, verify fail**

**Step 3: Implement BehaviorRecognizerV3**

- Sliding window of recent poses + positions per track
- Classification logic based on:
  - Movement distance over window → walking vs stationary
  - Posture sequence → climbing/fallen detection
  - Duration at same position → prolonged stay
- Configurable thresholds

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add plugins/indoor_fence/detection/behavior_recognizer.py plugins/indoor_fence/tests/test_behavior.py
git commit -m "feat(indoor_fence): add behavior recognizer with temporal analysis"
```

---

### Task 10: Add Auto Fence Generator

**Files:**
- Create: `plugins/indoor_fence/detection/auto_fence_generator.py`
- Test: `plugins/indoor_fence/tests/test_auto_fence.py`

**Step 1: Write the failing test**

```python
"""Tests for automatic fence generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.detection.auto_fence_generator import (
    AutoFenceGenerator, FenceZone, FenceLevel,
)
from plugins.indoor_fence.detection.object_detector import DetectionBox


def test_fence_generation_from_equipment():
    gen = AutoFenceGenerator(
        warning_buffer_m=0.5,
        danger_buffer_m=0.3,
    )
    equipment = DetectionBox(
        x1=100, y1=100, x2=300, y2=400,
        confidence=0.9,
        class_id=1,
        class_name="cabinet",
    )
    fences = gen.generate_from_detection(equipment, image_size=(640, 480))
    assert len(fences) >= 2  # warning + danger zones
    warning = [f for f in fences if f.level == FenceLevel.WARNING][0]
    danger = [f for f in fences if f.level == FenceLevel.DANGER][0]
    # Warning zone should be larger than danger zone
    assert warning.area > danger.area


def test_fence_zone_contains():
    zone = FenceZone(
        vertices=[(0, 0), (4, 0), (4, 4), (0, 4)],
        level=FenceLevel.WARNING,
        equipment_id="cab_1",
    )
    assert zone.contains(2.0, 2.0) is True
    assert zone.contains(5.0, 5.0) is False


def test_fence_merge():
    gen = AutoFenceGenerator(warning_buffer_m=0.5, danger_buffer_m=0.3)
    eq1 = DetectionBox(x1=100, y1=100, x2=200, y2=300, confidence=0.9, class_id=1, class_name="cabinet")
    eq2 = DetectionBox(x1=180, y1=100, x2=280, y2=300, confidence=0.9, class_id=1, class_name="cabinet")
    fences1 = gen.generate_from_detection(eq1, image_size=(640, 480))
    fences2 = gen.generate_from_detection(eq2, image_size=(640, 480))
    # Overlapping fences should be merge-able
    merged = gen.merge_overlapping(fences1 + fences2)
    assert len(merged) <= len(fences1) + len(fences2)
```

**Step 2: Run test, verify fail**

**Step 3: Implement AutoFenceGenerator**

- `FenceLevel` enum: WARNING, DANGER, CRITICAL
- `FenceZone`: vertices, level, equipment_id, area property, contains() method
- `generate_from_detection(det, image_size)`: expands bbox by buffer amounts, converts to ground coordinates
- `merge_overlapping(zones)`: merges overlapping zones of same level
- Buffer computation uses morphological dilation concept

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add plugins/indoor_fence/detection/auto_fence_generator.py plugins/indoor_fence/tests/test_auto_fence.py
git commit -m "feat(indoor_fence): add automatic fence generation from equipment detection"
```

---

### Task 11: Add Model Manager

**Files:**
- Create: `plugins/indoor_fence/detection/model_manager.py`
- Test: `plugins/indoor_fence/tests/test_model_manager.py`

**Step 1: Write the failing test**

```python
"""Tests for model lifecycle manager."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.detection.model_manager import (
    ModelManager, ModelInfo, ModelStatus,
)


def test_model_manager_creation():
    mm = ModelManager()
    assert mm.list_models() == []


def test_register_model():
    mm = ModelManager()
    mm.register("yolov8n", "/path/to/model.onnx", model_type="detector")
    models = mm.list_models()
    assert len(models) == 1
    assert models[0].model_id == "yolov8n"
    assert models[0].status == ModelStatus.REGISTERED


def test_model_not_found():
    mm = ModelManager()
    assert mm.get("nonexistent") is None


def test_model_status_transitions():
    mm = ModelManager()
    mm.register("test_model", "/path/to/model.onnx", model_type="detector")
    mm.set_status("test_model", ModelStatus.LOADED)
    info = mm.get("test_model")
    assert info.status == ModelStatus.LOADED


def test_model_status_enum():
    assert ModelStatus.REGISTERED.value == "registered"
    assert ModelStatus.LOADED.value == "loaded"
    assert ModelStatus.FAILED.value == "failed"
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat(indoor_fence): add model lifecycle manager"
```

---

## Phase 3: Fusion & Tracking Layer (Tasks 12-15)

### Task 12: Implement Extended Kalman Filter (EKF)

**Files:**
- Create: `plugins/indoor_fence/core/fusion/__init__.py`
- Create: `plugins/indoor_fence/core/fusion/ekf_fusion.py`
- Test: `plugins/indoor_fence/tests/test_ekf.py`

**Step 1: Write the failing test**

```python
"""Tests for Extended Kalman Filter fusion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from plugins.indoor_fence.core.fusion.ekf_fusion import EKF6DOF


def test_ekf_creation():
    ekf = EKF6DOF()
    state = ekf.get_state()
    assert len(state) == 6  # x, y, z, vx, vy, vz
    assert all(s == 0.0 for s in state)


def test_ekf_predict():
    ekf = EKF6DOF()
    # Set initial state with velocity
    ekf.set_state([1.0, 2.0, 0.0, 0.5, 0.0, 0.0])
    ekf.predict(dt=1.0)
    state = ekf.get_state()
    # x should have moved by vx * dt
    assert abs(state[0] - 1.5) < 0.01
    assert abs(state[1] - 2.0) < 0.01


def test_ekf_update_camera():
    ekf = EKF6DOF()
    ekf.set_state([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    # Camera observes (x, y) only
    measurement = np.array([1.1, 2.1])
    R = np.diag([0.01, 0.01])
    ekf.update_camera(measurement, R)
    state = ekf.get_state()
    # State should move toward measurement
    assert abs(state[0] - 1.1) < 0.1
    assert abs(state[1] - 2.1) < 0.1


def test_ekf_update_uwb():
    ekf = EKF6DOF()
    ekf.set_state([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    # UWB observes (x, y, z)
    measurement = np.array([1.2, 2.2, 0.5])
    R = np.diag([0.04, 0.04, 0.04])
    ekf.update_uwb(measurement, R)
    state = ekf.get_state()
    # z should update toward 0.5
    assert state[2] > 0.0


def test_ekf_update_lidar():
    ekf = EKF6DOF()
    ekf.set_state([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    # LiDAR observes (x, y) in ground plane
    measurement = np.array([0.9, 1.9])
    R = np.diag([0.005, 0.005])
    ekf.update_lidar(measurement, R)
    state = ekf.get_state()
    assert abs(state[0] - 0.9) < 0.15


def test_ekf_multi_sensor_fusion():
    ekf = EKF6DOF()
    ekf.set_state([0.0, 0.0, 0.0, 1.0, 0.5, 0.0])

    for i in range(10):
        ekf.predict(dt=0.1)
        # Simulate noisy camera observation
        true_x = (i + 1) * 0.1
        true_y = (i + 1) * 0.05
        cam_obs = np.array([true_x + np.random.randn() * 0.05,
                            true_y + np.random.randn() * 0.05])
        ekf.update_camera(cam_obs, np.diag([0.0025, 0.0025]))

    state = ekf.get_state()
    # Should track roughly toward (1.0, 0.5, 0)
    assert abs(state[0] - 1.0) < 0.3
    assert abs(state[1] - 0.5) < 0.3


def test_ekf_get_covariance():
    ekf = EKF6DOF()
    cov = ekf.get_covariance()
    assert cov.shape == (6, 6)
    # Diagonal should be positive
    assert all(cov[i, i] > 0 for i in range(6))
```

**Step 2: Run test, verify fail**

**Step 3: Implement EKF6DOF**

Six-state EKF `[x, y, z, vx, vy, vz]`:
- `predict(dt)`: constant velocity model, F matrix
- `update_camera(z, R)`: H_cam observes [x, y]
- `update_lidar(z, R)`: H_lidar observes [x, y]
- `update_uwb(z, R)`: H_uwb observes [x, y, z]
- `update_imu(z, R)`: H_imu observes [vx, vy, vz] via acceleration integration
- Process noise Q configurable
- All implemented using numpy arrays

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add plugins/indoor_fence/core/fusion/ plugins/indoor_fence/tests/test_ekf.py
git commit -m "feat(indoor_fence): implement 6-DOF Extended Kalman Filter for multi-sensor fusion"
```

---

### Task 13: Implement NLOS Detector & Dynamic Weight Manager

**Files:**
- Create: `plugins/indoor_fence/core/fusion/nlos_detector.py`
- Create: `plugins/indoor_fence/core/fusion/weight_manager.py`
- Test: `plugins/indoor_fence/tests/test_nlos_weights.py`

**Step 1: Write the failing test**

```python
"""Tests for NLOS detection and dynamic weight management."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.fusion.nlos_detector import NLOSDetector
from plugins.indoor_fence.core.fusion.weight_manager import (
    WeightManager, SensorWeight,
)
from plugins.indoor_fence.protocols import SensorType


def test_nlos_detector_los():
    detector = NLOSDetector()
    # Line-of-sight: low delay spread, high first-path power
    prob = detector.classify(
        first_path_power=-70.0,
        total_power=-65.0,
        delay_spread_ns=5.0,
    )
    assert prob < 0.3  # Should be classified as LOS


def test_nlos_detector_nlos():
    detector = NLOSDetector()
    # Non-line-of-sight: high delay spread, low first-path power
    prob = detector.classify(
        first_path_power=-90.0,
        total_power=-65.0,
        delay_spread_ns=50.0,
    )
    assert prob > 0.7  # Should be classified as NLOS


def test_weight_manager_default():
    wm = WeightManager()
    weights = wm.get_weights()
    assert SensorType.CAMERA in weights
    assert SensorType.LIDAR in weights
    assert all(0.0 <= w.weight <= 1.0 for w in weights.values())


def test_weight_manager_sensor_failure():
    wm = WeightManager()
    wm.report_sensor_status(SensorType.CAMERA, healthy=False)
    weights = wm.get_weights()
    assert weights[SensorType.CAMERA].weight < weights[SensorType.LIDAR].weight


def test_weight_manager_nlos_degrades_uwb():
    wm = WeightManager()
    wm.report_nlos(SensorType.UWB, nlos_probability=0.9)
    weights = wm.get_weights()
    assert weights[SensorType.UWB].weight < 1.0


def test_weight_manager_night_mode():
    wm = WeightManager()
    wm.set_environment(low_light=True)
    weights = wm.get_weights()
    # Camera should be degraded in low light
    assert weights[SensorType.CAMERA].weight < weights[SensorType.LIDAR].weight
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat(indoor_fence): add NLOS detector and dynamic sensor weight manager"
```

---

### Task 14: Refactor Multi-Sensor Fusion Manager

**Files:**
- Create: `plugins/indoor_fence/core/fusion/sensor_fusion_v3.py`
- Test: `plugins/indoor_fence/tests/test_fusion_v3.py`

**Step 1: Write the failing test**

```python
"""Tests for V3 multi-sensor fusion manager."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import time
from plugins.indoor_fence.core.fusion.sensor_fusion_v3 import (
    SensorFusionV3,
)
from plugins.indoor_fence.protocols import SensorType, SensorData, FusionOutput


def test_fusion_v3_creation():
    fusion = SensorFusionV3()
    assert fusion is not None


def test_fusion_camera_only():
    fusion = SensorFusionV3()
    ts = time.time()
    camera_data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        data={"detections": [{"id": 0, "position_2d": (3.0, 2.0), "bbox": [100, 200, 200, 400]}]},
        confidence=0.9,
    )
    outputs = fusion.update({SensorType.CAMERA: camera_data})
    assert len(outputs) >= 1
    assert isinstance(outputs[0], FusionOutput)


def test_fusion_camera_lidar():
    fusion = SensorFusionV3()
    ts = time.time()
    camera_data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        data={"detections": [{"id": 0, "position_2d": (3.0, 2.0)}]},
        confidence=0.9,
    )
    lidar_data = SensorData(
        sensor_type=SensorType.LIDAR,
        timestamp=ts,
        data={"clusters": [{"id": 0, "position_2d": (3.1, 2.1)}]},
        confidence=0.95,
    )
    outputs = fusion.update({
        SensorType.CAMERA: camera_data,
        SensorType.LIDAR: lidar_data,
    })
    assert len(outputs) >= 1
    # Fused position should be between camera and lidar
    pos = outputs[0].position_3d
    assert 2.9 <= pos[0] <= 3.2


def test_fusion_with_uwb():
    fusion = SensorFusionV3()
    ts = time.time()
    uwb_data = SensorData(
        sensor_type=SensorType.UWB,
        timestamp=ts,
        data={"positions": [{"id": 0, "x": 3.0, "y": 2.0, "z": 0.5}]},
        confidence=0.85,
    )
    camera_data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        data={"detections": [{"id": 0, "position_2d": (3.0, 2.0)}]},
        confidence=0.9,
    )
    outputs = fusion.update({
        SensorType.CAMERA: camera_data,
        SensorType.UWB: uwb_data,
    })
    assert len(outputs) >= 1
    # Should have z component from UWB
    assert outputs[0].position_3d[2] > 0.0


def test_fusion_tracking_over_time():
    fusion = SensorFusionV3()
    for i in range(10):
        ts = time.time() + i * 0.1
        camera_data = SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=ts,
            data={"detections": [{"id": 0, "position_2d": (1.0 + i * 0.1, 2.0)}]},
            confidence=0.9,
        )
        outputs = fusion.update({SensorType.CAMERA: camera_data})

    assert len(outputs) >= 1
    # Track should have velocity
    assert outputs[0].velocity_3d[0] != 0.0
```

**Step 2-5: Implement, test, commit**

The `SensorFusionV3` class:
- Manages per-track EKF instances
- Routes sensor data to appropriate EKF update methods
- Handles data association (matching sensor observations to existing tracks)
- Uses WeightManager to adjust observation noise
- Returns `List[FusionOutput]` per update cycle

```bash
git commit -m "feat(indoor_fence): implement V3 multi-sensor fusion manager with EKF"
```

---

### Task 15: Upgrade 3D Multi-Target Tracker

**Files:**
- Create: `plugins/indoor_fence/core/tracking/__init__.py`
- Create: `plugins/indoor_fence/core/tracking/multi_target_tracker_v3.py`
- Create: `plugins/indoor_fence/core/tracking/hungarian.py`
- Test: `plugins/indoor_fence/tests/test_tracker_v3.py`

**Step 1: Write the failing test**

```python
"""Tests for V3 multi-target tracker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.tracking.multi_target_tracker_v3 import (
    MultiTargetTrackerV3, TrackState,
)
from plugins.indoor_fence.protocols import FusionOutput, SensorType


def test_tracker_creation():
    tracker = MultiTargetTrackerV3(max_age=30, min_hits=3)
    assert tracker.get_active_tracks() == []


def test_tracker_single_target():
    tracker = MultiTargetTrackerV3(max_age=30, min_hits=1)
    for i in range(5):
        outputs = [FusionOutput(
            track_id=0, position_3d=(float(i), 2.0, 0.0),
            velocity_3d=(1.0, 0.0, 0.0), confidence=0.9,
            sources=[SensorType.CAMERA],
        )]
        tracks = tracker.update(outputs)

    active = tracker.get_active_tracks()
    assert len(active) >= 1
    assert active[0].state == TrackState.ACTIVE


def test_tracker_multiple_targets():
    tracker = MultiTargetTrackerV3(max_age=30, min_hits=1)
    for i in range(5):
        outputs = [
            FusionOutput(track_id=0, position_3d=(float(i), 2.0, 0.0),
                        velocity_3d=(1.0, 0.0, 0.0), confidence=0.9,
                        sources=[SensorType.CAMERA]),
            FusionOutput(track_id=1, position_3d=(8.0 - float(i), 4.0, 0.0),
                        velocity_3d=(-1.0, 0.0, 0.0), confidence=0.9,
                        sources=[SensorType.CAMERA]),
        ]
        tracks = tracker.update(outputs)

    active = tracker.get_active_tracks()
    assert len(active) == 2


def test_tracker_lost_target():
    tracker = MultiTargetTrackerV3(max_age=3, min_hits=1)
    # Target appears
    for i in range(3):
        tracker.update([FusionOutput(
            track_id=0, position_3d=(float(i), 2.0, 0.0),
            confidence=0.9, sources=[SensorType.CAMERA],
        )])

    # Target disappears
    for i in range(5):
        tracker.update([])

    active = tracker.get_active_tracks()
    assert len(active) == 0  # Should be deleted after max_age
```

**Step 2-5: Implement, test, commit**

The tracker uses the Hungarian algorithm for association and manages track lifecycle (init -> confirm -> active -> lost -> delete). Supports 3D state space.

```bash
git commit -m "feat(indoor_fence): implement V3 3D multi-target tracker"
```

---

## Phase 4: State Machine & Rule Engine (Tasks 16-18)

### Task 16: Expand State Machine for V3

**Files:**
- Modify: `plugins/indoor_fence/core/state_machine.py`
- Test: `plugins/indoor_fence/tests/test_state_machine_v3.py`

**Step 1: Write the failing test**

```python
"""Tests for V3 state machine with expanded states."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.state_machine import StateMachine, PersonState
from plugins.indoor_fence.core.geometry import Point2D
from plugins.indoor_fence.core.zone_config import ZoneConfigLoader


def test_person_state_has_new_states():
    assert hasattr(PersonState, 'CLIMBING')
    assert hasattr(PersonState, 'PROLONGED_STAY')
    assert hasattr(PersonState, 'FALLEN')
    assert hasattr(PersonState, 'MULTI_PERSON')


def test_climbing_state():
    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 3.0),
        metadata={"behavior": "climbing"}
    )
    assert result.state == PersonState.CLIMBING


def test_fallen_state():
    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 1.0),
        metadata={"behavior": "fallen"}
    )
    assert result.state == PersonState.FALLEN


def test_prolonged_stay_state():
    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 3.0),
        metadata={"behavior": "prolonged_stay"}
    )
    assert result.state == PersonState.PROLONGED_STAY


def test_behavior_overrides_position():
    """Behavior-based states should take priority when applicable."""
    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)
    # Person in normal zone but fallen
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 1.0),
        metadata={"behavior": "fallen"}
    )
    assert result.state == PersonState.FALLEN  # Behavior overrides normal position
```

**Step 2-5: Implement, test, commit**

Add new states to PersonState enum and update StateMachine.evaluate_person to check behavior metadata. Priority: FALLEN > CLIMBING > behavior states > position states.

```bash
git commit -m "feat(indoor_fence): expand state machine with behavior-based states"
```

---

### Task 17: Build Rule Engine

**Files:**
- Create: `plugins/indoor_fence/core/rules/__init__.py`
- Create: `plugins/indoor_fence/core/rules/rule_engine.py`
- Create: `plugins/indoor_fence/core/rules/risk_scorer.py`
- Test: `plugins/indoor_fence/tests/test_rules.py`

**Step 1: Write the failing test**

```python
"""Tests for rule engine and risk scorer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.rules.rule_engine import (
    RuleEngine, Rule, RuleAction,
)
from plugins.indoor_fence.core.rules.risk_scorer import RiskScorer
from plugins.indoor_fence.protocols import PersonStateV3, RiskAssessment


def test_rule_creation():
    rule = Rule(
        rule_id="cross_line_alarm",
        condition="person_state == 'cross_line'",
        action=RuleAction.ALARM_RED,
        priority=10,
        cooldown_seconds=5.0,
    )
    assert rule.rule_id == "cross_line_alarm"


def test_rule_engine_evaluate():
    engine = RuleEngine()
    engine.add_rule(Rule(
        rule_id="cross_line",
        condition="person_state == 'cross_line'",
        action=RuleAction.ALARM_RED,
        priority=10,
        cooldown_seconds=0,
    ))
    actions = engine.evaluate({
        "person_state": "cross_line",
        "zone_id": "cabinet_1",
    })
    assert RuleAction.ALARM_RED in actions


def test_rule_engine_cooldown():
    engine = RuleEngine()
    engine.add_rule(Rule(
        rule_id="test",
        condition="person_state == 'on_line'",
        action=RuleAction.ALARM_YELLOW,
        priority=5,
        cooldown_seconds=10.0,
    ))
    # First evaluation triggers
    actions1 = engine.evaluate({"person_state": "on_line"})
    assert RuleAction.ALARM_YELLOW in actions1
    # Immediate re-evaluation should be cooled down
    actions2 = engine.evaluate({"person_state": "on_line"})
    assert RuleAction.ALARM_YELLOW not in actions2


def test_rule_engine_from_yaml(tmp_path):
    yaml_content = """
rules:
  - rule_id: cross_line
    condition: "person_state == 'cross_line'"
    action: alarm_red
    priority: 10
    cooldown_seconds: 5
  - rule_id: climbing
    condition: "person_state == 'climbing'"
    action: alarm_red
    priority: 9
    cooldown_seconds: 3
"""
    yaml_file = tmp_path / "rules.yaml"
    yaml_file.write_text(yaml_content)
    engine = RuleEngine.from_yaml(str(yaml_file))
    assert len(engine.rules) == 2


def test_risk_scorer():
    scorer = RiskScorer()
    score = scorer.score(
        person_state=PersonStateV3.CROSS_LINE,
        distance_to_danger=0.05,
        is_authorized=False,
        behavior="normal",
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.7  # Cross line + unauthorized should be high risk


def test_risk_scorer_low_risk():
    scorer = RiskScorer()
    score = scorer.score(
        person_state=PersonStateV3.NORMAL,
        distance_to_danger=2.0,
        is_authorized=True,
        behavior="normal",
    )
    assert score < 0.3  # Normal + authorized + far from danger
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat(indoor_fence): add configurable rule engine and multi-dimensional risk scorer"
```

---

### Task 18: Add Adaptive Thresholds

**Files:**
- Create: `plugins/indoor_fence/core/rules/adaptive_threshold.py`
- Test: `plugins/indoor_fence/tests/test_adaptive.py`

**Step 1: Write the failing test**

```python
"""Tests for adaptive threshold module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.rules.adaptive_threshold import AdaptiveThreshold


def test_adaptive_threshold_creation():
    at = AdaptiveThreshold(
        name="warning_distance",
        initial_value=0.3,
        min_value=0.1,
        max_value=1.0,
    )
    assert at.current_value == 0.3


def test_adaptive_threshold_feedback():
    at = AdaptiveThreshold(
        name="warning_distance",
        initial_value=0.3,
        min_value=0.1,
        max_value=1.0,
        learning_rate=0.1,
    )
    # Report false positives -> threshold should increase
    for _ in range(20):
        at.report_event(false_positive=True)
    assert at.current_value > 0.3


def test_adaptive_threshold_true_positive():
    at = AdaptiveThreshold(
        name="warning_distance",
        initial_value=0.5,
        min_value=0.1,
        max_value=1.0,
        learning_rate=0.1,
    )
    # Report true positives (missed alarms) -> threshold should decrease
    for _ in range(20):
        at.report_event(missed_alarm=True)
    assert at.current_value < 0.5


def test_adaptive_threshold_bounds():
    at = AdaptiveThreshold(
        name="test",
        initial_value=0.5,
        min_value=0.2,
        max_value=0.8,
        learning_rate=0.5,
    )
    # Push to boundary
    for _ in range(100):
        at.report_event(false_positive=True)
    assert at.current_value <= 0.8

    for _ in range(100):
        at.report_event(missed_alarm=True)
    assert at.current_value >= 0.2
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat(indoor_fence): add adaptive threshold with feedback learning"
```

---

## Phase 5: Standalone Service & Training (Tasks 19-23)

### Task 19: Data Recorder

**Files:**
- Create: `plugins/indoor_fence/standalone/data_recorder.py`
- Test: `plugins/indoor_fence/tests/test_data_recorder.py`

**Step 1: Write the failing test**

```python
"""Tests for multi-sensor data recorder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import time
from plugins.indoor_fence.standalone.data_recorder import DataRecorder
from plugins.indoor_fence.protocols import SensorData, SensorType


def test_recorder_creation():
    rec = DataRecorder()
    assert rec.is_recording is False


def test_recorder_start_stop(tmp_path):
    rec = DataRecorder(output_dir=str(tmp_path))
    rec.start()
    assert rec.is_recording is True

    data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=time.time(),
        data={"test": True},
        confidence=0.9,
    )
    rec.record(data)
    rec.record(data)
    rec.stop()

    assert rec.is_recording is False
    # Should have created output files
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) >= 1


def test_recorder_multiple_sensors(tmp_path):
    rec = DataRecorder(output_dir=str(tmp_path))
    rec.start()

    for sensor in [SensorType.CAMERA, SensorType.LIDAR, SensorType.UWB]:
        rec.record(SensorData(
            sensor_type=sensor,
            timestamp=time.time(),
            data={"sensor": sensor.value},
            confidence=0.9,
        ))

    rec.stop()
    # Should have separate files per sensor
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) >= 1


def test_recorder_metadata(tmp_path):
    rec = DataRecorder(output_dir=str(tmp_path))
    rec.start(metadata={"scene": "test", "operator": "test_user"})
    rec.stop()

    # Should have metadata file
    meta_files = list(tmp_path.glob("*metadata*.json"))
    assert len(meta_files) >= 1
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat(indoor_fence): add multi-sensor data recorder with metadata"
```

---

### Task 20: Data Replayer

**Files:**
- Create: `plugins/indoor_fence/standalone/data_replayer.py`
- Test: `plugins/indoor_fence/tests/test_data_replayer.py`

**Step 1: Write the failing test**

```python
"""Tests for data replay engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import json
import time
from plugins.indoor_fence.standalone.data_replayer import DataReplayer
from plugins.indoor_fence.protocols import SensorData, SensorType


@pytest.fixture
def replay_dir(tmp_path):
    """Create sample recording data."""
    data_file = tmp_path / "camera.jsonl"
    with open(data_file, "w") as f:
        for i in range(10):
            entry = SensorData(
                sensor_type=SensorType.CAMERA,
                timestamp=1000.0 + i * 0.1,
                data={"frame": i, "detections": []},
                confidence=0.9,
            )
            f.write(entry.model_dump_json() + "\n")
    return tmp_path


def test_replayer_load(replay_dir):
    replayer = DataReplayer(str(replay_dir))
    assert replayer.total_frames >= 10


def test_replayer_next(replay_dir):
    replayer = DataReplayer(str(replay_dir))
    data = replayer.next()
    assert data is not None
    assert data[SensorType.CAMERA].data["frame"] == 0

    data = replayer.next()
    assert data[SensorType.CAMERA].data["frame"] == 1


def test_replayer_reset(replay_dir):
    replayer = DataReplayer(str(replay_dir))
    replayer.next()
    replayer.next()
    replayer.reset()
    data = replayer.next()
    assert data[SensorType.CAMERA].data["frame"] == 0


def test_replayer_exhausted(replay_dir):
    replayer = DataReplayer(str(replay_dir))
    for _ in range(10):
        replayer.next()
    data = replayer.next()
    assert data is None  # No more frames
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat(indoor_fence): add data replay engine for recorded sensor streams"
```

---

### Task 21: Training Pipeline Scaffold

**Files:**
- Create: `plugins/indoor_fence/standalone/training_pipeline.py`
- Test: `plugins/indoor_fence/tests/test_training.py`

**Step 1: Write the failing test**

```python
"""Tests for training pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.standalone.training_pipeline import (
    TrainingPipeline, TrainingConfig, TrainingStatus, DatasetInfo,
)


def test_training_config():
    config = TrainingConfig(
        model_type="yolov8n",
        epochs=10,
        batch_size=16,
        learning_rate=0.001,
    )
    assert config.model_type == "yolov8n"


def test_pipeline_creation():
    pipeline = TrainingPipeline()
    assert pipeline.status == TrainingStatus.IDLE


def test_register_dataset(tmp_path):
    pipeline = TrainingPipeline(data_dir=str(tmp_path))
    # Create a minimal dataset directory
    ds_dir = tmp_path / "ds1"
    ds_dir.mkdir()
    (ds_dir / "images").mkdir()
    (ds_dir / "labels").mkdir()
    # Create a dummy image and label
    import numpy as np
    np.save(str(ds_dir / "images" / "0001.npy"), np.zeros((480, 640, 3)))
    (ds_dir / "labels" / "0001.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    info = pipeline.register_dataset("ds1", str(ds_dir))
    assert isinstance(info, DatasetInfo)
    assert info.num_images >= 1


def test_training_status_enum():
    assert TrainingStatus.IDLE.value == "idle"
    assert TrainingStatus.TRAINING.value == "training"
    assert TrainingStatus.COMPLETED.value == "completed"
    assert TrainingStatus.FAILED.value == "failed"
```

**Step 2-5: Implement, test, commit**

The training pipeline is a scaffold that manages datasets, training configs, and status. Actual training execution is a placeholder that can be connected to PyTorch/ultralytics later.

```bash
git commit -m "feat(indoor_fence): add training pipeline scaffold for model fine-tuning"
```

---

### Task 22: Simulator & Training API Routes

**Files:**
- Create: `plugins/indoor_fence/standalone/simulator_routes.py`
- Create: `plugins/indoor_fence/standalone/training_routes.py`
- Test: `plugins/indoor_fence/tests/test_api_routes.py`

**Step 1: Write the failing test**

```python
"""Tests for simulator and training API routes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from fastapi.testclient import TestClient
from plugins.indoor_fence.standalone.simulator_routes import create_simulator_router
from plugins.indoor_fence.standalone.training_routes import create_training_router
from plugins.indoor_fence.adapters.simulator import Simulator
from plugins.indoor_fence.standalone.training_pipeline import TrainingPipeline
from fastapi import FastAPI


@pytest.fixture
def sim_client():
    app = FastAPI()
    sim = Simulator.default()
    app.include_router(create_simulator_router(sim), prefix="/api/simulator")
    return TestClient(app)


@pytest.fixture
def training_client(tmp_path):
    app = FastAPI()
    pipeline = TrainingPipeline(data_dir=str(tmp_path))
    app.include_router(create_training_router(pipeline), prefix="/api/training")
    return TestClient(app)


def test_simulator_start(sim_client):
    resp = sim_client.post("/api/simulator/start", json={
        "num_persons": 2,
        "sensor_types": ["camera", "lidar"],
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_simulator_step(sim_client):
    sim_client.post("/api/simulator/start", json={
        "num_persons": 1,
        "sensor_types": ["camera"],
    })
    resp = sim_client.post("/api/simulator/step")
    assert resp.status_code == 200
    assert "data" in resp.json()


def test_simulator_scenarios(sim_client):
    resp = sim_client.get("/api/simulator/scenarios")
    assert resp.status_code == 200
    assert "scenarios" in resp.json()


def test_training_status(training_client):
    resp = training_client.get("/api/training/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat(indoor_fence): add simulator and training API routes"
```

---

### Task 23: Refactor Standalone App & Plugin Integration

**Files:**
- Modify: `plugins/indoor_fence/standalone/app.py`
- Modify: `plugins/indoor_fence/plugin.py`
- Modify: `plugins/indoor_fence/__init__.py`
- Modify: `plugins/indoor_fence/adapters/__init__.py`
- Modify: `plugins/indoor_fence/core/__init__.py`
- Test: `plugins/indoor_fence/tests/test_standalone_v3.py`

**Step 1: Write the failing test**

```python
"""Tests for V3 standalone application."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest


def test_plugin_import():
    """Verify all V3 modules can be imported."""
    from plugins.indoor_fence import IndoorFencePlugin
    from plugins.indoor_fence.protocols import (
        SensorData, SensorType, PersonStateV3, RiskAssessment,
    )
    from plugins.indoor_fence.adapters.uwb_adapter import UWBAdapter
    from plugins.indoor_fence.adapters.imu_adapter import IMUAdapter
    from plugins.indoor_fence.adapters.simulator import Simulator
    from plugins.indoor_fence.detection.object_detector import ObjectDetector
    from plugins.indoor_fence.detection.pose_estimator import PoseEstimatorV3
    from plugins.indoor_fence.detection.behavior_recognizer import BehaviorRecognizerV3
    from plugins.indoor_fence.core.fusion.ekf_fusion import EKF6DOF
    from plugins.indoor_fence.core.fusion.sensor_fusion_v3 import SensorFusionV3
    from plugins.indoor_fence.core.rules.rule_engine import RuleEngine
    from plugins.indoor_fence.core.rules.risk_scorer import RiskScorer
    from plugins.indoor_fence.standalone.data_recorder import DataRecorder
    from plugins.indoor_fence.standalone.data_replayer import DataReplayer
    from plugins.indoor_fence.standalone.training_pipeline import TrainingPipeline


def test_plugin_create_standalone():
    from plugins.indoor_fence.plugin import IndoorFencePlugin
    plugin = IndoorFencePlugin.create_standalone()
    assert plugin is not None
    assert plugin.id == "indoor_fence"


def test_plugin_standalone_routes():
    from plugins.indoor_fence.plugin import IndoorFencePlugin
    plugin = IndoorFencePlugin.create_standalone()
    routes = plugin.get_standalone_routes()
    assert isinstance(routes, list)
    # V3 should have simulator and training routes
    route_paths = [r["path"] for r in routes]
    assert any("simulator" in p for p in route_paths)
    assert any("training" in p or "data" in p for p in route_paths)


def test_standalone_server_startup():
    """Test that the standalone server can be created without crash."""
    from plugins.indoor_fence.plugin import IndoorFencePlugin
    from darkbreaker_sdk.standalone import StandalonePluginRunner

    plugin = IndoorFencePlugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent.parent / "standalone" / "templates",
        plugin_static_dir=Path(__file__).parent.parent / "standalone" / "static",
    )
    assert runner.app is not None
```

**Step 2: Run test, verify fail**

**Step 3: Integrate all V3 modules**

Update `plugin.py`:
- Wire SensorFusionV3 (EKF-based) into the infer pipeline
- Wire BehaviorRecognizerV3 into detection pipeline
- Wire RuleEngine into postprocess pipeline
- Add simulator and training routes to `get_standalone_routes()`
- Update `create_standalone()` to initialize all V3 components

Update `__init__.py`: export new V3 classes
Update `adapters/__init__.py`: export UWBAdapter, IMUAdapter
Update `core/__init__.py`: export new fusion, tracking, rules modules

Update `standalone/app.py`: register simulator and training routers

**Step 4: Run all tests**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add plugins/indoor_fence/
git commit -m "feat(indoor_fence): integrate all V3 modules into plugin and standalone app"
```

---

## Phase 6: V3 Config & Final Integration (Tasks 24-26)

### Task 24: Update Default Configuration for V3

**Files:**
- Modify: `plugins/indoor_fence/configs/default.yaml`
- Create: `plugins/indoor_fence/configs/scenarios/cross_line.json`
- Create: `plugins/indoor_fence/configs/scenarios/multi_person.json`

Update default.yaml to add:
- `uwb:` section (enabled: false by default, protocol: simulate)
- `imu:` section (enabled: false by default)
- `detection:` section (pose_enabled, behavior_enabled, auto_fence_enabled)
- `fusion.algorithm:` "ekf" (default) or "simple"
- `rules:` section with default rules
- `recording:` section
- `training:` section

Create scenario files for cross_line and multi_person simulation.

**Commit:**
```bash
git commit -m "feat(indoor_fence): update V3.0 default config with all new modules"
```

---

### Task 25: End-to-End Integration Test

**Files:**
- Create: `plugins/indoor_fence/tests/test_integration.py`

**Step 1: Write the integration test**

```python
"""End-to-end integration test for V3 indoor fence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from plugins.indoor_fence.plugin import IndoorFencePlugin
from darkbreaker_sdk.interfaces.base_plugin import PluginContext
from darkbreaker_sdk.schemas.common import ROI


def test_full_pipeline_simulation():
    """Run the full V3 pipeline with simulated sensors."""
    plugin = IndoorFencePlugin.create_standalone()
    assert plugin.init({
        "camera": {"enabled": True, "source": "simulate"},
        "lidar": {"enabled": True, "device_ip": "simulate"},
        "uwb": {"enabled": False},
        "fusion": {"algorithm": "ekf"},
    })

    context = PluginContext(
        task_id="test-001",
        site_id="test-site",
        device_id="test-device",
    )

    # Run multiple inference cycles
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(10):
        results = plugin.infer(frame, [], context)
        assert isinstance(results, list)

    # Postprocess should generate alarms for any violations
    alarms = plugin.postprocess(results, [])
    assert isinstance(alarms, list)

    # Health check
    health = plugin.healthcheck()
    assert health is not None

    # Cleanup
    plugin.cleanup()


def test_simulator_driven_pipeline():
    """Run V3 pipeline driven by simulator data."""
    from plugins.indoor_fence.adapters.simulator import Simulator, SimulatorConfig
    from plugins.indoor_fence.protocols import SensorType

    config = SimulatorConfig(
        num_persons=2,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
        path_type="cross_line",
        yellow_line_y=2.5,
    )
    sim = Simulator(config)

    plugin = IndoorFencePlugin.create_standalone()
    plugin.init({})

    context = PluginContext(
        task_id="sim-001",
        site_id="sim-site",
        device_id="sim-device",
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    all_results = []
    for _ in range(20):
        sim_data = sim.step()
        results = plugin.infer(frame, [], context)
        all_results.extend(results)

    assert len(all_results) > 0
    plugin.cleanup()
```

**Step 2: Run test**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/test_integration.py -v`

**Step 3: Fix any failures**

**Step 4: Commit**

```bash
git commit -m "test(indoor_fence): add end-to-end V3 integration tests"
```

---

### Task 26: Verify Standalone Startup & Run Full Test Suite

**Step 1: Run the full test suite**

Run: `cd /Users/ronan/Desktop/DarkBreaker && python3 -m pytest plugins/indoor_fence/tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify standalone startup**

Run: `cd /Users/ronan/Desktop/DarkBreaker && timeout 8 python3 -m plugins.indoor_fence 2>&1 || true`
Expected: Server starts, banner printed, no errors

**Step 3: Final commit**

```bash
git add -A plugins/indoor_fence/
git commit -m "feat(indoor_fence): complete V3.0 architecture upgrade

- Fix Python 3.9 compatibility in SDK runner
- Add V3 data protocols (Pydantic models)
- Enhance BaseAdapter with recording, replay, simulation modes
- Add UWB adapter with NLOS detection
- Add IMU adapter
- Add unified multi-sensor simulator with scenarios
- Extract detection module with YOLO detector
- Add pose estimator with posture classification
- Add behavior recognizer with temporal analysis
- Add automatic fence generator
- Add model lifecycle manager
- Implement 6-DOF EKF for multi-sensor fusion
- Add NLOS detector and dynamic weight manager
- Implement V3 multi-sensor fusion manager
- Upgrade 3D multi-target tracker
- Expand state machine with behavior states
- Add configurable rule engine
- Add multi-dimensional risk scorer
- Add adaptive thresholds with feedback learning
- Add data recorder and replayer
- Add training pipeline scaffold
- Add simulator and training API routes
- Full integration tests"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1 | 1-6 | Fix startup, protocols, adapters (UWB, IMU, simulator) |
| Phase 2 | 7-11 | Detection module (YOLO, pose, behavior, auto-fence, model mgr) |
| Phase 3 | 12-15 | Fusion & tracking (EKF, NLOS, weights, 3D tracker) |
| Phase 4 | 16-18 | State machine expansion, rule engine, adaptive thresholds |
| Phase 5 | 19-23 | Standalone services (recorder, replayer, training, API routes) |
| Phase 6 | 24-26 | Config update, integration tests, final verification |

Total: **26 tasks**, each following TDD (test first → implement → verify → commit).
