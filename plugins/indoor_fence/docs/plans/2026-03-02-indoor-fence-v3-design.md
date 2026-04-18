# Indoor Fence V3.0 Architecture Design

## Background

The indoor_fence plugin V2.1 is a multi-person safety monitoring system for electrical substations, using 2D LiDAR + camera fusion. The upgrade to V3.0 addresses deficiencies identified in the algorithm improvement proposal document:

1. Sensor limited to 2D plane positioning (no height/3D awareness)
2. Static zone config and thresholds (no adaptive capability)
3. Simple visual detection (no advanced DL models, pose, or behavior)
4. Basic fusion algorithm (simple Kalman, no EKF/UKF)
5. No simulation, data recording, or self-learning pipeline

**Current startup bug**: SDK runner uses Python 3.10+ union syntax (`str | None`) but runtime is Python 3.9.6.

## Architecture Overview

Seven-layer modular architecture with cross-cutting concerns:

```
Layer 7: Standalone Service     (FastAPI + WebSocket + Simulator + Training API)
Layer 6: Output & Control       (Alarm engine + Light + Audit + Push)
Layer 5: Rule Engine            (Adaptive thresholds + Multi-dim rules)
Layer 4: State Machine          (3D zone logic + Behavior state + Risk assessment)
Layer 3: Fusion & Tracking      (EKF/UKF multi-sensor fusion + 3D tracking)
Layer 2: Detection & Recognition(DL detection + Pose + Behavior + Auto-fence)
Layer 1: Sensor Adaptation      (Camera + LiDAR + UWB + IMU + BLE)

Cross-cutting: Config Manager | Data Recorder/Replayer | Training Pipeline | Health Monitor
```

## Module Layout

```
plugins/indoor_fence/
├── plugin.py                    # Main plugin class (refactored)
├── manifest.json
├── requirements.txt
├── __init__.py
├── __main__.py
├── run_standalone.py
│
├── adapters/                    # Layer 1: Sensor Adaptation
│   ├── base_adapter.py          # Unified interface (refactored)
│   ├── camera_adapter.py        # Camera (enhanced)
│   ├── lidar_adapter.py         # 2D LiDAR (enhanced simulation)
│   ├── uwb_adapter.py           # NEW: UWB 3D positioning
│   ├── imu_adapter.py           # NEW: IMU inertial sensor
│   ├── light_adapter.py         # Light control (retained)
│   └── simulator.py             # NEW: Unified simulation data generator
│
├── detection/                   # Layer 2: Detection & Recognition
│   ├── __init__.py
│   ├── object_detector.py       # NEW: Unified detection interface
│   ├── yolo_detector.py         # YOLO detector (extracted from camera_adapter)
│   ├── pose_estimator.py        # NEW: Pose estimation (MoveNet/HRNet)
│   ├── behavior_recognizer.py   # NEW: Behavior recognition (Transformer/ST-GCN)
│   ├── auto_fence_generator.py  # NEW: Automatic fence generation
│   └── model_manager.py         # NEW: Model lifecycle management
│
├── core/                        # Layers 3-5
│   ├── __init__.py
│   ├── geometry.py              # 2D/3D geometry (enhanced)
│   ├── zone_config.py           # Zone configuration (enhanced for 3D)
│   ├── state_machine.py         # State machine (expanded states)
│   ├── config_manager.py        # Configuration management
│   │
│   ├── fusion/                  # Layer 3: Fusion & Tracking
│   │   ├── __init__.py
│   │   ├── ekf_fusion.py        # NEW: Extended Kalman Filter
│   │   ├── ukf_fusion.py        # NEW: Unscented Kalman Filter
│   │   ├── sensor_fusion.py     # Refactored: Multi-sensor fusion manager
│   │   ├── nlos_detector.py     # NEW: NLOS detection & compensation
│   │   └── weight_manager.py    # NEW: Dynamic weight adjustment
│   │
│   ├── tracking/                # Layer 3: Multi-target tracking
│   │   ├── __init__.py
│   │   ├── multi_target_tracker.py  # Refactored: Enhanced 3D tracking
│   │   ├── kalman_filter.py         # Refactored: 3D state support
│   │   └── hungarian.py             # Hungarian algorithm
│   │
│   └── rules/                   # Layer 5: Rule Engine
│       ├── __init__.py
│       ├── rule_engine.py       # NEW: Configurable rule engine
│       ├── adaptive_threshold.py # NEW: Self-adaptive thresholds
│       └── risk_scorer.py       # NEW: Multi-dimensional risk scoring
│
├── standalone/                  # Layers 6-7
│   ├── app.py                   # FastAPI app (refactored)
│   ├── simulator_routes.py      # NEW: Simulation control API
│   ├── training_routes.py       # NEW: Data upload/training API
│   ├── data_recorder.py         # NEW: Multi-sensor data recording
│   ├── data_replayer.py         # NEW: Data replay engine
│   ├── training_pipeline.py     # NEW: Model training pipeline
│   ├── configs/
│   │   └── zone.yaml
│   └── templates/
│       └── indoor_fence.html
│
├── tests/
│   ├── conftest.py
│   ├── test_adapters.py         # Adapter unit tests
│   ├── test_detection.py        # Detection module tests
│   ├── test_fusion.py           # Fusion algorithm tests
│   ├── test_state_machine.py    # State machine tests
│   ├── test_rules.py            # Rule engine tests
│   ├── test_standalone.py       # Standalone mode tests
│   ├── test_simulator.py        # Simulator tests
│   └── test_integration.py      # End-to-end integration tests
│
├── configs/
│   ├── default.yaml             # Default configuration (v3.0 schema)
│   └── scenarios/               # NEW: Simulation scenarios
│       ├── basic_patrol.json
│       ├── cross_line.json
│       └── multi_person.json
│
├── scripts/
│   └── benchmark.py
├── data/
└── logs/
```

## Layer 1: Sensor Adaptation

### BaseAdapter Interface (Enhanced)

```python
class BaseAdapter(ABC):
    def connect(self) -> bool
    def disconnect(self) -> None
    def get_data(self) -> Optional[SensorData]
    def get_health(self) -> AdapterHealth
    def start_recording(self, path: str) -> None
    def stop_recording(self) -> None
    def load_replay(self, path: str) -> None
    def get_weight(self) -> float  # Dynamic trust weight
    @property
    def is_simulated(self) -> bool
    @property
    def mode(self) -> AdapterMode  # LIVE / SIMULATED / REPLAY
```

### UWB Adapter

- Protocols: DW1000/DW3000 series, custom UDP
- Output: 3D coordinates `(x, y, z)` + accuracy + timestamp
- NLOS detection via CIR feature analysis
- Simulation: noisy 3D trajectory generation
- Quick deployment: mobile/grid UWB layout support

### Unified Simulator

- Configurable: person count, path patterns (random/fixed/crossing/climbing)
- Synchronized multi-sensor data streams (time-aligned)
- Scenario scripts: JSON-defined behavior sequences
- Fault injection: occlusion, sensor failure, NLOS conditions

## Layer 2: Detection & Recognition

### Object Detector

- Unified `ObjectDetector` interface, multi-backend (ONNX/TensorRT/OpenVINO)
- Default: YOLOv8n, upgradable to YOLOv7/PP-YOLOE/ResNeXt-FPN
- Categories: person, dangerous equipment, safety helmet/uniform
- Output: bbox, confidence, class, foot_point

### Pose Estimator

- Lightweight: MoveNet Lightning (edge) / HRNet-W32 (server)
- 17 keypoints output
- Posture classification: standing, bending, climbing, crouching, fallen
- Optional — falls back to bbox-only mode

### Behavior Recognizer

- Sliding window temporal model: recent N frames of keypoints -> classification
- Categories: normal walk, prolonged stay, climbing, crossing, fallen/anomaly
- Architecture: lightweight Transformer or ST-GCN
- Configurable time thresholds per behavior type

### Auto Fence Generator

- Morphological operations on detection bounding boxes
- Buffer zone computation around dangerous equipment
- Multi-layer alert zones (warning/danger/critical)
- Integration with 3D model/point cloud slicing
- Real-time update as equipment positions change

## Layer 3: Fusion & Tracking

### EKF Fusion

- State vector: `[x, y, z, vx, vy, vz]` (6-DOF)
- Observation models per sensor:
  - Camera: 2D projection (H_cam)
  - LiDAR: 2D range-bearing (H_lidar)
  - UWB: 3D position (H_uwb)
  - IMU: acceleration (H_imu)
- Per-sensor R matrix (observation noise) independently configurable
- Dynamic weight: auto-adjust R based on sensor health + environment

### NLOS Compensation

- ML classifier on UWB CIR features for NLOS identification
- Auto-inflate observation noise for NLOS measurements
- Visual tracking loss -> UWB weight boost
- Night/low-light -> UWB + LiDAR weight increase

### 3D Multi-Target Tracking

- Extended from 2D to 3D state space
- Support for stairs, scaffolding, elevated areas
- Track lifecycle: init -> confirm -> active -> lost -> delete

## Layer 4: State Machine (Enhanced)

### Person States

```python
class PersonState(Enum):
    NORMAL = "normal"
    ON_LINE = "on_line"
    CROSS_LINE = "cross_line"
    MISPLACED = "misplaced"
    HIGH_RISK = "high_risk"
    CLIMBING = "climbing"
    PROLONGED_STAY = "prolonged_stay"
    FALLEN = "fallen"
    MULTI_PERSON = "multi_person"
```

### Equipment States (New)

- Door-not-closed, cabinet anomaly, equipment displacement

### 3D Zone Logic

- Volume-based zones (not just 2D polygons)
- Height-aware risk assessment
- Per-equipment-type, per-zone configurable rules

## Layer 5: Rule Engine

### Configurable Rules

Each rule: `(condition_expr, action, priority, cooldown_seconds)`

### Adaptive Thresholds

- Historical event frequency statistics
- Auto-adjust warning/danger distances
- Balance alert sensitivity vs false positive rate

### Risk Scoring

`risk_score = position_risk * behavior_risk * authorization_factor`

## Layers 6-7: Output & Standalone Service

### Standalone Service APIs

**Simulation:**
- `POST /api/simulator/start` — Start simulation scene
- `POST /api/simulator/inject` — Inject sensor data
- `GET /api/simulator/scenarios` — List predefined scenarios
- `POST /api/simulator/scenario/{id}/run` — Run scenario script

**Data Recording & Training:**
- `POST /api/data/start_recording` — Start recording all sensor streams
- `POST /api/data/upload` — Upload annotated data
- `POST /api/training/start` — Start model training task
- `GET /api/training/status` — Query training progress
- `POST /api/training/deploy` — Deploy trained model

**Health Monitoring:**
- Per-sensor independent health status
- Auto-recovery: sensor failure -> degrade (reduce weight / switch to simulation)
- Prometheus-style metrics output

## Data Protocols

All inter-layer communication via Pydantic models:

```python
class SensorData(BaseModel):
    sensor_type: str
    timestamp: float
    data: Dict[str, Any]
    confidence: float
    is_simulated: bool = False

class DetectionResult(BaseModel):
    track_id: int
    position_3d: Tuple[float, float, float]
    velocity_3d: Tuple[float, float, float]
    bbox: Optional[Tuple[int, int, int, int]]
    pose_keypoints: Optional[List[Tuple[float, float, float]]]
    behavior: Optional[str]
    confidence: float
    fusion_sources: List[str]

class RiskAssessment(BaseModel):
    person_state: PersonState
    risk_score: float
    zone_id: Optional[str]
    violations: List[str]
    recommended_action: str
```

## Python 3.9 Compatibility

All code MUST use:
- `Optional[X]` instead of `X | None`
- `Union[X, Y]` instead of `X | Y`
- `from __future__ import annotations` at file top
- `Dict`, `List`, `Tuple` from `typing` module

## Key Design Decisions

1. **Preserve working code** — Camera, LiDAR, light adapters retain core logic; detection logic extracted from camera_adapter into detection/ module
2. **Simulation-first** — Every adapter has built-in simulation mode; complete system can run without any hardware
3. **Data recording built-in** — All sensor streams can be recorded and replayed for debugging and training
4. **Modular upgrade** — Each layer independently testable and replaceable
5. **Config-driven** — All thresholds, rules, and behaviors configurable via YAML/API
