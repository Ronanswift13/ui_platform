# DarkBreaker Plugin SDK Architecture Refactoring Design

**Date**: 2026-02-18
**Status**: Approved
**Approach**: Plugin SDK Layer (Approach A)

---

## 1. Problem Statement

Current plugins have hard dependencies on `platform_core`:
- Import `BasePlugin` from `platform_core.plugin_manager.base`
- Import schemas (`RecognitionResult`, `Alarm`, `ROI`, etc.) from `platform_core.schema.models`
- Cannot run independently without the full platform installed
- No plugin-level dependency management (`requirements.txt`)
- No standalone UI for individual plugin demonstration
- No standardized demo/script/test structure across plugins

**Goal**: Every plugin (17 total: 10 outdoor + 7 indoor) must be able to:
1. Run independently with its own FastAPI server and Bootstrap+Jinja2 UI
2. Be debugged, tested, and demoed without the platform
3. Integrate seamlessly into the main platform UI when enabled
4. Have its own requirements.txt, demo scripts, benchmark tools
5. Support future paid subscription model (architecture-ready)

---

## 2. Architecture Overview

### Dependency Flow (Before vs After)

**Before:**
```
plugins/* ──depends──> platform_core (monolithic)
apps/*    ──depends──> platform_core
```

**After:**
```
darkbreaker_sdk        (lightweight, ~20 files, zero platform dependency)
    ^                       ^
    |                       |
plugins/*              platform_core (slimmed, orchestration only)
    ^                       ^
    |                       |
    +----------- apps/* ---+
```

### Key Principle
- `darkbreaker_sdk` defines ALL interfaces and schemas
- Plugins depend ONLY on `darkbreaker_sdk`
- `platform_core` depends on `darkbreaker_sdk` (dependency inversion)
- `platform_core/schema/models.py` becomes a compatibility re-export layer

---

## 3. darkbreaker_sdk Package Design

```
darkbreaker_sdk/
├── __init__.py                 # Version, top-level exports
├── interfaces/
│   ├── __init__.py
│   ├── base_plugin.py         # BasePlugin ABC + PluginManifest + PluginContext
│   ├── base_adapter.py        # BaseAdapter ABC for device adapters
│   └── lifecycle.py           # PluginStatus, HealthStatus, PluginCapability enums
├── schemas/
│   ├── __init__.py
│   ├── detection.py           # BoundingBox, RecognitionResult
│   ├── alarm.py               # Alarm, AlarmLevel, AlarmRule, AlarmStatus
│   ├── plugin_io.py           # PluginOutput
│   └── common.py              # ROI, ROIType, Evidence, EvidenceType
├── standalone/
│   ├── __init__.py
│   ├── runner.py              # StandalonePluginRunner - generic FastAPI server
│   ├── templates/
│   │   ├── base_standalone.html
│   │   ├── plugin_dashboard.html
│   │   └── components/
│   │       ├── video_panel.html
│   │       ├── detection_list.html
│   │       ├── alarm_panel.html
│   │       ├── config_editor.html
│   │       └── stats_panel.html
│   └── static/
│       ├── css/standalone.css
│       └── js/standalone.js
└── utils/
    ├── __init__.py
    ├── logging.py             # Standard Python logging wrapper
    ├── config.py              # YAML/JSON config loader
    └── model_loader.py        # ONNX Runtime model loader utility
```

### 3.1 interfaces/base_plugin.py

Extracted from `platform_core/plugin_manager/base.py`. Key changes:
- Zero dependency on `platform_core`
- Import schemas from `darkbreaker_sdk.schemas`
- Add `create_standalone()` classmethod for direct instantiation (no PluginManager)
- Add `get_standalone_routes()` method for plugin-specific API endpoints

### 3.2 interfaces/base_adapter.py

New abstract base class formalizing the adapter pattern from `indoor_fence`:
```python
class BaseAdapter(ABC):
    def connect() -> bool
    def disconnect() -> None
    def status -> AdapterStatus
    def is_simulated -> bool
```

### 3.3 standalone/runner.py

Generic FastAPI mini-server that can host ANY plugin:
```python
class StandalonePluginRunner:
    def __init__(self, plugin_class, plugin_config=None):
        self.app = FastAPI()
        self.plugin = plugin_class.create_standalone(plugin_config)
        self._register_routes()
        self._register_plugin_routes()

    def run(self, host="0.0.0.0", port=8080):
        uvicorn.run(self.app, host=host, port=port)
```

Standard routes provided by runner:
- `GET /` - Plugin dashboard UI
- `GET /api/status` - Plugin health/status
- `POST /api/detect` - Run detection on uploaded image
- `GET /api/config` - Get current config
- `PUT /api/config` - Update config
- `WS /ws/stream` - Real-time detection stream
- Plugin-specific routes from `get_standalone_routes()`

---

## 4. Plugin Standard Structure (per plugin)

Every plugin follows this structure:

```
plugins/{plugin_id}/
├── manifest.json              # Plugin metadata (existing, enhanced)
├── requirements.txt           # NEW: Plugin-specific Python dependencies
├── setup.py                   # NEW: Optional pip-installable setup
├── __init__.py
├── plugin.py                  # Main entry (refactored: imports from darkbreaker_sdk)
├── core/                      # Algorithm core (PRESERVED, no changes to algorithms)
│   └── ...
├── adapters/                  # Device adapters (if applicable)
│   └── ...
├── models/                    # Pre-trained models (existing)
│   └── ...
├── configs/                   # Default configs (existing)
│   └── default.yaml
├── standalone/                # NEW: Independent running capability
│   ├── __init__.py
│   ├── app.py                 # FastAPI standalone application
│   ├── templates/             # Plugin-specific Jinja2 templates
│   │   └── {plugin_id}.html   # Custom dashboard for this plugin
│   └── static/                # Plugin-specific CSS/JS
│       └── {plugin_id}.js
├── demo/                      # NEW: Demo scripts (or enhanced from existing)
│   ├── __init__.py
│   ├── run_demo.py            # Main demo entry point
│   ├── demo_*.py              # Individual feature demos
│   └── sample_data/           # Sample images/data for demo
├── scripts/                   # NEW: Utility scripts
│   ├── benchmark.py           # Performance benchmarking
│   ├── export_config.py       # Config export tool
│   └── test_model.py          # Model validation script
├── tests/                     # Tests (enhanced)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core.py           # Core algorithm tests
│   ├── test_plugin.py         # Plugin interface tests
│   └── test_standalone.py     # NEW: Standalone running tests
└── README.md                  # Plugin documentation
```

### 4.1 Plugin Independence Requirements

Each plugin MUST support these execution modes:

| Mode | Command | Description |
|------|---------|-------------|
| **Standalone Server** | `python -m plugins.{id}.standalone.app` | Full web UI with detection |
| **Demo** | `python -m plugins.{id}.demo.run_demo` | Console-based feature demos |
| **Unit Test** | `pytest plugins/{id}/tests/` | Independent test suite |
| **Benchmark** | `python -m plugins.{id}.scripts.benchmark` | Performance profiling |
| **Platform Integration** | Via PluginManager | Normal platform operation |

---

## 5. platform_core Compatibility Layer

### 5.1 schema/models.py (backward compatibility)

```python
"""Backward compatibility - re-export from darkbreaker_sdk"""
from darkbreaker_sdk.schemas import *
from darkbreaker_sdk.schemas.detection import BoundingBox, RecognitionResult
from darkbreaker_sdk.schemas.alarm import Alarm, AlarmLevel, AlarmRule, AlarmStatus
from darkbreaker_sdk.schemas.plugin_io import PluginOutput
from darkbreaker_sdk.schemas.common import ROI, ROIType, Evidence, EvidenceType
# ... all existing exports preserved
```

### 5.2 plugin_manager/base.py (backward compatibility)

```python
"""Backward compatibility - re-export from darkbreaker_sdk"""
from darkbreaker_sdk.interfaces import *
from darkbreaker_sdk.interfaces.base_plugin import BasePlugin
from darkbreaker_sdk.interfaces.lifecycle import (
    PluginStatus, PluginCapability, HealthStatus, PluginManifest, PluginContext
)
```

### 5.3 New: plugin_manager/installer.py

Plugin enable/disable controller (MATLAB-style):
```python
class PluginInstaller:
    def list_available() -> list[PluginInfo]    # All plugins with status
    def enable(plugin_id) -> bool                # Enable plugin
    def disable(plugin_id) -> bool               # Disable plugin
    def get_enabled() -> list[str]               # Currently enabled plugins
    def check_dependencies(plugin_id) -> list    # Check plugin deps
```

---

## 6. Migration Strategy (per plugin)

For each of the 17 plugins:

### Step 1: Update imports
```python
# Before:
from platform_core.plugin_manager.base import BasePlugin, HealthStatus, ...
from platform_core.schema.models import Alarm, RecognitionResult, ...

# After:
from darkbreaker_sdk.interfaces import BasePlugin, HealthStatus, ...
from darkbreaker_sdk.schemas import Alarm, RecognitionResult, ...
```

### Step 2: Add create_standalone() classmethod
```python
@classmethod
def create_standalone(cls, config=None):
    """Create plugin instance without PluginManager"""
    manifest = PluginManifest.from_file(Path(__file__).parent / "manifest.json")
    instance = cls(manifest, Path(__file__).parent)
    instance.init(config or cls._default_config())
    return instance
```

### Step 3: Add get_standalone_routes()
```python
def get_standalone_routes(self):
    """Plugin-specific API routes for standalone mode"""
    routes = []
    # Plugin-specific endpoints
    return routes
```

### Step 4: Create standalone/app.py
### Step 5: Create standalone/templates/{plugin_id}.html
### Step 6: Create demo/ scripts
### Step 7: Create scripts/ utilities
### Step 8: Add requirements.txt
### Step 9: Enhance tests/

---

## 7. Plugin Inventory & Standalone UI Requirements

### Indoor Plugins (7)

| # | Plugin ID | Standalone UI Features |
|---|-----------|----------------------|
| 1 | indoor_fence | Video feed, zone overlay, person tracking, authorization panel, alarm log |
| 2 | animal_detection | Video feed, species detection boxes, thermal view, deterrent controls, stats |
| 3 | fire_detection | Video feed, flame/smoke detection, thermal overlay, suppression controls |
| 4 | slam_mapping | 3D point cloud viewer, occupancy grid, loop closure status |
| 5 | temperature_monitoring | Thermal heatmap, hotspot list, trend charts, threshold settings |
| 6 | device_monitoring | Equipment status cards, health index gauges, fault prediction |
| 7 | multimodal_fusion | Multi-stream view, fusion confidence, cross-modal alignment |

### Outdoor Plugins (10)

| # | Plugin ID | Standalone UI Features |
|---|-----------|----------------------|
| 8 | transformer_inspection | Image upload/camera, defect detection overlay, thermal analysis |
| 9 | switch_inspection | State recognition display, position detection, logic verification |
| 10 | busbar_inspection | Insulator detection, hardware fitting inspection, small-target enhancement |
| 11 | capacitor_inspection | Structural inspection, bulging/leakage detection |
| 12 | meter_reading | Meter image upload, reading display, perspective correction preview |
| 13 | bird_monitoring | Video feed, species ID, risk zones, deterrent controls, database |
| 14 | acoustic_monitoring | Waveform display, spectrogram, partial discharge events, anomaly alerts |
| 15 | gas_detection | Concentration gauges, trend prediction chart, leakage alerts |
| 16 | hyperspectral_detection | Spectral bands viewer, analysis results, comparison view |
| 17 | multimodal_fusion | (same as indoor, shared plugin) |

---

## 8. Standalone UI Template Architecture

### Base Template (SDK-provided)
```
base_standalone.html
├── Header: Plugin name, version, status indicator
├── Main Area (plugin-customizable):
│   ├── Left: Video/Image input panel
│   ├── Center: Detection visualization
│   └── Right: Controls & config
├── Bottom: Detection log, alarm history
└── Footer: Connection status, FPS, model info
```

### Shared Components (SDK-provided)
- `video_panel.html` - Camera/video/image upload input
- `detection_list.html` - Scrollable detection results
- `alarm_panel.html` - Real-time alarm display
- `config_editor.html` - Dynamic config form from schema
- `stats_panel.html` - Performance statistics

Each plugin extends the base template and adds custom panels.

---

## 9. Main Platform Integration Changes

### 9.1 Plugin Selector UI (MATLAB-style)

New page: `ui/templates/pages/plugin_manager.html`
- Grid of plugin cards with enable/disable toggles
- Categories: Indoor / Outdoor
- Each card shows: icon, name, version, description, status
- Dependencies warning if disabling a required plugin

### 9.2 Dynamic Dashboard Loading

Main dashboard (`indoor_center.html`, `outdoor_center_v4.html`) loads only enabled plugins:
```javascript
// Dynamic component loading based on enabled plugins
async function loadEnabledPlugins() {
    const enabled = await fetch('/api/plugins/enabled');
    for (const plugin of enabled) {
        loadPluginComponent(plugin.id, plugin.ui_config);
    }
}
```

---

## 10. Success Criteria

1. All 17 plugins can start standalone: `python -m plugins.{id}.standalone.app`
2. All 17 plugins have working demo scripts: `python -m plugins.{id}.demo.run_demo`
3. All 17 plugins pass tests independently: `pytest plugins/{id}/tests/`
4. All 17 plugins have `requirements.txt` with pinned dependencies
5. Main platform still works with all plugins enabled (backward compatibility)
6. `platform_core/schema/models.py` re-exports from SDK (zero breakage)
7. Plugin selector UI allows enable/disable of individual plugins
8. No algorithm code is modified during refactoring (core/ preserved)

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking existing API endpoints | Compatibility layer in platform_core re-exports all symbols |
| Algorithm regression | Core algorithm code is NOT modified, only import paths change |
| Missing dependencies | Each plugin gets requirements.txt, tested in isolation |
| UI inconsistency | SDK provides shared base template, plugins extend it |
| Performance degradation | Benchmark scripts verify no overhead from SDK layer |
