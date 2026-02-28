# Plugin Microapp Architecture Design

## Goal
Make each of the 16 plugins a true standalone microapp that:
1. Can be opened in VS Code as an independent project and run directly
2. Contains its own simplified platform services (inference, models, data, alarms)
3. Has complete UI including voltage level selection, device config, settings
4. Can be distributed as plugin_folder + SDK to team members
5. Still works as a managed plugin when loaded by the main platform via `python run.py`

## Design: SDK Enhancement Approach

### Layer 1: SDK Services (`darkbreaker_sdk/services/`)
Add lightweight, self-contained versions of key platform_core services:
- `LocalInferenceEngine` — ONNX/PyTorch inference without platform dependencies
- `LocalModelRegistry` — Model discovery from plugin's `models/` directory
- `LocalDataManager` — SQLite/JSON-based local data persistence
- `LocalAlarmManager` — Local alarm lifecycle management
- `LocalConfigManager` — Configuration + voltage level + device type management

### Layer 2: Import Path Fix
- Each plugin's `__main__.py` uses relative imports + smart path detection
- Works from project root (`python -m plugins.xxx`) AND plugin directory

### Layer 3: `run_standalone.py` Per Plugin
- Zero-config entry point in each plugin root directory
- VS Code users can right-click → Run Python File

### Layer 4: Enhanced StandalonePluginRunner
- Platform-level UI: voltage level selector, device config, model settings
- Enhanced `base_standalone.html` with shared sidebar/header/settings panel

### Layer 5: Main Platform Unchanged
- `python run.py` continues to work as-is
- No changes to `platform_core/plugin_manager/`

## Implementation Steps
1. Create `darkbreaker_sdk/services/` with 5 service modules
2. Enhance `StandalonePluginRunner` with service injection
3. Update `base_standalone.html` template with platform-level UI
4. Update all 16 plugins' `__main__.py` for dual-mode imports
5. Add `run_standalone.py` to all 16 plugins
6. Test: each plugin runs from its own directory and from project root
