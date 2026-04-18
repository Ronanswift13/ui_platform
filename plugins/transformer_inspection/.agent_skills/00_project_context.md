# 00 Project Context

## Verified Facts

- `plugin.py` is the runtime entrypoint declared by `manifest.json`.
- `detector_enhanced.py` is the active detector backend used by `plugin.py`.
- `defect_detector.py` and `thermal_analyzer.py` exist, but they are not called by `Plugin.infer()`.
- `configs/default.yaml` is present and supplies `model.*`, `inference.*`, `thermal.*`, and `alarm.*`.
- `tests/` now contains standalone, plugin, detector, config, fallback, and governance contract tests.
- HF governance scripts live under `scripts/`:
  - `run_targeted_tests.sh`
  - `run_regression_tests.sh`
  - `run_quality_gate.sh`
  - `collect_root_cause.sh`

## Current Runtime Shape

```text
plugin.py
  -> detector_enhanced.py
  -> darkbreaker_sdk interfaces/schemas

detector_enhanced.py
  -> numpy / cv2
  -> optional ai_models.*

defect_detector.py / thermal_analyzer.py
  -> local side modules
  -> not automatically wired into plugin.py
```

## What Is Actually Verified

- `Plugin.create_standalone()` works with `configs/default.yaml`.
- `infer()` returns SDK `RecognitionResult` objects for:
  - defect ROIs
  - `breather` / `silica` state ROIs
  - `oil_level` / `meter` / `gauge` state ROIs
  - thermal over-temperature when `thermal.enabled=true` and a thermal frame is provided
- `demo/run_demo.py` is a smoke entrypoint, not a replay benchmark.
- detector fallback path works when optional deep-learning modules are absent or empty.

## What Is Not Yet Verified

- No real image replay dataset exists.
- No field accuracy baseline is enforced inside governance scripts.
- No valve-state runtime path is wired in the current main chain.
- `defect_detector.py` and `thermal_analyzer.py` are not runtime-governed by the HF scripts yet.

## Safe Working Boundary

- Edit only files inside `plugins/transformer_inspection`.
- Read-only reference to `plugins/switch_inspection` and `plugins/busbar_inspection` is acceptable for structure reuse.
- Any change that would require another plugin, `ui/`, `platform_core/`, or `darkbreaker_sdk/` is a blocker.

## Useful Local Commands

```bash
./scripts/run_targeted_tests.sh plugin
./scripts/run_regression_tests.sh
./scripts/run_quality_gate.sh
./scripts/collect_root_cause.sh
```
