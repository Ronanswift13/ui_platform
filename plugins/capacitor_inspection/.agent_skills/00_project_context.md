# 00 Project Context

## Verified Facts

- `plugin.py` is the runtime entrypoint declared by `manifest.json`.
- `detector_enhanced.py` is the active detector backend used by `plugin.py`.
- `configs/default.yaml` is present and provides `model.*`, `inference.*`, `structural_integrity.*`, `intrusion_detection.*`, and `capacitor_bank.*`.
- `tests/` now contains standalone, plugin, detector, config, fallback, and governance contract tests.
- HF-style governance scripts live under `scripts/`:
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
```

## What Is Actually Verified

- `Plugin.create_standalone()` works with `configs/default.yaml`.
- `infer()` returns SDK `RecognitionResult` objects for:
  - structural-defect ROIs
  - intrusion ROIs
- `demo/run_demo.py` is a smoke entrypoint, not a replay benchmark.
- detector fallback path still exists when optional deep-learning modules are absent or empty.

## What Is Not Yet Verified

- No real image replay dataset exists.
- No field accuracy baseline is enforced inside governance scripts.
- Intrusion positive cases are covered by contract tests, not by a replay corpus.
- Traditional CV thresholds in the detector are not fully configuration-driven yet.

## Safe Working Boundary

- Edit only files inside `plugins/capacitor_inspection`.
- Read-only reference to `plugins/switch_inspection`, `plugins/transformer_inspection`, and `plugins/busbar_inspection` is acceptable for structure reuse.
- Any change that would require another plugin, `ui/`, `platform_core/`, or `darkbreaker_sdk/` is a blocker.

## Useful Local Commands

```bash
./scripts/run_targeted_tests.sh plugin
./scripts/run_regression_tests.sh
./scripts/run_quality_gate.sh
./scripts/collect_root_cause.sh
```
