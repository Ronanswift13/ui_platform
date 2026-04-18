# 00 Project Context

## Verified Facts

- `plugin.py` is the runtime entrypoint declared by `manifest.json`.
- `detector_enhanced.py` is the only detector implementation currently present.
- `switch_consistency.py` exists as a tested helper module, but it is not called by `Plugin.infer()`.
- `configs/default.yaml` is present and provides thresholds, rule definitions, gauge enablement, and ROI metadata.
- `tests/` contains standalone, consistency, config, detector, plugin, and governance contract tests.
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

switch_consistency.py
  -> standalone helper module with its own tests
  -> not automatically wired into plugin.py
```

## What Is Actually Verified

- `Plugin.create_standalone()` works with `configs/default.yaml`.
- `infer(..., context=None)` is supported only for standalone/demo smoke.
- `demo/run_demo.py` is a smoke entrypoint, not a replay benchmark.
- detector fallback path works without optional external model services.

## What Is Not Yet Verified

- No real image replay dataset exists.
- No现场性能基准 has been enforced inside governance scripts.
- No platform-level consistency orchestration is wired from this plugin.

## Safe Working Boundary

- Edit only files inside `plugins/switch_inspection`.
- Read-only reference to `plugins/busbar_inspection` is acceptable for structure reuse.
- Any change that would require `ui/`, `platform_core/`, `darkbreaker_sdk/`, or another plugin is a blocker.

## Useful Local Commands

```bash
./scripts/run_targeted_tests.sh plugin
./scripts/run_regression_tests.sh
./scripts/run_quality_gate.sh
./scripts/collect_root_cause.sh
```
