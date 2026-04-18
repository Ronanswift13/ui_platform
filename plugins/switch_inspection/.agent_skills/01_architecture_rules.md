# 01 Architecture Rules

## Layering

1. `plugin.py` owns SDK adaptation, ROI extraction, result assembly, and alarm assembly.
2. `detector_enhanced.py` owns switch-state heuristics, clarity evaluation, interlock evaluation, and gauge-reading adapters.
3. `switch_consistency.py` stays independent unless an explicit task asks to wire it into runtime.
4. `standalone/` and `demo/` are runtime shells only; they do not define business rules.

## Dependency Boundaries

- `plugin.py` may depend on `darkbreaker_sdk` and `detector_enhanced.py`.
- `detector_enhanced.py` must not depend on `darkbreaker_sdk` or `standalone/`.
- `switch_consistency.py` must not import `plugin.py`.
- Scripts and commands may read docs/tests, but must not mutate files outside this plugin.

## Contract Adapters That Must Stay Stable

- `detector_enhanced.py` must keep these methods for `plugin.py`:
  - `evaluate_clarity()`
  - `recognize_indicator_state()`
  - `recognize_linkage_state()`
  - `validate_interlock()`
  - `read_gauge()`
- `plugin.py` may accept `context=None` only for standalone/demo/test smoke.

## Red Lines

- Do not rename `manifest.json` core fields: `id`, `entrypoint`, `plugin_class`.
- Do not fold `switch_consistency.py` into `plugin.py` without dedicated tests and explicit approval.
- Do not move business thresholds out of `configs/default.yaml` into ad-hoc constants.
- Do not claim replay coverage or production sensor integration when only local tests exist.
