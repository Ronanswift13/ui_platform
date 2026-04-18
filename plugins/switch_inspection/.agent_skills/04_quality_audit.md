# 04 Quality Audit

## Blocking Checks

1. `./scripts/run_quality_gate.sh` must pass.
2. `detector_enhanced.py` must not import `darkbreaker_sdk` or `standalone`.
3. `plugin.py`, `detector_enhanced.py`, `switch_consistency.py` must not contain `print()`.
4. No `except:`, no `except Exception: pass`.
5. `.agent_skills/08_task_routing.md` must mention all four governance scripts.

## High-Risk Review Points

1. `plugin.py` and `detector_enhanced.py` must stay contract-aligned.
2. `context=None` support must remain clearly limited to standalone/demo/test smoke.
3. `switch_consistency.py` must not be documented as runtime-integrated unless code changes prove it.
4. Interlock output severity must continue to come from `configs/default.yaml`.
5. Gauge reading must remain disabled by default unless config changes.

## Audit Commands

```bash
./scripts/run_quality_gate.sh
./scripts/run_targeted_tests.sh quality
./scripts/run_regression_tests.sh
```

## Audit Output Format

```text
[AUDIT_RESULT] PASS|FAIL
[BLOCKERS] NONE | <items>
[HIGH_RISK] NONE | <items>
[ACTION_ITEMS] <ordered actions>
[EVIDENCE] <key command outputs>
```

## What Audit Must Not Claim

- real replay coverage
- platform-level consistency orchestration
- field-proven accuracy metrics
