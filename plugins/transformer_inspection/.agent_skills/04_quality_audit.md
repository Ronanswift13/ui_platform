# 04 Quality Audit

## Blocking Checks

1. `./scripts/run_quality_gate.sh` must pass.
2. `detector_enhanced.py` must not import `darkbreaker_sdk` or standalone code.
3. `plugin.py` and `detector_enhanced.py` must not contain `print()`.
4. No `except:`, no `except Exception: pass`.
5. `.agent_skills/08_task_routing.md` must mention all four governance scripts.
6. `.claude/commands/*.md` must not reference `CLAUDE.md` or “脚本不存在”.

## High-Risk Review Points

1. `plugin.py` and `detector_enhanced.py` must stay contract-aligned.
2. State routing must remain truthfully limited to silica gel and oil-level paths unless code proves more.
3. `defect_detector.py` / `thermal_analyzer.py` must not be documented as active main-chain modules.
4. Thermal behavior must continue to depend on `thermal.enabled` and an actual thermal frame.
5. Documentation must not claim replay coverage or field-proven accuracy.

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
- valve-state runtime verification
- field-proven visual accuracy metrics
