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
2. Intrusion route must keep using `timestamp` and not regress to the old positional misuse.
3. Documentation must not claim replay coverage or stable field intrusion accuracy.
4. Traditional CV thresholds are still technical debt and should remain visible in audit output.

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
- field-proven intrusion accuracy
- full configuration cleanup of every traditional CV threshold
