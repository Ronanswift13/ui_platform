# 06 Refactor Policy

## Allowed Low-Risk Refactors

1. Add adapter logic between `plugin.py` and `detector_enhanced.py`.
2. Keep plugin and detector aligned on nested `inference.*` config.
3. Update tests, scripts, commands, README, PROJECT_CARD, and `.agent_skills`.
4. Replace debug `print()` calls with logging.
5. Clarify docs when historical capability wording is wider than the verified runtime path.

## High-Risk Changes

1. Changing `manifest.json` identity fields.
2. Renaming published defect labels or alarm severities without explicit need.
3. Swapping out the main detector path instead of preserving fallback behavior.
4. Wiring `defect_detector.py`, `thermal_analyzer.py`, or valve recognition into runtime without tests and docs.

## Execution Order

1. Reproduce the issue with a targeted test or script.
2. Make the smallest repair or implementation.
3. Run the closest targeted script.
4. Run regression if runtime code, scripts, commands, or routing changed.
5. Append `.agent_skills/07_learning_log.md` for any new root cause.

## Rollback Triggers

1. `create_standalone()` cannot instantiate the plugin.
2. `healthcheck()` flips from healthy to unhealthy.
3. `infer()` stops returning a list.
4. Regression turns red after a supposed governance-only change.
