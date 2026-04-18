# 03 Test Strategy

## Test Entry Points

- `./scripts/run_targeted_tests.sh standalone`
- `./scripts/run_targeted_tests.sh plugin`
- `./scripts/run_targeted_tests.sh detector`
- `./scripts/run_targeted_tests.sh config`
- `./scripts/run_targeted_tests.sh quality`
- `./scripts/run_regression_tests.sh`

## Current Test Inventory

- `tests/test_standalone.py`
- `tests/test_plugin_contract.py`
- `tests/test_plugin_postprocess.py`
- `tests/test_detector_contract.py`
- `tests/test_fallback_chain.py`
- `tests/test_config_contract.py`
- `tests/test_quality_gate_contract.py`

## What Each Script Means

- `run_targeted_tests.sh`
  - fast module-scoped validation
  - fails if a requested module has no mapped tests
- `run_regression_tests.sh`
  - full pytest
  - demo smoke
  - optional regression marker run
  - optional benchmark via `RUN_BENCHMARK=1`
  - optional mypy/flake8/bandit if installed
- `run_quality_gate.sh`
  - governance asset presence
  - architecture/anti-pattern scan
  - regression gate
  - basic security scan
- `collect_root_cause.sh`
  - snapshots config/docs/skills
  - records plugin-local git diff
  - captures targeted + regression outputs

## Honest Limits

- There is no real image replay corpus yet.
- Regression currently validates code contracts and smoke behavior, not field accuracy.
- Benchmarking is opt-in and should not be quoted as a default CI-like gate.
- Valve-state runtime behavior is not part of the current tested surface.

## Writeback Rules

- behavior change -> update tests first or in the same change
- governance/script change -> update `.agent_skills/08_task_routing.md`
- new failure mode -> append `.agent_skills/07_learning_log.md`
