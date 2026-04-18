# 08 Task Routing

All `/implement`, `/repair`, and `/audit` work for `switch_inspection` must read this file first.

## Shared Pre-Read

1. `PROJECT_CARD.md`
2. `README.md`
3. `.agent_skills/00_project_context.md`
4. `.agent_skills/01_architecture_rules.md`

## implement

### Read next

1. `.agent_skills/02_algorithm_contract.md`
2. `.agent_skills/03_test_strategy.md`
3. `configs/default.yaml`

### Execute

1. decide the target module: `standalone|plugin|detector|consistency|config|quality`
2. update tests when behavior changes
3. run `./scripts/run_targeted_tests.sh <module>`
4. if production code, scripts, commands, README, PROJECT_CARD, or routing changed, run `./scripts/run_regression_tests.sh`
5. if a new failure pattern appeared, update `.agent_skills/07_learning_log.md`

## repair

### Read next

1. `.agent_skills/02_algorithm_contract.md`
2. `.agent_skills/06_refactor_policy.md`
3. `.agent_skills/07_learning_log.md`

### Execute

1. reproduce the failure with a test or script
2. run the closest targeted module:
   - `./scripts/run_targeted_tests.sh plugin`
   - `./scripts/run_targeted_tests.sh detector`
   - `./scripts/run_targeted_tests.sh consistency`
   - `./scripts/run_targeted_tests.sh config`
3. make the smallest repair
4. run `./scripts/run_regression_tests.sh` if runtime code or scripts changed
5. if the failure remains unclear or regression fails, run `./scripts/collect_root_cause.sh`
6. append `.agent_skills/07_learning_log.md`

### Root-cause writeback

- `collect_root_cause.sh` stores evidence under `data/root_cause/<timestamp>/`

## audit

### Read next

1. `.agent_skills/04_quality_audit.md`
2. `.agent_skills/05_security_boundary.md`

### Execute

1. run `./scripts/run_quality_gate.sh`
2. if extra evidence is needed, run:
   - `./scripts/run_targeted_tests.sh quality`
   - `./scripts/run_regression_tests.sh`
3. report blocker / high-risk / action-items separately
4. do not claim replay coverage or platform integration that is not present

## Quick Map

| Task | Minimum script | Escalation script | Evidence path |
|------|----------------|-------------------|---------------|
| implement | `run_targeted_tests.sh <module>` | `run_regression_tests.sh` | test output |
| repair | `run_targeted_tests.sh <module>` | `collect_root_cause.sh` | `data/root_cause/<timestamp>/` |
| audit | `run_quality_gate.sh` | `run_regression_tests.sh` | gate + regression output |
