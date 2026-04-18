# 06 Refactor Policy

## Low-Risk Changes Allowed

- update tests
- update scripts
- update commands and `.agent_skills/*`
- fix plugin/detector contract adapters
- convert noisy `print()` logging to `logging`
- fix config-key drift without changing business semantics

## High-Risk Changes Requiring Explicit Approval

- rewrite detector fusion logic
- change rule semantics in `logic_validation.rules`
- wire `switch_consistency.py` into main runtime flow
- change manifest identity fields
- introduce new cross-plugin or platform dependencies

## Execution Order

1. prove the problem with tests or script output
2. make the smallest local change
3. run targeted tests
4. run regression if production code, scripts, or routing changed
5. log the lesson in `07_learning_log.md` if the failure was non-trivial
