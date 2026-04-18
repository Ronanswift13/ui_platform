# 08 任务路由 — UI

## 共享前置

1. `CLAUDE.md`
2. `PROJECT_CARD.md`
3. `.agent_skills/00_project_context.md`
4. `.agent_skills/01_architecture_rules.md`

## implement

### 追加读取
- `.agent_skills/02_ui_contract.md`
- `.agent_skills/05_security_boundary.md`
- `.agent_skills/04_quality_audit.md`（审查条目）

### 执行顺序
1. 重述目标
2. 编码（遵循三态覆盖）
3. `./scripts/check_ui_contract.sh`
4. `./scripts/check_three_state_coverage.sh`

### 回写
- `.agent_skills/07_learning_log.md`

## repair

### 追加读取
- `.agent_skills/04_quality_audit.md`（定位违规）
- `.agent_skills/03_test_strategy.md`
- `.agent_skills/07_learning_log.md`（最近 2 条）

### 执行顺序
1. 定位违规项
2. 修复
3. `./scripts/check_ui_contract.sh`

### 回写
- `.agent_skills/07_learning_log.md`（必须）

## audit

### 追加读取
- `.agent_skills/04_quality_audit.md`（完整清单）
- `.agent_skills/03_test_strategy.md`
- `.agent_skills/02_ui_contract.md`（验收条件）

### 执行顺序
1. `./scripts/check_ui_contract.sh`
2. `./scripts/check_three_state_coverage.sh`
3. `./scripts/run_quality_gate.sh`

### 回写
- `.agent_skills/04_quality_audit.md`（审计发现）
