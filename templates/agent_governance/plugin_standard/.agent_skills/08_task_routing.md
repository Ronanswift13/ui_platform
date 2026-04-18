# 08 任务路由 — {{PLUGIN_DISPLAY_NAME}}

## 当前治理前提

- 治理等级: STD
- 有 `.agent_skills/00~08`
- 脚本体系不完整（部分或全部缺失）
- 验证入口: `python -m pytest tests/ -q`

## 共享前置（所有任务类型）

1. `PROJECT_CARD.md`（如存在）
2. `CLAUDE.md`（如存在）
3. `.agent_skills/00_project_context.md`
4. `.agent_skills/01_architecture_rules.md`

## implement

### 追加读取
- `.agent_skills/02_algorithm_contract.md`
- `.agent_skills/03_test_strategy.md`

### 执行顺序
1. 重述目标
2. 先写测试
3. 最小实现
4. `python -m pytest tests/ -q`

### 回写
- `.agent_skills/07_learning_log.md`

## repair

### 追加读取
- `.agent_skills/07_learning_log.md`
- `.agent_skills/06_refactor_policy.md`

### 执行顺序
1. 记录现象
2. 先写失败测试
3. 最小因果修复
4. `python -m pytest tests/ -q`
5. 追加根因到 `07_learning_log.md`

## audit

### 追加读取
- `.agent_skills/04_quality_audit.md`
- `.agent_skills/05_security_boundary.md`

### 执行顺序
1. 按审计清单逐条检查
2. `python -m pytest tests/ -q`

## upgrade

### 追加读取
- `manifest.json`
- `.agent_skills/02_algorithm_contract.md`

### 执行顺序
1. 评估变更影响
2. 更新依赖/契约
3. `python -m pytest tests/ -q`

## 升级路径

补齐 `.claude/commands/` + 质量门禁脚本体系即可升级到 HF。
