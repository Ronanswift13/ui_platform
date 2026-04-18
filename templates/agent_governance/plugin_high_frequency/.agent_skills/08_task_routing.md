# 08 任务路由 — {{PLUGIN_DISPLAY_NAME}}

## 共享前置（所有任务类型）

每次任务开始前必须读取:
1. `PROJECT_CARD.md`
2. `CLAUDE.md`
3. `.agent_skills/00_project_context.md`
4. `.agent_skills/01_architecture_rules.md`

## implement（功能/规则实现）

### 追加读取
- `.agent_skills/02_algorithm_contract.md`
- `.agent_skills/03_test_strategy.md`
- `configs/default.yaml`

### 执行顺序
1. 重述目标与约束
2. 先写测试
3. 最小实现
4. `./scripts/run_targeted_tests.sh <module>`
5. `./scripts/run_regression_tests.sh`

### 回写
- `.agent_skills/07_learning_log.md`

## repair（缺陷修复）

### 追加读取
- `.agent_skills/02_algorithm_contract.md`
- `.agent_skills/07_learning_log.md`
- `.agent_skills/06_refactor_policy.md`

### 执行顺序
1. 记录现象与复现
2. 先写失败测试
3. 最小因果修复
4. `./scripts/run_targeted_tests.sh <module>`
5. `./scripts/run_regression_tests.sh`
6. （重大缺陷）`./scripts/collect_root_cause.sh`

### 回写
- `.agent_skills/07_learning_log.md`（必须）

## audit（质量审计）

### 追加读取
- `.agent_skills/04_quality_audit.md`
- `.agent_skills/05_security_boundary.md`

### 执行顺序
1. `./scripts/run_quality_gate.sh`
2. 安全扫描
3. 按 `04_quality_audit.md` 逐条审查

### 回写
- `.agent_skills/04_quality_audit.md`（审计发现）

## upgrade（依赖/契约升级）

### 追加读取
- `manifest.json`
- `.agent_skills/02_algorithm_contract.md`
- `.agent_skills/03_test_strategy.md`

### 执行顺序
1. 评估变更影响
2. 更新依赖/契约
3. `./scripts/run_targeted_tests.sh all`
4. `./scripts/run_regression_tests.sh`
5. `./scripts/run_quality_gate.sh`

### 回写
- `manifest.json`
- `.agent_skills/07_learning_log.md`

## 阻断条件

命中以下任一 → 停止并报告:
1. `PROJECT_CARD.md` 业务目标与实现冲突
2. 需调整 `manifest.json` 核心字段
3. `run_regression_tests.sh` 返回非 0

<!-- BUSINESS: 补充本插件专属的阻断条件 -->
