# 08_task_routing — 任务路由表

本文件定义四类标准任务的完整执行路径：必读文件、执行脚本、回写文件。
所有 `/implement`、`/repair`、`/audit`、`/upgrade` 命令必须先加载本文件。

---

## 通用前置（所有任务类型共享）

**必读（按序）：**
1. `PROJECT_CARD.md` — 确认业务目标与约束未变
2. `CLAUDE.md` — 确认固定指令与阻断条件
3. `.agent_skills/00_project_context.md` — 项目上下文
4. `.agent_skills/01_architecture_rules.md` — 架构红线

**阻断检查：**
- 若 `PROJECT_CARD.md` 业务目标与当前实现冲突 → 停止并报告
- 若改动需修改 `manifest.json` 核心字段 → 停止并报告

---

## implement（功能/规则实现）

### 必读
| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | 通用前置 | 确认约束 |
| 2 | `.agent_skills/02_algorithm_contract.md` | 算法输入输出契约 |
| 3 | `.agent_skills/03_test_strategy.md` | 测试分层与命名 |
| 4 | `configs/default.yaml` | 当前配置基线 |

### 执行顺序
1. **先写测试** — 在 `tests/` 下新增或更新契约测试
2. **最小实现** — 不跨越层级边界（detector 不依赖 SDK，plugin 不含算法逻辑）
3. **targeted 验证** — `./scripts/run_targeted_tests.sh <module>`
4. **regression 验证** — `./scripts/run_regression_tests.sh`

### 回写文件
| 文件 | 条件 |
|------|------|
| `tests/test_*.py` | 必须：新增/更新的测试 |
| 实现模块 (`plugin.py` / `detector_enhanced.py` / `config_adapter.py` / `reason_code_mapper.py`) | 必须：最小改动 |
| `configs/default.yaml` | 若新增配置项 |
| `.agent_skills/07_learning_log.md` | 若遇到非显然问题 |

---

## repair（缺陷修复）

### 必读
| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | 通用前置 | 确认约束 |
| 2 | `.agent_skills/02_algorithm_contract.md` | 确认预期行为 |
| 3 | `.agent_skills/07_learning_log.md` | 检查是否已知问题 |
| 4 | `.agent_skills/06_refactor_policy.md` | 确认修复边界，不做无关重构 |

### 执行顺序
1. **记录现象** — 现象、复现步骤、首次失败边界
2. **先写失败测试** — 测试必须能复现缺陷
3. **最小因果修复** — 只修当前根因，不扩散改动
4. **targeted 验证** — `./scripts/run_targeted_tests.sh <module>`
5. **regression 验证** — `./scripts/run_regression_tests.sh`
6. **根因收集** — 若为重大缺陷，运行 `./scripts/collect_root_cause.sh`

### 回写文件
| 文件 | 条件 |
|------|------|
| `tests/test_*.py` | 必须：失败测试转绿 |
| 缺陷所在模块 | 必须：最小修复 |
| `.agent_skills/07_learning_log.md` | **必须**：追加根因条目 |
| `.coveragerc` | 若修复暴露新的覆盖率缺口 |

---

## audit（质量审计）

### 必读
| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | 通用前置 | 确认约束 |
| 2 | `.agent_skills/04_quality_audit.md` | 审计清单与命令 |
| 3 | `.agent_skills/05_security_boundary.md` | 安全边界 |
| 4 | `.coveragerc` | 当前覆盖率门槛 |

### 执行顺序
1. **架构检查** — `./scripts/run_quality_gate.sh`（含架构检查 + 反模式扫描 + 测试 + 静态分析）
2. **人工审查** — 按 `04_quality_audit.md` 逐项检查
3. **分级输出** — 阻断项 / 高风险项 / 建议项

### 回写文件
| 文件 | 条件 |
|------|------|
| 无生产代码改动 | 审计只读，不修代码 |
| 审计报告（输出到对话） | 必须：含分级结论 |
| `.agent_skills/07_learning_log.md` | 若发现系统性问题 |

**关键约束：** 未给出实际执行命令与结果时，不得宣称通过。

---

## upgrade（依赖/契约升级）

### 必读
| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | 通用前置 | 确认约束 |
| 2 | `manifest.json` | 当前版本与依赖声明 |
| 3 | `.agent_skills/02_algorithm_contract.md` | 确认契约变更范围 |
| 4 | `.agent_skills/01_architecture_rules.md` | 确认层级边界 |
| 5 | `requirements.txt` | 当前依赖版本 |

### 执行顺序
1. **影响评估** — 列出受影响的模块与测试
2. **先更新测试** — 按新契约更新测试期望值
3. **适配实现** — 按层级从底层（detector）到顶层（plugin）逐层适配
4. **targeted 验证** — `./scripts/run_targeted_tests.sh all`
5. **regression 验证** — `./scripts/run_regression_tests.sh`
6. **quality gate** — `./scripts/run_quality_gate.sh`

### 回写文件
| 文件 | 条件 |
|------|------|
| `tests/test_*.py` | 必须：适配新契约 |
| 受影响的实现模块 | 必须 |
| `manifest.json` | 若版本号变更 |
| `requirements.txt` | 若依赖版本变更 |
| `configs/default.yaml` | 若配置结构变更 |
| `.agent_skills/02_algorithm_contract.md` | 若算法契约变更 |
| `.agent_skills/07_learning_log.md` | 必须：记录升级决策与风险 |

---

## 快速参考：任务 -> 脚本映射

| 任务 | 最小验证 | 完整验证 | 交付验证 |
|------|----------|----------|----------|
| implement | `run_targeted_tests.sh <module>` | `run_regression_tests.sh` | — |
| repair | `run_targeted_tests.sh <module>` | `run_regression_tests.sh` | `collect_root_cause.sh` |
| audit | `run_quality_gate.sh` | — | — |
| upgrade | `run_targeted_tests.sh all` | `run_regression_tests.sh` | `run_quality_gate.sh` |
