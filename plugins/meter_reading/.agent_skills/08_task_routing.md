# 08_task_routing

本文件定义 `meter_reading` 的标准任务路由：必读文件、模块映射、执行脚本、回写文件。
所有 `/implement`、`/repair`、`/audit`、`/upgrade` 命令必须先加载本文件。

---

## 模块映射（meter_reading 当前真实入口）

| 模块 | 主要实现文件 | targeted 参数 | 主测试文件 |
|---|---|---|---|
| analog | `detector_enhanced.py` | `analog` | `tests/test_analog_meter.py` |
| digital | `detector_enhanced.py` | `digital` | `tests/test_digital_ocr.py` |
| led | `detector_enhanced.py` | `led` | `tests/test_led_indicator.py` |
| validation | `detector_enhanced.py` | `validation` | `tests/test_input_validation.py`, `tests/test_confidence.py` |
| plugin | `plugin.py` | `plugin` | `tests/test_plugin_integration.py` |
| contract | `plugin.py`, `detector_enhanced.py`, `configs/default.yaml` | `contract` | `tests/test_output_structure.py` |
| all | `plugin.py`, `detector_enhanced.py`, `tests/` | `all` | `tests/` |

---

## 通用前置（所有任务共享）

### 必读（按序）

1. `PROJECT_CARD.md`
2. `CLAUDE.md`
3. `.agent_skills/00_project_context.md`
4. `.agent_skills/01_architecture_rules.md`

### 阻断检查

- 若 `manifest.json` 核心字段需要修改 -> 停止并报告
- 若改动会新增/删除表计类型或变更业务量程 -> 先人工确认
- 若任务声称“已做回归”，但 `tests/regression/` 或 `tests/fixtures/` 仍为空 -> 结论无效

---

## implement（功能/规则实现）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/03_test_strategy.md`
4. `configs/default.yaml`

### 执行顺序

1. 先判定落在哪个模块：`analog / digital / led / validation / plugin / contract`
2. 先写或更新对应测试文件
3. 实施最小改动，不跨越层级边界
4. 运行 `./scripts/run_targeted_tests.sh <module>`
5. 若触及 `plugin.py`、`detector_enhanced.py`、`configs/default.yaml`、metadata 契约或命令/脚本入口，再运行 `./scripts/run_regression_tests.sh`
6. 若契约或测试重点变化，同步更新 `.agent_skills/02_algorithm_contract.md` 与 `.agent_skills/03_test_strategy.md`

### 回写文件

| 文件 | 条件 |
|---|---|
| `tests/test_*.py` | 必须：新增/更新测试 |
| `plugin.py` / `detector_enhanced.py` / `configs/default.yaml` | 按最小实现修改 |
| `.agent_skills/02_algorithm_contract.md` | 若契约变化 |
| `.agent_skills/03_test_strategy.md` | 若测试分层或重点变化 |
| `.agent_skills/07_learning_log.md` | 若遇到非显然问题 |

---

## repair（缺陷修复）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/06_refactor_policy.md`
4. `.agent_skills/07_learning_log.md`

### 执行顺序

1. 记录症状、复现路径、首次失败边界
2. 选择受影响模块并先写失败测试
3. 实施最小因果修复，不做搭车重构
4. 运行 `./scripts/run_targeted_tests.sh <module>`
5. 运行 `./scripts/run_regression_tests.sh`
6. 若为复杂缺陷或需留痕，运行 `./scripts/collect_root_cause.sh`
7. 必须追加 `.agent_skills/07_learning_log.md`

### 回写文件

| 文件 | 条件 |
|---|---|
| 受影响的 `tests/test_*.py` | 必须：失败测试转绿 |
| 缺陷所在实现文件 | 必须：最小修复 |
| `.agent_skills/07_learning_log.md` | 必须 |
| `.agent_skills/04_quality_audit.md` | 若发现可复用审查规则 |

---

## audit（质量审计）

### 必读

1. 通用前置
2. `.agent_skills/04_quality_audit.md`
3. `.agent_skills/05_security_boundary.md`
4. `.coveragerc`

### 执行顺序

1. 运行 `./scripts/run_quality_gate.sh`
2. 检查 targeted / regression / static / security 的真实输出
3. 人工核对：
   - 状态集是否漂移
   - metadata 必填字段是否漂移
   - 回归是否真实执行或被显式 skip
   - `METER_RANGES` / OCR / LED 语义是否意外变化
4. 按阻断项 / 高风险项 / 建议项输出结论

### 回写文件

| 文件 | 条件 |
|---|---|
| 生产代码 | 审计默认只读，不修代码 |
| 审计报告（输出到对话） | 必须 |
| `.agent_skills/07_learning_log.md` | 若发现系统性问题 |

**关键约束：** 若 `tests/regression/` 或 `tests/fixtures/` 为空，审计必须明确写出“数据集回归未执行”。

---

## upgrade（依赖/契约升级）

### 必读

1. 通用前置
2. `manifest.json`
3. `requirements.txt`
4. `.agent_skills/02_algorithm_contract.md`
5. `.agent_skills/03_test_strategy.md`

### 执行顺序

1. 列出影响的模块、配置键、测试文件
2. 先更新测试期望
3. 按层级适配实现
4. 运行 `./scripts/run_targeted_tests.sh all`
5. 运行 `./scripts/run_regression_tests.sh`
6. 运行 `./scripts/run_quality_gate.sh`
7. 记录升级风险与后续动作到 `.agent_skills/07_learning_log.md`

---

## 快速参考：任务 -> 脚本映射

| 任务 | 最小验证 | 完整验证 | 交付补充 |
|---|---|---|---|
| implement | `run_targeted_tests.sh <module>` | `run_regression_tests.sh` | 合同/测试策略同步 |
| repair | `run_targeted_tests.sh <module>` | `run_regression_tests.sh` | `collect_root_cause.sh` + learning log |
| audit | `run_quality_gate.sh` | 人工审计 | 明确回归是否 skip |
| upgrade | `run_targeted_tests.sh all` | `run_regression_tests.sh` | `run_quality_gate.sh` |
