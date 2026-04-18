# 08_task_routing — 任务路由表

## 通用前置（所有任务类型共享）

**必读（按序）：**
1. `.agent_skills/00_project_context.md` — 项目上下文与治理等级
2. `.agent_skills/01_architecture_rules.md` — 架构红线

**阻断检查：**
- 若改动需修改 `manifest.json` 核心字段 → 停止并报告
- 若改动涉及协议启用（SNMP/Modbus）→ 必须人工确认

---

## implement（功能/规则实现）

### 必读
1. 通用前置
2. `.agent_skills/02_algorithm_contract.md` — 输入输出契约
3. `.agent_skills/03_test_strategy.md` — 测试分层
4. `configs/default.yaml` — 当前配置

### 执行顺序
1. 先写测试
2. 最小实现（不跨越层级边界：detector 不依赖 SDK，plugin 不含计算逻辑）
3. `cd plugins/device_monitoring && ./scripts/run_sanity_checks.sh`
4. 触及 detector、健康指数、工单或 replay 时再跑 `./scripts/run_targeted_tests.sh`

### 回写文件
- `tests/test_*.py`（必须）
- 实现模块（`plugin.py` / `detector.py`）
- `configs/default.yaml`（若新增配置项）
- `.agent_skills/07_learning_log.md`（若遇非显然问题）

---

## repair（缺陷修复）

### 必读
1. 通用前置
2. `.agent_skills/02_algorithm_contract.md` — 确认预期行为
3. `.agent_skills/07_learning_log.md` — 检查是否已知问题
4. `.agent_skills/06_refactor_policy.md` — 不做无关重构

### 执行顺序
1. 记录现象
2. 先写失败测试
3. 最小因果修复
4. `cd plugins/device_monitoring && ./scripts/run_sanity_checks.sh`
5. 触及 detector 或 replay 时补跑 `./scripts/run_targeted_tests.sh`
6. 追加 learning_log 条目

---

## audit（质量审计）

### 必读
1. 通用前置
2. `.agent_skills/04_quality_audit.md` — 审计清单
3. `.agent_skills/05_security_boundary.md` — 安全边界

### 执行顺序
1. 运行反模式扫描命令（见 04_quality_audit.md）
2. 运行测试
3. 输出分级结论（阻断 / 高风险 / 建议）

**约束：** 未执行实际命令与查看结果时，不得宣称通过。

---

## 快速参考

| 任务 | 最小验证命令 |
|------|-------------|
| implement | `cd plugins/device_monitoring && ./scripts/run_sanity_checks.sh` |
| repair | `cd plugins/device_monitoring && ./scripts/run_sanity_checks.sh` + 必要 targeted |
| audit | 反模式扫描 + 测试 |

---

## 当前完成项与后续补齐项

以下第一批治理项已完成，可按**标准治理基线**处理：

1. [x] `tests/test_detector.py`（L0 DeviceHealthCalculator 边界测试 + 工单触发）
2. [x] `tests/test_process_contract.py`（L1 `detect()`/process 统一输出壳）
3. [x] `tests/test_device_replay.py`（现场式设备数据 replay）
4. [x] `detector.py` 中 CPU 使用率阈值提取到 config
5. [x] `scripts/run_targeted_tests.sh`
6. [x] `scripts/run_sanity_checks.sh`

后续仍可补：`tests/test_scan_devices.py`、`scripts/run_quality_gate.sh`、`.coveragerc`、真实脱敏设备遥测 replay。
