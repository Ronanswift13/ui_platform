# 08_task_routing — 任务路由表

## 通用前置（所有任务类型共享）

**必读（按序）：**
1. `.agent_skills/00_project_context.md` — 项目上下文与治理等级
2. `.agent_skills/01_architecture_rules.md` — 架构红线

**阻断检查：**
- 若改动需修改 `manifest.json` 核心字段 → 停止并报告

---

## implement（功能/规则实现）

### 必读
1. 通用前置
2. `.agent_skills/02_algorithm_contract.md` — 输入输出契约
3. `.agent_skills/03_test_strategy.md` — 测试分层
4. `configs/default.yaml` — 当前配置

### 执行顺序
1. 先写测试
2. 最小实现（不跨越层级边界）
3. `cd plugins/acoustic_monitoring && ./scripts/run_sanity_checks.sh`
4. 触及 detector/analyzer/WAV 回放时再跑 `./scripts/run_targeted_tests.sh`
5. 触及质量门或反模式清理时跑 `./scripts/run_quality_gate.sh`

### 回写文件
- `tests/test_*.py`（必须）
- 实现模块（`plugin.py` / `detector.py` / `analyzer.py`）
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
4. `cd plugins/acoustic_monitoring && ./scripts/run_sanity_checks.sh`
5. 若缺陷位于算法层或回放链路，补跑 `./scripts/run_targeted_tests.sh`
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
| implement | `cd plugins/acoustic_monitoring && ./scripts/run_sanity_checks.sh` |
| repair | `cd plugins/acoustic_monitoring && ./scripts/run_sanity_checks.sh` + 必要 targeted |
| audit | `cd plugins/acoustic_monitoring && ./scripts/run_quality_gate.sh` |

---

## 当前完成项与后续补齐项

以下第一批质量治理项已完成，可按**标准治理基线**处理：

1. [x] `tests/test_detector.py`（L0 detector 边界与模型 registry 路径）
2. [x] `tests/test_analyzer.py`（诊断输出结构）
3. [x] `tests/test_config_contract.py`（manifest/YAML 合同）
4. [x] `tests/test_process_contract.py`（统一输出壳）
5. [x] `tests/test_real_audio_replay.py`（WAV 容器回放）
6. [x] `scripts/run_targeted_tests.sh`
7. [x] `scripts/run_sanity_checks.sh`
8. [x] `scripts/run_quality_gate.sh`

后续仍可补：真实现场音频样本、覆盖率门槛、真实模型资产 preflight。当前不要把合成 WAV replay 伪装成真实传感器闭环。
