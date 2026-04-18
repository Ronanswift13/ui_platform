# 08_task_routing — 任务路由表

## 通用前置（所有任务类型共享）

**必读（按序）：**
1. `.agent_skills/00_project_context.md` — 项目上下文与治理等级
2. `.agent_skills/01_architecture_rules.md` — 架构红线

**阻断检查：**
- 若改动需修改 `manifest.json` 核心字段 → 停止并报告
- 若改动涉及灭火联动配置（`suppression.*`）→ **必须人工确认**
- 若改动涉及疏散路线 → 必须人工确认

---

## implement（功能/规则实现）

### 必读
1. 通用前置
2. `.agent_skills/02_algorithm_contract.md` — 输入输出契约
3. `.agent_skills/03_test_strategy.md` — 测试分层
4. `configs/default.yaml` — 当前配置

### 执行顺序
1. 先写测试
2. 最小实现（不跨越层级边界：detector 不控制设备，plugin 不含推理算法）
3. `python -m pytest plugins/fire_detection/tests/ -q`

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
4. `python -m pytest plugins/fire_detection/tests/ -q`
5. 追加 learning_log 条目

---

## audit（质量审计）

### 必读
1. 通用前置
2. `.agent_skills/04_quality_audit.md` — 审计清单
3. `.agent_skills/05_security_boundary.md` — 安全边界（**本插件必读，涉及灭火联动安全**）

### 执行顺序
1. 运行反模式扫描命令（见 04_quality_audit.md）
2. **灭火联动安全专项审查**（见 05_security_boundary.md 第 3 节）
3. 运行测试
4. 输出分级结论（阻断 / 高风险 / 建议）

**约束：** 未执行实际命令与查看结果时，不得宣称通过。

---

## 快速参考

| 任务 | 最小验证命令 |
|------|-------------|
| implement | `python -m pytest plugins/fire_detection/tests/ -q` |
| repair | `python -m pytest plugins/fire_detection/tests/ -q` |
| audit | 反模式扫描 + 灭火安全审查 + 测试 |

---

## 后续补齐项

当以下条件满足后，本插件可升级为**标准治理**：

1. [ ] 补齐 `tests/test_tracker.py`（L0 FireTracker 跟踪 + 扩散分析）
2. [ ] 补齐 `tests/test_fire_level.py`（L0 火灾等级判定边界）
3. [ ] 补齐 `tests/test_sensor_fusion.py`（L0 融合置信度计算）
4. [ ] 补齐 `tests/test_plugin_detect.py`（L1 `detect()` 全链路）
5. [ ] 补齐 `tests/test_suppression.py`（L0 灭火联动触发 + 安全开关验证）
6. [ ] 新建 `scripts/run_targeted_tests.sh`
7. [ ] 新建 `scripts/run_regression_tests.sh`
8. [ ] 新建 `scripts/run_quality_gate.sh`
9. [ ] 建立 `.coveragerc`（覆盖率门槛 >= 60%）

**建议**：鉴于本插件涉及物理设备安全（灭火联动），测试补齐优先级应高于其他轻量插件。

当前 `scripts/` 仅有 `benchmark.py`，不强行生成门禁脚本。
