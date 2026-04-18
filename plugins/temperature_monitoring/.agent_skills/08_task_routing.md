# 08_task_routing — 任务路由表

## 通用前置（所有任务类型共享）

**必读（按序）：**
1. `.agent_skills/00_project_context.md` — 项目上下文与治理等级
2. `.agent_skills/01_architecture_rules.md` — 架构红线

**阻断检查：**
- 若改动需修改 `manifest.json` 核心字段 → 停止并报告
- 若改动涉及联动配置（`linkage.*`）→ 必须人工确认
- 若改动涉及通风控制 → 必须人工确认

---

## implement（功能/规则实现）

### 必读
1. 通用前置
2. `.agent_skills/02_algorithm_contract.md` — 输入输出契约
3. `.agent_skills/03_test_strategy.md` — 测试分层
4. `configs/default.yaml` — 当前配置

### 执行顺序
1. 先写测试
2. 最小实现（不跨越层级边界：detector 不依赖 SDK，plugin 不含算法逻辑）
3. `python -m pytest plugins/temperature_monitoring/tests/ -q`

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
4. `python -m pytest plugins/temperature_monitoring/tests/ -q`
5. 追加 learning_log 条目

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
| implement | `python -m pytest plugins/temperature_monitoring/tests/ -q` |
| repair | `python -m pytest plugins/temperature_monitoring/tests/ -q` |
| audit | 反模式扫描 + 测试 |

---

## 后续补齐项

当以下条件满足后，本插件可升级为**标准治理**：

1. [ ] 补齐 `tests/test_detector.py`（L0 热点检测 + severity + 趋势分析）
2. [ ] 补齐 `tests/test_heatmap.py`（L0 三条 heatmap 生成路径）
3. [ ] 补齐 `tests/test_plugin_detect.py`（L1 `detect()` 全链路）
4. [ ] 补齐 `tests/test_zone_match.py`（L0 区域匹配 + threshold_offset）
5. [ ] 将 `detector.py` 中 15-75°C 硬编码映射提取到配置
6. [ ] 新建 `scripts/run_targeted_tests.sh`
7. [ ] 新建 `scripts/run_regression_tests.sh`
8. [ ] 新建 `scripts/run_quality_gate.sh`
9. [ ] 建立 `.coveragerc`（覆盖率门槛 >= 60%）

当前 `scripts/` 仅有 `benchmark.py`，不强行生成门禁脚本。
