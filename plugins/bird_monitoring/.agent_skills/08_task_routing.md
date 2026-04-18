# 08_task_routing — 任务路由表

## 通用前置（所有任务类型共享）

**必读（按序）：**
1. `.agent_skills/00_project_context.md` — 项目上下文与治理等级
2. `.agent_skills/01_architecture_rules.md` — 架构红线

**阻断检查：**
- 若改动需修改 `manifest.json` 核心字段 → 停止并报告
- 若改动涉及驱鸟硬件控制策略 → 停止并报告；当前能力为 blocked

---

## implement（功能/规则实现）

### 必读
1. 通用前置
2. `.agent_skills/02_algorithm_contract.md` — 输入输出契约
3. `.agent_skills/03_test_strategy.md` — 测试分层
4. `configs/default.yaml` — 当前配置

### 执行顺序
1. 先写/补测试
2. 最小实现（不跨越层级边界）
3. `python -m pytest plugins/bird_monitoring/tests/ -q`

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
4. `python -m pytest plugins/bird_monitoring/tests/ -q`
5. 追加 learning_log 条目

---

## audit（质量审计）

### 必读
1. 通用前置
2. `.agent_skills/04_quality_audit.md` — 审计清单
3. `.agent_skills/05_security_boundary.md` — 安全边界

### 执行顺序
1. 运行反模式扫描命令（见 04_quality_audit.md）
2. 运行测试（若存在）
3. 输出分级结论（阻断 / 高风险 / 建议）

**约束：** 未执行实际命令与查看结果时，不得宣称通过。

---

## 快速参考

| 任务 | 最小验证命令 |
|------|-------------|
| implement | `scripts/run_targeted_tests.sh all` |
| repair | `scripts/run_targeted_tests.sh all` |
| audit | `scripts/run_quality_gate.sh`（反模式 + pytest + 覆盖率） |
| L0 纯逻辑 | `scripts/run_targeted_tests.sh l0` |
| L1 集成 | `scripts/run_targeted_tests.sh l1` |
| L2 合成 replay | `scripts/run_targeted_tests.sh l2` |
| 跨插件回归 | `scripts/run_regression_tests.sh` |
| 真实 ONNX preflight gating | `scripts/check_real_model.py /path/to/bird_yolov8n.onnx` |
| 真实 fixture intake gating | `scripts/validate_fixture.py [--sample-id ...]` |
| 重新生成合成 fixture | `python3 plugins/bird_monitoring/tests/regression/build_synthetic_fixtures.py` |
| 仅 preflight 契约 | `python -m pytest plugins/bird_monitoring/tests/test_real_dl_preflight.py -q` |
| 仅质量门三态 | `python -m pytest plugins/bird_monitoring/tests/test_quality_tristate.py -q` |
| opt-in 实验检测器（仅手测） | `BIRD_ENABLE_ENHANCED_DETECTOR=1 python -c "from plugins.bird_monitoring.detector import BirdDetectorEnhanced; BirdDetectorEnhanced({})"` |

### scripts/ 入口职责矩阵

| 脚本 | 职责 | 退出码语义 |
|---|---|---|
| `scripts/run_targeted_tests.sh` | 分层执行 L0/L1/L2/all 测试 | 透传 pytest |
| `scripts/run_quality_gate.sh` | 反模式扫描 + pytest + coverage 三段门 | 1=反模式, 2=pytest, 3=覆盖率, 4=环境缺失 |
| `scripts/run_regression_tests.sh` | 6 个对照插件跨插件回归 | 透传 pytest |
| `scripts/check_real_model.py` | 真实 ONNX 交付 gating；调用 `BirdDetector._preflight_onnx()`，**不伪造** | 0=passed, 1=failed, 2=未执行/缺文件, 3=参数错 |
| `scripts/validate_fixture.py` | 真实 bird 图片 intake gating；按 `expected_results.json` 校验文件，**不写 JSON** | 0=全通过, 1=校验失败, 2=参数错, 3=cv2 缺失 |
| `tests/regression/build_synthetic_fixtures.py` | 合成 `.npy`（仅 simulation 可复现的两个 label） | 0=正常 |

---

## Agent prompt 槽位

任务接力时，按场景挑选 prompt：

| 场景 | Prompt | 关键产出 |
|---|---|---|
| 收到真实 ONNX 模型，要切 real_dl | `prompts/real_dl_preflight_prompt.md` | preflight 全绿 / runtime_mode=real_dl / replay 至少 1 例真实样本 |
| 开始采集真实 bird 图片 fixture | `prompts/fixture_collection_prompt.md` | `tests/fixtures/{normal,anomaly,boundary,quality_fail}/*.jpg` + `expected_results.json` 槽位状态升级 |
| 加固 standalone UI 的 runtime_mode / 质量门可视化 | `prompts/ui_hardening_prompt.md` | 三类 runtime_mode 徽章 + 质量门三态 + 训练占位下载 |

每个 prompt 的强制约束已写入对应文件，agent 必须遵守。

---

## 后续补齐项

当前已达到标准治理基础测试层 + real_dl preflight 契约 + 质量门三态 + 合成 replay：`tests/` 目录已有 77 个测试。

下一阶段补齐项：

1. [ ] 真实 YOLOv8 ONNX 模型交付 → `scripts/check_real_model.py` 全绿 → runtime_mode=real_dl
2. [ ] 真实 bird 图片 fixture 采集（按 `prompts/fixture_collection_prompt.md`）→ `scripts/validate_fixture.py` 全绿 → 解锁 `bird_*` replay
3. [x] `scripts/run_targeted_tests.sh`（按模块分层执行入口）
4. [x] `scripts/run_regression_tests.sh`（全量回归）
5. [x] `scripts/run_quality_gate.sh`（反模式扫描 + 测试 + 覆盖率）
6. [x] `.coveragerc`（覆盖率门槛 = 60%，omit experimental / demo / tests）
7. [ ] 将 `BIRD_DATABASE` 外部化到数据文件
8. [ ] real_dl 通过后，评估 `experimental/enhanced_detector.py` 是否合回主链

`scripts/` 已建立；新增脚本必须同时：
1. 在「scripts/ 入口职责矩阵」中登记
2. 在 `tests/test_scripts_contract.py::TestRoutingDocumentsScripts` 测试中固定文件名引用
