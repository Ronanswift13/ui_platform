# 03_test_strategy

## 1. 固定母版规则（测试治理）

1. 每个硬约束至少对应 1 个自动化测试。
2. 每个 bug 修复必须新增防回归测试。
3. 测试分层执行：L0 快速、L1 集成、L2 回归。
4. 测试脚本必须返回明确退出码（0 通过，非 0 失败）。

## 2. 本项目差异规则（busbar_inspection）

### 2.1 分层定义

- **L0 Targeted**：契约与纯逻辑测试，目标 <= 2 分钟。
- **L1 Integration**：`Plugin.create_standalone()` + `infer/postprocess/healthcheck` 全链路。
- **L2 Regression**：样本集回放、原因码稳定性、性能回归。

### 2.2 覆盖目标（本项目硬门槛）

- 总体覆盖率 >= 70%
- `plugin.py` 覆盖率 >= 85%
- `detector_enhanced.py` 关键契约函数覆盖率 >= 75%
- 新增模块覆盖率 >= 90%

> 说明：当前仓库测试存量较少，本门槛用于“第一轮补齐后”的合并门禁。

## 3. `scripts/run_targeted_tests.sh` 职责说明（强制）

脚本职责固定为“快速失败 + 模块化执行”：

1. 接收模块参数：`standalone/plugin/detector/quality/config/all`。
2. 仅运行 L0/L1 快速测试，不运行慢速回归。
3. 模块无测试文件时返回非 0，阻断“假绿色”。
4. 输出执行摘要（模块名、执行文件、通过/失败）。

建议调用：

```bash
./scripts/run_targeted_tests.sh standalone
./scripts/run_targeted_tests.sh plugin
./scripts/run_targeted_tests.sh all
```

## 4. `scripts/run_regression_tests.sh` 职责说明（强制）

脚本职责固定为“发布前全量门禁”：

1. 先跑 `run_targeted_tests.sh all`，失败即停止。
2. 执行全量 `pytest`（包含 `regression` 标记）。
3. 执行静态与安全检查（工具存在则强制执行）。
4. 输出阶段化结果与最终 PASS/FAIL。

建议调用：

```bash
./scripts/run_regression_tests.sh
```

## 5. 必测用例矩阵（第一轮必须补齐）

| 用例ID | 模块 | 输入 | 期望 |
|---|---|---|---|
| T-001 | plugin | 空 ROI 列表 | 返回空 list，不抛异常 |
| T-002 | plugin | 非法 ROI 框 | 跳过该 ROI，不影响其他 ROI |
| T-003 | detector | 模糊图像 | 质量门禁失败，原因码=103 |
| T-004 | detector | 过曝图像 | 质量门禁失败，原因码=101 |
| T-005 | detector | 低对比图像 | 质量门禁失败，原因码=101 |
| T-006 | detector | 正常图像 + 小目标 | 输出变焦建议且 action 非 NONE |
| T-007 | detector | 多重叠框 | NMS 后数量下降 |
| T-008 | contract | 内部码 1001/1002/1004 | 外部码映射 103/101/102 |
| T-009 | contract | YAML 修改 `thresholds.conf_thr` | 算法读取值同步变化 |
| T-010 | integration | infer + postprocess | 告警等级和文案符合契约 |

## 6. AI 自动闭环 / 人工确认

### 可自动闭环

- 新增与维护 `tests/test_*.py`
- 运行 targeted/regression 脚本
- 输出失败堆栈并自动修复测试相关问题

### 必须人工确认

- 回归样本集是否覆盖真实现场（雨雾/逆光/遮挡）
- 性能门槛是否满足上线工况
- 告警误报成本与漏报成本权衡

## 7. 失败阻断条件

满足任一条件即阻断合并：

1. `run_targeted_tests.sh all` 非 0。
2. `run_regression_tests.sh` 非 0。
3. 必测矩阵任何用例缺失。
4. 覆盖率低于门槛。
