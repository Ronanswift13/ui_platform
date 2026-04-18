# 02 — 插件成熟度矩阵

> 最后更新: 2026-04-17

## 成熟度等级

| 等级 | 含义 | 最低要求 |
|---|---|---|
| **L3 完整态** | 四件套齐全 | plugin.py + manifest.json + configs/ + tests/ |
| **L2 基本态** | 缺少 configs 或 tests | plugin.py + manifest.json + (configs 或 tests) |
| **L1 骨架态** | 仅有入口 | plugin.py + manifest.json |
| **L0 占位态** | 无插件骨架 | 仅 __init__.py + 单体模块 |

### 标准治理基线标注

`L3 标准治理基线` 是 `L3 完整态` 的子状态，表示插件不仅具备四件套，还具备当前任务要求的本地 agent skill、contract/smoke 测试、最小质量门或 targeted 脚本、统一输入输出合同说明。

注意：`标准治理基线` 不等于完整高频开发级，也不等于已有覆盖率门禁、真实硬件/协议联调、UI/cockpit 接线或在线训练闭环。

## 插件矩阵

| 插件 | plugin.py | manifest | configs/ | tests/ | .py 数 | 等级 |
|---|:---:|:---:|:---:|:---:|---:|---|
| acoustic_monitoring | ✓ | ✓ | ✓ | ✓ | 20 | **L3 标准治理基线** |
| animal_detection | ✓ | ✓ | ✓ | ✓ | 6 | **L3** |
| bird_monitoring | ✓ | ✓ | ✓ | ✓ | 7 | **L3 标准治理** |
| busbar_inspection | ✓ | ✓ | ✓ | ✓ | 8 | **L3** |
| capacitor_inspection | ✓ | ✓ | ✓ | ✓ | 5 | **L3** |
| device_monitoring | ✓ | ✓ | ✓ | ✓ | 17 | **L3 标准治理基线** |
| fire_detection | ✓ | ✓ | ✓ | ✓ | 5 | **L3** |
| indoor_fence | ✓ | ✓ | ✓ | ✓ | 9 | **L3** |
| meter_reading | ✓ | ✓ | ✓ | ✓ | 5 | **L3** |
| switch_inspection | ✓ | ✓ | ✓ | ✓ | 6 | **L3** |
| temperature_monitoring | ✓ | ✓ | ✓ | ✓ | 5 | **L3** |
| transformer_inspection | ✓ | ✓ | ✓ | ✓ | 8 | **L3** |
| gas_detection | ✓ | ✓ | ✓ | ✓ | 15 | **L3 标准治理基线** |
| hyperspectral_detection | ✓ | ✓ | ✗ | ✓ | 3 | **L2** |
| multimodal_fusion | ✓ | ✓ | ✗ | ✓ | 6 | **L2** |
| slam_mapping | ✓ | ✓ | ✗ | ✓ | 5 | **L2** |
| action_event_monitoring | ✓ | ✓ | ✓ | ✓ | 13 | **L3 标准治理基线** |
| **radar** | ✗ | ✗ | ✗ | ✗ | 2 | **L0** |
| **thermal** | ✗ | ✗ | ✗ | ✗ | 2 | **L0** |

## 汇总

- **L3 完整态 / 标准治理基线**: 14 个 (74%)
- **L2 基本态**: 3 个 (16%) — hyperspectral_detection, multimodal_fusion, slam_mapping
- **L0 占位态**: 2 个 (10%) — radar, thermal

## L0 → L1 升级路径

radar 和 thermal 需要补齐:
1. `plugin.py` — 继承 `EnhancedBasePlugin`，实现 `init/process/process_async`
2. `manifest.json` — 声明 name/version/dependencies
3. `configs/default.yaml` — 默认参数
4. `tests/test_plugin.py` — 冒烟测试

## L3 → L3 标准治理 升级路径

条件（基于 bird_monitoring 2026-04-16 实践）：
1. 生产代码 `print()` 清零
2. 业务阈值全部从 YAML 读取
3. 未识别 / 未知 → 显式 `unknown_*` 回退（无伪数据）
4. 输入质量门（`quality_failed` label）
5. 复核路径（`review_required` label + WARNING 告警）
6. runtime_mode 外部可观测（healthcheck + 每条 result metadata）
7. 训练占位元数据（`training_placeholders`）
8. manifest 能力三分完整（`verified_capabilities / experimental_capabilities / blocked_capabilities`）
9. 未初始化 infer 返回 `error/9000`
10. 测试矩阵覆盖 4 维度（生命周期 / 质量 / 核心逻辑 / 契约），≥ 30 测试

**参考**：`plugins/bird_monitoring/.agent_skills/engineering-baseline-playbook.md`（七步转换法 + 准出清单）
**参考学习日志**：`plugins/bird_monitoring/.agent_skills/07_learning_log.md`（2026-04-16 Entry）

### 标准治理状态

| 插件 | 标准治理 | 学习日志条目数 |
|---|:---:|:---:|
| busbar_inspection | ✓ (金标准) | 见其 07_learning_log.md |
| bird_monitoring | ✓ (2026-04-16) | 1 |
| acoustic_monitoring | ✓ (B 类时序传感基线，2026-04-17) | 1 |
| gas_detection | ✓ (B 类时序传感基线，2026-04-17) | 见其 07_learning_log.md |
| device_monitoring | ✓ (B 类时序传感基线，2026-04-17) | 1 |
| action_event_monitoring | ✓ (B 类事件监测基线，2026-04-17) | 见其 07_learning_log.md |
| 其他 L3 插件 | 未验证，建议按 playbook 复核 | — |

## B 类时序传感/数值异常插件当前基线

| 插件 | 当前事实 | 最小质量门 | 未完成边界 |
|---|---|---|---|
| acoustic_monitoring | detector/analyzer/config/process/standalone/WAV replay 测试已补齐 | `plugins/acoustic_monitoring/scripts/run_sanity_checks.sh`、`run_targeted_tests.sh`、`run_quality_gate.sh` | 真实现场音频、覆盖率门槛、真实模型资产 preflight |
| gas_detection | `configs/default.yaml` 已存在；predictor + analyzer 已进入 `process()` 主输出合同；24 样本趋势路径已回归锁定 | `plugins/gas_detection/scripts/run_sanity_checks.sh`、`plugins/gas_detection/tests/test_trend_contract.py` | DGA 详细分析、结果库写入、targeted/quality gate 脚本 |
| device_monitoring | detector 边界、健康指数/工单阈值、现场式 replay fixture 已补齐；CPU 使用率阈值已配置化 | `plugins/device_monitoring/scripts/run_sanity_checks.sh`、`run_targeted_tests.sh` | `scan_devices()` 专项测试、真实脱敏遥测、quality gate/覆盖率 |
| action_event_monitoring | requirements/demo/entrypoints/Plugin 别名/standalone smoke 已补齐；已进入全局 integration 清单和 installer `monitoring/8097` 映射 | `plugins/action_event_monitoring/scripts/run_sanity_checks.sh`、`plugins/action_event_monitoring/tests/` | UI/dashboard/cockpit 前端入口、真实协议 smoke、CandidateEvent/人工复核 API |

### 根级同步要求

当任一插件从 L2 升到 L3，或从普通 L3 升到 `L3 标准治理基线` 时，必须同步更新：

1. 插件本地 `.agent_skills/00_project_context.md`、`03_test_strategy.md`、`08_task_routing.md`。
2. 本文件的插件矩阵、汇总统计和标准治理状态。
3. 若涉及全局 standalone/installer 接线，必须同时核对 `platform_core/plugin_manager/installer.py`、`tests/integration/test_plugin_standalone.py`、`tests/integration/test_all_standalone_boot.py`。
