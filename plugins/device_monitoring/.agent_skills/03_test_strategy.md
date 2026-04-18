# 03_test_strategy

## 1. 固定母版规则

1. 每个硬约束至少对应 1 个自动化测试。
2. 每个 bug 修复必须新增防回归测试。
3. 测试分层执行：L0 快速、L1 集成。
4. 测试脚本必须返回明确退出码。

## 2. 当前测试现状

| 文件 | 层级 | 内容 |
|------|------|------|
| `tests/test_config_contract.py` | L0 | manifest/YAML/config schema |
| `tests/test_process_contract.py` | L1 | 单设备、多设备、空输入、阈值改动、metadata、趋势降级 |
| `tests/test_detector.py` | L0 | CPU 阈值配置化、健康指数边界、工单触发/去重、异常分数值域 |
| `tests/test_device_replay.py` | L1 | `tests/fixtures/field_like_device_replay.json` 现场式遥测回放到告警/工单路径 |
| `tests/test_standalone.py` | L1 | `create_standalone` + `healthcheck` + `infer` + runner/smoke |

当前缺口：`scan_devices()` 专项测试和真实设备脱敏样本仍未补齐。现有 `field_like_device_replay.json` 是更接近现场的合成/脱敏式 fixture，不得写成真实硬件闭环。

## 3. 分层定义

- **L0 Targeted**：纯逻辑测试，无外部依赖，目标 < 2 分钟。
  - `DeviceHealthCalculator.calculate()` — 各指标边界值
  - 状态判定（health_index → status 映射）
  - 工单触发条件（`health_index < th_alarm` → 生成 ticket）
  - 异常评分值域（`anomaly_score` ∈ `[0, 1]`）
- **L1 Integration**：`Plugin.create_standalone()` 全链路。
  - `detect()` 正常读数 / 空读数 / 异常读数
  - `scan_devices()` 模拟扫描
  - `healthcheck` / `shutdown` / `reinit`

## 4. 最小测试入口

```bash
# 运行全部测试
python -m pytest plugins/device_monitoring/tests/ -q

# 最小质量门
cd plugins/device_monitoring && ./scripts/run_sanity_checks.sh

# detector/replay 快速回归
cd plugins/device_monitoring && ./scripts/run_targeted_tests.sh
```

## 5. 测试命名约定

```
test_{module}_{scenario}_{expected_behavior}
```

示例：`test_calculator_high_cpu_temp_deducts_score`、`test_detect_empty_readings_returns_success`

## 6. 后续补齐项（优先级排序）

1. **`tests/test_scan_devices.py`** — scan_devices 模拟扫描输出结构验证。
2. **真实脱敏遥测 replay** — 替换或追加到 `tests/fixtures/`，保留来源说明且不得包含凭据。
3. **quality gate 脚本** — 反模式扫描 + py_compile + pytest，稳定后再考虑覆盖率。
