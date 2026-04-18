# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`（吞异常）。
2. 禁止在生产代码新增 `print()`。
3. 禁止硬编码业务阈值（须来自 config）。
4. 禁止输出越界 `health_index` 或 `anomaly_score`。
5. 禁止提交未覆盖的新逻辑分支。

## 2. 本项目高优先级审计项

1. **cpu_usage 阈值配置化回归**：`detector.py` 已从 `thresholds.cpu_usage_warning_percent` / `cpu_usage_alarm_percent` 读取，后续不得退回硬编码。
2. **health_index 值域**：`DeviceHealthCalculator.calculate()` 最后 `max(0, min(100, ...))` 是否覆盖所有路径。
3. **anomaly_score 值域**：`_calc_anomaly()` 返回值是否 clamp 到 `[0, 1]`。
4. **工单去重**：`_create_ticket()` 是否对同一设备重复创建工单有防护。
5. **核心 detect/process 覆盖**：`test_process_contract.py` 与 `test_device_replay.py` 已覆盖核心路径，后续新增工单或健康规则必须同步测试。

## 3. 反模式清单

| 反模式 | 检测方法 | 严重度 |
|--------|----------|--------|
| 硬编码阈值 | `rg '> 90\|> 70\|> 85\|< 50' detector.py` 人工审查，确认只是默认兜底或测试边界 | 高 |
| 吞异常 | `rg 'except.*pass' *.py` | 阻断 |
| print 调试 | `rg '\bprint\(' *.py --glob '!scripts/*' --glob '!demo/*'` | 阻断 |
| health_index 未 clamp | 审查 `calculate()` 所有 return 路径 | 高 |
| anomaly_score 未 clamp | 审查 `_calc_anomaly()` return | 高 |
| 未使用配置项 | 对比 YAML 键与代码读取 | 中 |

## 4. 审计命令

```bash
# 反模式扫描
rg 'except.*pass' plugins/device_monitoring/*.py
rg '\bprint\(' plugins/device_monitoring/*.py --glob '!scripts/*' --glob '!demo/*'

# 硬编码阈值排查
rg '> 90|> 70|> 85|< 50' plugins/device_monitoring/detector.py

# 配置使用检查
rg 'self\.weights\.|self\.thresholds\.|self\.config\.' plugins/device_monitoring/detector.py | head -30

# 测试执行
python -m pytest plugins/device_monitoring/tests/ -q

# 最小质量门
cd plugins/device_monitoring && ./scripts/run_sanity_checks.sh

# detector/replay targeted
cd plugins/device_monitoring && ./scripts/run_targeted_tests.sh
```
