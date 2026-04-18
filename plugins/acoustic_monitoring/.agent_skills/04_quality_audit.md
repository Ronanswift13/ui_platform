# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`（吞异常）。
2. 禁止在生产代码新增 `print()`。
3. 禁止硬编码业务阈值（须来自 config）。
4. 禁止输出越界 `anomaly_score` 或 `confidence`（必须 `[0, 1]`）。
5. 禁止提交未覆盖的新逻辑分支。

## 2. 本项目高优先级审计项

1. **阈值来源一致性**：`detector.py` 中所有阈值是否从 `config` 读取，而非局部常量。
2. **anomaly_score 值域**：所有检测路径输出的 score 是否 clamp 到 `[0, 1]`。
3. **severity 映射稳定性**：`AcousticAnomalyType.get_severity()` 是否覆盖所有类型。
4. **降级路径可观测**：mock 音频路径是否标记 `data_source`。
5. **累计告警逻辑**：`alarm_accumulation_count` 是否正确递增/重置。

## 3. 反模式清单

| 反模式 | 检测方法 | 严重度 |
|--------|----------|--------|
| 硬编码阈值 | `rg '\b0\.\d+\b' detector.py` 人工审查是否为阈值 | 阻断 |
| 吞异常 | `rg 'except.*pass' *.py` | 阻断 |
| print 调试 | `rg '\bprint\(' *.py --glob '!scripts/*' --glob '!demo/*'` | 阻断 |
| score 未 clamp | 审查 `detect()` 返回路径 | 高 |
| 未使用的配置项 | 对比 YAML 键与代码读取 | 中 |

## 4. 审计命令

```bash
# 反模式扫描
rg 'except.*pass' plugins/acoustic_monitoring/*.py
rg '\bprint\(' plugins/acoustic_monitoring/*.py --glob '!scripts/*' --glob '!demo/*'

# 配置使用检查
rg 'config\.' plugins/acoustic_monitoring/detector.py | head -30

# 测试执行
python -m pytest plugins/acoustic_monitoring/tests/ -q
```
