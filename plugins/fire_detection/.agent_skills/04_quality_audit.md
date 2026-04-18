# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`（吞异常）。
2. 禁止在生产代码新增 `print()`。
3. 禁止硬编码业务阈值（须来自 config）。
4. 禁止输出越界 `confidence` 或非法 `bbox`。
5. 禁止提交未覆盖的新逻辑分支。

## 2. 本项目高优先级审计项

1. **灭火联动安全性**：`suppression.auto_sprinkler_enabled` 和 `auto_power_cutoff` 默认 `false` — 必须确保代码中不存在绕过配置开关的路径。
2. **fusion_confidence 值域**：融合计算输出是否 clamp 到 `[0, 1]`。
3. **fire_level 判定一致性**：`config.fire_level.*` 阈值与代码中判定逻辑是否对齐。
4. **zone 多边形安全**：区域匹配是否处理空多边形或退化多边形。
5. **test_standalone 测试 infer 而非 detect**：当前测试未覆盖核心 `detect()` 路径。
6. **spread_rate 符号语义**：正值表示扩大、负值表示缩小 — 是否在所有消费端一致使用。

## 3. 反模式清单

| 反模式 | 检测方法 | 严重度 |
|--------|----------|--------|
| 硬编码阈值 | `rg '\b0\.\d+\b' detector.py` 人工审查 | 高 |
| 吞异常 | `rg 'except.*pass' *.py` | 阻断 |
| print 调试 | `rg '\bprint\(' *.py --glob '!scripts/*' --glob '!demo/*'` | 阻断 |
| fusion_confidence 未 clamp | 审查融合计算 return 路径 | 高 |
| 灭火开关绕过 | `rg 'sprinkler\|power_cutoff' plugin.py detector.py` 审查控制流 | 阻断 |
| 未使用配置项 | 对比 YAML 键与代码读取 | 中 |

## 4. 审计命令

```bash
# 反模式扫描
rg 'except.*pass' plugins/fire_detection/*.py
rg '\bprint\(' plugins/fire_detection/*.py --glob '!scripts/*' --glob '!demo/*'

# 灭火联动安全审查
rg 'sprinkler\|power_cutoff\|auto_.*enabled' plugins/fire_detection/plugin.py plugins/fire_detection/detector.py

# 阈值来源检查
rg 'self\.config\[' plugins/fire_detection/detector.py | head -20

# 测试执行
python -m pytest plugins/fire_detection/tests/ -q
```
