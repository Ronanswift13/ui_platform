# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`（吞异常）。
2. 禁止在生产代码新增 `print()`。
3. 禁止硬编码业务阈值（须来自 config）。
4. 禁止输出越界 `confidence` 或非法 `bbox`。
5. 禁止提交未覆盖的新逻辑分支。

## 2. 本项目高优先级审计项

1. **thermal_frame 归一化硬编码**：`detector.py` 约第 164 行 `frame / 255.0 * 60 + 15` 将值映射到 15-75°C — 此范围为硬编码，应考虑配置化或至少文档化。
2. **severity 全覆盖**：`_temp_severity()` 是否覆盖所有 threshold 分支，无遗漏。
3. **z-score 除零防护**：`std + 1e-6` 防护是否在所有路径生效。
4. **test_standalone 测试 infer 而非 detect**：当前测试未覆盖核心 `detect()` 路径。
5. **联动事件安全**：`linkage.ventilation_control` 默认 false — 确保代码中不存在绕过开关的路径。
6. **heatmap.tolist() 性能**：大尺寸 heatmap 序列化可能造成性能瓶颈。

## 3. 反模式清单

| 反模式 | 检测方法 | 严重度 |
|--------|----------|--------|
| 硬编码温度映射 | `rg '255\|15.*75' detector.py` | 高 |
| 吞异常 | `rg 'except.*pass' *.py` | 阻断 |
| print 调试 | `rg '\bprint\(' *.py --glob '!scripts/*' --glob '!demo/*'` | 阻断 |
| severity 分支遗漏 | 审查 `_temp_severity()` 和 `_assess_status()` 所有 return 路径 | 高 |
| 联动开关绕过 | `rg 'ventilation\|linkage' plugin.py detector.py` 审查控制流 | 高 |
| 未使用配置项 | 对比 YAML 键与代码读取 | 中 |

## 4. 审计命令

```bash
# 反模式扫描
rg 'except.*pass' plugins/temperature_monitoring/*.py
rg '\bprint\(' plugins/temperature_monitoring/*.py --glob '!scripts/*' --glob '!demo/*'

# 硬编码检查
rg '255|15.*75|0\.02|2\.5' plugins/temperature_monitoring/detector.py

# 配置使用检查
rg 'self\.th_|self\.hs_|self\.config' plugins/temperature_monitoring/detector.py | head -20

# 测试执行
python -m pytest plugins/temperature_monitoring/tests/ -q
```
