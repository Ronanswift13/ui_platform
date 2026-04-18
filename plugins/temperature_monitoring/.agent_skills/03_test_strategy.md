# 03_test_strategy

## 1. 固定母版规则

1. 每个硬约束至少对应 1 个自动化测试。
2. 每个 bug 修复必须新增防回归测试。
3. 测试分层执行：L0 快速、L1 集成。
4. 测试脚本必须返回明确退出码。

## 2. 当前测试现状

| 文件 | 层级 | 内容 |
|------|------|------|
| `tests/test_standalone.py` | L1 | `create_standalone` + `healthcheck` + `infer`（返回列表） + `runner` 创建 |
| `tests/conftest.py` | — | 仅路径配置 |

**缺口**：
- 无 L0 单测（热点检测、severity 映射、趋势分析、区域匹配）
- test_standalone 测试 `infer()` 而非核心 `detect()`
- 无 heatmap 生成路径测试（thermal / sensor / simulate 三条路径）
- 无联动事件触发测试

## 3. 分层定义

- **L0 Targeted**：纯逻辑测试，不依赖模型文件或 GPU，目标 < 2 分钟。
  - `_temp_severity()` — 给定温度值，验证 severity 级别
  - `_detect_hotspots()` — 给定已知 heatmap，验证热点位置和温度
  - `_match_zone()` — 给定坐标和区域定义，验证匹配
  - `_analyze_trend()` — 给定历史序列，验证 rise_rate 和 direction
  - `_get_heatmap()` 三条路径覆盖
- **L1 Integration**：`Plugin.create_standalone()` 全链路。
  - `detect()` 有 thermal_frame / 有 sensor_readings / 两者均无
  - healthcheck / shutdown / reinit

## 4. 最小测试入口

```bash
# 运行全部测试
python -m pytest plugins/temperature_monitoring/tests/ -q

# 仅 standalone 集成测试
python -m pytest plugins/temperature_monitoring/tests/test_standalone.py -q
```

## 5. 测试命名约定

```
test_{module}_{scenario}_{expected_behavior}
```

示例：`test_severity_above_critical_returns_critical`、`test_detect_no_input_uses_simulation`

## 6. 待补齐项（优先级排序）

1. **`tests/test_detector.py`** — L0 热点检测 + severity + 趋势分析 — **最高优先**
2. **`tests/test_heatmap.py`** — L0 三条 heatmap 生成路径
3. **`tests/test_plugin_detect.py`** — L1 `detect()` 全链路（替代当前只测 `infer()` 的不足）
4. **`tests/test_zone_match.py`** — L0 区域匹配 + threshold_offset
