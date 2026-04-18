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
- 无 L0 单测（FireTracker、火灾等级判定、传感器融合、区域匹配）
- test_standalone 测试 `infer()` 而非核心 `detect()`
- 无灭火联动逻辑测试
- 无降级路径测试

## 3. 分层定义

- **L0 Targeted**：纯逻辑测试，不依赖模型文件或 GPU，目标 < 2 分钟。
  - `FireTracker` — IoU 匹配、轨迹创建/过期、扩散速率
  - 火灾等级判定 — 给定已知 confidence + area_ratio，验证 FireLevel
  - 传感器融合 — 给定权重和传感器读数，验证 fusion_confidence
  - 区域匹配 — 点在多边形内检测
  - `FireDetection` / `SensorReading` / `FireAssessment` 数据类约束
- **L1 Integration**：`Plugin.create_standalone()` 全链路。
  - `detect()` 正常帧 / 空帧 / 带传感器数据
  - healthcheck / shutdown / reinit

## 4. 最小测试入口

```bash
# 运行全部测试
python -m pytest plugins/fire_detection/tests/ -q

# 仅 standalone 集成测试
python -m pytest plugins/fire_detection/tests/test_standalone.py -q
```

## 5. 测试命名约定

```
test_{module}_{scenario}_{expected_behavior}
```

示例：`test_tracker_new_detection_creates_track`、`test_fire_level_large_area_returns_critical`

## 6. 待补齐项（优先级排序）

1. **`tests/test_tracker.py`** — L0 FireTracker 跟踪逻辑与扩散分析 — **最高优先**
2. **`tests/test_fire_level.py`** — L0 火灾等级判定边界
3. **`tests/test_sensor_fusion.py`** — L0 融合置信度计算
4. **`tests/test_plugin_detect.py`** — L1 `detect()` 全链路（替代当前只测 `infer()` 的不足）
5. **`tests/test_suppression.py`** — L0 灭火联动触发条件与安全边界
