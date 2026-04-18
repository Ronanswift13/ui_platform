# 00_project_context

## 0. 治理等级

**最小治理** — 理由：仅 1 个 standalone 测试文件（测试 `infer()` 而非核心 `detect()`）、仅有 benchmark 脚本、无质量门禁。检测逻辑为确定性公式（z-score 热点 + 阈值分级），复杂度适中。

## 1. 固定母版规则（跨插件统一）

1. **接口契约冻结**：插件必须实现 `init/detect/infer/healthcheck`，签名与 SDK 一致。
2. **配置优先**：阈值必须来自 `configs/default.yaml`，不允许在推理主路径硬编码。
3. **降级可观测**：降级路径必须输出 `failure_reason` + `metadata`。
4. **输出可校验**：`temperature` 为物理量（°C）；`severity` 必须覆盖所有分支。
5. **测试分层**：至少包含 L0（单测）、L1（集成）分层执行入口。

## 2. 本项目差异规则（temperature_monitoring 专属）

1. **非图像检测插件**：核心输入为热成像帧 `thermal_frame`（可选）或传感器阵列读数 `sensor_readings`（可选），两者均缺失时生成模拟热力图。
2. **热力图为中间表示**：所有输入统一转换为 2D `np.ndarray` 温度矩阵（heatmap），后续分析在此基础上进行。
3. **z-score 热点检测**：使用统计方法检测异常高温区域，非 ML 主路径。
4. **温升趋势预测**：基于历史缓冲的线性/ARIMA/LSTM 预测，默认 linear。
5. **跨模块联动**：`linkage` 配置可触发 `fire_detection` / `fence_plugin` / 通风控制联动事件。
6. **区域化监控**：`zones` 配置定义监控区域（矩形 region），支持 `threshold_offset` 区域阈值偏移。

## 3. 当前目录与职责边界

```
temperature_monitoring/
├── plugin.py                   # SDK 适配层（init、detect 编排、告警生成、infer 适配）
├── detector.py                 # 算法层（TemperatureDetector：heatmap 生成、热点检测、
│                               #   趋势分析、状态评估、联动检查）
├── configs/default.yaml        # 运行参数唯一来源（传感器/阈值/热点/预测/区域/联动）
├── manifest.json               # 插件注册信息
├── tests/
│   ├── conftest.py             # 路径配置
│   └── test_standalone.py      # L1 集成测试（create_standalone + healthcheck + infer）
├── scripts/benchmark.py        # 性能基准测试
├── standalone/                 # 独立运行 Web 仪表盘
│   ├── app.py
│   └── templates/
├── demo/run_demo.py            # 演示脚本
└── .agent_skills/              # AI 代理规则（本目录）
```

## 4. 模块职责边界

| 模块 | 职责 | 不应包含 |
|------|------|----------|
| `plugin.py` | SDK 适配、配置加载、detect() 编排、告警生成、infer() → RecognitionResult 转换 | 热力图计算、z-score 分析、趋势预测 |
| `detector.py` | heatmap 生成（热成像/传感器/模拟）、z-score 热点检测、趋势分析、状态评估、联动事件检查 | SDK schema、告警级别映射 |
| `standalone/app.py` | Web 仪表盘运行 | 算法逻辑 |

### 关键数据流

```
thermal_frame / sensor_readings / (模拟)
    → _get_heatmap() → 2D np.ndarray (温度矩阵)
    → 基础统计 (max/min/avg)
    → _detect_hotspots() → List[Hotspot] (z-score + 轮廓分析)
    → _analyze_trend() → TempTrend (历史缓冲 + 预测)
    → _assess_status() → status (normal/warning/alarm/critical)
    → _check_linkage() → linkage_events
    → TemperatureResult
```

## 5. AI 自动闭环 vs 人工确认

### 可自动闭环
- `.agent_skills/` 规则维护
- `tests/` 测试补齐与执行
- 配置键一致性检查
- 反模式扫描

### 需人工确认
- 温度阈值调整（`thresholds.*` — 影响告警触发）
- 联动配置变更（`linkage.*` — 跨插件联动）
- 通风控制启用（`ventilation_control` — 涉及物理设备）
- 预测方法切换（`prediction.method` — LSTM 需模型文件）
- 区域定义变更（`zones.*`）

## 6. 可执行校验命令

```bash
# 配置可解析
python -c "import yaml; yaml.safe_load(open('plugins/temperature_monitoring/configs/default.yaml'))"

# 插件可导入
python -c "from plugins.temperature_monitoring.plugin import TemperatureMonitoringPlugin; print(TemperatureMonitoringPlugin.__name__)"

# 基础测试
python -m pytest plugins/temperature_monitoring/tests/test_standalone.py -q

# 性能基准
python -m plugins.temperature_monitoring.scripts.benchmark
```
