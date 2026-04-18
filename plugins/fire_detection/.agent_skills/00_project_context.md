# 00_project_context

## 0. 治理等级

**最小治理** — 理由：仅 1 个 standalone 测试文件、仅有 benchmark 脚本、无质量门禁。但本插件功能复杂度高（多模态融合、灭火联动、疏散路线），一旦补齐测试基线应尽快升至**标准治理**。

## 1. 固定母版规则（跨插件统一）

1. **接口契约冻结**：插件必须实现 `init/detect/infer/healthcheck`，签名与 SDK 一致。
2. **配置优先**：阈值必须来自 `configs/default.yaml`，不允许在推理主路径硬编码。
3. **降级可观测**：降级路径必须输出 `failure_reason` + `metadata`。
4. **输出可校验**：`confidence` ∈ `[0, 1]`；`bbox` 归一化坐标 ∈ `[0, 1]`。
5. **测试分层**：至少包含 L0（单测）、L1（集成）分层执行入口。

## 2. 本项目差异规则（fire_detection 专属）

1. **业务目标**：室内消防监测——火焰/烟雾实时检测、火势跟踪、多传感器融合、主动灭火联动、应急疏散。
2. **多模态输入**：可见光帧 `frame` + 热成像 `thermal_frame`（可选）+ 传感器 `sensor_data`（烟雾/温度/CO/湿度，可选）。
3. **火灾等级评估**：6 级 `FireLevel`（none → smoldering → small → medium → large → critical），阈值由 `fire_level.*` 配置。
4. **主动灭火联动**：`suppression` 配置控制自动喷淋/断电/声光报警，涉及物理设备动作。
5. **区域监控**：`zones` 配置定义监控区域多边形与优先级，检测结果关联 zone_id。
6. **演练模式**：`drill.enabled` 支持消防演练场景模拟。

## 3. 当前目录与职责边界

```
fire_detection/
├── plugin.py                   # SDK 适配层（init、detect 编排、告警生成、灭火联动、疏散路线）
├── detector.py                 # 算法层（FireDetector YOLO 检测 + FireTracker 跟踪 +
│                               #   传感器融合 + 火灾等级评估 + 扩散分析）
├── configs/default.yaml        # 运行参数唯一来源（模型/检测/融合/灭火/区域/疏散/演练）
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
| `plugin.py` | SDK 适配、配置加载、detect() 编排、告警生成、灭火联动触发、疏散路线推荐、训练数据上传 | YOLO 推理、IoU 跟踪、融合算法 |
| `detector.py` | YOLO 推理、FireTracker 火势跟踪、传感器融合评分、火灾等级判定、扩散速率分析 | SDK schema、告警级别映射、灭火设备控制 |
| `standalone/app.py` | Web 仪表盘运行 | 算法逻辑 |

### 关键数据流

```
frame [+ thermal_frame] [+ sensor_data]
    → plugin.detect()
        → detector.detect(frame) → visual detections
        → FireTracker.update() → tracked detections + spread_rate
        → sensor_fusion() → fusion_confidence
        → fire_level_assessment() → FireLevel
        → suppression_decision() → actions
    → plugin: alarms + evacuation_routes
    → 输出 Dict
```

## 5. AI 自动闭环 vs 人工确认

### 可自动闭环
- `.agent_skills/` 规则维护
- `tests/` 测试补齐与执行
- 配置键一致性检查
- 反模式扫描

### 需人工确认
- **灭火联动配置变更**（`suppression.*` — 涉及喷淋/断电物理动作）
- 火灾等级阈值调整（`fire_level.*`）
- 传感器融合权重变更
- 监控区域 / 疏散路线配置变更
- 模型文件路径变更（manifest.json）

## 6. 可执行校验命令

```bash
# 配置可解析
python -c "import yaml; yaml.safe_load(open('plugins/fire_detection/configs/default.yaml'))"

# 插件可导入
python -c "from plugins.fire_detection.plugin import FireDetectionPlugin; print(FireDetectionPlugin.__name__)"

# 基础测试
python -m pytest plugins/fire_detection/tests/test_standalone.py -q

# 性能基准
python -m plugins.fire_detection.scripts.benchmark
```
