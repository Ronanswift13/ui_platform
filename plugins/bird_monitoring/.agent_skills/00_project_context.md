# 00_project_context

## 0. 治理等级

**标准治理（部分达到 busbar 初步完善水平）** — 升级自 2026-04-16（见 `07_learning_log.md` 本日 Entry）。
达成条件：
- `tests/` 目录 58 个测试全通过，覆盖生命周期 / runtime truth / simulation 空检测 / 质量 / 风险 / 契约 / standalone 仿真隔离 / 目录职责
- 17 处 `print()` 清零，统一日志
- 阈值双源问题修复（全部从 YAML 读取）
- 伪数据风险清零（未知物种 → `unknown_bird`，不再默认 sparrow）
- runtime truth 外部可观测（healthcheck + 每条 result 的 metadata）
- 能力三分已回灌 PROJECT_CARD；真实检测仍 blocked
- PROJECT_CARD + README 齐全

下一台阶「高阶治理」的触发条件：real_dl preflight + 真实 fixture/replay + 质量门三态 + docs/prompts 升级记录。

## 1. 固定母版规则（跨插件统一）

1. **接口契约冻结**：插件必须实现 `init/process/healthcheck`，签名与 SDK 一致。
2. **配置优先**：阈值必须来自 `configs/default.yaml`，不允许在推理主路径硬编码。
3. **降级可观测**：降级路径必须输出 `failure_reason` + `metadata`。
4. **输出可校验**：`confidence` 必须满足值域 `[0, 1]`；`bbox` 归一化坐标 `[0, 1]`。
5. **测试分层**：至少包含 L0（单测）、L1（集成）分层执行入口。

## 2. 本项目差异规则（bird_monitoring 专属）

1. **业务目标**：输电线路鸟类检测、种类识别、风险评估、**驱离建议**（非驱离控制）。
2. **输入为图像帧**：`np.ndarray` (H×W×3 BGR) + `ROI` 列表。
3. **唯一生产检测器**：`plugin.py -> detector.py::BirdDetector`；`BirdDetectorEnhanced` 与 `experimental/advanced_bird_detector.py` 均为 experimental/demo。
4. **内置鸟类数据库**：`BIRD_DATABASE` 内置 8 种（外部 JSON 加载未实现，blocked）。
5. **风险评估复合评分**：距离 + 物种 + 翼展 + 行为，四维权重**全部从 YAML 读取**。
6. **驱离语义边界**：**不控制任何硬件**，仅输出 `deterrent_suggestion` JSON 建议（action/methods/reason）。README 与 manifest `blocked_capabilities` 明示。
7. **runtime truth 外显**：`simulation`（默认）/ `real_dl`（需 ONNX 文件）/ `traditional_fallback`（未启用）。healthcheck 暴露模型路径、文件存在性、session 和 real_model 状态。
8. **9 种 label 契约**：`no_bird / bird_safe / bird_warning / bird_danger / bird_critical / review_required / unknown_bird / quality_failed / error`，新增 label 必须更新 `test_standalone.py::test_infer_returns_results` 白名单。

## 3. 当前目录与职责边界

```
bird_monitoring/
├── plugin.py                       # SDK 适配层（质量门、风险编排、告警生成、鸟类数据库）
├── detector.py                     # 生产入口 BirdDetector；BirdDetectorEnhanced 为 legacy/experimental
├── advanced_bird_detector.py       # 兼容 shim，旧 import 跳转到 experimental
├── experimental/
│   └── advanced_bird_detector.py   # experimental/demo（含随机检测，禁止生产接入）
├── main.py                         # CLI 入口（train/infer 独立脚本）
├── configs/
│   ├── default.yaml                # 运行参数唯一来源
│   └── standalone.yaml             # standalone 专用覆盖配置
├── manifest.json                   # 插件注册信息
├── standalone/                     # 独立运行 Web 仪表盘
│   ├── app.py
│   ├── bird_simulator.py           # standalone_simulation 演示仿真器
│   ├── templates/
│   └── static/
├── docs/                           # 升级说明槽位
├── prompts/                        # agent prompt 槽位
├── demo/run_demo.py                # 演示脚本
├── notebooks/                      # 调试 notebook
├── data/results.db                 # 检测结果存储
└── .agent_skills/                  # AI 代理规则（本目录）
```

**注意**：`scripts/` 目录仍为空（未建构建/校验脚本）；`tests/` 已就位（conftest + 5 测试文件）。

```
bird_monitoring/tests/
├── conftest.py                     # plugin_instance / sample_frame / make_context / make_roi
├── test_standalone.py              # 22 测试：生命周期 + runtime truth + simulation 空检测 + standalone 仿真隔离
├── test_risk_assessment.py         # 10 测试：风险 + 物种识别
├── test_quality_assessment.py      # 9 测试：质量门
├── test_plugin_contract.py         # 13 测试：结果结构 + 告警契约 + 驱离建议
└── test_directory_contract.py      # 4 测试：入口与目录职责契约
```

## 4. 模块职责边界

| 模块 | 职责 | 不应包含 |
|------|------|----------|
| `plugin.py` | SDK 适配、配置解析、鸟类数据库查询、风险评分编排、告警生成 | YOLO 推理、跟踪算法细节 |
| `detector.py::BirdDetector` | ONNX 推理、空检测 simulation、图像质量评估、runtime truth | SDK schema、告警级别映射、硬件控制 |
| `detector.py::BirdDetectorEnhanced` | legacy/experimental 增强检测代码 | 默认生产加载、硬件控制 |
| `experimental/advanced_bird_detector.py` | PyTorch demo、鸟种分类/轨迹预测概念验证 | 生产 import、真实结果宣称 |
| `main.py` | CLI train/infer 入口 | 不应被其他模块 import |
| `standalone/app.py` | Web 仪表盘运行、挂载隔离仿真路由 | 生产算法逻辑 |
| `standalone/bird_simulator.py` | UI demo 仿真 | 生产 import、真实精度证据 |

## 5. AI 自动闭环 vs 人工确认

### 可自动闭环
- `.agent_skills/` 规则维护
- `tests/` 新建与测试补齐
- 配置键一致性检查
- 反模式扫描

### 需人工确认
- 鸟类数据库增删（`BIRD_DATABASE` 变更）
- 风险评估权重调整（影响告警触发策略）
- 任何驱鸟设备控制策略变更（当前 blocked）
- `experimental/advanced_bird_detector.py` / `BirdDetectorEnhanced` 是否正式启用/弃用
- 模型文件路径变更（manifest.json）

## 6. 可执行校验命令

```bash
# 配置可解析
python -c "import yaml; yaml.safe_load(open('plugins/bird_monitoring/configs/default.yaml'))"

# 插件可导入
python -c "from plugins.bird_monitoring.plugin import BirdMonitoringPlugin; print(BirdMonitoringPlugin.__name__)"

# standalone 可创建
python -c "from plugins.bird_monitoring.plugin import BirdMonitoringPlugin as P; p = P.create_standalone(); print(p.id)"
```

## 7. 已知技术债（2026-04-16 更新）

### 已清零 ✅
- ~~阈值双源~~：已全部从 YAML 读取，类常量只做 fallback（2026-04-16）
- ~~无测试~~：58 个测试已就位（2026-04-16）
- ~~print 污染~~：17 处 `print()` 清零（2026-04-16）
- ~~伪数据风险~~：未知物种 → `unknown_bird`；生产 simulation 不造鸟（2026-04-16）
- ~~硬件驱离误入生产~~：驱离入口改为 suggestion-only，硬件执行阻断（2026-04-16）

### 待解决 ⏳
1. `BIRD_DATABASE` 仍硬编码在 plugin.py（迁移 YAML 需人工确认，未阻塞）
2. `BirdDetectorEnhanced` 仍在 `detector.py` 内，需后续移入 `experimental/` 或加显式 opt-in
3. 无真实 ONNX 模型（blocked，需模型训练）
4. 无 regression 精度基线（blocked，需 fixture 图片）
5. Standalone HTML 仅显示计数，未显示逐条结果 / 风险面板 / 驱离建议
