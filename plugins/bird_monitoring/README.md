# bird_monitoring 插件

输变电站室外场景鸟类监控插件。当前阶段定位是 **低数据、无真实模型、强工程契约**：优先保证运行真值、质量门、复核路径和“不伪造检测”。

## 1. 当前运行模式

| 模式 | 状态 | 行为 |
|------|------|------|
| `simulation` | 当前默认 | 无 ONNX 模型时返回空检测；有效输入输出 `no_bird`，质量失败输出 `quality_failed` |
| `real_dl` | blocked | 需要真实 ONNX、onnxruntime session、class map 和输出契约验收 |
| `traditional_fallback` | 未启用 | 当前没有传统视觉兜底 |
| `standalone_simulation` | demo only | 仅用于 standalone `/api/simulator/*`，与真实 `/api/detect` 和 `plugin.infer()` 隔离 |

```python
from plugins.bird_monitoring.plugin import BirdMonitoringPlugin

plugin = BirdMonitoringPlugin.create_standalone()
health = plugin.healthcheck()
print(health.details["runtime_mode"])        # simulation
print(health.details["real_model_loaded"])   # False
```

## 2. 生产主链

```text
plugin.py -> detector.py::BirdDetector -> _simulate_detection() / real ONNX session
```

- `BirdDetector` 是唯一生产检测器入口。
- `detector.py::BirdDetectorEnhanced` 为 legacy/experimental，不再由 plugin 自动加载。
- `experimental/advanced_bird_detector.py` 含随机检测生成，只能作为 experimental/demo，不得进入生产推理主链。
- 顶层 `advanced_bird_detector.py` 仅是兼容 shim，用于保留旧 import 路径。
- 驱离能力仅输出 `deterrent_suggestion`；`trigger_deterrent()` 和 detector 内驱离控制均不执行硬件动作。

## 3. 入口 / 目录分层

| 路径 | 定位 |
|------|------|
| `plugin.py` | SDK 生产入口 |
| `detector.py::BirdDetector` | 唯一可信生产算法入口 |
| `standalone/app.py` | standalone UI runner；挂载真实 `/api/detect` 和隔离 `/api/simulator/*` |
| `run_standalone.py` / `__main__.py` | 兼容启动入口，保留外部调用语义 |
| `main.py` | train / infer CLI 工具，不是 standalone server |
| `demo/run_demo.py` | 本地演示脚本，可使用随机图像，不作为生产证据 |
| `standalone/bird_simulator.py` | UI 演示仿真器，所有输出标记 `standalone_simulation` |
| `experimental/` | 概念验证与 legacy 实验代码 |
| `docs/` / `prompts/` | 后续升级说明与 agent prompt 槽位 |

## 4. 输入 / 输出契约

### 输入
- `frame`: `np.ndarray` BGR 图像
- `rois`: `List[ROI]`
- `context`: `PluginContext`

### 输出 labels

| label | 含义 |
|-------|------|
| `no_bird` | 未检测到鸟；当前 simulation 的正常结果 |
| `bird_safe` / `bird_warning` / `bird_danger` / `bird_critical` | 真实检测链形成后按风险评分输出 |
| `review_required` | 检测或物种置信度不足，需要人工复核 |
| `unknown_bird` | 检测到鸟但物种映射未知 |
| `quality_failed` | 输入质量不达标，阻断推理 |
| `error` | 插件未初始化或运行异常 |

每条结果 metadata 至少包含：
- `runtime_mode`
- `input_quality`
- `training_placeholders`

检测型结果额外包含：
- `species_confidence`
- `review_required` / `review_reason`
- `risk_score` / `risk_level`
- `deterrent_suggestion`

## 5. 告警契约

| label | AlarmLevel |
|-------|------------|
| `review_required` | WARNING |
| `unknown_bird` | WARNING |
| `quality_failed` | WARNING |
| `bird_danger` | ERROR |
| `bird_critical` | CRITICAL |
| `bird_safe` / `bird_warning` / `no_bird` | 不生成告警 |

## 6. Standalone 启动

```bash
python -m plugins.bird_monitoring
# 或
python plugins/bird_monitoring/run_standalone.py
```

默认地址：`http://localhost:8092`

当前 standalone 调用真实插件 runner：
- 真实检测：SDK runner 的 `/api/detect`，runtime 由 `plugin.infer()` / detector 决定。
- 演示仿真：`/api/simulator/*`，runtime 固定为 `standalone_simulation`，不得作为真实检测结果。

## 7. 测试

```bash
python3 -m pytest plugins/bird_monitoring/tests/ -q
```

当前基线：**58 passed**。

覆盖维度：
- 生命周期与 standalone 创建
- runtime truth healthcheck 字段
- simulation 空检测不伪造鸟
- 输入质量失败路径
- 风险评分纯逻辑
- 结果 metadata 与训练占位
- 告警分级
- 驱离建议
- standalone 仿真隔离
- 目录职责契约

未覆盖：
- 真实图片 replay
- 真实 ONNX 推理
- real_dl preflight
- 精度 regression

## 8. 能力分层

**verified**
- sdk_lifecycle_and_standalone
- simulation_empty_detection
- input_quality_fail_path
- runtime_truth_basic
- review_required_alarm_contract
- training_placeholder_metadata
- deterrent_suggestion_output
- standalone_simulation_demo

**experimental**
- risk_scoring_contract
- species_identification_mapping
- legacy `BirdDetectorEnhanced`
- `experimental/advanced_bird_detector.py`

**blocked**
- real_dl_onnx_inference
- replay_regression
- wingspan_estimation_from_bbox
- behavior_classification
- deterrent_hardware_control

## 9. 已知边界

- 当前无真实模型，不能宣称具备真实鸟类检测能力。
- simulation 只允许空检测，不允许生成合成鸟 bbox。
- `standalone_simulation` 可以生成演示鸟类结果，但只能走 `/api/simulator/*`，不得进入生产主链。
- 物种识别、风险评估、驱离建议目前是契约完备和逻辑占位，不代表已有真实视觉精度。
- 外部鸟类数据库和相机标定尚未完成。
