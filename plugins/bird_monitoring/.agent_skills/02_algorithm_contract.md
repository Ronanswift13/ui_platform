# 02_algorithm_contract

> 最后更新：2026-04-17（real_dl preflight + 质量门三态 + experimental 隔离）

## 1. 生产主链

```text
plugin.py
  -> detector.py::BirdDetector
    -> real ONNX session  → _preflight_onnx() → real_dl  | preflight fail → simulation
    -> _simulate_detection()（默认；必须返回 []）
```

- `plugin.py` 不再自动加载 `BirdDetectorEnhanced`。
- `experimental/advanced_bird_detector.py` 为 experimental/demo；含随机检测生成，禁止进入生产主链。
- 顶层 `advanced_bird_detector.py` 仅为旧 import 兼容 shim。
- `detector.py::BirdDetectorEnhanced` 仅是 shim 类；除非环境变量 `BIRD_ENABLE_ENHANCED_DETECTOR=1`，否则 `__new__` 抛 `RuntimeError`，并将真实实现委派给 `experimental/enhanced_detector.py::EnhancedBirdDetector`。
- `_preflight_onnx()` 必须不抛异常；任何失败写入 `report.issues` 后 `report.passed=False`，`runtime_mode` 自动回落 `simulation`，`healthcheck.details.preflight` 完整暴露。

## 2. 输入契约

```python
plugin.infer(frame: np.ndarray, rois: list[ROI], context: PluginContext)
```

- `frame`: H×W×3 BGR `uint8`。
- `rois`: 平台传入 ROI；空列表返回 `[]`。
- `context`: `PluginContext`。

质量门为三态：

| status | 触发 | 行为 |
|---|---|---|
| `pass` | 所有指标在 hard / soft 阈值之上 | 正常推理 |
| `soft_fail` | 任一指标低于 `soft_*_threshold` 但未触发 hard 条件 | 推理继续，但每条 detection 强制 `review_required=True`，原因 `质量软失败` |
| `hard_fail` | `frame is None` / 空数组 / `min_dimension` 不足 / detector hard issues | 阻断推理，输出 `label="quality_failed"` |

- `quality.metadata` 必须暴露 `status / hard_issues / soft_issues / clarity_score / brightness_score / overall_score`。
- soft_fail 不得伪装成 pass；hard_fail 不得被吞为 no_bird。
- `is_valid` 字段保留（向后兼容），等价 `status != "hard_fail"`。

## 3. 输出 labels

| label | 含义 | 当前状态 | 告警 |
|-------|------|----------|------|
| `no_bird` | 未检测到鸟 | verified；simulation 正常输出 | — |
| `bird_safe` | 检测到鸟，风险可接受 | experimental；需真实检测 | — |
| `bird_warning` | 警戒风险 | experimental；需真实检测 | — |
| `bird_danger` | 危险风险 | experimental；需真实检测 | ERROR |
| `bird_critical` | 紧急风险 | experimental；需真实检测 | CRITICAL |
| `review_required` | 检测/物种置信度不足 | verified contract | WARNING |
| `unknown_bird` | 物种映射未知 | verified contract | WARNING |
| `quality_failed` | 输入质量不达标 | verified | WARNING |
| `error` | 未初始化或异常 | verified | — |

新增 label 必须同步：
- `tests/test_standalone.py` 白名单
- `postprocess()` 告警映射
- README / PROJECT_CARD

## 4. metadata 必填字段

每条 `RecognitionResult.metadata` 必含：
- `runtime_mode`
- `input_quality`
- `training_placeholders`

`training_placeholders` 至少包含：
- `hard_negative_candidate`
- `hard_positive_candidate`
- `suggested_label_for_dataset`
- `annotation_status`
- `model_placeholder`

检测型结果额外包含：
- `species_id` / `species_name` / `species_confidence`
- `risk_score` / `risk_level`
- `review_required` / `review_reason`
- `deterrent_suggestion`

## 5. runtime truth

允许值：
- `simulation`
- `real_dl`
- `traditional_fallback`（当前不启用）
- `standalone_simulation`（仅用于 standalone `/api/simulator/*` 演示路由）

`healthcheck.details` 必须暴露：
- `runtime_mode`
- `model_path_configured`
- `model_path_resolved`
- `model_file_exists`
- `real_model_loaded`
- `onnx_session_ready`
- `fallback_enabled`
- `simulation_enabled`
- `model_load_error`
- `preflight`（dict：`{performed, passed, checks, issues}`，simulation 默认 `performed=False`）

当前事实：
- `runtime_mode=simulation`
- `model_file_exists=false`
- `real_model_loaded=false`
- `onnx_session_ready=false`
- `fallback_enabled=false`
- `simulation_enabled=true`

## 6. 风险 / 物种 / 驱离契约边界

### 物种识别
- `SPECIES_LABEL_MAP` 命中内置 8 种时返回 species_id。
- 未命中必须返回 `unknown_bird`，禁止默认成 `sparrow` 或其他具体鸟种。
- 真实物种分类精度未验证，能力归类为 experimental。

### 风险评估
- 风险公式使用 YAML 权重：距离、物种、翼展、行为。
- 距离当前来自检测器字段或默认值，未做相机标定。
- 翼展和行为只参与规则框架，真实视觉能力 blocked。

### 驱离建议
- 只允许输出 `deterrent_suggestion`。
- `trigger_deterrent()` 必须返回 `False`，不得访问硬件。
- detector 层不得执行 HTTP / Modbus / serial / GPIO / MQTT 控制。

## 7. 降级策略

| 场景 | 行为 | label |
|------|------|-------|
| 无 ONNX 模型 | simulation 空检测 | `no_bird` |
| ONNX 存在但 preflight 失败 | 写 `preflight.issues`，回落 simulation | `no_bird` |
| 质量门 hard_fail（None / 空 / 过小 / 过暗 hard / 模糊 hard） | 阻断 | `quality_failed` |
| 质量门 soft_fail | 推理继续 + 强制 review | `bird_*` 或 `no_bird` 但带 `review_required=True` |
| 低检测置信度 | 旁路复核 | `review_required` |
| 物种未知 | 保守输出未知 | `unknown_bird` |
| 未初始化 infer | 结构化错误 | `error` |
| experimental enhanced detector 未 opt-in | shim 抛 RuntimeError，不接入 | 无影响 |

## 8. 禁止行为

1. simulation 不得生成合成鸟 bbox。
2. experimental/demo 代码不得进入生产主链；`BirdDetectorEnhanced` shim 不得绕过 `BIRD_ENABLE_ENHANCED_DETECTOR` 检查。
3. 未完成能力不得标记 verified。
4. 驱离硬件控制不得出现在生产路径。
5. `standalone_simulation` 不得和真实 `/api/detect` 混用。
6. 删除 metadata 字段属于 breaking change，只能新增字段或测试。
7. `_preflight_onnx()` 失败时不得静默 promote 到 real_dl；必须暴露 issues。
8. 合成 fixture（`tests/regression/build_synthetic_fixtures.py`）只允许覆盖 simulation 可复现的 label，禁止用于伪造 `bird_*` 精度基线。
