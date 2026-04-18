# transformer_inspection

`transformer_inspection` 是一个面向主变巡视场景的 enhanced-detector 插件。当前主链路是 `plugin.py -> detector_enhanced.py`，负责外观缺陷识别、呼吸器硅胶状态识别、油位计读数和可选热像阈值告警；`defect_detector.py` 与 `thermal_analyzer.py` 仍是旁路模块，没有接入 `Plugin.infer()` 主链路。

## Verified Entry Points

- `python3 demo/run_demo.py`
- `python3 -m pytest tests -q`
- `./scripts/run_targeted_tests.sh <module>`
- `./scripts/run_regression_tests.sh`
- `./scripts/run_quality_gate.sh`

## Runtime Contract

- `Plugin.create_standalone()` 从 `manifest.json` 和 `configs/default.yaml` 启动插件。
- `infer(frame, rois, context)` 接受 BGR `numpy.ndarray`、ROI 列表和 `PluginContext`。
- 为了 demo 和 smoke test，`context=None` 会自动填一个 standalone 上下文；平台运行时仍应传入真实 `PluginContext`。
- 当前 ROI 语义主要依赖 `roi.name` / `roi.id` / `roi.metadata`：
  - 包含 `breather` / `silica` 的 ROI 走 `recognize_silica_gel()`
  - 包含 `oil_level` / `meter` / `gauge` 的 ROI 走 `detect_oil_level()`
  - 其他 ROI 默认只走 `detect_defects()`
- 热成像路径只在 `thermal.enabled=true` 且 `context.metadata["thermal_frame"]` 提供热像时生效。

## Output and Degradation

- 缺陷结果由 detector 的 `Detection` dataclass 适配成 `RecognitionResult`。
- 硅胶状态只输出：
  - `silica_gel_normal`
  - `silica_gel_abnormal`
  - `silica_gel_unknown`
- 油位计结果输出 `oil_level_reading`，`value` 为 `level_ratio`，状态字符串放在 metadata。
- 当前阀门状态标签仍保留在历史配置里，但没有接入运行时主链路，不应写成已验证能力。
- 若可选深度模型不可用，detector 回退到 OpenCV/颜色规则链路。

## Dependencies and Configuration

- 基础依赖：`numpy`、`opencv-python`、`pyyaml`、`darkbreaker-sdk`
- 可选依赖：
  - `ai_models.deep_learning.yolov8_vit`
  - `ai_models.deep_learning.segformer`
  - `ai_models.deep_learning.gabor_texture`
  - 外部 `model_registry`
- 当前 detector 与 plugin 都读取 `configs/default.yaml` 的 `inference.*` 阈值。

## HF Governance Assets

- `.agent_skills/00~08`
- `.claude/commands/implement.md`
- `.claude/commands/repair.md`
- `.claude/commands/audit.md`
- `scripts/run_targeted_tests.sh`
- `scripts/run_regression_tests.sh`
- `scripts/run_quality_gate.sh`
- `scripts/collect_root_cause.sh`

## Current Limits

- 还没有真实图像回放集；`run_regression_tests.sh` 目前是“全量 pytest + demo smoke + 可选静态扫描”。
- `scripts/benchmark.py` 是手动性能检查入口，不在默认 HF 门禁里。
- `defect_detector.py`、`thermal_analyzer.py` 不是当前主链路事实源。
- 阀门状态识别还停留在历史口径，当前不要把它作为已验证能力引用。
