# switch_inspection

`switch_inspection` 是一个面向断路器、隔离开关、接地开关巡视场景的 enhanced-detector 插件。当前主链路是 `plugin.py -> detector_enhanced.py`，负责状态识别、清晰度门禁和最小五防逻辑校验；`switch_consistency.py` 作为独立一致性校核模块存在并有测试，但默认没有接入 `Plugin.infer()` 主链路。

## Verified Entry Points

- `python3 demo/run_demo.py`
- `python3 -m pytest tests -q`
- `./scripts/run_targeted_tests.sh <module>`
- `./scripts/run_regression_tests.sh`
- `./scripts/run_quality_gate.sh`

## Runtime Contract

- `Plugin.create_standalone()` 从 `manifest.json` 和 `configs/default.yaml` 启动插件。
- `infer(frame, rois, context)` 接受 BGR `numpy.ndarray`、ROI 列表和 `PluginContext`。
- 为了 demo 和 smoke test，`context=None` 时会自动填一个 standalone 上下文；平台运行时仍应传入真实 `PluginContext`。
- 识别 ROI 主要看 `roi.name` 或 `roi.roi_type` 的字符串值，当前支持：
  - `breaker_indicator`
  - `isolator_indicator`
  - `grounding_indicator`
  - `breaker_linkage`
  - `isolator_linkage`
  - `grounding_handle`
  - `gauge_pressure`
  - `gauge_density`
  - `clarity_anchor`

## Output and Degradation

- 清晰度低于 `image_quality.min_clarity_score` 时输出 `clarity_low`，并给出 `suggested_action=REFOCUS_OR_RECAPTURE`。
- 状态识别通过 detector 适配层输出 `state/confidence/evidence`，再由 `plugin.py` 组装成 `RecognitionResult`。
- `logic_validation` 根据 `configs/default.yaml` 中的最小五防规则输出 `logic_error` 或 `logic_warning`。
- `gauge_reading.enabled=false` 时不会产生 SF6 表计结果，也不会伪造失败告警。

## Dependencies and Configuration

- 基础依赖：`numpy`、`opencv-python`、`pyyaml`、`darkbreaker-sdk`
- 可选依赖：
  - `ai_models.deep_learning.yolov8_vit`
  - 外部 `model_registry`
  - 外部 `fusion_engine`
- 当前默认配置没有绑定真实深度模型路径，因此主链路通常运行在 fallback 模式：颜色/OCR/角度证据优先。

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

- 还没有真实图像回放集；`run_regression_tests.sh` 目前是“全量 pytest + demo smoke + 可选静态扫描”，不是现场样本级回放。
- `switch_consistency.py` 已测试，但尚未并入 `Plugin.infer()` 主链路。
- 五防规则只覆盖 `configs/default.yaml` 中声明的 3 条最小规则，不代表完整运维规则库。
