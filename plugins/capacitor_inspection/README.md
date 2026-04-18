# capacitor_inspection

`capacitor_inspection` 是一个面向电容器组巡视场景的 enhanced-detector 插件。当前主链路是 `plugin.py -> detector_enhanced.py`，负责结构完整性检测和区域入侵检测；`plugin.py` 只做 SDK 适配和 ROI 路由，算法与跟踪保持在 detector 层。

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
- 当前 ROI 语义主要依赖 `roi.roi_type` 和 `roi.name`：
  - `ROIType.DEFECT` 或名称包含 `capacitor_bank/capacitor_unit/fuse/connecting_bar/insulator` 走结构缺陷链路
  - `ROIType.INTRUSION` 或名称包含 `fence/warning_zone/restricted_zone` 走入侵检测链路

## Output and Degradation

- 结构缺陷结果由 detector 的 `CapacitorDetection` dataclass 适配成 `RecognitionResult`。
- 入侵结果由 detector 的 `IntrusionDetection` dataclass 适配成 `RecognitionResult`，metadata 会保留 `zone/track_id/duration_sec/confirmed`。
- 若可选深度模型不可用，结构检测回退到传统 CV；入侵检测在当前默认配置下缺少正向传统样本能力，常见结果是空列表。
- 当前没有真实 intrusion replay 样本集，因此“入侵正检效果”仍需人工补足事实源。

## Dependencies and Configuration

- 基础依赖：`numpy`、`opencv-python`、`pyyaml`、`darkbreaker-sdk`
- 可选依赖：
  - `ai_models.deep_learning.yolov8_vit`
  - `ai_models.deep_learning.thermal_visible_registration`
  - `ai_models.deep_learning.yolov8_obb`
  - 外部 `model_registry`
- `plugin.py` 与 `detector_enhanced.py` 当前都读取 `configs/default.yaml` 的 `inference.*` 阈值。

## Governance Assets

- `.agent_skills/00~08`
- `.claude/commands/implement.md`
- `.claude/commands/repair.md`
- `.claude/commands/audit.md`
- `scripts/run_targeted_tests.sh`
- `scripts/run_regression_tests.sh`
- `scripts/run_quality_gate.sh`
- `scripts/collect_root_cause.sh`

## Current Limits

- 还没有真实图像 replay 数据集；`run_regression_tests.sh` 目前是“全量 pytest + demo smoke + 可选静态扫描”。
- Intrusion positive cases 目前主要由契约测试覆盖，不是现场样本回放。
- 传统 CV 路径里仍有一批阈值是硬编码技术债，当前已被 quality audit 明确标注。
