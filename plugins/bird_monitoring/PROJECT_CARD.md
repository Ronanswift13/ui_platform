# PROJECT_CARD: bird_monitoring

## 1. 项目名称
鸟类监控插件（bird_monitoring）

## 2. 当前结论（2026-04-17 工程契约对齐）
**对齐 busbar_inspection 工程基线**（除真实模型/真实图片 fixture 外）。

已达到的部分：SDK 接口、空检测 simulation 降级、质量失败结构化、runtime truth 基础字段、复核/告警契约、训练占位元数据、standalone 生命周期测试、standalone 仿真演示路由与生产 detect 路由隔离、仿真页 runtime_mode 显式标记 standalone_simulation、仿真场景覆盖 8 个 label 路径、**real_dl ONNX preflight 契约**（路径/session/shape/输出通道/类别映射）、**质量门三态 `pass / soft_fail / hard_fail`**（soft_fail 触发 review_required，hard_fail 阻断为 quality_failed）、**`BirdDetectorEnhanced` 完全迁出生产 detector**（仅在 `BIRD_ENABLE_ENHANCED_DETECTOR=1` 下经 shim 委派 `experimental/enhanced_detector.py`）、**最小合成 replay baseline**（覆盖 simulation 可复现的 `no_bird_clear` / `quality_dark`）、**docs/ 真实升级决策 + prompts/ 三套 agent 升级 prompt**、**standalone UI 质量门三态徽章 + 训练占位 JSON 下载**。

未达到的部分：无真实 ONNX 模型、无真实 bird 图片 fixture（`bird_safe / unknown / review` 的 replay 仍 blocked），物种识别/翼展/行为分类仍 experimental 或 blocked。

## 3. 输入源
- 摄像头帧：`BGR np.ndarray`，单帧推理。
- ROI 列表：`List[ROI]`（`darkbreaker_sdk.schemas.ROI`）。
- 配置文件：`configs/default.yaml`。
- 鸟类数据库：内置 8 种 dict；外部 YAML 路径可加载，但尚无真实数据资产验收。

## 4. 输出目标
- 统一结构化结果：`RecognitionResult`，label ∈ `{no_bird, bird_safe, bird_warning, bird_danger, bird_critical, review_required, unknown_bird, quality_failed, error}`。
- 当前生产主链在无模型时只输出 `no_bird` 或失败类结果，不生成合成鸟。
- 风险评估：距离 + 物种 + 翼展 + 行为权重框架；距离为启发式，翼展/行为未落地。
- 驱离：仅输出 `deterrent_suggestion` JSON；硬件控制为 blocked。
- 告警：`review_required / unknown_bird / quality_failed → WARNING`，`bird_danger → ERROR`，`bird_critical → CRITICAL`。

## 5. 生产主链与隔离边界

| 文件 | 当前定位 | 状态 |
|------|----------|------|
| `plugin.py` | SDK 适配层、质量门三态、风险评分编排、告警生成 | production |
| `detector.py::BirdDetector` | 唯一生产检测器入口；ONNX 存在时 real_dl + preflight，否则 simulation 空检测 | production |
| `detector.py::BirdDetectorEnhanced` | shim 类，仅 `BIRD_ENABLE_ENHANCED_DETECTOR=1` 时委派 experimental | shim only |
| `experimental/enhanced_detector.py` | 旧增强检测器实现（3D 距离/轨迹/威胁评估） | experimental opt-in |
| `experimental/advanced_bird_detector.py` | PyTorch 概念验证，含随机检测生成 | experimental/demo only |
| `advanced_bird_detector.py` | 顶层兼容 shim，旧 import 跳转到 experimental | compatibility only |
| `standalone/app.py` | 真实插件 runner + 挂载 `/api/simulator/*` 隔离路由 | standalone |
| `run_standalone.py` / `__main__.py` | 外部兼容启动入口，委托 standalone runner | compatibility launcher |
| `standalone/bird_simulator.py` | standalone 演示仿真器；输出全部带 `runtime_mode=standalone_simulation` | standalone-only |
| `standalone/templates/bird_monitoring.html` | 仿真+真实双模式 UI，runtime_mode + 质量门三态徽章，训练占位 JSON 下载 | standalone-only |
| `standalone/static/*` | standalone UI 静态资源目录，目前为占位说明 | standalone assets |
| `docs/` | `real_dl_upgrade_decision.md` + `quality_gate_tristate.md` 决策记录 | knowledge |
| `prompts/` | `real_dl_preflight_prompt.md` / `fixture_collection_prompt.md` / `ui_hardening_prompt.md` | agent prompts |
| `tests/regression/build_synthetic_fixtures.py` | 合成 `.npy` fixture 生成器，仅覆盖 simulation 可复现的两个 label | tooling |
| `tests/replay/expected_results.json` | replay 预期表，bird_* 槽位仍 `planned_blocked_by_model` | partial replay |

## 6. runtime truth
- `runtime_mode=simulation`：默认现状；模型文件不存在，`_simulate_detection()` 返回 `[]`。
- `runtime_mode=real_dl`：blocked；需 ONNX 文件、onnxruntime session、类别契约与输出格式验收。
- `traditional_fallback=false`：当前无传统视觉检测兜底。
- `runtime_mode=standalone_simulation`：**已实现**；仅由 `standalone/bird_simulator.py` 通过 `/api/simulator/*` 路由产生，不进入 `plugin.infer()`。每条仿真 detection / alarm / payload 顶层均显式打标。
- `healthcheck.details` 已暴露 `model_path_configured / model_file_exists / real_model_loaded / onnx_session_ready / fallback_enabled / simulation_enabled`。

## 7. 能力三分表

| 能力 | 分类 | 证据 / 边界 |
|------|------|-------------|
| sdk_lifecycle_and_standalone | **verified** | create_standalone / healthcheck / cleanup / code_hash 测试 |
| simulation_empty_detection | **verified** | 生产主链固定 `BirdDetector`，无模型只返回 `no_bird` |
| input_quality_tristate | **verified** | `pass / soft_fail / hard_fail` 三态 + soft 触发 review_required，hard 阻断 quality_failed（`tests/test_quality_tristate.py`） |
| real_dl_preflight_contract | **verified contract** | `_preflight_onnx()` 校验 input/output rank、shape、output 通道 = 4+len(CLASSES)、CLASSES 数量；失败回落 simulation 并写 `healthcheck.preflight`（`tests/test_real_dl_preflight.py`） |
| enhanced_detector_experimental_only | **verified** | `BirdDetectorEnhanced` 仅在 `BIRD_ENABLE_ENHANCED_DETECTOR=1` 时通过 shim 委派 `experimental/enhanced_detector.py`（`tests/test_directory_contract.py`） |
| runtime_truth_basic | **verified** | healthcheck 暴露模型路径、文件存在性、session 状态、preflight 报告 |
| review_required_alarm_contract | **verified** | 低置信度/unknown/质量软失败 可进入复核，postprocess WARNING |
| training_placeholder_metadata | **verified** | 正常、质量失败、错误结果均含 `training_placeholders` |
| deterrent_suggestion_output | **verified** | 仅建议 JSON，`trigger_deterrent()` 恒不触发硬件 |
| standalone_simulation_demo | **verified demo** | `/api/simulator/*` 与生产 `/api/detect` 隔离，显式 `standalone_simulation` |
| standalone_ui_quality_visualization | **verified** | 模板含 `q-status` 三态徽章 + `downloadTrainingPlaceholders` JSON 下载 |
| synthetic_replay_baseline | **verified (partial)** | `no_bird_clear` / `quality_dark` 合成 `.npy` 精度回归通过 |
| risk_scoring_contract | **experimental** | 纯逻辑测试覆盖，但依赖启发式距离和占位行为/翼展 |
| species_identification_mapping | **experimental** | 类名映射可用，无真实分类模型/图片验证 |
| real_bird_image_replay | **blocked** | 无真实 bird 图片，`bird_*` 样本槽 `planned_blocked_by_model` |
| real_dl_onnx_inference | **blocked** | 无 ONNX 模型；preflight 契约已就位但需真实模型触发 |
| wingspan_estimation_from_bbox | **blocked** | 需模型输出 + 相机标定 |
| behavior_classification | **blocked** | 需时序跟踪/行为模型 |
| deterrent_hardware_control | **blocked** | 明确不属于本插件当前职责 |

## 8. 测试状态
- 插件内：`90 passed`（2026-04-17 scripts/ 入口 + 覆盖率门补齐）。
- 跨插件回归：`424 passed, 8 skipped`，无退化。
- 覆盖率：62%（fail_under=60，由 `.coveragerc` 控制；experimental / demo / scripts / tests omit）。
- 新增覆盖：`tests/test_scripts_contract.py`（13 例：scripts 存在 + shebang + +x / `run_targeted_tests.sh` 未知 layer exit 2 / `check_real_model.py` 缺失模型 exit 2 / `validate_fixture.py` planned 槽位报 slot_pending exit 1 / 08_task_routing.md 注册检查）。
- 未覆盖：真实 bird 图片 replay（`bird_safe / unknown / review` 仍 blocked）、真实 ONNX 端到端、相机标定后的真实距离。

## 9. 下一阶段优先级
1. 交付真实 YOLOv8 ONNX 模型 → `_preflight_onnx()` 通过 → runtime_mode 切到 `real_dl`。
2. 采集真实 bird fixture（按 `prompts/fixture_collection_prompt.md` 槽位）解锁 `bird_*` replay。
3. 在 `real_dl` 通过后，扩展 replay regression 覆盖 `bird_safe / review_required / unknown_bird`。
4. 在 docs/ 中追加首次真实模型上线后的精度基线决策记录。
5. 评估 `experimental/enhanced_detector.py` 是否需要在 real_dl 通过后正式合并回 detector 主链。
