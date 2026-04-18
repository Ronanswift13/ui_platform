# PROJECT_CARD: busbar_inspection

## 1. 项目名称
母线自主巡视插件（busbar_inspection）

## 2. 项目类型
plugin_update（已有插件规则化治理 + 工程骨架补齐）

## 3. 输入源
- 巡检图像帧：`BGR np.ndarray`，单帧推理，支持 1080p/4K。
- ROI 列表：平台传入 `ROI`（归一化 `x/y/width/height`）。
- 插件上下文：`task_id/site_id/device_id/component_id/timestamp`。
- 配置文件：`configs/default.yaml`。

## 4. 输出目标
- 缺陷识别结果：`RecognitionResult`（`pin_missing/crack/foreign_object/quality_failed`）。
- 告警结果：`Alarm`（按标签映射到 `ERROR/WARNING`）。
- 质量解释：`failure_reason` + `metadata.quality` + `metadata.suggested_action`。
- 质量门禁三态：`quality_gate_status=pass|soft_fail|hard_fail`。
- 运行模式：`runtime_mode=real_dl|traditional_fallback|quality_blocked|standalone_simulation`。
- 复核状态：`review_status=confirmed|review_required|blocked`。
- 变焦建议：`metadata.suggested_zoom`、`metadata.suggested_action`。
- 健康状态：`healthcheck()` 输出计数、真实 `runtime_mode`、模型路径与回退状态。

## 5. 关键约束
### 工程约束
- 必须遵循 `darkbreaker_sdk.interfaces.BasePlugin` 契约。
- `plugin.py` 仅做 SDK 适配，不承载核心检测算法。
- `detector_enhanced.py` 不得依赖 `darkbreaker_sdk`。
- 所有阈值必须来自 `configs/default.yaml` 映射。

### 业务约束
- 单帧多 ROI 独立处理，单 ROI 异常不得拖垮整帧。
- 质量门禁失败必须可解释（原因码 + 建议动作）。
- 质量门禁必须区分 `PASS / SOFT_FAIL / HARD_FAIL`，且 `quality_gate_status` 不能拿 `runtime_mode` 偷代。
- `SOFT_FAIL` 不得提前阻断；`HARD_FAIL` 才允许直接输出 `quality_failed`。
- 对目标过小场景必须给出变焦建议，不允许仅返回失败。
- 当前 runtime supported labels 冻结为 `pin_missing/crack/foreign_object/quality_failed`。
- `broken_part` 与 `fitting_loose`（兼容别名 `loose_fitting`）仅可标记为 `blocked`，不得宣称已稳定支持。
- simulator 必须作为真实 `check_quality_gate()` 的校准器，场景期望要能和真实门禁结果逐项对齐。

### 安全约束
- 不访问外部网络。
- 不持久化原始图像到未授权目录。
- 日志中不得输出敏感标识符原文拼接。

## 6. 验收标准（本轮治理）
- `plugins/busbar_inspection/tests` 全量通过。
- tri-state 契约测试通过：`PASS / SOFT_FAIL / HARD_FAIL` 行为可观测、可区分。
- simulator 对齐测试通过：`quality_blur/quality_occlude/quality_overexpose/rainy_inspection/normal_clear` 与真实 `check_quality_gate()` 一致。
- real_dl preflight 测试通过：缺模型、invalid ONNX、label contract mismatch 均能 fail fast 并保持 `traditional_fallback`。
- 如存在真实可打开且契约兼容的 ONNX fixture，必须验证 `runtime_mode=real_dl`、`onnx_session_ready=true`、`real_model_loaded=true`。
- `healthcheck()` 必须同时暴露 runtime truth 字段与 preflight 字段，不得只留日志。
- P0/P1 runtime truth 与标签冻结测试不得回退。
- `.agent_skills/04_quality_audit.md`、`.agent_skills/07_learning_log.md` 与可复用 reference 已同步更新。

## 7. 禁止事项
- 禁止修改 SDK 接口签名。
- 禁止新增硬编码业务阈值到推理主路径。
- 禁止使用 `except: pass`。
- 禁止在生产路径新增 `print()`。
- 禁止删除既有降级路径（深度学习 -> 传统方法）。

## 8. 已知参考物
- `README.md`：业务目标与输出字段说明。
- `plugin.py`：SDK 适配层。
- `detector_enhanced.py`：检测与质量门禁核心实现。
- `configs/default.yaml`：阈值与运行参数。
- `tests/test_standalone.py`：当前可运行测试基线。

## 9. 当前任务
- 保持 P0/P1 的 runtime truth 与标签冻结成果，不回退。
- 以 tri-state 收口质量门禁：`PASS / SOFT_FAIL / HARD_FAIL`。
- 建立 `quality_failed`、`review_required`、`blocked` 的可解释输出路径。
- 让 simulator 成为真实质量门禁的校准器，而不是只做视觉演示。
- 在不扩标签、不改 SDK 的前提下，为 real_dl 建立 ONNX preflight 门禁，验证”可找到资产 / 可建立 session / 类别兼容 / 输出兼容 / 可失败回退”。

## 10. Phase 2 能力三分表 (2026-04-16 审计)

| 能力 | 分类 | 证据 |
|------|------|------|
| defect_detection_pin_missing | **verified** | 标签契约测试+tri-state 门禁测试通过；reason_code 映射可审计 |
| defect_detection_crack | **verified** | 同上；traditional_crack_validation 路径有测试 |
| defect_detection_foreign_object | **verified** | 同上；标签冻结在 runtime supported labels 内 |
| quality_gate_tristate | **verified** | PASS/SOFT_FAIL/HARD_FAIL 行为测试全通过 |
| small_object_enhancement | experimental | config 支持 small_object_enhancement 开关，代码存在，无回归数据 |
| tile_detection_4k | experimental | tile_detection/tile_size/tile_overlap 配置存在，代码路径存在 |
| noise_filtering | experimental | filter_classes+min_static_frames 配置，代码存在但无场景验证 |
| broken_part | **blocked** | 标签已冻结为 blocked，不得宣称已支持 |
| fitting_loose | **blocked** | 标签已冻结为 blocked（含兼容别名 loose_fitting） |
| real_dl_onnx_inference | **blocked** | preflight 框架已建立但无真实 ONNX 模型文件 |

**runtime_mode_support**: real_dl=❌ traditional_fallback=✅ simulation=✅
**测试状态**: 全量通过 (0 failures)；fixtures/regression 仅 README 占位
**关键差距**: 本批插件中治理最成熟，但 real_dl 仍为 blocked
