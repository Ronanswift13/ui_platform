# 03_test_strategy

> 最后更新：2026-04-17（scripts/ 入口 + 覆盖率门）

## 1. 固定母版规则

1. 每个硬约束至少对应 1 个自动化测试。
2. 每个 bug 修复必须新增防回归测试。
3. 测试分层执行：L0 快速、L1 集成、L2 regression（合成 fixture 已就位）。
4. 测试脚本必须返回明确退出码。
5. 每个 scripts/ 入口必须在 `tests/test_scripts_contract.py` 注册并校验退出码语义。

## 2. 当前测试现状（已就位）

- **90 个测试**，9 个文件，约 1.5s 执行时间。
- 插件内全部绿灯；跨插件回归 `424 passed, 8 skipped`，无退化。
- 覆盖率 62%（`.coveragerc::fail_under=60`）。
- 入口：`scripts/run_targeted_tests.sh all` ；分层 `l0 / l1 / l2`。

### 测试文件矩阵

| 文件 | 数量 | 维度 | 关键断言 |
|---|---:|---|---|
| test_standalone.py | 23 | 生命周期 + runtime truth + standalone simulator + 模板契约 | create_standalone / 生产主链使用 BirdDetector / healthcheck real_model/session 字段 / simulation 不造鸟 / simulator 标记 `standalone_simulation` / 未初始化 infer 返回 `error/9000` / 模板含 `q-status` 三态徽章 + `downloadTrainingPlaceholders` |
| test_quality_assessment.py | 9 | 输入质量（基础） | 合法帧 pass / None 失败 / 空帧失败 / 极小帧质量失败 / 暗帧质量失败 / detector 层质量评估 / runtime_mode |
| test_quality_tristate.py | 6 | 质量门三态 | pass→正常推理 / 极暗→hard_fail 输出 quality_failed / 极小→hard_fail / soft_fail 强制 review_required / soft_fail 不输出 quality_failed / 告警等级映射 |
| test_real_dl_preflight.py | 7 | ONNX preflight 契约 | _FakeSession 合法 shape 通过 / input_h_mismatch 标记 issue / output channel 与 4+len(CLASSES) 不匹配 / dynamic dim "batch" 忽略 / preflight 失败 runtime_mode 回落 simulation / healthcheck.preflight 暴露 |
| test_risk_assessment.py | 10 | 风险 + 物种 | 安全距离→safe / 近距离老鹰→danger/critical / 未知鸟近→danger / 未知鸟远→safe / score→level 全边界 / 物种识别正/未知/空/大小写 |
| test_plugin_contract.py | 13 | 结果契约 + 告警 | 必需字段 / 失败结果 training_placeholders / runtime_mode in metadata / 多 ROI / 空 ROI / review→WARNING / unknown→WARNING / quality_failed→WARNING / danger→ERROR / safe 不告警 / 驱离建议 |
| test_replay_baseline.py | 3 | L2 合成 fixture replay | `no_bird_clear.npy` 输出 no_bird / `quality_dark.npy` 输出 quality_failed / `bird_*` 槽 `planned_blocked_by_model` 契约不变 |
| test_directory_contract.py | 6 | 目录职责 + experimental opt-in | 入口存在 / advanced shim 指向 experimental / simulator 与生产解耦 / docs-prompts-regression 占位 / **enhanced detector 必须 BIRD_ENABLE_ENHANCED_DETECTOR=1** / **enhanced 主体在 experimental/，detector.py 仅 shim** |
| test_scripts_contract.py | 13 | scripts/ 入口契约 | scripts/ 与 .coveragerc 存在 / 5 脚本 +x + shebang 合法 / `run_targeted_tests.sh bogus` exit 2 / `check_real_model.py` 缺模型 exit 2 / `validate_fixture.py` planned 槽 exit 1 + 报 slot_pending / 5 脚本必须在 08_task_routing.md 注册 |

### conftest.py 共享夹具

- `plugin_dir` — 插件根路径
- `default_config` — 最小默认配置（含 quality 段）
- `plugin_instance` — `BirdMonitoringPlugin.create_standalone(config=default_config)`
- `sample_frame` — 640×480 BGR 随机
- `tiny_frame` — 10×10（质量门必败）
- `dark_frame` — 640×480 全 5（亮度不足）
- `make_context(task_id, **kwargs)` — PluginContext 工厂
- `make_roi(roi_id, x, y, w, h)` — ROI 工厂（含 name / component_id / roi_type=DEFECT）

## 3. 分层定义（现状）

- **L0 Targeted**（纯逻辑 <2s）：test_risk_assessment、test_quality_assessment、test_quality_tristate、test_real_dl_preflight
- **L1 Integration**（走 plugin 全链路 ~1s）：test_standalone、test_plugin_contract、test_directory_contract
- **L2 Regression**（合成 fixture 已就位 / 真实图片仍 blocked）：test_replay_baseline；通过 `tests/regression/build_synthetic_fixtures.py` 产出 `no_bird_clear.npy` / `quality_dark.npy`，`bird_*` 真实样本仍受 `planned_blocked_by_model` 阻断

## 4. 测试命名约定

```
test_{module}_{scenario}_{expected_behavior}
```

示例（本项目已采用）：
- `test_safe_distance_returns_safe`
- `test_close_eagle_returns_critical_or_danger`
- `test_unknown_bird_close_returns_danger`
- `test_score_to_level_boundaries`
- `test_review_required_generates_warning`

## 5. 新增硬约束 → 测试追加规则

| 若你新增/修改 | 必须更新 |
|---|---|
| 新 label | `test_standalone.py::test_infer_returns_results` 白名单 |
| 新质量门规则 | `test_quality_assessment.py` + `test_quality_tristate.py` 三态分支 |
| 新质量门阈值（hard / soft） | `configs/default.yaml::quality` + `test_quality_tristate.py` 阈值断言 |
| 新风险权重或阈值 | `test_risk_assessment.py` 边界测试 |
| 新 metadata 必填字段 | `test_plugin_contract.py::TestResultStructure` 断言 |
| 新告警分级规则 | `test_plugin_contract.py::TestAlarmContract` |
| 新 runtime_mode 值 | `test_standalone.py::test_healthcheck_after_init` 白名单 |
| 生产检测器加载策略 | `test_standalone.py::test_production_chain_uses_base_detector` |
| simulation 语义 | `test_standalone.py::test_simulation_chain_does_not_emit_bird_detection` |
| 目录迁移或入口调整 | `test_directory_contract.py` |
| standalone simulator 变更 | `test_standalone.py::TestSimulatorIsolation` |
| ONNX 输入/输出 shape 或 CLASSES 数量 | `test_real_dl_preflight.py` _FakeIO shape stub |
| `BirdDetectorEnhanced` shim 行为 | `test_directory_contract.py::test_enhanced_detector_requires_explicit_opt_in` |
| 新增合成 fixture | `tests/regression/build_synthetic_fixtures.py` + `test_replay_baseline.py` 断言 |
| `tests/replay/expected_results.json` 槽位 | `test_replay_baseline.py` 阻断契约断言 |
| standalone UI 模板 quality 三态 / 训练占位下载 | `test_standalone.py::test_template_exposes_quality_tristate_and_placeholder_download` |
| 新增 / 重命名 scripts/ 入口 | `test_scripts_contract.py::TestScriptsLayout` + `TestRoutingDocumentsScripts` 列表 + `08_task_routing.md` 矩阵 |

## 6. 跨插件导入冲突防御（经验教训）

**历史故障**：`from plugin import BirdDetection` 在多插件同目录下被 pytest 解析为其他插件的 plugin.py，导致 AttributeError。

**防御规则**：**所有测试 import 必须使用完整路径**。
- ✅ `from plugins.bird_monitoring.plugin import BirdMonitoringPlugin, BirdDetection`
- ✅ `from plugins.bird_monitoring.detector import BirdDetector`
- ❌ `from plugin import BirdMonitoringPlugin`
- ❌ `from detector import BirdDetector`

该规则已写入 conftest 和全部 test_*.py 文件，违反者在跨插件并行 pytest 时必然失败。

## 7. 可运行的验收命令

```bash
# 插件内全部测试
python3 -m pytest plugins/bird_monitoring/tests/ -v

# 仅 standalone（冒烟）
python3 -m pytest plugins/bird_monitoring/tests/test_standalone.py -q

# 仅质量门三态
python3 -m pytest plugins/bird_monitoring/tests/test_quality_tristate.py -q

# 仅 ONNX preflight 契约
python3 -m pytest plugins/bird_monitoring/tests/test_real_dl_preflight.py -q

# L2 合成 replay regression（自动 lazy 生成 .npy）
python3 -m pytest plugins/bird_monitoring/tests/test_replay_baseline.py -q

# 重新生成合成 fixture
python3 plugins/bird_monitoring/tests/regression/build_synthetic_fixtures.py

# 跨插件回归（验证本插件不污染他人）
python3 -m pytest plugins/bird_monitoring/tests/ plugins/busbar_inspection/tests/ \
    plugins/animal_detection/tests/ plugins/fire_detection/tests/ \
    plugins/transformer_inspection/tests/ plugins/switch_inspection/tests/ -q
```

## 8. 未覆盖但已声明 blocked 的

- 真实 ONNX 推理端到端（blocked：无模型；preflight 契约已就位）
- 真实 bird 图片 replay（blocked：`tests/replay/expected_results.json` 中 `bird_*` 仍为 `planned_blocked_by_model`）
- 相机标定后的真实距离（blocked：无标定数据）
- 翼展 / 行为分类（blocked：无模型）

这些不视为测试缺口，在 manifest `blocked_capabilities` 中已声明。
