# 07_learning_log

用于记录本插件重要故障、根因与预防动作。每次 `/repair` 或重大质量问题修复后必须追加。

## Entry Template
- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

---

### Entry 2026-04-16 — 工程基线转换（最小治理 → 标准治理）

**Context**
- bird_monitoring 在 Phase 2 审计中被判为"8 个插件中治理最薄弱"：tests/ 为空、无 fixture、无 README、无 PROJECT_CARD（Phase 2 新建）、17 处 `print()`、阈值双源、伪数据风险（未知物种默认 sparrow）。
- 本次按 DarkBreaker 阶段目标「低数据 / 弱模型 / 强工程治理」做基线转换：没有模型也必须把契约、降级、可观测、复核路径、训练占位全部做对。

**Symptoms（审计发现）**
1. **伪数据风险**：`_identify_species()` 对未匹配类名默认返回 `sparrow`，导致下游风险评估基于错误物种。
2. **静默失败**：过暗 / 极小 / 模糊图像直接进入推理，无质量门。
3. **低置信度直通**：低置信度检测被当作正常结果输出，无复核路径。
4. **硬件耦合误导**：`deterrent_control` 暗示插件操控声学/激光硬件，实际硬件不存在。
5. **日志污染**：生产代码 12 处 `print()`（plugin.py）+ 5 处（detector.py）。
6. **阈值双源**：`RISK_THRESHOLDS` 类常量与 YAML `risk_assessment` 冲突。
7. **runtime_mode 不可观测**：simulation / real_dl 无外部可读字段。
8. **无训练闭环**：没有为未来模型迭代收集的占位数据结构。

**Root cause**
- 早期版本把"接入真实 YOLOv8"当作唯一目标，遗漏了"模型缺席时仍需交付工程契约"这一阶段性事实。
- `deterrent` 命名词义过重，误把"建议"当作"控制"。
- 测试基线缺席使任何改动都无保障，导致技术债累积。

**Fix（本轮完成）**
1. **物种回退**：引入 `SPECIES_LABEL_MAP` 枚举，未命中返回 `("unknown_bird", 0.0)`；新增 `_assess_unknown_bird_risk()` 做保守估计。
2. **质量门**：`detector.assess_image_quality()`（min_dimension / 灰度均值 / Laplacian 方差）+ `plugin._assess_input_quality()` + `_make_quality_fail_result()` 输出 `label="quality_failed"`。
3. **复核路径**：`BirdDetection.review_required/review_reason` + postprocess 生成 WARNING 告警；阈值来自 `config.quality.review_confidence_threshold`。
4. **驱离语义纠偏**：`deterrent_control` → `deterrent_suggestion`，输出 `{action, methods, reason}` JSON，**绝不**对接硬件；README 明示该边界。
5. **日志化**：17 处 `print()` 全换为 `logger.info/warning/error`。
6. **阈值配置化**：距离 / 物种 / 翼展 / 行为权重全部从 `config.risk_assessment.*_weight` 读取，类常量只做 fallback。
7. **runtime_mode 可观测**：`detector.runtime_mode` property + `healthcheck.details["runtime_mode"]` + 每个 result `metadata["runtime_mode"]`。
8. **训练占位**：`_make_training_placeholders()` 为每个结果附 `{hard_negative_candidate, hard_positive_candidate, suggested_label_for_dataset, annotation_status, model_placeholder}`。
9. **测试矩阵**：4 文件 / 39 测试（test_standalone / test_risk_assessment / test_quality_assessment / test_plugin_contract），4 维度覆盖生命周期、质量、风险、契约。
10. **manifest 三分升级**：`verified_capabilities` 从 `[]` 扩到 8 项；`blocked_capabilities` 细化为 4 项（真实 DL、硬件驱离、翼展、行为）。
11. **文档**：新建 README.md，重写 PROJECT_CARD.md（含 Phase 2/3 三分表）。

**Prevention**
- 任何"依赖真实模型的能力"必须先在 manifest 声明为 `blocked_capabilities`，代码同时提供 simulation 降级路径。
- 任何"依赖外部硬件的能力"必须以 `_suggestion` / `_advisory` / `_hint` 命名，不得用 `_control` / `_trigger`。
- 未初始化的 infer 必须返回 `label="error", failure_reason="9000"`（SDK 约定），不得 crash。
- 所有 result 必须含 `runtime_mode` + `training_placeholders`，否则契约测试失败。
- 新增 label 必须更新 `test_standalone.py::test_infer_returns_results` 的白名单。
- 质量门阈值变更必须同步 `test_quality_assessment.py` 的断言（tiny/dark/None 分支）。

**Follow-up（未完成，标记来源阶段）**
- ⏳ 真实 ONNX 模型接入（需模型训练）
- ⏳ 真实距离估计（需相机标定）
- ⏳ 精度 regression 基线（需 fixture 图片）
- ⏳ Standalone HTML 升级（结果面板、上传、风险/物种/驱离/训练占位展示）
- ⏳ 外部鸟类数据库 JSON 加载（当前仍为内置 8 种 dict）

**Validation**
- 插件内：39 passed / 0 failed / 0.38s
- 跨插件回归（6 个插件）：373 passed, 8 skipped / 3.33s，无退化

**Artifacts changed**
- 修改：plugin.py（~600 行重写）、detector.py、configs/default.yaml、manifest.json、PROJECT_CARD.md
- 新建：.gitignore、README.md、tests/conftest.py、tests/test_standalone.py、tests/test_risk_assessment.py、tests/test_quality_assessment.py、tests/test_plugin_contract.py

---

### Entry 2026-04-16 — 主链收敛与 busbar 对照整改

**Context**
- 本轮按 busbar_inspection 同等级工程完成度做对照扫描，重点检查主链统一性、runtime 真值、质量门、review 路径、standalone 隔离、测试可信度。
- 结论为 **部分达到**：bird_monitoring 已具备基础工程契约，但缺少 busbar 已具备的 real_dl preflight、质量门三态、真实 replay/fixture、仿真隔离路由和完整 runtime truth 矩阵。

**Symptoms**
1. `plugin.py` 动态加载逻辑会优先选择 `BirdDetectorEnhanced`，导致生产主链不是唯一可信的 `BirdDetector`。
2. `detector.py` 保留 `RepelController` HTTP/Modbus 风格硬件控制语义，与“仅输出驱离建议”边界冲突。
3. `advanced_bird_detector.py` 使用 `np.random` 生成鸟类检测，只能是 demo/experimental，不能进入生产链。
4. 质量门发现极小 / 过暗 / 模糊问题后没有稳定阻断，可能继续输出 `no_bird`。
5. `quality_failed` / `error` 结果缺少 `training_placeholders`，与“每条 result 必含训练占位”契约不一致。
6. `quality_failed` 文档声明 WARNING，但 postprocess 实际输出 INFO。
7. replay baseline 槽位仍是 planned，没有真实图片，不能作为视觉精度证据。

**Root cause**
- 早期代码把“增强检测器 / 高级检测器 / 驱离控制”放在同一插件目录和同一 detector 文件里，但没有生产/实验边界。
- 测试更多验证合同字段，缺少针对 simulation 不造鸟、质量门阻断、runtime truth、硬件副作用的防回归断言。

**Fix**
1. `plugin.py` 生产加载器固定为 `detector.py::BirdDetector`。
2. `detector.py::RepelController` 改为生产禁用，不加载设备、不执行命令。
3. `_assess_input_quality()` 对尺寸、亮度、清晰度问题置 `is_valid=False`。
4. `_make_quality_fail_result()` 与 `_make_error_result()` 补齐 `training_placeholders`。
5. `quality_failed` 告警级别统一为 WARNING。
6. `trigger_deterrent()` 保留兼容入口但恒返回 `False`。
7. `healthcheck.details` 增加模型路径、模型存在性、ONNX session、real model、fallback/simulation 字段。
8. 新增/补强测试，插件内测试提升到 44 passed。
9. 回灌 PROJECT_CARD、README、algorithm contract、refactor policy，明确 verified / experimental / blocked。

**Prevention**
- 后续 agent 不得把 `advanced_bird_detector.py` 或 `BirdDetectorEnhanced` 接入 `plugin.infer()`，除非先完成显式 opt-in、测试和人工确认。
- simulation 只能输出 `no_bird` 或失败类结果，禁止为了演示输出鸟类 bbox。
- 任何驱离相关变更必须保持 suggestion-only；硬件控制一律 blocked。
- 宣称 verified 必须有代码路径和测试证据；纯规则/占位/计划槽位只能标 experimental 或 blocked。
- 新增 standalone 仿真必须使用独立路由并标记 `runtime_mode=standalone_simulation`。

**Follow-up**
- 建立 busbar 同款 real_dl preflight。
- 采集最小真实 bird fixture/replay。
- 将 legacy/experimental 检测器移入 `experimental/`。
- 引入质量门三态 `pass / soft_fail / hard_fail`。
- 持续检查 manifest / README / PROJECT_CARD 能力三分，避免再次把 experimental 或 blocked 包装成 verified。

**Validation**
- `python3 -m pytest plugins/bird_monitoring/tests/ -q` → 44 passed, 1 warning。

**Artifacts changed**
- 修改：plugin.py、detector.py、README.md、PROJECT_CARD.md、.agent_skills/02_algorithm_contract.md、.agent_skills/06_refactor_policy.md、.agent_skills/07_learning_log.md
- 修改：tests/test_standalone.py、tests/test_quality_assessment.py、tests/test_plugin_contract.py

---

### Entry 2026-04-16 — 目录收敛与 experimental 隔离

**Context**
- 本轮按 busbar_inspection 的目录分层方式，对 bird_monitoring 的 production runtime、standalone demo、simulator/mock、tests、docs、prompts、experimental 进行工程诊断。
- 目标不是重写算法，而是让后续 agent 能快速判断主入口、真实算法、独立 UI、仿真器和不可扩散文件。

**Symptoms**
1. `main.py`、`__main__.py`、`run_standalone.py`、`demo/run_demo.py`、`standalone/app.py` 同时存在，但 README 未明确哪个是 canonical standalone runner。
2. 顶层 `advanced_bird_detector.py` 是概念验证代码且含随机检测逻辑，虽然文档标为 experimental，但仍贴在生产主链文件旁边。
3. `standalone/static/`、`docs/`、`prompts/`、`tests/regression/` 缺少显式占位，不利于后续 UI/demo/fixture 扩展。
4. standalone 仿真器已实现，但旧文档仍写“未实现”，且 `unknown_species` 场景实际输出 `review_required` 而非 `unknown_bird`。

**Fix**
1. 将高级概念验证代码迁到 `experimental/advanced_bird_detector.py`。
2. 顶层 `advanced_bird_detector.py` 改为兼容 shim，保留旧 import 路径但不承载算法。
3. 新增 `experimental/README.md`、`standalone/static/README.md`、`docs/README.md`、`prompts/README.md`、`tests/regression/README.md`。
4. 新增 `tests/test_directory_contract.py`，固定入口与目录职责边界。
5. 修正 standalone simulator label 优先级，使 `unknown_species` 场景产出 `unknown_bird`。
6. 回灌 PROJECT_CARD、README、refactor policy、algorithm contract、routing/test/architecture 文档。

**Prevention**
- `standalone/app.py` 是 standalone runner 的实现入口；`run_standalone.py` 与 `__main__.py` 是兼容启动器，不得删除或改语义。
- `main.py` 是 train/infer CLI，不是 Web server。
- `demo/run_demo.py` 可以使用随机图像，但输出不得进入生产契约或测试精度基线。
- `standalone/bird_simulator.py` 只能通过 `/api/simulator/*` 输出 `standalone_simulation`，不得被 `plugin.py` / `detector.py` import。
- `experimental/advanced_bird_detector.py` 只能作为概念验证，除非完成显式 opt-in、runtime truth、测试和人工确认，否则不得接入生产。

**Follow-up**
- 继续拆离 `detector.py::BirdDetectorEnhanced` 或加显式 opt-in guard。
- 在 `docs/` 补真实升级决策记录，在 `prompts/` 补 real_dl preflight / fixture replay / UI demo hardening prompt。
- 建立真实 fixture 后再启用 `tests/regression/` 精度回归。

**Validation**
- `python3 -m pytest plugins/bird_monitoring/tests/ -q` → 58 passed, 1 warning。

**Artifacts changed**
- 移动：advanced_bird_detector.py → experimental/advanced_bird_detector.py
- 新建：advanced_bird_detector.py 兼容 shim、experimental/、docs/、prompts/、standalone/static/、tests/regression/、tests/test_directory_contract.py
- 修改：standalone/bird_simulator.py、README.md、PROJECT_CARD.md、.agent_skills/02_algorithm_contract.md、.agent_skills/03_test_strategy.md、.agent_skills/06_refactor_policy.md、.agent_skills/07_learning_log.md、.agent_skills/08_task_routing.md

---

### Entry 2026-04-17 — real_dl preflight + 质量门三态 + experimental 隔离 + 最小 replay

**Context**
- 上一轮 (2026-04-16) 落地了主链收敛与目录隔离，但留下 5 个 follow-up：real_dl preflight、真实 fixture/replay、`BirdDetectorEnhanced` 拆离、质量门三态、docs/prompts 实质化。
- 本轮目标是把这 5 项中无需真实模型/真实图片就能做的部分全部落地，余下部分留 blocked 但补好契约和 prompt。

**Symptoms**
1. `detector.py::BirdDetector._load_model()` 加载完 ONNX session 后没有契约校验；任何模型放进去都会被当作 real_dl，输出 shape / 类别数 / 输入尺寸不匹配会在第一次推理 crash。
2. `detector.py::BirdDetectorEnhanced` 与生产 `BirdDetector` 同住 `detector.py`，有人轻易导入并误判为生产路径。
3. 质量门只有 `pass / fail` 二态：极小 / 极暗 → fail 直接 `quality_failed`，但中等模糊或亮度临界则被当成 pass，下游低置信度检测无 review 兜底。
4. `tests/regression/` 只有 README，没有任何 baseline；replay 槽位空，没有 simulation 可复现的样本垫底。
5. `docs/` / `prompts/` 是空 README，agent 接到“接入真实模型 / 采集 fixture / 加固 UI”任务时没有任何模板可依赖。
6. standalone 模板只展示二态质量结果，没有 soft_fail 中间态徽章；训练占位字段已生成但 UI 无下载入口。

**Root cause**
- 早期把 ONNX 加载本身当作 real_dl 通过的充分条件，未把契约校验设计为独立步骤。
- `BirdDetectorEnhanced` 是一份历史增强代码，不是“增强 = experimental”这个边界没有被 code-level 强制。
- 质量门最初只服务 `quality_failed` 阻断，没有为 review 路径开放 soft 通道。
- 没有“合成 fixture 也算 baseline”的共识，导致 L2 一直空白。

**Fix**
1. **`detector.py`**：
   - 新增 `_preflight_onnx()`，校验 input rank / shape / 与 `self.input_size` 兼容（"batch" 维度忽略）、output 通道 = `4 + len(CLASSES)`、`CLASSES` 数 ≥ 8；任何异常写 `report.issues`，`report.passed=False`。
   - `_load_model()` 调用 preflight；失败则关闭 session 回落 simulation。
   - `get_runtime_status()` 暴露 `preflight=dict(self._preflight_report)`。
   - `BirdDetectorEnhanced` 改为 `__new__` shim：除非 `os.environ.get("BIRD_ENABLE_ENHANCED_DETECTOR")=="1"` 否则抛 `RuntimeError("experimental detector is opt-in only ...")`，opt-in 后委派 `experimental/enhanced_detector.py::EnhancedBirdDetector`。
2. **`experimental/enhanced_detector.py`** 新建：原 `BirdDetectorEnhanced` 全部主体（3D 距离 / 轨迹 / 威胁评估），从 `plugins.bird_monitoring.detector` import `BirdDetector` 作为基类。
3. **`plugin.py::_assess_input_quality()`**：返回 `{status, is_valid, issues, hard_issues, soft_issues, overall_score, ...}`；`status ∈ {pass, soft_fail, hard_fail}`；hard 条件包含 None / 空 / `min_dimension` 不足 / detector hard issues；soft 条件来自新阈值 `soft_overall_threshold / soft_clarity_threshold / soft_brightness_threshold`。
4. **`plugin.py::infer()`**：只在 `status=="hard_fail"` 时调 `_make_quality_fail_result`；`soft_fail` 时强制每条 detection `review_required=True / review_reason="质量软失败"`，no_bird 也带上同样的 metadata。
5. **`configs/default.yaml::quality`** 新增 soft 阈值。
6. **`tests/regression/build_synthetic_fixtures.py`** 新建：生成 `no_bird_clear.npy`（640×480 均匀噪声）与 `quality_dark.npy`（全黑+微噪），仅覆盖 simulation 可复现的两个 label。
7. **`tests/test_replay_baseline.py`** 新建：3 例（no_bird_clear / quality_dark / `bird_*` 槽位 `planned_blocked_by_model` 契约）。
8. **`tests/test_quality_tristate.py`** 新建：6 例覆盖三态及 review 升级。
9. **`tests/test_real_dl_preflight.py`** 新建：7 例用 `_FakeSession + _FakeIO` 校验 preflight 全分支。
10. **`tests/test_directory_contract.py`** 新增：`test_enhanced_detector_requires_explicit_opt_in`、`test_enhanced_detector_body_lives_in_experimental`。
11. **`tests/test_standalone.py`** 新增 `test_template_exposes_quality_tristate_and_placeholder_download`。
12. **`standalone/templates/bird_monitoring.html`**：新增 `<span id="q-status">` 三态徽章（pass=success / soft_fail=warning / hard_fail=danger）+ "下载 placeholders.json" 按钮 + `setLastTrainingPlaceholders()` + `window.downloadTrainingPlaceholders` Blob 下载；`runRealDetection` 把 `q.status / q.soft_issues` 传入 `renderQuality`。
13. **`docs/real_dl_upgrade_decision.md`** + **`docs/quality_gate_tristate.md`** 新建。
14. **`prompts/real_dl_preflight_prompt.md`** + **`prompts/fixture_collection_prompt.md`** + **`prompts/ui_hardening_prompt.md`** 新建。

**Prevention**
- 任何加入 ONNX 的检测器必须在 session 加载后立即跑 preflight；不得只靠 session.run() 异常兜底。
- `experimental/` 下的检测器若需暴露给生产，必须保留 `__new__`-based opt-in shim，并在 `tests/test_directory_contract.py` 里固定 RuntimeError 分支。
- 质量门新增任何指标必须同步：(a) hard / soft 阈值字段、(b) `test_quality_tristate.py` 三态分支、(c) UI 模板徽章。
- 合成 fixture 只允许覆盖 simulation 可复现的 label；`bird_*` 真实样本必须由 `prompts/fixture_collection_prompt.md` 路径采集。
- 模板新增结构化字段必须在 `test_standalone.py` 的模板字符串断言里固定。

**Follow-up**
- 真实 YOLOv8 ONNX 模型交付 → 跑 `prompts/real_dl_preflight_prompt.md` 的清单。
- 真实 bird 图片采集 → 跑 `prompts/fixture_collection_prompt.md` 的清单 → 把 `expected_results.json` 中的 `bird_*` 槽位从 `planned_blocked_by_model` 改为 `collected`。
- 真实模型上线后再决定 `experimental/enhanced_detector.py` 是否合回主链。

**Validation**
- 插件内：`python3 -m pytest plugins/bird_monitoring/tests/ -q` → **77 passed, 1 warning / 1.33s**。
- 跨插件回归（6 个插件）：**331 passed, 8 skipped**，无退化。

**Artifacts changed**
- 修改：`detector.py`（preflight + shim）、`plugin.py`（三态质量门 + soft review）、`configs/default.yaml`（soft 阈值）、`standalone/templates/bird_monitoring.html`（三态徽章 + 下载）、`tests/test_directory_contract.py`、`tests/test_standalone.py`、`PROJECT_CARD.md`、`.agent_skills/02_algorithm_contract.md`、`.agent_skills/03_test_strategy.md`、`.agent_skills/07_learning_log.md`、`.agent_skills/08_task_routing.md`。
- 新建：`experimental/enhanced_detector.py`、`tests/regression/build_synthetic_fixtures.py`、`tests/test_quality_tristate.py`、`tests/test_real_dl_preflight.py`、`tests/test_replay_baseline.py`、`docs/real_dl_upgrade_decision.md`、`docs/quality_gate_tristate.md`、`prompts/real_dl_preflight_prompt.md`、`prompts/fixture_collection_prompt.md`、`prompts/ui_hardening_prompt.md`。

---

### Entry 2026-04-17 — scripts/ 入口 + 覆盖率门 + 真实交付 gating CLI

**Context**
- 上一轮（real_dl preflight + 质量门三态 + experimental 隔离 + 最小 replay）剩 5 项 risk：真实 ONNX 模型未交付 / 真实 bird fixture 未采集 / `BirdDetectorEnhanced` 仍 experimental / 物种翼展行为 blocked / `scripts/` 目录与 `.coveragerc` 缺失。
- 真实模型与真实图片受外部输入约束，不能在不伪造闭环的前提下解决；本轮聚焦把这两项**变成可执行的 gating 工具**（agent 一旦拿到模型/图片就能跑），同时把 `scripts/` 与覆盖率门补齐。

**Symptoms**
1. `scripts/` 目录不存在；08_task_routing.md 长期挂着 4 个待办：`run_targeted_tests.sh / run_regression_tests.sh / run_quality_gate.sh / .coveragerc`，没有统一的入口让人或 CI 一键执行分层测试 / 反模式扫描 / 覆盖率门。
2. 没有针对真实 ONNX 模型交付的 gating CLI：交付者只能复制粘贴 `BirdDetector(...).get_runtime_status()` 的 hack 代码自检，preflight 报告没有结构化输出口。
3. 没有针对真实 bird 图片 fixture 的 intake CLI：采集者无法在不读 `expected_results.json` 源码的情况下确认槽位状态、cv2 可读性和最小尺寸。
4. `04_quality_audit.md` 的反模式命令依赖 `rg`，在没安装 ripgrep 的开发机上无法执行。

**Root cause**
- 项目阶段长期处在「单元测试足够 → 不需要门禁脚本」的惯性中；当 7 个测试文件 / 77 测试规模出现后，分层执行与覆盖率门变成必要，但没有人补齐。
- 外部交付物（模型 / 图片）的接入流程只在 prompts/ 里描述，没有可运行的工具兜底，agent 容易在第一次接入时绕开 preflight 校验。

**Fix**
1. **`scripts/run_targeted_tests.sh`** 新建：`l0 / l1 / l2 / all` 四档；`l0` 仅 risk/quality/quality_tristate/preflight 纯逻辑；`l1` 走 plugin 全链路；`l2` 跑合成 fixture replay。
2. **`scripts/run_quality_gate.sh`** 新建：三段门 — 反模式扫描 → pytest → coverage；扫描器优先 `rg`，回落 `grep -E`；退出码 1/2/3/4 分别对应反模式/pytest/覆盖率/环境缺失。`--no-coverage` 跳过第三段。
3. **`scripts/run_regression_tests.sh`** 新建：跨 6 个对照插件回归，自动跳过缺失目录。
4. **`.coveragerc`** 新建：`source = plugins/bird_monitoring`，omit `experimental/ / advanced_bird_detector.py / main.py / __main__.py / run_standalone.py / demo/ / notebooks/ / tests/ / scripts/`；`fail_under = 60`；`exclude_lines` 排除 `experimental detector is opt-in only` shim 分支。当前实测 62%。
5. **`scripts/check_real_model.py`** 新建：CLI gating 工具，接收 ONNX 路径，构造 `BirdDetector({"model": {"path": ...}})`，跑真实 `_load_model + _preflight_onnx`，输出 preflight 报告（人类格式或 `--json`）；退出码 `0/1/2/3` 严格反映 passed / failed / 未执行（缺文件、缺 ort、加载失败）/ 参数错。**不伪造任何推理结果**。
6. **`scripts/validate_fixture.py`** 新建：CLI intake 工具，按 `tests/replay/expected_results.json` 槽位校验 fixture 文件存在性、cv2 可读性、最小宽高（默认 32，与 quality 门 `min_dimension` 对齐）；planned 状态文件缺失时报 `slot_pending` 而非 `file_missing`；如果文件已存在但 `collection_status` 仍 `planned*`，报 `collection_status_not_updated`。**不写 expected_results.json**。
7. **`tests/test_scripts_contract.py`** 新建（13 例）：scripts 与 .coveragerc 存在 / 5 脚本 +x + shebang / `run_targeted_tests.sh bogus` exit 2 / `check_real_model.py` 缺路径 exit 2 / `validate_fixture.py` planned 槽 exit 1 + 报 slot_pending / 5 脚本必须在 08_task_routing.md 注册（防孤儿）。
8. **`08_task_routing.md`**：新增「scripts/ 入口职责矩阵」与退出码语义；快速参考表全部切换到脚本入口；后续补齐项中 4 项打勾、新增 enhanced 评估项。
9. **`04_quality_audit.md`**：审计命令一条龙改为 `scripts/run_quality_gate.sh`；保留底层 `rg` 等效命令作为 troubleshooting 参考。
10. **`03_test_strategy.md`**：测试矩阵补 `test_scripts_contract.py`；总数 77 → 90；新增「新增/重命名 scripts/ 入口」追加规则。
11. **`PROJECT_CARD.md`** 第 8 节：测试 77 → 90、跨插件 331 → 424、覆盖率 62%。

**Prevention**
- 任何新增 scripts/ 入口必须同时：(a) 加 `+x` 与 shebang；(b) 在 `08_task_routing.md::scripts/ 入口职责矩阵` 登记；(c) 在 `tests/test_scripts_contract.py::SHELL_SCRIPTS / PYTHON_SCRIPTS` 列表加名字（自动触发存在性 + shebang + +x + 路由注册检查）。
- 任何外部交付物（模型 / 图片 / 标定数据）必须先有 CLI gating 工具再宣布通过；`check_real_model.py` 与 `validate_fixture.py` 是模板。
- 反模式扫描必须支持 `rg` 与 `grep -E` 双路径，避免开发机环境差异破坏 audit 流程。
- 覆盖率门 `fail_under` 调整必须同步 `tests/test_scripts_contract.py::test_coveragerc_exists_with_fail_under` 断言。

**Follow-up**
- 真实 YOLOv8 ONNX 交付 → `scripts/check_real_model.py model.onnx` 全绿 → `runtime_mode=real_dl`。
- 真实 bird 图片采集 → `scripts/validate_fixture.py` 全绿 → 改 `expected_results.json::collection_status` → 解锁 `bird_*` replay。
- real_dl 验证后再决定 `experimental/enhanced_detector.py` 是否合回主链。
- 物种 / 翼展 / 行为分类 仍 blocked by 真实模型，prompts/ 槽位已就绪。

**Validation**
- 插件内：`scripts/run_targeted_tests.sh all` → **90 passed**, 1.5s。
- 跨插件回归：`scripts/run_regression_tests.sh` → **424 passed, 8 skipped**，无退化。
- 质量门：`scripts/run_quality_gate.sh` → 反模式 0 / pytest 90 passed / coverage **62%**（fail_under=60 通过）。
- gating CLI 手测：
  - `scripts/check_real_model.py /tmp/no_such.onnx` → exit 2 + `[FAIL] 模型文件不存在`
  - `scripts/validate_fixture.py` → exit 1 + 5 个槽位全部报 `slot_pending`（仍 planned，符合事实）

**Artifacts changed**
- 新建：`.coveragerc`、`scripts/run_targeted_tests.sh`、`scripts/run_quality_gate.sh`、`scripts/run_regression_tests.sh`、`scripts/check_real_model.py`、`scripts/validate_fixture.py`、`tests/test_scripts_contract.py`。
- 修改：`PROJECT_CARD.md`、`.agent_skills/03_test_strategy.md`、`.agent_skills/04_quality_audit.md`、`.agent_skills/07_learning_log.md`、`.agent_skills/08_task_routing.md`。

---

### Entry Template (for future entries)

- **Date**:
- **Context**:
- **Symptom**:
- **Root cause**:
- **Fix**:
- **Prevention**:
- **Follow-up**:
- **Validation**:
