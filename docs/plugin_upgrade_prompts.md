# DarkBreaker 插件统一改造 Prompt 完整体系

> 生成日期：2026-04-15
> 依据：《DarkBreaker平台可执行升级报告 V1.0》 + 《DarkBreaker 插件统一改造 Prompt（母版）》
> 覆盖：全部 19 个插件，8 个升级阶段，4 个层级

---

## 使用说明

```
结构 = 阶段 Prompt + 母版 Prompt + 插件参数卡
```

1. 根据当前所处阶段，选择对应的 **阶段 Prompt Overlay**
2. 拼接 **统一母版 Prompt**（见第一部分）
3. 附上目标插件的 **插件参数卡**（见第三部分）
4. 如果是样本上传判别模式，额外附加 **样本上传判别 Prompt**（见第四部分）

---

## 第一部分：统一母版 Prompt（Part A）

```text
你当前要处理的插件路径是：
{PLUGIN_PATH}

插件参数卡如下：
{PLUGIN_CARD}

你的任务不是泛泛"优化精度"，而是将该插件改造成符合 DarkBreaker 当前阶段统一交付标准的工程插件。

【DarkBreaker 当前阶段目标】

DarkBreaker 当前处于"低数据 / 弱模型 / 强工程治理"阶段。
目标不是假装已经完成真实深度学习交付，而是先建立统一的：
1. 可上传样本并分析
2. 可解释识别/异常判别
3. 可输出详细告警信息
4. 可支持逐项结果一一对应
5. 可对误判/不确定结果给出复核提示
6. 可为未来真实模型训练和工程交付预留占位符

【必须遵守的工程边界】
1. plugin.py 仅做 SDK/接口适配，不承载复杂核心算法。
2. 核心识别、后处理、冲突仲裁、样本诊断，优先放 detector / analyzer / engine / core 层。
3. 所有新增阈值必须进入 configs/default.yaml 或当前插件已有配置体系。
4. 不允许在生产路径新增 except: pass。
5. 不允许在生产路径新增 print()。
6. 不要夸大当前真实模型能力，不要把未来能力写成当前事实。
7. 需要引用当前目录中的 .agent_skills 约束、README、manifest、tests、standalone 事实。
8. 如果插件已有 standalone，则必须优先考虑"上传样本 → 输出逐项结果 + 详细告警 + 错误判别信息"的改造。
9. 所有新增逻辑必须补测试。
10. 如果插件目前仍是占位态，则不要直接套完整识别插件改造模式，而应先补最小治理底座。

【首先必须审计】
请先审计并明确指出：
1. 当前主链路文件和真实入口
2. 当前能力与 manifest/README 是否一致
3. 当前是否具备上传样本后的统一输出能力
4. 当前是否缺少：
   - 输入质量评估
   - 冲突仲裁
   - review_required / uncertain
   - 一一对应逐项清单
   - 详细告警
   - 未来训练占位符
5. 当前 tests 是否只覆盖冒烟而未覆盖真实样本/回放
6. 当前 standalone/UI 是否只能展示粗结果而不能展示逐项细节

【本轮允许的改造方向】

A. 输入与质量评估层
   - 增加样本输入校验
   - 增加样本质量评估
   - 对不满足识别条件的输入输出可解释失败，不允许沉默失败

B. 判别与后处理层
   - 保留现有模型链与 fallback 链
   - 增加候选结果结构化输出
   - 增加冲突仲裁
   - 对不确定结果输出 review_required / uncertain
   - 禁止把未知类别默认硬判成某个主类

C. 统一结果表达层
   - sample_summary
   - itemized_results
   - diagnostic_panel
   - alarms
   - training_placeholders

D. UI / standalone / 调试视图
   如当前插件具备 standalone 或前端展示能力，请补齐：
   - 一一对应识别清单
   - 错误判别/冲突诊断窗口
   - 样本质量信息
   - 图像/样本级 summary 与 item 级 detail 分离
   - future-model / 数据回灌预留字段

E. future-model 占位能力
   为后续真实工程交付预留：
   - hard_negative_candidate
   - hard_positive_candidate
   - suggested_label_for_dataset
   - annotation_status
   - feature_placeholder / model_placeholder

【最终必须输出】
1. [AUDIT_RESULT] PASS|FAIL
2. [BLOCKERS]
3. [HIGH_RISK]
4. [ACTION_ITEMS]
5. 改动文件清单
6. 每个文件的修改目的
7. 新增测试清单
8. 当前工程能解决的问题
9. 必须依赖真实模型/数据才能解决的问题
10. 本轮验收标准

【输出要求】
1. 必须引用实际文件和关键位置。
2. 不要只给建议，要给可执行方案。
3. 不要把未来设想写成当前事实。
4. 结论必须具体、工程化、可交付。
```

---

## 第二部分：阶段 Prompt Overlay（按 Phase 选用）

### Phase 1：项目去冗余与结构收口（P0）

```text
【阶段覆盖 Overlay — Phase 1: 去冗余与结构收口】

当前阶段优先级：P0
当前阶段目标：让团队能明确区分源码、运行产物、历史旧版、缓存垃圾、占位模块。

你在本阶段的职责是对该插件做结构审计与冗余盘点，不是做功能改造。

必做项：
1. 盘点该插件目录下是否存在以下类型的冗余：
   - 旧版 checkpoint / .pt / .onnx 文件残留
   - 旧版日志 / 覆盖率报告 / htmlcov
   - 不再使用的旧版脚本、demo 残留
   - 与其他插件重复的工具函数
2. 检查该插件的 .gitignore 是否覆盖了运行产物
3. 输出该插件的状态总表行：
   - 插件名称 | 成熟度等级 | 已启用状态 | 是否有真实模型 | 是否有真实夹具 | 是否纳入第一批

门禁：
- 未完成去冗余收口前，不允许开始该插件的大规模功能升级
- 不允许删除任何 evidence/ 下的文件
- 旧产物先移动到 archive/legacy_2026Q2/，不直接删除
- 所有路径改造先做引用扫描（grep 确认无其他文件引用该路径）

风险关注：
- 删除历史产物时误删真实依赖
- tests 对旧路径有隐式依赖

最终输出：
1. 冗余文件清单（分类：可归档 / 需确认 / 不可删）
2. 引用依赖扫描结果
3. 插件状态总表行
4. 建议 .gitignore 补充项
```

### Phase 2：A 类插件能力边界收口（P0）

```text
【阶段覆盖 Overlay — Phase 2: 能力边界收口】

当前阶段优先级：P0
当前阶段目标：把"插件宣称能力"收口到"已验证能力"。

本阶段仅适用于第一批 A 类插件：
transformer_inspection, switch_inspection, busbar_inspection, capacitor_inspection, meter_reading
以及扩展观察对象：animal_detection, bird_monitoring, fire_detection

必做项：
1. 审计该插件的 manifest.json / README.md / PROJECT_CARD.md / standalone 说明文档
2. 对每个声称的 capability，判定为以下之一：
   - verified：有 tests/fixtures 证明、regression 可跑通
   - experimental：代码存在但测试不充分或仅 smoke test
   - blocked：代码不存在或依赖未满足
3. 在 manifest.json 中新增或修订以下字段：
   - verified_capabilities: []
   - experimental_capabilities: []
   - blocked_capabilities: []
   - required_fixtures: []
   - runtime_mode_support: { real_dl: bool, traditional_fallback: bool, simulation: bool }
   - current_known_limits: []
4. 同步更新 PROJECT_CARD.md 和 README.md（若存在）

门禁：
- 未完成能力口径统一，不允许对外发布"第一批已交付"
- manifest 声称的稳定能力不得超出 tests/fixtures 证明范围
- 一律以 PROJECT_CARD + fixtures + regression 为准

风险关注：
- 产品文档与算法现实脱节
- 试点现场按未验证能力验收

最终输出：
1. 能力三分表（verified / experimental / blocked）
2. manifest.json 修改 diff
3. PROJECT_CARD.md 修改 diff
4. 当前能力差距分析
```

### Phase 3：A 类最小真实回放集建设（P0）

```text
【阶段覆盖 Overlay — Phase 3: 最小真实回放集建设】

当前阶段优先级：P0
当前阶段目标：让第一批插件从"能跑"升级为"可试点回放验证"。

每插件最低要求：
- 20~50 张真实样本（或等效时序数据窗口）
- 覆盖类型：
  - 正常样本
  - 典型异常样本
  - 边界/误检样本
  - 质量门禁失败样本
- 结果基准文件：expected_results.json 或 expected_labels.yaml

必做项：
1. 审计当前 tests/fixtures/ 和 data/ 目录，评估现有夹具质量
2. 设计最小回放集目录结构：
   tests/replay/
   ├── normal/          # 正常样本
   ├── anomaly/         # 典型异常
   ├── boundary/        # 边界/误检
   ├── quality_fail/    # 质量门禁失败
   └── expected_results.json
3. 编写回放测试脚本 tests/test_replay.py：
   - 遍历回放集
   - 调用插件主链路
   - 与 expected_results.json 对比
   - 输出 pass/fail/drift 报告
4. 标注每个样本类型的来源（真实采集 / 合成 / 脱敏）

推荐优先顺序：
1. meter_reading（算法合同和测试基础较强）
2. switch_inspection
3. transformer_inspection
4. busbar_inspection
5. capacitor_inspection

门禁：
- 没有最小真实回放集的插件，不得纳入第一批验收
- 仅依赖 mock / smoke / unit test 的插件，不能标记为"试点可验证"

最终输出：
1. 当前夹具质量评估
2. 回放集目录结构设计
3. test_replay.py 框架代码
4. 样本缺口清单（需人工采集/标注的部分）
5. expected_results.json 模板
```

### Phase 4：A 类统一输出契约（P0）

```text
【阶段覆盖 Overlay — Phase 4: 统一输出契约】

当前阶段优先级：P0
当前阶段目标：消除插件间行为不一致，便于平台/UI/告警/报告统一消费。

必须统一的字段：

runtime 相关：
- runtime_mode: "real_dl" | "traditional_fallback" | "quality_blocked" | "simulation"

质量门禁相关：
- quality_gate_status: "pass" | "soft_fail" | "hard_fail"

复核状态：
- review_status: "confirmed" | "review_required" | "blocked"

可解释性字段：
- failure_reason: str | null
- suggested_action: str | null
- fallback_level: int
- model_loaded: bool
- preflight_ok: bool
- runtime_truth: str  # 当前实际运行模式的真实描述

必做项：
1. 审计该插件当前输出结构，列出与统一契约的差异
2. 在 detector / analyzer / engine 层增加统一输出适配：
   - 不改变内部算法逻辑
   - 在输出层包装为统一字段
3. 在 tests/ 中增加契约层断言：
   - test_output_contract.py
   - 验证每个必选字段存在且语义正确
4. 更新 standalone/templates 以展示新增统一字段

门禁：
- 第一批插件未统一输出契约前，不允许统一接 UI 验收
- quality_gate 不允许继续出现插件私有语义泛化冒用

策略：先兼容输出（新字段 + 旧字段并存），再逐步收口旧字段

最终输出：
1. 当前输出结构 vs 统一契约差异表
2. 适配层代码修改
3. test_output_contract.py
4. standalone 模板更新
```

### Phase 5：A 类最小 training 闭环落地（P1）

```text
【阶段覆盖 Overlay — Phase 5: 最小 training 闭环】

当前阶段优先级：P1
当前阶段目标：只为第一批交付插件建立真实可跑的训练链路。

必做项：
1. 确认该插件的 plugin_id 在训练侧的映射：
   - transformer -> transformer_inspection
   - switch -> switch_inspection
   - busbar -> busbar_inspection
   - capacitor -> capacitor_inspection
   - meter -> meter_reading
2. 确认该插件对应的 task_type：
   - detection / classification / ocr / thermal_anomaly
3. 验证以下训练链路可跑通：
   - upload validate → ingest route → preprocess → train → export → registry → deploy
4. 检查 model_integration.py 中是否存在旧路径硬编码
5. 确认所有训练产物统一走 plugin_id + task_type + version 命名

门禁：
- 未统一 plugin_id 前，不允许继续扩散 legacy alias 到新逻辑
- 第一批交付不要求 B/C 类全部训练器真落地
- training_api.py 仅保留 legacy visual-only
- 新业务统一走 training_api_v2.py

不纳入本阶段：
- multimodal_feature_fusion
- multimodal_decision_fusion
- action_event_sequence_recognition
- multivariate_sensor_anomaly
- equipment_health_prediction

最终输出：
1. plugin_id / task_type 映射确认
2. 训练链路各环节可用性检查报告
3. 旧路径硬编码清单
4. 修复方案
```

### Phase 6：B 类基础闭环（P2）

```text
【阶段覆盖 Overlay — Phase 6: B 类基础闭环】

当前阶段优先级：P2
当前阶段目标：先让 B 类"可评估"，再让它"可规模训练"。

本阶段适用插件：
- acoustic_monitoring（首批试点）
- gas_detection（首批试点）
- device_monitoring
- temperature_monitoring
- action_event_monitoring

B 类插件特征：
- 输入以音频窗口/时间序列/事件流/指标快照为主
- 输出 time_range / window_id / itemized anomalies
- 需要趋势诊断和复测建议
- 当前 TemporalTrainer 仍以计划生成/占位产物为主

必做项：
1. 补上传示例包（每个插件至少一个可用的数据上传 bundle）
2. 固化 manifest 样例（确保 input/output schema 与实际一致）
3. 生成基础评测报表框架：
   - false_alarm_rate
   - recall
   - detection_latency
   - lead_time（预警提前量）
   - stability_under_missing_data
4. 至少完成一个 baseline 真跑闭环：
   - 用规则/统计方法作为 baseline
   - 不要求深度模型
   - 但必须有明确的输入→输出→评测链路

门禁：
- 没有基线评测，不允许声称时序训练体系已可交付
- 没有数据模板，不允许要求外部标注/采集团队直接投产使用
- B 类不抢占 A 类第一批资源

最终输出：
1. 数据上传 bundle 模板
2. 评测报表框架代码
3. baseline 闭环验证报告
4. 与 A 类差异说明（哪些能力当前不具备、预计什么时候可以补齐）
```

### Phase 7：C 类契约收口与规则融合稳定化（P2）

```text
【阶段覆盖 Overlay — Phase 7: C 类契约收口】

当前阶段优先级：P2
当前阶段目标：让 multimodal_fusion 先变成稳定 orchestrator，再考虑重训练。

本阶段仅适用：multimodal_fusion

必做项：
1. 明确 modality → plugin_id 映射表：
   - visual → transformer_inspection / busbar_inspection / ...
   - thermal → temperature_monitoring / transformer_inspection thermal module
   - acoustic → acoustic_monitoring
   - gas → gas_detection
   - ultrasonic → （待定）
   - hyperspectral → hyperspectral_detection
   - vibration → （待定）
2. 固定缺模态策略：
   - 每种模态缺失时的降级行为
   - 最少需要几种模态才能出融合结论
   - 缺模态时 confidence 的衰减规则
3. 固定 rule + model hybrid 行为：
   - 规则融合（当前可交付）
   - 模型融合（future placeholder）
   - 明确哪些诊断规则是 rule-based、哪些是 model-based
4. 固定 samples.jsonl 对齐方式与时间对齐容忍度

门禁：
- 不允许在输入来源不稳定时先追求复杂特征融合模型
- 不允许插件依赖图与训练配置图长期分叉
- MultimodalTrainer 当前仍偏计划式，不纳入第一批真训练

最终输出：
1. modality → plugin_id 映射表
2. 缺模态降级策略文档
3. rule/model 行为边界定义
4. 诊断规则清单（标注 rule-based vs model-placeholder）
```

### Phase 8：L0 占位插件补齐最小骨架（P3）

```text
【阶段覆盖 Overlay — Phase 8: L0 占位插件最小骨架】

当前阶段优先级：P3
当前阶段目标：补齐最小四件套，不做重算法开发。

本阶段适用：thermal, radar

这些插件当前状态：
- thermal：仅有 __init__.py + enhanced_thermal_analyzer.py + README.md，无 manifest / plugin.py / configs / tests
- radar：仅有 __init__.py + mmwave_radar_adapter.py + README.md，无 manifest / plugin.py / configs / tests

必做项（最小四件套）：
1. plugin.py — 继承 EnhancedBasePlugin，实现最小 init/process 骨架
2. manifest.json — 完整但标记所有 capability 为 blocked
3. configs/default.yaml — 最小配置
4. tests/test_plugin.py — 最小冒烟测试（init 成功、process 返回合法结构）

不要做：
- 不要生成 .agent_skills/（升级报告明确禁止）
- 不要做重算法开发
- 不要把现有 adapter/analyzer 硬套为主检测链路
- 不要纳入第一批资源竞争

门禁：
- 在第一批交付前，L0 不进入主链资源竞争
- 只做最小骨架

最终输出：
1. plugin.py 骨架代码
2. manifest.json
3. configs/default.yaml
4. tests/test_plugin.py
5. 与现有 adapter/analyzer 的关系说明
```

---

## 第三部分：全量插件参数卡（Part B）

### ━━━━━ A 类：视觉缺陷 / 状态识别（第一批必选） ━━━━━

#### 1) busbar_inspection

```text
[PLUGIN_CARD]
plugin_id: busbar_inspection
plugin_name: 母线自主巡视插件
plugin_family: visual_defect
plugin_type: 远距小目标缺陷识别 / ROI缺陷检测 / 场景级误判抑制
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/busbar_inspection
current_stage: engineering_baseline
primary_input: image
primary_output: defects + alarms + quality + zoom_suggestion

core_entry_files:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - configs/default.yaml
  - standalone/app.py
  - standalone/templates/busbar_inspection.html
  - tests/
  - README.md
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 detector_enhanced.py
  - 已存在 tests
  - 已存在 README
  - 已存在 core/ 目录

must_read_first:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - configs/default.yaml
  - standalone/app.py
  - standalone/templates/busbar_inspection.html
  - tests/test_standalone.py
  - tests/test_quality_gate_contract.py
  - .agent_skills/02_algorithm_contract.md
  - .agent_skills/04_quality_audit.md

primary_capabilities:
  - crack / foreign_object / pin_missing 检测
  - quality gate
  - zoom suggestion
  - standalone 上传检测

current_known_risks:
  - foreign_object / mixed_defects 易偏向 crack
  - 需要逐项识别清单
  - 需要错误判别窗口
  - 需要仿真 vs 真实模式对照

must_have_ui:
  - itemized_results
  - diagnostic_panel
  - alarms
  - quality_panel
  - zoom_panel
  - simulation_vs_live_compare

must_not_do:
  - 不要把 mixed_defects 作为对象级 label 强行接入当前契约
  - 不要把未知类别默认回退到 crack

future_placeholders:
  - crack_verifier_placeholder
  - foreign_object_conflict_resolver_placeholder
  - dataset_feedback_placeholder

acceptance_focus:
  - 上传图片后逐项输出识别清单
  - 不确定结果必须可解释
  - UI 必须能承载 review_required / candidate_labels / detection_id
```

#### 2) capacitor_inspection

```text
[PLUGIN_CARD]
plugin_id: capacitor_inspection
plugin_name: 电容器自主巡视插件
plugin_family: visual_defect
plugin_type: 结构完整性 + 入侵检测
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/capacitor_inspection
current_stage: engineering_baseline
primary_input: image
primary_output: structural_defects + intrusion_events + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - configs/default.yaml
  - standalone/
  - tests/
  - README.md
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 detector_enhanced.py
  - 已存在 tests
  - 已存在 README

must_read_first:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - configs/default.yaml
  - tests/
  - README.md
  - .agent_skills/02_algorithm_contract.md
  - .agent_skills/04_quality_audit.md

primary_capabilities:
  - capacitor bank structural inspection
  - tilt / collapse / missing part detection
  - intrusion detection

current_known_risks:
  - 结构缺陷与入侵事件结果表达可能混杂
  - intrusion 正样本能力可能不足
  - 需要"结构类结果"和"入侵类结果"分层展示
  - fallback 链中可能存在硬编码阈值技术债

must_have_ui:
  - itemized_results
  - structural_summary_panel
  - intrusion_event_panel
  - alarms
  - diagnostic_panel

must_not_do:
  - 不要把 intrusion 和 structural defect 混成同一种 item 语义
  - 不要把 README 中未验证能力写成已验证

future_placeholders:
  - tilt_classifier_placeholder
  - collapse_detector_placeholder
  - intrusion_reid_placeholder

acceptance_focus:
  - 上传图片后至少能区分 structural_defect 与 intrusion_event
  - 每条结果要有 item_id / category / severity / suggested_action
```

#### 3) switch_inspection

```text
[PLUGIN_CARD]
plugin_id: switch_inspection
plugin_name: 开关间隔自主巡视插件
plugin_family: visual_state
plugin_type: 状态识别 + 五防逻辑校验 + 图像质量门禁
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/switch_inspection
current_stage: engineering_baseline
primary_input: image
primary_output: state_results + logic_validation + quality_results + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - switch_consistency.py
  - configs/default.yaml
  - standalone/
  - tests/
  - README.md
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 detector_enhanced.py
  - 已存在 switch_consistency.py（逻辑校验模块）
  - 已存在 tests
  - 已存在 README

must_read_first:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - switch_consistency.py
  - configs/default.yaml
  - tests/
  - README.md
  - .agent_skills/

primary_capabilities:
  - breaker / isolator / grounding state recognition
  - clarity gate
  - minimal logic validation

current_known_risks:
  - 逻辑校验模块可能未完全并入主链
  - 状态输出与逻辑错误输出需要分层
  - 需要上传样本后的逐项清单

must_have_ui:
  - state_result_panel
  - logic_validation_panel
  - image_quality_panel
  - alarms
  - itemized_results

must_not_do:
  - 不要把逻辑校验结果和视觉状态结果混成同一类 defect
  - 不要把历史规则写成当前已验证全覆盖能力

future_placeholders:
  - state_classifier_placeholder
  - linkage_logic_engine_placeholder
  - gauge_reading_extension_placeholder

acceptance_focus:
  - 上传图片后输出状态 + 逻辑 + 质量三类结果
  - 每类结果都要有独立 reason / severity / recommended_action
```

#### 4) transformer_inspection

```text
[PLUGIN_CARD]
plugin_id: transformer_inspection
plugin_name: 主变自主巡视插件
plugin_family: visual_defect
plugin_type: 外观缺陷 + 状态识别 + 热像辅助
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/transformer_inspection
current_stage: engineering_baseline
primary_input: image | thermal_image(optional)
primary_output: defect_results + silica_gel_state + oil_level_reading + thermal_alarm + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - thermal_analyzer.py
  - defect_detector.py
  - configs/default.yaml
  - standalone/
  - tests/
  - README.md
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 detector_enhanced.py
  - 已存在 thermal_analyzer.py（热像分析）
  - 已存在 defect_detector.py（缺陷检测）
  - 已存在 tests
  - 已存在 README

must_read_first:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - thermal_analyzer.py
  - defect_detector.py
  - configs/default.yaml
  - tests/
  - README.md
  - .agent_skills/

primary_capabilities:
  - 外观缺陷识别
  - 硅胶状态识别
  - 油位读取
  - 可选热像告警

current_known_risks:
  - 旁路模块（thermal_analyzer / defect_detector）可能未接主链
  - 需要将多种结果拆成 itemized_results
  - 需要避免把未接主链能力写成已验证能力
  - 已验证能力面窄于可能的宣称范围

must_have_ui:
  - itemized_results
  - silica_gel_panel
  - oil_level_panel
  - defect_panel
  - thermal_panel
  - alarms

must_not_do:
  - 不要把 defect_detector.py / thermal_analyzer.py 的未来设想写成当前事实
  - 不要把阀门状态历史能力写成已验证能力

future_placeholders:
  - silica_classifier_placeholder
  - oil_meter_parser_placeholder
  - thermal_fusion_placeholder

acceptance_focus:
  - 一张上传图像可拆多类 item 结果
  - 每个 item 有独立 confidence / review_status / suggested_action
```

#### 5) meter_reading

```text
[PLUGIN_CARD]
plugin_id: meter_reading
plugin_name: 表计读数识别插件
plugin_family: visual_state
plugin_type: 表计检测 + 关键点定位 + OCR读数 + 多表类型适配
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/meter_reading
current_stage: engineering_baseline
primary_input: image
primary_output: reading_value + confidence + meter_type + recommended_action

core_entry_files:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - configs/default.yaml
  - standalone/
  - tests/
  - README.md
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 detector_enhanced.py
  - 已存在 tests
  - 已存在 README
  - 支持多种表计类型：pressure_gauge, temperature_gauge, oil_level_gauge, sf6_density_gauge, digital_display, led_indicator
  - Fallback 链：HRNet keypoint → HoughCircle → HoughLine

must_read_first:
  - manifest.json
  - plugin.py
  - detector_enhanced.py
  - configs/default.yaml
  - tests/
  - README.md
  - .agent_skills/

primary_capabilities:
  - 表计检测与定位
  - 关键点/刻度识别
  - OCR 数字读数
  - 多表类型适配（指针式/数字式/LED）

current_known_risks:
  - 量程、透视、眩光、OCR 边界需要现场级样本验证
  - 多子模型（detector / keypoint / OCR）需统一 model role 管理
  - 算法合同和测试基础较强，但缺真实试点回放集

must_have_ui:
  - itemized_results（逐表读数清单）
  - meter_type_panel（表类型识别结果）
  - reading_detail_panel（读数详情：量程/角度/置信度）
  - quality_panel（图像清晰度/角度偏差）
  - alarms
  - diagnostic_panel

must_not_do:
  - 不要把不同表计类型的读数逻辑硬编码在 plugin.py 中
  - 不要把 fallback 链的中间结果隐藏
  - 不要把 OCR 对 LED 的识别能力写成已充分验证

future_placeholders:
  - keypoint_refiner_placeholder
  - ocr_model_placeholder
  - scale_calibration_placeholder
  - perspective_correction_placeholder

acceptance_focus:
  - 作为第一批样板插件重点推进
  - 上传表计图片后输出完整读数 + 置信度 + 使用链路说明
  - 率先使用统一 model role + registry 管理
```

### ━━━━━ A 类：扩展观察对象 ━━━━━

#### 6) animal_detection

```text
[PLUGIN_CARD]
plugin_id: animal_detection
plugin_name: 动物入侵检测插件
plugin_family: visual_security
plugin_type: 小动物检测 + 热成像验证 + 行为/驱离联动
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/animal_detection
current_stage: hybrid
primary_input: image | thermal_image(optional)
primary_output: detections + species + alive_validation + risk + deterrent_suggestion + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - configs/default.yaml
  - standalone/
  - tests/
  - README.md
  - .agent_skills/
  - core/detector.py
  - core/tracker.py
  - core/thermal_validator.py
  - core/deterrent.py
  - core/event_schema.py
  - core/statistics.py
  - core/onnx_inference.py

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 tests
  - 已存在 README
  - 核心实现在 core/ 目录下，非 detector_enhanced.py 模式

must_read_first:
  - manifest.json
  - plugin.py
  - configs/default.yaml
  - README.md
  - tests/
  - standalone/
  - .agent_skills/
  - core/detector.py
  - core/tracker.py
  - core/thermal_validator.py
  - core/deterrent.py
  - core/event_schema.py

primary_capabilities:
  - animal detection
  - species classification
  - thermal validation
  - deterrent suggestion
  - intrusion statistics
  - behavior tracking

current_known_risks:
  - species / alive / deterrent 结果可能表达分散
  - 上传样本模式下需要详细说明"是否建议驱离"
  - 模型依赖：animal_yolov8, species_classifier, thermal_animal

must_have_ui:
  - itemized_results
  - species_panel
  - thermal_validation_panel
  - risk_panel
  - deterrent_panel
  - alarms

must_not_do:
  - 不要把热验证结果缺失时伪装成已验证活体
  - 不要夸大 species classifier 精度

future_placeholders:
  - species_refiner_placeholder
  - behavior_classifier_placeholder
  - deterrent_decision_model_placeholder

acceptance_focus:
  - 上传图片后逐项输出 species / risk / action
  - 热像缺失时必须明确降级说明
```

#### 7) bird_monitoring

```text
[PLUGIN_CARD]
plugin_id: bird_monitoring
plugin_name: 鸟类监控插件
plugin_family: visual_security
plugin_type: 鸟类识别 + 风险评估 + 驱离建议
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/bird_monitoring
current_stage: engineering_baseline
primary_input: image
primary_output: detections + species_guess + risk_assessment + deterrent_suggestion + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - detector.py
  - advanced_bird_detector.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 tests
  - 已存在 detector.py + advanced_bird_detector.py（双检测器）
  - 无 README 作为强事实源时，优先以 manifest + plugin + tests 为准

must_read_first:
  - manifest.json
  - plugin.py
  - detector.py
  - advanced_bird_detector.py
  - configs/default.yaml
  - tests/
  - standalone/
  - .agent_skills/

primary_capabilities:
  - bird detection
  - species identification
  - risk assessment
  - deterrent control suggestion

current_known_risks:
  - 物种识别能力可能未充分验证
  - 需要把"检测"与"风险评估"分层输出
  - 双检测器（detector.py vs advanced_bird_detector.py）主链路需确认

must_have_ui:
  - itemized_results
  - species_guess_panel
  - risk_panel
  - deterrent_panel
  - alarms

must_not_do:
  - 不要把 species identification 写成已高精度能力
  - 不要省略风险解释

future_placeholders:
  - species_embedding_placeholder
  - line_risk_estimator_placeholder
  - deterrent_policy_placeholder

acceptance_focus:
  - 上传图片后至少明确输出 bird / unknown_bird / review_required
  - 风险和驱离建议必须可解释
```

#### 8) fire_detection

```text
[PLUGIN_CARD]
plugin_id: fire_detection
plugin_name: 消防监测插件
plugin_family: visual_safety
plugin_type: 火焰/烟雾/火点检测 + 多传感器融合
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/fire_detection
current_stage: hybrid
primary_input: image | thermal_image(optional) | sensor_data(optional)
primary_output: fire/smoke detections + fire_level + fusion_confidence + suppression_actions + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - detector.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 tests
  - 已存在 detector.py
  - 模型依赖：fire_smoke_yolov8, fire_classification, thermal_anomaly

must_read_first:
  - manifest.json
  - plugin.py
  - detector.py
  - configs/default.yaml
  - tests/
  - standalone/
  - .agent_skills/

primary_capabilities:
  - fire detection
  - smoke detection
  - thermal anomaly detection
  - suppression suggestion
  - multi_sensor_fusion
  - evacuation_guidance

current_known_risks:
  - 视觉结果、热像结果、传感器结果需要统一融合表达
  - 安全告警类插件必须保证 detailed alarm 足够完整
  - 灭火/疏散建议需要明确标注为规则推理而非模型输出

must_have_ui:
  - itemized_results
  - fire_level_panel
  - fusion_panel
  - suppression_action_panel
  - alarms
  - diagnostic_panel

must_not_do:
  - 不要把灭火动作建议写成已验证的自动执行能力
  - 不要隐藏传感器数据缺失对融合置信度的影响
  - 安全类插件不允许沉默失败

future_placeholders:
  - fire_classifier_placeholder
  - smoke_density_estimator_placeholder
  - thermal_fire_fusion_placeholder

acceptance_focus:
  - 上传图片后输出 fire_level + 逐项检测清单
  - 每个检测结果必须包含 severity / confidence / reason / action
  - 融合置信度必须标注数据来源
```

### ━━━━━ B 类：时序传感 / 数值异常 ━━━━━

#### 9) acoustic_monitoring

```text
[PLUGIN_CARD]
plugin_id: acoustic_monitoring
plugin_name: 声学监测插件
plugin_family: audio_timeseries
plugin_type: 局部放电超声检测 + 异常声学特征识别
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/acoustic_monitoring
current_stage: engineering_baseline
primary_input: audio(array) + device_id + roi_id
primary_output: anomaly_detected + anomaly_type + anomaly_score + confidence + severity + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - detector.py
  - analyzer.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone（含 audio_manager.py）
  - 已存在 detector.py + analyzer.py（双入口）
  - 已存在 tests
  - 模型依赖：audio_anomaly_transformer, ultrasonic_pd_detector

must_read_first:
  - manifest.json
  - plugin.py
  - detector.py
  - analyzer.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

primary_capabilities:
  - partial_discharge_detection（局放检测）
  - acoustic_monitoring（声学异常监测）

current_known_risks:
  - 音频窗口化处理与时序对齐可能未完整
  - TemporalTrainer 当前以计划生成为主
  - 需要明确输出 time_range / window_id 级别的逐项结果
  - 基线评测标准（false_alarm_rate / detection_latency）未固化

must_have_ui:
  - itemized_anomalies（按时间窗口逐项）
  - spectrogram_panel
  - anomaly_score_panel
  - trend_panel
  - alarms

must_not_do:
  - 不要把音频 transformer 模型的能力写成已验证稳定
  - 不要隐藏窗口化丢失的上下文信息
  - 不要把规则阈值硬编码在 detector.py 主链路中

future_placeholders:
  - pd_classifier_placeholder
  - acoustic_feature_extractor_placeholder
  - temporal_context_model_placeholder

acceptance_focus:
  - B 类首批试点插件
  - 上传音频后输出逐窗口异常检测结果
  - 至少完成一个基于规则/统计的 baseline 闭环
```

#### 10) gas_detection

```text
[PLUGIN_CARD]
plugin_id: gas_detection
plugin_name: 气体检测插件
plugin_family: structured_timeseries
plugin_type: 多气体浓度监测 + 泄漏检测 + 趋势预测
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/gas_detection
current_stage: engineering_baseline
primary_input: device_id + gas_readings(SF6/H2/CO/C2H2/CH4/C2H4/C2H6) + environmental(temperature/humidity/pressure)
primary_output: gas_status + predictions + leak_detection + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - analyzer.py
  - predictor.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 analyzer.py + predictor.py（分析 + 预测双模块）
  - 已存在 tests
  - 模型依赖：sf6_forecast(LSTM), multi_gas_forecast(Transformer), equipment_health_trend

must_read_first:
  - manifest.json
  - plugin.py
  - analyzer.py
  - predictor.py
  - configs/default.yaml
  - tests/
  - standalone/
  - .agent_skills/

primary_capabilities:
  - gas_concentration_monitoring
  - leakage_detection
  - 支持气体：SF6, H2, CO, C2H2, CH4, C2H4, C2H6

current_known_risks:
  - 预测模型（LSTM/Transformer）当前可能以占位/计划为主
  - 多气体关联分析需要真实历史数据验证
  - 泄漏检测阈值需要配置化

must_have_ui:
  - gas_concentration_panel（多气体浓度仪表盘）
  - trend_panel（趋势预测曲线）
  - leak_alert_panel
  - itemized_anomalies
  - alarms
  - diagnostic_panel

must_not_do:
  - 不要把预测模型精度写成已验证
  - 不要隐藏环境因素（温湿度/气压）对判别的影响
  - 气体阈值不允许硬编码

future_placeholders:
  - sf6_forecast_model_placeholder
  - multi_gas_correlator_placeholder
  - leak_source_estimator_placeholder

acceptance_focus:
  - B 类首批试点插件
  - 上传气体数据后输出逐气体种类的浓度/趋势/异常判定
  - 泄漏检测结果需包含 severity / confidence / recommended_action
```

#### 11) device_monitoring

```text
[PLUGIN_CARD]
plugin_id: device_monitoring
plugin_name: 设备状态监控插件
plugin_family: structured_timeseries
plugin_type: 设备健康指数 + 故障预测 + 异常检测
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/device_monitoring
current_stage: engineering_baseline
primary_input: device_readings(array: cpu_temp/cpu_usage/memory_usage/network_quality/...)
primary_output: device_status + health_index + anomaly_score + predicted_failure + recommendations

core_entry_files:
  - manifest.json
  - plugin.py
  - detector.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 detector.py
  - 已存在 tests
  - 模型依赖：device_anomaly_autoencoder, failure_predictor_lstm, health_calibrator

must_read_first:
  - manifest.json
  - plugin.py
  - detector.py
  - configs/default.yaml
  - tests/
  - standalone/
  - .agent_skills/

primary_capabilities:
  - device_status_monitoring
  - health_index_calculation
  - fault_prediction
  - anomaly_detection
  - maintenance_scheduling

current_known_risks:
  - 健康指数计算逻辑可能为规则权重，需说明非模型产出
  - 故障预测模型（LSTM）可能为占位
  - 多指标综合判断需要明确权重来源

must_have_ui:
  - device_dashboard（设备状态总览）
  - health_index_panel
  - anomaly_timeline
  - prediction_panel
  - maintenance_schedule_panel
  - alarms

must_not_do:
  - 不要把规则权重计算伪装成 ML 模型输出
  - 不要把故障预测精度写成已验证
  - 健康指数权重不允许硬编码

future_placeholders:
  - anomaly_autoencoder_placeholder
  - failure_predictor_placeholder
  - health_calibrator_placeholder

acceptance_focus:
  - 上传设备指标数据后输出健康指数 + 异常清单 + 维护建议
  - 每条建议需包含 priority / evidence / action
```

#### 12) temperature_monitoring

```text
[PLUGIN_CARD]
plugin_id: temperature_monitoring
plugin_name: 温度监测插件
plugin_family: structured_timeseries
plugin_type: 热像分析 + 热点检测 + 温度趋势预测
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/temperature_monitoring
current_stage: engineering_baseline
primary_input: thermal_frame(image/matrix) + sensor_readings(array) + zones
primary_output: heatmap + max_temp/min_temp/avg_temp + hotspots + trend + predictions + alarms

core_entry_files:
  - manifest.json
  - plugin.py
  - detector.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 detector.py
  - 已存在 tests
  - 模型依赖：thermal_anomaly_cnn, temp_lstm_predictor

must_read_first:
  - manifest.json
  - plugin.py
  - detector.py
  - configs/default.yaml
  - tests/
  - standalone/
  - .agent_skills/

primary_capabilities:
  - thermal_imaging
  - hotspot_detection
  - temperature_trend_analysis
  - heatmap_generation
  - temperature_prediction

current_known_risks:
  - 热像分析与传感器读数的融合逻辑需明确
  - 温度预测模型可能为占位
  - 不同设备类型的温度阈值标准需配置化

must_have_ui:
  - heatmap_panel
  - hotspot_list（逐热点清单）
  - trend_chart（温度趋势图）
  - prediction_panel
  - zone_summary_panel
  - alarms

must_not_do:
  - 不要把温度阈值硬编码
  - 不要把 CNN 热异常检测写成已验证能力
  - 不要隐藏传感器数据与热像数据的不一致

future_placeholders:
  - thermal_anomaly_cnn_placeholder
  - temp_predictor_placeholder
  - cross_device_correlation_placeholder

acceptance_focus:
  - 上传热像/温度数据后输出热点清单 + 趋势分析 + 异常判定
  - 每个热点需包含 location / temperature / severity / trend / action
```

#### 13) action_event_monitoring

```text
[PLUGIN_CARD]
plugin_id: action_event_monitoring
plugin_name: 动作事件监控插件
plugin_family: structured_timeseries
plugin_type: 协议订阅 + 动作/信号变位监测 + SOE记录
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/action_event_monitoring
current_stage: engineering_baseline
primary_input: protocol_subscriptions(OPC_UA/MQTT/IEC104)
primary_output: action_events + signal_changes + soe_records

core_entry_files:
  - manifest.json
  - plugin.py
  - configs/default.yaml
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 最小化实现：无 standalone / 无 tests / 无 detector
  - 重度依赖 platform_core 协议适配层
  - manifest 较精简

must_read_first:
  - manifest.json
  - plugin.py
  - configs/default.yaml
  - .agent_skills/

primary_capabilities:
  - action_monitoring
  - soe_collection
  - signal_change_detection
  - protocol_subscription (OPC UA / MQTT / IEC104)

current_known_risks:
  - 当前为最小化实现，缺少 tests 和 standalone
  - 强依赖 platform_core 的协议适配
  - 事件序列识别能力可能仅为契约占位

must_have_ui:
  - event_timeline（事件时间线）
  - signal_change_panel
  - soe_table
  - protocol_status_panel
  - alarms

must_not_do:
  - 不要把协议适配层逻辑放入 plugin.py
  - 不要把事件序列识别能力写成已验证
  - 不要硬编码协议地址

future_placeholders:
  - event_sequence_recognizer_placeholder
  - action_pattern_classifier_placeholder
  - soe_anomaly_detector_placeholder

acceptance_focus:
  - 能接收协议数据并输出结构化事件清单
  - 每条事件需包含 timestamp / event_type / source / severity
  - 补齐最小 tests
```

### ━━━━━ C 类：融合诊断 ━━━━━

#### 14) multimodal_fusion

```text
[PLUGIN_CARD]
plugin_id: multimodal_fusion
plugin_name: 多模态融合诊断插件
plugin_family: fusion
plugin_type: 多模态数据融合 + 规则+模型混合诊断 + 缺模态降级
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/multimodal_fusion
current_stage: engineering_baseline
primary_input: device_id + modalities(visual/thermal/acoustic/ultrasonic/gas/hyperspectral/vibration)
primary_output: overall_status + confidence + detections + modality_contributions + diagnostic_report

core_entry_files:
  - manifest.json
  - plugin.py
  - fusion_engine.py
  - fusion_engine_enhanced.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 fusion_engine.py + fusion_engine_enhanced.py（双引擎）
  - 已存在 tests
  - 支持 7 种模态
  - 已有诊断规则：transformer overheating / insulation degradation / poor contact
  - 模型依赖：multimodal_fusion, multimodal_feature_fusion, multimodal_decision_fusion

must_read_first:
  - manifest.json
  - plugin.py
  - fusion_engine.py
  - fusion_engine_enhanced.py
  - configs/default.yaml
  - tests/
  - standalone/
  - .agent_skills/

primary_capabilities:
  - multi_modality_fusion（规则融合 — 当前可交付）
  - diagnostic_rules（变压器过热 / 绝缘劣化 / 接触不良）
  - missing_modality_degradation

current_known_risks:
  - 高度依赖上游插件成熟度
  - MultimodalTrainer 当前仍偏计划式
  - modality → plugin_id 映射可能存在不一致
  - 双引擎（fusion_engine vs fusion_engine_enhanced）主链路需确认

must_have_ui:
  - overall_status_panel
  - modality_contribution_chart
  - evidence_chain_panel
  - diagnostic_rule_panel（标注 rule-based vs model-based）
  - missing_modality_warning
  - alarms

must_not_do:
  - 不要在输入来源不稳定时追求复杂特征融合模型
  - 不要把模型融合写成当前已验证能力
  - 不要让插件依赖图与训练配置图分叉

future_placeholders:
  - feature_fusion_model_placeholder
  - decision_fusion_model_placeholder
  - cross_modality_attention_placeholder

acceptance_focus:
  - 当前定位为稳定 orchestrator，不追求重训练
  - 规则融合可稳定运行
  - 缺模态降级策略固化且可测试
  - modality → plugin_id 映射明确
```

### ━━━━━ 其他完整插件 ━━━━━

#### 15) indoor_fence

```text
[PLUGIN_CARD]
plugin_id: indoor_fence
plugin_name: 室内电子围栏插件
plugin_family: spatial_security
plugin_type: 人员检测 + 多目标跟踪 + 区域入侵 + 授权校验 + 激光雷达围栏
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/indoor_fence
current_stage: engineering_baseline
primary_input: camera_frame + lidar_scan(2D) + optional(depth/IMU/UWB)
primary_output: person_detections + tracking_ids + zone_intrusion_events + authorization_violations

core_entry_files:
  - manifest.json
  - plugin.py
  - detector.py
  - protocols.py
  - configs/default.yaml
  - standalone/
  - tests/
  - README.md
  - .agent_skills/
  - core/
  - adapters/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 tests
  - 已存在 README
  - 已存在 core/ 和 adapters/ 目录
  - V2.1.0 stable；保留 V3 研究级跟踪/融合路径
  - 空间感知类，不要硬套 bbox 缺陷模板

must_read_first:
  - manifest.json
  - plugin.py
  - detector.py
  - protocols.py
  - configs/default.yaml
  - core/
  - tests/
  - README.md
  - .agent_skills/

primary_capabilities:
  - person_detection
  - multi_target_tracking
  - zone_intrusion
  - authorization_check
  - lidar_fence

current_known_risks:
  - V2/V3 双版本共存可能导致主链路混淆
  - 空间事件输出格式不同于视觉缺陷类
  - 授权校验需要对接外部权限系统

must_have_ui:
  - zone_map_panel（区域地图 + 入侵可视化）
  - person_tracking_panel
  - intrusion_event_list
  - authorization_status_panel
  - alarms

must_not_do:
  - 不要硬套普通 bbox 缺陷模板
  - 不要把 V3 研究路径的能力写成当前稳定能力
  - 输出轨迹、区域事件、授权校验、空间关系

future_placeholders:
  - reid_model_placeholder
  - fusion_tracker_placeholder
  - behavior_analyzer_placeholder

acceptance_focus:
  - 上传视频帧/雷达数据后输出区域入侵事件清单
  - 每条事件需包含 zone_id / person_id / timestamp / authorization_status
```

#### 16) slam_mapping

```text
[PLUGIN_CARD]
plugin_id: slam_mapping
plugin_name: SLAM建图插件
plugin_family: spatial_security
plugin_type: 点云处理 + 3D建图 + 路径规划 + 地面沉降检测
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/slam_mapping
current_stage: engineering_baseline
primary_input: lidar/radar point_cloud
primary_output: 3d_map + device_positions + path_planning_routes + ground_settlement

core_entry_files:
  - manifest.json
  - plugin.py
  - semantic_slam_plugin.py
  - configs/default.yaml
  - standalone/
  - tests/
  - .agent_skills/

current_file_structure_notes:
  - 已存在 .agent_skills
  - 已存在 standalone
  - 已存在 tests
  - 已存在 semantic_slam_plugin.py（语义SLAM实现）

must_read_first:
  - manifest.json
  - plugin.py
  - semantic_slam_plugin.py
  - configs/default.yaml
  - tests/
  - .agent_skills/

primary_capabilities:
  - point_cloud_processing
  - 3d_mapping
  - path_planning
  - ground_settlement_detection

current_known_risks:
  - SLAM 实时性需求与离线分析需区分
  - 地面沉降检测可能为规则推理
  - 空间类输出格式需与 indoor_fence 协调

must_have_ui:
  - 3d_map_viewer
  - path_planning_panel
  - settlement_alert_panel
  - device_position_panel
  - alarms

must_not_do:
  - 不要把 SLAM 实时能力写成已验证
  - 不要硬套视觉缺陷模板

future_placeholders:
  - semantic_slam_model_placeholder
  - settlement_predictor_placeholder
  - loop_closure_model_placeholder

acceptance_focus:
  - 上传点云数据后输出 3D 地图 + 设备定位 + 异常区域标注
```

#### 17) hyperspectral_detection

```text
[PLUGIN_CARD]
plugin_id: hyperspectral_detection
plugin_name: 高光谱检测插件
plugin_family: visual_defect
plugin_type: 高光谱成像分析 + 光谱缺陷检测
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/hyperspectral_detection
current_stage: engineering_baseline
primary_input: hyperspectral_data(224_bands, 400-2500nm)
primary_output: spectral_analysis_results + defect_detections

core_entry_files:
  - manifest.json
  - plugin.py

current_file_structure_notes:
  - 最小化实现
  - 无 standalone / 无 tests / 无 .agent_skills / 无 configs 目录
  - 默认配置内嵌：224 bands, PCA to 30 components

must_read_first:
  - manifest.json
  - plugin.py

primary_capabilities:
  - hyperspectral_analysis
  - defect_detection (spectral domain)

current_known_risks:
  - 极简实现，缺少基本治理结构
  - 高光谱数据格式标准化未明确
  - 与视觉类插件的缺陷语义需要对齐

must_have_ui:
  - spectral_analysis_panel
  - band_selection_panel
  - defect_overlay_panel
  - alarms

must_not_do:
  - 不要把光谱分析写成已充分验证的缺陷检测能力
  - 不要忽略数据格式标准化问题

future_placeholders:
  - spectral_feature_extractor_placeholder
  - hyperspectral_classifier_placeholder

acceptance_focus:
  - 先补齐最小治理底座（configs/ + tests/）
  - 上传高光谱数据后输出光谱特征 + 异常标注
```

### ━━━━━ L0 占位态 ━━━━━

#### 18) thermal（占位态）

```text
[PLUGIN_CARD]
plugin_id: thermal
plugin_name: 热像分析工具（占位）
plugin_family: placeholder
plugin_type: 热像数据分析适配器
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/thermal
current_stage: placeholder_L0
primary_input: thermal_image
primary_output: N/A（尚无完整输出定义）

current_files:
  - __init__.py
  - enhanced_thermal_analyzer.py
  - README.md

missing_files:
  - plugin.py（缺失）
  - manifest.json（缺失）
  - configs/（缺失）
  - tests/（缺失）
  - standalone/（缺失）

must_read_first:
  - enhanced_thermal_analyzer.py
  - README.md

current_status:
  - 仅为热像分析工具/适配器
  - 不是完整插件
  - 不生成 .agent_skills/

must_not_do:
  - 不要生成 .agent_skills/（升级报告明确禁止）
  - 不要做重算法开发
  - 不要纳入第一批资源竞争
  - 不要把 enhanced_thermal_analyzer.py 硬套为主检测链路

upgrade_scope:
  - 仅补最小四件套骨架
  - plugin.py + manifest.json + configs/default.yaml + tests/test_plugin.py
```

#### 19) radar（占位态）

```text
[PLUGIN_CARD]
plugin_id: radar
plugin_name: 毫米波雷达适配器（占位）
plugin_family: placeholder
plugin_type: 毫米波雷达数据适配
plugin_path: /Users/ronan/Desktop/DarkBreaker/plugins/radar
current_stage: placeholder_L0
primary_input: mmwave_radar_data
primary_output: N/A（尚无完整输出定义）

current_files:
  - __init__.py
  - mmwave_radar_adapter.py
  - README.md

missing_files:
  - plugin.py（缺失）
  - manifest.json（缺失）
  - configs/（缺失）
  - tests/（缺失）
  - standalone/（缺失）

must_read_first:
  - mmwave_radar_adapter.py
  - README.md

current_status:
  - 仅为雷达数据适配/转换工具
  - 不是完整插件
  - 不生成 .agent_skills/

must_not_do:
  - 不要生成 .agent_skills/（升级报告明确禁止）
  - 不要做重算法开发
  - 不要纳入第一批资源竞争
  - 不要把 mmwave_radar_adapter.py 硬套为主检测链路

upgrade_scope:
  - 仅补最小四件套骨架
  - plugin.py + manifest.json + configs/default.yaml + tests/test_plugin.py
```

---

## 第四部分：样本上传判别模式 Prompt（专用版，可选附加）

适用场景：没有充分训练数据，但希望先用已收集样本上传后完成识别、缺陷输出、详细告警。

```text
你当前处理的是 DarkBreaker 插件的"样本上传辅助判别模式"升级任务。

目标不是伪造真实模型精度，而是在当前缺乏数据训练和深度学习条件下，先把插件升级到如下可交付状态：
1. 用户可上传样本（图片 / 热像 / 音频 / 时序数据）
2. 插件可先完成输入质量评估
3. 插件可输出逐项识别/异常判别清单
4. 插件可对每个结果给出详细告警信息
5. 插件可对误判风险和不确定结果给出 review_required 提示
6. 插件可为未来真实模型训练预留结构化占位字段

【当前阶段定位】
- 当前阶段是"工程可交付优先"
- 当前阶段允许：规则 / fallback / 传统算法 / 弱模型 / AI辅助判别
- 当前阶段不允许：
  - 把不确定结果包装成高精度真实模型结论
  - 夸大未验证能力
  - 隐藏失败原因
  - 沉默返回空结果

【必须具备的输出能力】

A. 样本级摘要
   - sample_id, sample_type, input_quality, processing_status
   - overall_risk, review_required

B. 逐项结果清单
   - item_id, roi_id / zone_id / time_window_id
   - pred_label, candidate_labels, confidence, severity
   - reason_code, review_status, suggested_action, evidence

C. 详细诊断
   - primary_judgement, secondary_judgement, score_gap
   - conflict_type, limitation_note, diagnostic_reason

D. 详细告警
   - level, title, message
   - related_item_ids, recommended_action

E. 未来训练占位符
   - hard_negative_candidate, hard_positive_candidate
   - suggested_label_for_dataset, annotation_status

【实现原则】
1. plugin.py 继续只做适配层。
2. 主识别和判别逻辑放 detector/analyzer 层。
3. 所有阈值必须配置化。
4. 所有不确定结果必须可解释。
5. 所有新增逻辑必须测试覆盖。
6. standalone/UI 如存在，必须能展示逐项清单和错误判别信息。

【最终要求】
请给出：
1. 当前插件在"样本上传判别模式"下的升级方案
2. 需要修改的文件
3. 需要新增的输出字段
4. 需要新增的 UI 组件
5. 需要新增的测试
6. 哪些是当前阶段能完成的
7. 哪些必须等未来真实训练模型补齐
```

---

## 第五部分：快速使用指南

### 场景 1：对第一批 A 类插件做能力收口（当前推荐）

```
Phase 2 Overlay + 母版 Prompt + [busbar/capacitor/switch/transformer/meter] 参数卡
```

### 场景 2：对第一批 A 类插件做输出契约统一

```
Phase 4 Overlay + 母版 Prompt + [busbar/capacitor/switch/transformer/meter] 参数卡
```

### 场景 3：对 A 类插件做样本上传判别模式升级

```
Phase 3 Overlay + 母版 Prompt + 样本上传判别 Prompt + 对应插件参数卡
```

### 场景 4：启动 B 类时序插件基线闭环

```
Phase 6 Overlay + 母版 Prompt + [acoustic/gas/device/temperature/action_event] 参数卡
```

### 场景 5：C 类融合插件契约收口

```
Phase 7 Overlay + 母版 Prompt + multimodal_fusion 参数卡
```

### 场景 6：L0 占位插件补骨架

```
Phase 8 Overlay + [thermal/radar] 参数卡
（不使用母版 Prompt，直接用 Phase 8 Overlay 即可）
```

### 执行排期对照

| 周次 | Phase | 目标 | 适用插件 |
|------|-------|------|----------|
| Week 1 | Phase 1 + 2 | 去冗余 + 能力收口 | 全量（Phase 1）/ A类5个（Phase 2） |
| Week 2 | Phase 2 + 4 | 文档对齐 + 输出契约 | A类5个 |
| Week 3 | Phase 3 | 最小回放集 | A类5个 |
| Week 4 | Phase 5 | training 闭环 | A类5个 |
| Week 5+ | Phase 6 + 7 | B类基线 + C类契约 | B类5个 + multimodal_fusion |
| 后续 | Phase 8 | L0骨架 | thermal + radar |
