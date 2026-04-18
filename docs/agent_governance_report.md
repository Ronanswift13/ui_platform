# DarkBreaker Agent Governance Report

> 自动生成 — 扫描时间基于当前目录状态

## 1. 总体统计

| 指标 | 值 |
|------|-----|
| 扫描对象总数 | 21 |
| HF 等级数量 | 4 |
| STD 等级数量 | 15 |
| MIN 等级数量 | 0 |
| L0 等级数量 | 2 |

**按对象类型分布:**

| 类型 | 数量 | 成员 |
|------|------|------|
| placeholder | 2 | radar, thermal |
| plugin-enhanced-detector | 5 | busbar_inspection, capacitor_inspection, meter_reading, switch_inspection, transformer_inspection |
| plugin-light-detector | 6 | acoustic_monitoring, bird_monitoring, device_monitoring, fire_detection, indoor_fence, temperature_monitoring |
| plugin-minimal | 1 | action_event_monitoring |
| plugin-service-integration | 5 | animal_detection, gas_detection, hyperspectral_detection, multimodal_fusion, slam_mapping |
| root | 1 | root |
| ui | 1 | ui |

## 2. 各对象治理等级表

| 对象 | 范围 | 类型 | 等级 | agent_skills | commands | 脚本 | HF候选 | 工作量 | 下一步 | 关键文件 | 目录 |
|------|------|------|------|-------------|----------|------|----------|--------|----------|----------|------|
| root | root | root | **STD** | 9/9 | — | 0/5 | no | high | add/update .claude/commands | 3/7 | 3/4 |
| ui | ui | ui | **STD** | 9/9 | ✓ | 2/5 | no | high | align 08_task_routing.md with scripts | 2/7 | 0/4 |
| acoustic_monitoring | plugin | plugin-light-detector | **STD** | 9/9 | ✓ | 0/5 | no | medium | normalize run_quality_gate.sh | 3/7 | 3/4 |
| action_event_monitoring | plugin | plugin-minimal | **STD** | 9/9 | ✓ | 1/5 | no | high | align 08_task_routing.md with scripts | 2/7 | 1/4 |
| animal_detection | plugin | plugin-service-integration | **HF** | 9/9 | ✓ | 4/5 | no | low | align 08_task_routing.md with scripts | 5/7 | 4/4 |
| bird_monitoring | plugin | plugin-light-detector | **STD** | 9/9 | ✓ | 0/5 | no | medium | normalize run_quality_gate.sh | 3/7 | 3/4 |
| busbar_inspection | plugin | plugin-enhanced-detector | **HF** | 9/9 | ✓ | 4/5 | no | low | align 08_task_routing.md with scripts | 6/7 | 3/4 |
| capacitor_inspection | plugin | plugin-enhanced-detector | **STD** | 9/9 | ✓ | 0/5 | yes | high | normalize run_quality_gate.sh | 3/7 | 4/4 |
| device_monitoring | plugin | plugin-light-detector | **STD** | 9/9 | ✓ | 0/5 | no | medium | normalize run_quality_gate.sh | 3/7 | 3/4 |
| fire_detection | plugin | plugin-light-detector | **STD** | 9/9 | ✓ | 0/5 | no | medium | normalize run_quality_gate.sh | 3/7 | 3/4 |
| gas_detection | plugin | plugin-service-integration | **STD** | 9/9 | ✓ | 0/5 | no | high | normalize run_quality_gate.sh | 2/7 | 2/4 |
| hyperspectral_detection | plugin | plugin-service-integration | **STD** | 9/9 | ✓ | 0/5 | no | high | normalize run_quality_gate.sh | 2/7 | 2/4 |
| indoor_fence | plugin | plugin-light-detector | **HF** | 9/9 | ✓ | 4/5 | no | low | align 08_task_routing.md with scripts | 5/7 | 3/4 |
| meter_reading | plugin | plugin-enhanced-detector | **HF** | 9/9 | ✓ | 4/5 | no | low | align 08_task_routing.md with scripts | 5/7 | 4/4 |
| multimodal_fusion | plugin | plugin-service-integration | **STD** | 9/9 | ✓ | 0/5 | no | high | normalize run_quality_gate.sh | 2/7 | 2/4 |
| radar | plugin | placeholder | **L0** | — | — | 0/5 | no | high | stabilize plugin core facts before promotion | 1/7 | 0/4 |
| slam_mapping | plugin | plugin-service-integration | **STD** | 9/9 | ✓ | 0/5 | no | high | normalize run_quality_gate.sh | 2/7 | 2/4 |
| switch_inspection | plugin | plugin-enhanced-detector | **STD** | 9/9 | ✓ | 0/5 | yes | high | normalize run_quality_gate.sh | 4/7 | 4/4 |
| temperature_monitoring | plugin | plugin-light-detector | **STD** | 9/9 | ✓ | 0/5 | no | medium | normalize run_quality_gate.sh | 3/7 | 3/4 |
| thermal | plugin | placeholder | **L0** | — | — | 0/5 | no | high | stabilize plugin core facts before promotion | 1/7 | 0/4 |
| transformer_inspection | plugin | plugin-enhanced-detector | **STD** | 9/9 | ✓ | 0/5 | yes | high | normalize run_quality_gate.sh | 4/7 | 4/4 |

## 3. 缺失项统计

### 3.1 缺失 agent_skills 编号

- radar (缺 00, 01, 02, 03, 04, 05, 06, 07, 08)
- thermal (缺 00, 01, 02, 03, 04, 05, 06, 07, 08)

### 3.2 缺失 .claude/commands/

共 3 个对象: root, radar, thermal

### 3.3 缺失质量门禁脚本

| 脚本 | 缺失对象数 | 缺失列表 |
|------|-----------|----------|
| run_targeted_tests.sh | 17 | root, ui, acoustic_monitoring, action_event_monitoring, bird_monitoring, capacitor_inspection, device_monitoring, fire_detection, gas_detection, hyperspectral_detection, multimodal_fusion, radar, slam_mapping, switch_inspection, temperature_monitoring, thermal, transformer_inspection |
| run_regression_tests.sh | 17 | root, ui, acoustic_monitoring, action_event_monitoring, bird_monitoring, capacitor_inspection, device_monitoring, fire_detection, gas_detection, hyperspectral_detection, multimodal_fusion, radar, slam_mapping, switch_inspection, temperature_monitoring, thermal, transformer_inspection |
| run_quality_gate.sh | 16 | root, acoustic_monitoring, action_event_monitoring, bird_monitoring, capacitor_inspection, device_monitoring, fire_detection, gas_detection, hyperspectral_detection, multimodal_fusion, radar, slam_mapping, switch_inspection, temperature_monitoring, thermal, transformer_inspection |
| collect_root_cause.sh | 16 | root, acoustic_monitoring, action_event_monitoring, bird_monitoring, capacitor_inspection, device_monitoring, fire_detection, gas_detection, hyperspectral_detection, multimodal_fusion, radar, slam_mapping, switch_inspection, temperature_monitoring, thermal, transformer_inspection |
| run_sanity_checks.sh | 20 | root, ui, acoustic_monitoring, animal_detection, bird_monitoring, busbar_inspection, capacitor_inspection, device_monitoring, fire_detection, gas_detection, hyperspectral_detection, indoor_fence, meter_reading, multimodal_fusion, radar, slam_mapping, switch_inspection, temperature_monitoring, thermal, transformer_inspection |

### 3.4 缺失关键文件

| 文件 | 缺失对象数 |
|------|-----------|
| plugin.py | 4 |
| detector.py | 15 |
| detector_enhanced.py | 16 |
| manifest.json | 4 |
| README.md | 14 |
| PROJECT_CARD.md | 15 |
| CLAUDE.md | 15 |

## 4. HF 候选快速晋升规则

### 4.1 固化规则

当前优先固化一条高价值筛选规则：

- `governance_level == STD`

- `object_type == plugin-enhanced-detector`

- `has_agent_skills == true`

- 且对象自身目录具备真实核心事实源：`plugin.py` / `detector_enhanced.py` / `manifest.json` / `tests/` / `configs/`

- 满足以上条件 -> 标记为 `HF upgrade candidate`


### 4.2 HF 升级候选

| 对象 | 类型 | 工作量 | 推荐下一步 | 原因 |
|------|------|--------|------------|------|
| capacitor_inspection | plugin-enhanced-detector | high | normalize run_quality_gate.sh | STD + plugin-enhanced-detector + has_agent_skills=true + 核心事实源齐备 (plugin.py / detector_enhanced.py / manifest.json / tests / configs) |
| switch_inspection | plugin-enhanced-detector | high | normalize run_quality_gate.sh | STD + plugin-enhanced-detector + has_agent_skills=true + 核心事实源齐备 (plugin.py / detector_enhanced.py / manifest.json / tests / configs) |
| transformer_inspection | plugin-enhanced-detector | high | normalize run_quality_gate.sh | STD + plugin-enhanced-detector + has_agent_skills=true + 核心事实源齐备 (plugin.py / detector_enhanced.py / manifest.json / tests / configs) |

### 4.3 虽然是 STD，但当前不适合直接升 HF 的对象

| 对象 | 类型 | 当前不宜直升原因 | 建议下一步 |
|------|------|------------------|------------|
| acoustic_monitoring | plugin-light-detector | 当前对象类型为 plugin-light-detector，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |
| action_event_monitoring | plugin-minimal | 当前对象类型为 plugin-minimal，不适用 enhanced detector 型 HF 快速晋升规则 | align 08_task_routing.md with scripts |
| bird_monitoring | plugin-light-detector | 当前对象类型为 plugin-light-detector，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |
| device_monitoring | plugin-light-detector | 当前对象类型为 plugin-light-detector，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |
| fire_detection | plugin-light-detector | 当前对象类型为 plugin-light-detector，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |
| gas_detection | plugin-service-integration | 当前对象类型为 plugin-service-integration，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |
| hyperspectral_detection | plugin-service-integration | 当前对象类型为 plugin-service-integration，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |
| multimodal_fusion | plugin-service-integration | 当前对象类型为 plugin-service-integration，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |
| slam_mapping | plugin-service-integration | 当前对象类型为 plugin-service-integration，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |
| temperature_monitoring | plugin-light-detector | 当前对象类型为 plugin-light-detector，不适用 enhanced detector 型 HF 快速晋升规则 | normalize run_quality_gate.sh |

## 5. 推荐下一步动作

### L0 占位态对象

以下对象处于占位状态，需要评估是否计划激活:

- **radar** (placeholder)
- **thermal** (placeholder)

### 缺失 README.md 的插件

acoustic_monitoring, action_event_monitoring, bird_monitoring, capacitor_inspection, device_monitoring, fire_detection, gas_detection, hyperspectral_detection, indoor_fence, meter_reading, multimodal_fusion, slam_mapping, temperature_monitoring

### 缺失 PROJECT_CARD.md 的插件

acoustic_monitoring, action_event_monitoring, bird_monitoring, capacitor_inspection, device_monitoring, fire_detection, gas_detection, hyperspectral_detection, multimodal_fusion, radar, slam_mapping, switch_inspection, temperature_monitoring, thermal, transformer_inspection

## 6. 后续复用说明

### 可供 agent_task_router 复用的字段

- `object_type`: 直接作为路由分类依据，决定任务分发策略

- `governance_level`: 决定任务可接受的复杂度上限 (L0 不接受任何写操作)

- `has_agent_skills` + `agent_skills_files`: 判断是否具备上下文知识库

- `scripts_present`: 决定能否执行自动化质量门禁


### 可供 sync_agent_commands.py 复用的字段

- `has_claude_commands`: 识别哪些对象需要同步 command 模板

- `governance_level == 'STD'` 的对象: 优先批量补齐 commands

- `scripts_present` / `scripts_missing`: 决定 command 中可引用的脚本

- `hf_upgrade_candidate` / `estimated_upgrade_effort`: 识别哪些对象值得优先补 command 与脚本

- `recommended_next_action`: 直接决定同步器先补 commands 还是先补脚本/文档


### 为什么 enhanced detector 型插件是当前最优晋升对象

1. 这类对象通常已经有 `plugin.py + detector_enhanced.py + manifest.json` 这组稳定事实源，结构最适合模板化治理升级

2. 一旦同时具备 `tests/` 与 `configs/`，说明它已经有最基本的运行、验证与配置边界，补 HF 的收益高于重新补骨架

3. 它们从 STD 升到 HF 往往不需要重写算法，只需要把 commands、脚本和 task routing 对齐


### 为什么这条规则对后续 sync_agent_commands.py / sync_plugin_template.py 有帮助

1. 规则只依赖扫描字段和对象自身目录事实，适合脚本稳定复用，不依赖人工记忆插件名字

2. `hf_upgrade_candidate` 可直接作为同步优先级过滤器，减少对不成熟对象误下发 commands/template 的风险

3. `recommended_next_action` 让同步器可以先做最值钱的一步：补 commands、补质量门禁脚本、或修正文档/卡片不一致


### 如何判断"下一个最值得升级的插件"

1. 先看 `hf_upgrade_candidate == yes`

2. 再按 `estimated_upgrade_effort` 从 `low -> medium -> high` 排序

3. 同等条件下，优先选择 `recommended_next_action` 能直接由同步脚本完成的对象

4. 虽然是 STD 但缺少核心事实源、缺 tests/configs、或只是 service-integration 骨架的对象，暂不直升 HF
