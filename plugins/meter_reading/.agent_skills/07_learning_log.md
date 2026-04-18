# 07_learning_log

用于记录本插件的重要故障、根因、修复与可回灌经验。每次 `/repair` 或重大质量问题修复后必须追加。

## Entry Template

- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

- Date: 2026-03-10
- Context: 为 `meter_reading` 首次补齐 agentic 骨架
- Symptom: 核心算法已经较大，但缺少契约、脚本与自动化测试入口
- Root cause: 项目从原型直接进入功能迭代，治理骨架搭建滞后
- Fix: 建立 `00~07` skill、测试目录与基础脚本
- Prevention: 核心算法一旦进入长期维护状态，优先建设契约与测试，而不是继续堆功能
- Follow-up: 后续所有实现入口统一收口到 `08_task_routing.md`

---

- Date: 2026-03-19
- Context: V3.1 合约合规迭代
- Symptom: 状态集、OCR 清洗、量程缺失、metadata 输出与文档存在偏差
- Root cause: 文档先行演进，但未用测试把状态机和字段集合锁死
- Fix: 统一三态、补齐 metadata、移除幻象默认值、实现预处理与清洗函数
- Prevention: 枚举、metadata、fallback_level、OCR 规则都必须有独立测试锚点
- Follow-up: 将“枚举漂移”“metadata 漏字段”“幻象默认值”纳入通用审计规则

---

- Date: 2026-04-08
- Context: 对齐 `busbar_inspection / indoor_fence` 的入口治理方式
- Symptom: `scripts`、`.claude/commands` 与现有测试布局不一致，多个文件重复描述相同流程
- Root cause: 规则长期追加但没有单点路由表，导致命令、脚本、skill 分别维护自己的版本
- Fix: 新增 `08_task_routing.md`，重写 `00~07`，统一 `run_targeted_tests.sh / run_regression_tests.sh / run_quality_gate.sh / collect_root_cause.sh`
- Prevention: 任务流程只在 `08_task_routing.md` 维护；其他 skill 只保留职责与约束，不重复整套步骤
- Follow-up: 把“模块化 targeted + 显式 skip 空 regression”回流到 DarkBreaker 通用模板

## meter_reading 特有经验（保留在本插件）

1. 9 种表计类型、三条识别链路、LED 数值编码属于业务专有规则。
2. `HRNet -> HoughCircle -> HoughLine` 降级链和 `fallback_level` 语义属于模拟表专有合同。
3. `METER_RANGES` 注册表、角度范围、OCR 清洗规则、HSV 可分离性都直接绑定表计业务。
4. `reload_config()` 当前只热加载两个阈值，这属于插件特有实现事实。

## 可回流到 DarkBreaker 通用模板

1. 用 `08_task_routing.md` 作为命令入口的唯一任务路由表。
2. `run_targeted_tests.sh` 支持模块参数，并在缺少测试文件时快速失败。
3. `run_regression_tests.sh` 对空 `tests/regression/` 或空 `tests/fixtures/` 做显式 skip，而不是假通过。
4. `run_quality_gate.sh` 先做快速架构检查，再串联 regression gate。
5. 审计与命令输出必须带真实命令证据，不能只给流程摘要。
