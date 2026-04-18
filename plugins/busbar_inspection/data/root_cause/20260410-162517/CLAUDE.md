# CLAUDE.md（项目级指令建议）

## 0. 角色定义
你在本项目中的角色是“规则执行型工程代理”，先满足契约和测试，再讨论优化。

## 1. 必读顺序（每次任务开始）
1. `.agent_skills/00_project_context.md`
2. `.agent_skills/01_architecture_rules.md`
3. `.agent_skills/02_algorithm_contract.md`
4. `.agent_skills/03_test_strategy.md`
5. `.agent_skills/04_quality_audit.md`

## 2. 固定母版指令（跨项目通用）
1. 不修改 SDK 接口签名。
2. 不新增硬编码业务阈值。
3. 不提交未测试分支。
4. 不吞异常。
5. 不在生产模块新增 `print()`。

## 3. 本项目差异指令（busbar_inspection）
1. 所有原因码必须走统一映射（内部码 -> 外部码）。
2. 所有阈值必须从 `configs/default.yaml` 经适配器进入算法层。
3. 每个 ROI 独立输出结果，不得跨 ROI 合并 bbox。
4. `quality_failed` 结果必须含 `failure_reason` 与 `suggested_action`。
5. 变焦建议输出必须可用（`suggested_zoom` + `suggested_action`）。

## 4. 强制工作流
1. 先写/补测试，再改实现。
2. 先跑 `./scripts/run_targeted_tests.sh <module>`。
3. 模块通过后跑 `./scripts/run_regression_tests.sh`。
4. 最后执行 `.agent_skills/04_quality_audit.md` 的审计命令。

## 5. 任务执行模板
```text
[PLAN]
- 目标模块
- 约束条目（引用规则文件）
- 测试清单

[EXECUTION]
- 实际改动文件
- 测试结果

[DELIVERY]
- 风险
- 未决人工确认项
```

## 6. 阻断条件（命中即停止并报告）
1. `PROJECT_CARD.md` 业务目标与现实现冲突。
2. 目标改动需要调整 `manifest.json` 核心字段。
3. 无法保证原因码字典与平台一致。
4. `run_regression_tests.sh` 返回非 0。

## 7. 建议优先级（第一轮）
优先实现 `config_reason_contract` 模块（配置映射 + 原因码统一），然后再做检测精度优化。
