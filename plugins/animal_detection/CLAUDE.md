# CLAUDE.md（项目级指令）

## 0. 角色定义
你在本项目中的角色是"规则执行型工程代理"，先满足契约和测试，再讨论优化。

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

## 3. 本项目差异指令（animal_detection）
1. 所有阈值必须从 `configs/default.yaml` 进入算法层，不允许推理路径硬编码 (QR-1)。
2. confidence 必须经 `_sanitize_confidence()` 清洗后才可赋值 (QR-11)。
3. 枚举状态集（AnimalClass=8, EventType=10, RiskLevel=4）严格锁定 (QR-8)。
4. 降级链路必须保留并可审计：模型缺失→空检测，热成像缺失→跳过验证 (QR-3)。
5. 热验证和驱离功能在配置中默认关闭，文档标注为可选功能。

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
3. `run_regression_tests.sh` 返回非 0。

## 7. 建议优先级
1. 优先完善测试覆盖（core/ ≥ 80%），确保降级链路和枚举锁定有测试。
2. 将 bbox 越界衰减系数等中间环节硬编码值移入配置 (QR-1)。
3. 按动物类型差异化置信度阈值 (QR-5)。
