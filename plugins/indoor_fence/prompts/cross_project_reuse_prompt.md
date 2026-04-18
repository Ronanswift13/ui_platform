# 跨项目迁移提示词

对应 AATF 阶段 G（跨项目能力迁移）。

## 角色定义

你是一个负责将 indoor_fence 插件的 AATF 工程实践迁移到其他 DarkBreaker 插件的 AI 助手。

## 迁移源

本插件 (indoor_fence) 作为 AATF 参考实现，包含:
- `.agent_skills/` (8个技能文件，统一8段格式)
- `.claude/commands/` (5个工作流命令: bootstrap/implement/repair/audit/propagate)
- `prompts/` (4个提示词: root_cause/terminal_execute/web_modeling/cross_project_reuse)
- `scripts/` (4个自动化脚本: run_quality_gate/run_regression_tests/run_targeted_tests/collect_root_cause)
- `PROJECT_CARD.md` (9字段格式)
- `CLAUDE.md` (AATF 工作流入口)
- `docs/decision_records/` (ADR 模板)

## 可直接复用的文件（通用，不需修改）

| 文件 | 原因 |
|------|------|
| `.agent_skills/04_quality_audit.md` | 反模式清单通用 |
| `.agent_skills/05_security_boundary.md` | 安全规则通用 |
| `.agent_skills/06_refactor_policy.md` | 重构策略通用 |
| `.agent_skills/07_learning_log.md` | 经验回灌模板通用 |
| `.claude/commands/*` | 工作流命令通用 |
| `scripts/run_quality_gate.sh` | 质量门禁通用 |
| `scripts/collect_root_cause.sh` | 根因收集通用 |
| `prompts/root_cause_prompt.md` | 诊断框架通用（需改故障模式） |
| `docs/decision_records/000-template.md` | ADR 模板通用 |

## 需要定制的文件（每个项目不同）

| 文件 | 定制内容 |
|------|---------|
| `PROJECT_CARD.md` | 9个字段全部重写: 项目名称、类型、输入源、输出目标、约束、验收标准、禁止事项、参考物、当前任务 |
| `.agent_skills/00_project_context.md` | 目录结构、依赖列表、Mock策略、配置加载优先级 |
| `.agent_skills/01_architecture_rules.md` | 层级图、依赖方向、禁止修改列表 |
| `.agent_skills/02_algorithm_contract.md` | 降级策略、算法公式、阈值参数 |
| `.agent_skills/03_test_strategy.md` | Mock构造方法、边界用例清单、覆盖率目标 |
| `CLAUDE.md` | 项目概述、启动命令、常见任务 |
| `configs/default.yaml` | 全部参数 |
| `scripts/run_targeted_tests.sh` | 模块名和测试文件映射 |
| `prompts/terminal_execute_prompt.md` | 项目特定命令 |
| `prompts/web_modeling_prompt.md` | Web UI 规格 |

## 迁移步骤

1. **复制通用文件**
   ```bash
   cp -r indoor_fence/.agent_skills/04_quality_audit.md  target_plugin/.agent_skills/
   cp -r indoor_fence/.agent_skills/05_security_boundary.md  target_plugin/.agent_skills/
   cp -r indoor_fence/.agent_skills/06_refactor_policy.md  target_plugin/.agent_skills/
   cp -r indoor_fence/.agent_skills/07_learning_log.md  target_plugin/.agent_skills/
   cp -r indoor_fence/.claude/commands/  target_plugin/.claude/
   cp indoor_fence/scripts/run_quality_gate.sh  target_plugin/scripts/
   cp indoor_fence/scripts/collect_root_cause.sh  target_plugin/scripts/
   cp -r indoor_fence/docs/decision_records/  target_plugin/docs/
   ```

2. **创建 PROJECT_CARD.md** - 填写目标插件的 9 个字段

3. **创建 00-03 技能文件** - 使用 8 段格式模板，填入目标插件的具体内容

4. **创建 CLAUDE.md** - 参考 indoor_fence 的结构，修改项目概述和启动命令

5. **调整 scripts/run_targeted_tests.sh** - 修改模块名和测试文件映射

6. **运行 /bootstrap 验证** - 确认环境和测试基准

## 输出格式

对每个目标插件输出:
- 迁移清单（复制 / 定制 / 跳过）
- 每个需要定制的文件的具体修改说明
- 验证步骤
