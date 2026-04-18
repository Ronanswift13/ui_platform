# 00 项目上下文 — {{PLUGIN_DISPLAY_NAME}}

## 当前真实状态
- 插件目录: `plugins/{{PLUGIN_NAME}}/`
- 治理等级: MIN（最小治理）
- 主入口: `plugin.py`

### 固定模板规则
1. 本文件是项目上下文唯一权威来源。
2. 若本文件与代码实现冲突，以本文件为准并报告偏差。

## 当前实际能力

<!-- BUSINESS: 如实描述本插件当前具备的能力和缺失的部分 -->

## AI 自动化边界
- 可自动执行: 基础验证、代码审查
- 需人工确认: 所有结构性变更

## 升级路径
1. 补齐 tests/ 目录和基础测试
2. 补齐 `.agent_skills/01~08`
3. 补齐 scripts/run_targeted_tests.sh
4. 补齐 `.claude/commands/`
