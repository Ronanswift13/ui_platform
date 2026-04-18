# ADR-001: 采用 AATF 框架作为工程规范

## 状态
接受

## 日期
2026-03-07

## 上下文
indoor_fence 插件已有基础的 agent_skills (00-04) 和提示词，但缺乏统一的工程工作流框架。各技能文件格式不统一，没有标准化的工作流命令，经验回灌和受控扩散修复没有流程保障。需要一个可复制到其他 DarkBreaker 插件的标准化模板。

## 决策
采用 AATF (Adaptive Agentic Template Framework) 7 阶段框架：

| 阶段 | 名称 | 命令 |
|------|------|------|
| A | 起盘 | /bootstrap |
| B+C | 建模+实现 | /implement |
| D | 质量闸门 | /audit |
| E | 回灌 | /repair |
| F | 受控扩散 | /propagate |
| G | 跨项目迁移 | cross_project_reuse_prompt.md |

## 理由
- 统一所有插件的工程实践，降低跨项目切换成本
- 通过 `.claude/commands/` 将工作流编码为可执行命令
- 通过 8 段格式统一 agent_skills 的结构，便于 AI 消费
- 通过经验回灌 (07_learning_log.md) 防止重复犯错
- 通过受控扩散 (06_refactor_policy.md) 防止修复引入新 bug

替代方案考虑：
- 手动维护文档: 缺乏结构化，不可被 AI 自动消费
- 仅用 CLAUDE.md: 信息密度过高，缺乏分层

## 影响
- 新增 `.agent_skills/05-07` 三个文件
- 新增 `.claude/commands/` 五个命令
- 重构 00-04 为 8 段格式
- 重构 `PROJECT_CARD.md` 为 9 字段格式
- 增强 `CLAUDE.md` 为 AATF 工作流入口
- 不涉及业务逻辑代码变更

## 关联
- 相关 agent_skills: 全部 (00-07)
- 相关命令: 全部 (bootstrap/implement/repair/audit/propagate)
