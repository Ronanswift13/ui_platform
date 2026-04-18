# Cross Project Reuse Prompt

用于把 busbar_inspection 中验证过的规则迁移到其他插件。

## Checklist
- 哪些是母线特有假设（原因码、缺陷标签、变焦策略）？
- 哪些契约可复用（配置映射、质量门禁框架、测试分层）？
- 目标插件的输入输出 schema 是否一致？
- 迁移后最小验证集是什么？
- 需要同步更新哪些文档（PROJECT_CARD / CLAUDE / .agent_skills）？
