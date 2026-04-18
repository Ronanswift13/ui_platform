# CLAUDE.md — AI Agent 行为契约

> 本文件是 AI Agent (Claude Code) 处理 DarkBreaker 仓库任务时的首读路由文件。

## 1. 项目

输变电站 AI 自主巡视与监测平台，Python 3.10+，FastAPI + 插件化架构，19 个功能插件。

## 2. 快速导航

| 要做什么 | 去哪里 |
|---|---|
| 了解项目全貌 | `PROJECT_CARD.md` |
| 了解目录结构与模块关系 | `.agent_skills/01_architecture_map.md` |
| 查插件成熟度 | `.agent_skills/02_plugin_registry.md` |
| 跨插件任务路由 | `.agent_skills/05_task_routing.md` |
| 编码规范 | `.agent_skills/04_naming_conventions.md` |
| 配置体系 | `.agent_skills/07_config_hierarchy.md` |
| 测试策略 | `.agent_skills/06_testing_strategy.md` |
| 治理升级规则 | `.agent_skills/08_governance_rules.md` |

## 3. 核心约束

### 3.1 不要做的事

- **不要** 在占位态插件 (radar / thermal) 里生成 `.agent_skills/`
- **不要** 修改 `platform_core/` 下的接口签名，除非明确被要求
- **不要** 向 `configs/` 写入含密钥/凭证的内容
- **不要** 删除 `evidence/` 下的任何文件（证据链不可变）
- **不要** 在未跑通 `pytest` 的情况下声称"已完成"

### 3.2 必须做的事

- 修改插件时，先读其 `manifest.json` 确认版本与依赖
- 新建插件时，遵循最小四件套: `plugin.py` + `manifest.json` + `configs/` + `tests/`
- 跨插件修改时，先查 `.agent_skills/05_task_routing.md` 确认依赖关系
- 输出结果必须符合 `platform_core/schema/` 中的 UnifiedResult 格式

### 3.3 语言与格式

- 代码: Python，`black` 格式化 (line-length=100)，`ruff` lint
- 注释/文档: 中文优先，技术术语保留英文
- commit: 简洁英文，一句话 why

## 4. 常用命令

```bash
# 启动平台
python run.py

# 仅 API
python run.py --api --port 8000

# 测试
pytest tests/ -v

# lint
ruff check .
black --check .

# 类型检查
mypy apps/ platform_core/ plugins/
```

## 5. 插件开发速查

```python
from platform_core.plugin_manager.enhanced_base import EnhancedBasePlugin, TaskContext

class MyPlugin(EnhancedBasePlugin):
    def init(self, config: dict) -> bool: ...
    def process(self, inputs: dict, context: TaskContext) -> UnifiedResult: ...
    async def process_async(self, inputs: dict, context: TaskContext): ...
```

## 6. 分片知识库索引

```
.agent_skills/
├── 00_project_identity.md      # 项目身份与边界
├── 01_architecture_map.md      # 架构与目录映射
├── 02_plugin_registry.md       # 插件成熟度矩阵
├── 03_tech_stack.md            # 技术栈与依赖
├── 04_naming_conventions.md    # 命名与编码规范
├── 05_task_routing.md          # 跨插件任务路由
├── 06_testing_strategy.md      # 测试策略
├── 07_config_hierarchy.md      # 配置层级体系
└── 08_governance_rules.md      # 治理规则与升级条件
```
