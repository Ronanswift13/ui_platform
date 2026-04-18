# PROJECT_CARD: {{PLUGIN_NAME}}

## 1. 项目名称
{{PLUGIN_DISPLAY_NAME}}（{{PLUGIN_NAME}}）

## 2. 项目类型
<!-- BUSINESS: plugin_new / plugin_update / plugin_governance -->

## 3. 输入源
<!-- BUSINESS: 定义本插件的输入数据（类型、格式、来源） -->

## 4. 输出目标
<!-- BUSINESS: 定义本插件的输出结果（结构、枚举、格式） -->

## 5. 关键约束
### 工程约束
- 必须遵循 `darkbreaker_sdk.interfaces.BasePlugin` 契约。
- `plugin.py` 仅做 SDK 适配，不承载核心算法。
- `{{DETECTOR_FILE}}` 不得依赖 `darkbreaker_sdk`。
- 所有阈值必须来自 `configs/default.yaml` 映射。

### 业务约束
<!-- BUSINESS: 定义本插件的业务约束 -->

### 安全约束
- 不访问外部网络。
- 不持久化原始数据到未授权目录。
- 日志中不输出敏感标识符原文。

## 6. 验收标准
- `./scripts/run_targeted_tests.sh all` 通过。
- `./scripts/run_regression_tests.sh` 可执行且阶段化输出明确。
- `.agent_skills/00~08` 完整。
- `PROJECT_CARD.md`、`CLAUDE.md`、`.claude/commands/` 均为本项目定制。

## 7. 禁止事项
- 禁止修改 SDK 接口签名。
- 禁止新增硬编码业务阈值到推理主路径。
- 禁止使用 `except: pass`。
- 禁止在生产路径新增 `print()`。
<!-- BUSINESS: 补充本插件专属禁止事项 -->
