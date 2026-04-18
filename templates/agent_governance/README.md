# Agent Governance 模板层

结构母版源，供 `sync_agent_commands.py`、`sync_plugin_template.py`（待建）和新插件初始化使用。

## 目录

```
templates/agent_governance/
├── plugin_high_frequency/    # HF 级：完整脚本体系 + commands + skills
├── plugin_standard/          # STD 级：skills + 基础 commands，脚本占位
├── plugin_minimal/           # MIN 级：sanity check + 最小 skills
├── ui_governance/            # UI 专用治理结构
└── README.md                 # 本文件
```

## 模板层三类内容

| 标记 | 含义 | 处理方式 |
|------|------|----------|
| `{{变量名}}` | 需按插件实例替换 | 由同步脚本或人工替换 |
| `<!-- BUSINESS: ... -->` | 业务专属段，禁止默认复制 | 必须由人工填写 |
| 无标记的结构文本 | 通用结构，可直接复用 | 直接复制 |

## 变量清单

| 变量 | 含义 | 示例 |
|------|------|------|
| `{{PLUGIN_NAME}}` | 插件目录名 | `busbar_inspection` |
| `{{PLUGIN_DISPLAY_NAME}}` | 插件中文名 | `母线自主巡视插件` |
| `{{DETECTOR_FILE}}` | 检测器文件名 | `detector_enhanced.py` 或 `detector.py` |
| `{{MODULE_LIST}}` | targeted tests 模块列表 | 见各脚本内 case 语句 |
| `{{COV_TARGET}}` | pytest --cov 目标 | `plugins.busbar_inspection` |

## 为什么要从真实插件中抽离

1. **避免业务污染**：照搬 busbar 会引入原因码映射、ROI 隔离等业务规则到不相关插件
2. **降低同步成本**：修改通用规则只需改模板，再批量 sync，不必逐插件手工对齐
3. **升级路径清晰**：MIN → STD → HF 只需按模板层级递增补齐，不必猜测"哪些文件还缺"

## 后续脚本如何复用

- `sync_agent_commands.py`：可读取 `templates/agent_governance/*/commands/*.md` 作为模板源
- `sync_plugin_template.py`（待建）：读取对应层级模板，替换变量，写入目标插件
- `agent_task_router.py`：不直接使用模板，但路由逻辑中的对象类型 → 模板层级有稳定映射

## 照抄 busbar 会导致的典型错误

- `detector_enhanced.py` 层级边界规则被写入只有 `detector.py` 的插件
- 原因码（reason_code）契约被写入无此概念的插件
- ROI 隔离规则被写入非视觉类插件（如 gas_detection）
- 质量门禁中 `rg` 搜索 `detector_enhanced.py` 在无该文件的插件中报错
