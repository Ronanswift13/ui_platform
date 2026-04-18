# 00 项目上下文 — {{PLUGIN_DISPLAY_NAME}}

## 输入审计

### 当前真实状态
- 插件目录: `plugins/{{PLUGIN_NAME}}/`
- 治理等级: HF（高频治理）
- 主入口: `plugin.py`
- 检测器: `{{DETECTOR_FILE}}`

### 固定模板规则
1. 本文件是项目上下文唯一权威来源。
2. 所有 AI 任务启动前必须先读本文件 + `01_architecture_rules.md`。
3. 若本文件与代码实现冲突，以本文件为准并报告偏差。

## 目录结构

```
plugins/{{PLUGIN_NAME}}/
├── plugin.py                 # SDK 适配层
├── {{DETECTOR_FILE}}         # 检测/算法核心
├── configs/
│   └── default.yaml          # 运行参数与阈值
├── tests/                    # 测试用例
├── scripts/
│   ├── run_targeted_tests.sh
│   ├── run_regression_tests.sh
│   ├── run_quality_gate.sh
│   └── collect_root_cause.sh
├── .agent_skills/            # 治理知识库 (00~08)
├── .claude/commands/         # AI 命令入口
├── CLAUDE.md                 # 固定指令
├── PROJECT_CARD.md           # 项目卡片
└── manifest.json             # 插件元数据
```

## AI 自动化边界
- 可自动执行: 测试、审计、代码生成、文档更新
- 需人工确认: manifest.json 核心字段变更、API 签名变更、跨插件修改

<!-- BUSINESS: 补充本插件的业务输入源、输出目标、降级链路等专属上下文 -->
