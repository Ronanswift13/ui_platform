# 00 项目上下文 — UI 主控驾驶舱

## 当前真实状态
- 目录: `ui/`
- 治理等级: STD
- 技术栈: Jinja2 模板 + 原生 JS + CSS Grid

### 固定模板规则
1. 本文件是 UI 项目上下文唯一权威来源。
2. 所有 AI 任务启动前必须先读本文件 + `01_architecture_rules.md`。

## 目录结构

```
ui/
├── static/
│   ├── css/          # 样式（深蓝大屏主题）
│   ├── js/           # 脚本
│   └── images/       # 图片
├── templates/
│   ├── base.html     # 主布局
│   ├── pages/        # 页面模板
│   └── components/   # 可复用组件
├── scripts/          # 质量检查脚本
├── .agent_skills/    # 治理知识库
├── CLAUDE.md
└── PROJECT_CARD.md
```

## AI 自动化边界
- 可自动执行: 样式修改、组件开发、测试
- 需人工确认: API 接口变更、新增第三方依赖、导航结构变更
