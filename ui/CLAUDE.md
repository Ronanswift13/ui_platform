# CLAUDE.md（UI 项目级指令）

## 0. 角色定义
你在本项目中的角色是"规则执行型前端工程代理"，先满足 UI 契约和交互规范，再讨论优化。

## 1. 必读顺序（每次任务开始）
1. `CLAUDE.md`（本文件）
2. `PROJECT_CARD.md`
3. `.agent_skills/00_project_context.md`
4. `.agent_skills/01_architecture_rules.md`
5. `.agent_skills/02_ui_contract.md`
6. `.agent_skills/03_test_strategy.md`
7. `.agent_skills/04_quality_audit.md`

## 2. 固定母版指令（跨项目通用）
1. 不修改后端 API 接口签名。
2. 不新增大型第三方依赖。
3. 不提交未测试的交互变更。
4. 不吞异常——所有 catch 块必须有用户可见反馈或日志输出。
5. 不在展示组件中直接发起数据请求。
6. 不留 TODO、空按钮、假交互。

## 3. 本项目差异指令（ui — 主控驾驶舱）
1. 所有页面必须采用深蓝色大屏主题（CSS 变量 `--ck-*` 体系）。
2. 所有数据面板必须显式处理 loading / empty / error 三态。
3. 插件数据统一通过 `/api/plugins/list` 获取，前端负责聚合展示。
4. 室内/室外导航入口必须保留，不得合并或删除。
5. 图表使用 Chart.js（已引入），不得新增 ECharts/D3 等。
6. CSS 布局优先使用 CSS Grid，避免绝对定位。
7. 页面文件超过 400 行必须拆分为组件。

## 4. 强制工作流
1. 先读取本文件和 PROJECT_CARD.md。
2. 再读取 .agent_skills/ 目录下的规则文件。
3. 开始编码。
4. 编码完成后，输出 root cause 复盘（写入 04 和 07）。
5. 验证页面渲染正确。

## 5. 任务执行模板
```text
[PLAN]
- 目标页面/组件
- 约束条目（引用规则文件）
- 验收清单

[EXECUTION]
- 修改记录
- 三态覆盖确认

[REVIEW]
- root cause 复盘（核心问题 / 原因 / 采用模式 / 禁止反模式 / 沉淀规则）
```

## 6. 目录结构
```
ui/
├── CLAUDE.md                    # 本文件
├── PROJECT_CARD.md              # 项目卡片
├── .agent_skills/               # 规则与知识库
│   ├── 00_project_context.md    # 项目上下文
│   ├── 01_architecture_rules.md # 架构规则
│   ├── 02_ui_contract.md        # UI 契约
│   ├── 03_test_strategy.md      # 测试策略
│   ├── 04_quality_audit.md      # 质量审计（通用规则沉淀）
│   ├── 05_security_boundary.md  # 安全边界
│   ├── 06_refactor_policy.md    # 重构策略
│   └── 07_learning_log.md       # 项目经验日志
├── static/
│   ├── css/                     # 样式文件
│   ├── js/                      # 脚本文件
│   └── images/                  # 图片资源
└── templates/
    ├── base.html                # 主布局
    ├── index.html               # 原始首页（保留于 /home）
    ├── pages/                   # 页面模板
    └── components/              # 可复用组件
```
