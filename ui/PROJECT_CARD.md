# PROJECT_CARD

## 1. 项目名称
变电站监测平台主控驾驶舱 UI 重构

## 2. 项目类型
ui_refactor

## 3. 当前模块
主控驾驶舱仪表盘（cockpit dashboard）— 平台首页/主控中心

## 4. 输入源
* 现有页面代码路径: `ui/templates/index.html`, `ui/templates/pages/dashboard.html`, `ui/templates/pages/unified_dashboard.html`
* 当前接口定义: `/api/plugins/list`, `/api/plugins/enabled`, `/api/outdoor/*`, `/api/indoor/*`
* 现有组件目录: `ui/templates/components/`
* 路由配置: `apps/ui_server.py`
* UI 参考图: 云南电网生产运行支持系统（山火防控页面 + 视频巡检监控仪表盘）
* 插件系统: `plugins/` 目录下 16 个插件（室内 6 + 室外 10）

## 5. 输出目标
* 完成主控驾驶舱仪表盘重构，采用深色主题（参考云南电网风格）
* 保持原有室内/室外监测中心功能不变
* 主页实时统计和整理各个插件的监测情况
* UI 适配插件的调度和融合
* 补全 empty/loading/error 三态
* 保留原有变电站核心业务功能

## 6. 关键约束
* 不允许改动后端接口
* 不允许新增大型依赖（仅使用已有的 Bootstrap 5 + Chart.js）
* 不允许改动未授权目录（仅修改 ui/ 和 apps/ 路由注册）
* 不允许破坏现有路由（/outdoor, /indoor, /dashboard 等保持可用）
* 不允许删除已有核心业务功能

## 7. 验收标准
* 页面能正常运行
* lint/typecheck/test 通过
* 关键交互不退化
* 结构拆分清晰
* 样式统一（深蓝色大屏主题）
* 16 个插件的状态可在主控仪表盘实时查看

## 8. 禁止事项
* 禁止留 TODO
* 禁止空按钮或假交互
* 禁止把业务逻辑塞进纯展示组件
* 禁止一次性全仓重构

## 9. 当前任务
仅重构主控驾驶舱仪表盘页面（首页），不处理其他模块
