# 07 Learning Log

本文件记录 UI 模块的项目特定经验。每次重构后 Claude 必须将新发现的项目经验追加到本文件。

---

## [2026-04-03] UI 统一升级 Phase 1+2 — 基于变电站UI升级方案.docx

### 一、升级范围

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `apps/ui_server.py` | 路由重定向 + MODULES 补全 + 聚合 API | 旧路由 301 重定向；MODULES 从 7 个补齐到 16 个；新增 `/api/cockpit/overview` 和 `/api/plugins/registry` |
| `ui/static/css/main.css` | 全量重写 | `--ck-*` 变量提升到全局；导航栏深蓝主题；表单/表格/模态框/Badge 深蓝覆盖；三态 CSS 类 |
| `ui/static/js/main.js` | 全量重写 | AppState + EventBus；WSManager 断线重连；PerfTracker 性能埋点；renderPanel 三态函数 |
| `ui/static/js/cockpit.js` | 全量重写 | 移除 `_generateDefaultPlugins` 假数据；移除 `_simulateDataUpdate`；健康矩阵 `<a>` 跳转；接入 AppState 事件 |
| `ui/static/css/cockpit.css` | 变量提升 + body class 隔离 | `--ck-*` 色彩变量已移至 main.css；`body > .navbar` 改为 `body.cockpit-page > .navbar` |
| `ui/templates/base.html` | 导航栏主题 + footer WS 指示 | 移除 `bg-primary`；深蓝大屏导航栏；footer 增加 WebSocket 状态指示灯 |
| `ui/templates/pages/cockpit.html` | body class + 链接更新 | 增加 `cockpit-page` body class；侧边栏链接增加 `?plugin=xxx` 查询参数 |

### 二、关键教训

#### 教训 1：全局隐藏 `body > .navbar` 影响所有页面
- cockpit.css 原先用 `body > .navbar { display: none !important }` 隐藏 base.html 导航
- 这是全局选择器，会影响到引入 cockpit.css 但不是驾驶舱的页面
- **改为** `body.cockpit-page > .navbar`，仅在 cockpit 页面通过 JS 添加 body class 时生效

#### 教训 2：mock 数据比 error 态更危险
- `_generateDefaultPlugins()` 生成随机健康度和告警数，用户在生产环境看到"假数据"却以为真
- **改为** API 失败时设置 `CockpitState.pageState = 'error'`，健康矩阵显示"加载失败 + 重试按钮"

#### 教训 3：CSS 变量放在页面级 CSS 中导致其他页面无法共享
- 42 个 `--ck-*` 变量仅在 cockpit.css 中定义，outdoor/indoor 等页面无法使用
- **改为** 提升到 main.css 全局级别，cockpit.css 仅保留布局变量（sidebar-w, right-w 等）

#### 教训 4：MODULES 字典与 PLUGIN_META 不同步
- 后端 MODULES 只有 7 个插件，前端 PLUGIN_META 有 16 个
- `/module/{module_id}` 路由有 9 个插件无法访问
- **改为** MODULES 补齐全部 16 个插件，每个插件增加 `category` 和 `route` 字段

#### 教训 5：WebSocket 已有基础设施但被注释掉
- main.js 原有 `connectWebSocket()` 但在 DOMContentLoaded 中被注释为 `// connectWebSocket();`
- **改为** WSManager 自动连接，带指数退避重连和轮询降级

### 三、推广建议

1. **新页面必须使用 `--ck-*` 全局变量**，不得在页面级 CSS 中重新定义色彩变量
2. **所有数据面板必须使用三态**：`renderPanel(container, state, renderFn, options)`
3. **侧边栏/卡片链接必须带 `?plugin=xxx`** 查询参数，目标页面据此高亮对应 Tab
4. **WebSocket 连接状态必须在 UI 可见**，至少在 footer 有指示灯

---

## [2026-03-28] 主控驾驶舱（cockpit）新建 — 完整复盘

### 一、原页面失败原因（逐文件分析）

#### index.html（248 行）
| 问题 | 行号 | 影响 |
|------|------|------|
| 功能定位错误：单任务操作台，无全局概览 | 全文 | 管理者无法一眼看到 16 个插件的运行状态 |
| 任务模板下拉只有 3 项，与实际 16 个插件脱节 | 82-85 | 用户误以为平台只支持 3 种检测 |
| JS 内联在 HTML 中（68 行 JS） | 178-248 | 无法独立缓存、无法 lint |
| 2 个 TODO 注释（视频初始化、任务轮询） | 203, 241 | 功能实际未完成但界面展示正常 |
| `alert()` 占位交互 | 245 | 用户体验差，误以为是 bug |
| `fetch` 不检查 `response.ok` | 193 | API 返回 500 时静默失败 |
| 7 处 inline style（硬编码背景色、高度、字号） | 22,26,44,98,134,135,166 | 样式分散，无法统一管理 |

#### dashboard.html（914 行）
| 问题 | 行号 | 影响 |
|------|------|------|
| 551 行 CSS 内联在 `<style>` 块 | 6-551 | 文件膨胀 2.5 倍，无法浏览器缓存 |
| CSS 中 15 处硬编码颜色（`#888`、`#ccc`、`#666`） | 84,96,104,146... | 主题无法切换，搜索/替换困难 |
| CSS 变量无前缀（`--panel-bg`、`--accent-primary`） | 8-13 | 与 unified_dashboard 的 `--ud-*` 变量碰撞风险 |
| 多个 `onclick` 调用未定义函数 | 575-582 | 点击报错：`openSettings is not defined` |
| 模块导航硬编码 18 个模块项 | 588-740 | 新增/删除插件需要改 HTML |
| 仅有单模块视频视图，无聚合统计 | 744-774 | 同一时间只能看一个模块的画面 |

#### unified_dashboard.html（154 行）
| 问题 | 行号 | 影响 |
|------|------|------|
| HTML 骨架过于简单，依赖 JS 动态生成全部内容 | 56-132 | 首次加载白屏，SEO 不友好，调试困难 |
| 按"设备/区域"维度组织，非"插件"维度 | 51-63 | 与实际插件体系不对齐 |

### 二、这次采用的拆分模式

**文件分离模式（三文件对应一个页面）**：
```
cockpit.html  → 结构骨架（472行，纯 HTML + Jinja2 block）
cockpit.css   → 样式定义（1176行，独立文件，浏览器可缓存）
cockpit.js    → 交互逻辑（568行，模块化对象）
```

**对比 dashboard.html 的单文件混合**：
```
dashboard.html → CSS(551行) + HTML(350行) + JS引用(13行) = 914行单文件
```

**改进效果**：
- CSS 独立为 `.css` 文件 → 浏览器缓存生效，页面加载更快
- JS 独立为 `.js` 文件 → 可被 linter 检查，便于单独调试
- HTML 纯结构 → 可读性高，改布局不需要翻过 551 行 CSS

### 三、关键技术决策与教训

#### 决策 1：base.html 覆盖策略
- **错误尝试**：用 `display: none` 隐藏 base.html 的三个默认元素（navbar / .container-fluid / footer）
- **失败原因**：`.container-fluid.mt-3` 是 `{% block content %}` 的容器，隐藏它会隐藏 cockpit 的全部内容
- **正确方案**：navbar 和 footer 可以 `display: none`，但 `.container-fluid` 必须改为重置边距（`margin:0; padding:0; width:100%`）
- **规则编号**：B-001

#### 决策 2：插件数据降级机制
```javascript
async _loadPlugins() {
    try {
        const res = await fetch('/api/plugins/list');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        CockpitState.plugins = await res.json();
    } catch (e) {
        // 降级：生成 16 个默认插件数据
        CockpitState.plugins = this._generateDefaultPlugins();
    }
    this._renderPluginStatus();  // 无论真实/默认数据都能渲染
}
```
- **效果**：静态预览（npx serve）、开发环境（无后端）、演示场景均可正常展示
- **推广建议**：所有数据驱动页面都应有 fallback 数据策略

#### 决策 3：Chart.js 生命周期严格管理
```javascript
// 每次更新前先销毁旧实例
if (CockpitState.charts.alarmDist) CockpitState.charts.alarmDist.destroy();
CockpitState.charts.alarmDist = new Chart(canvas, { ... });

// 页面卸载时批量销毁
destroy() {
    Object.values(CockpitState.charts).forEach(c => { if (c && c.destroy) c.destroy(); });
}
```
- **不管理的后果**：Chart.js 在同一 canvas 上创建多个实例会叠加渲染，内存持续增长
- **规则编号**：R-004

#### 决策 4：CSS 变量分层设计
```css
/* 5 层背景色：从深到浅 */
--ck-bg-deep: #020b18;      /* 最底层（body） */
--ck-bg-primary: #041c32;   /* 侧边栏/右面板 */
--ck-bg-secondary: #06283d; /* 子区域 */
--ck-bg-card: #0a3553;      /* 卡片 */
--ck-bg-card-hover: #0d4068;/* 卡片 hover */

/* 3 层文字色：从亮到暗 */
--ck-text-primary: #e8f4fd;   /* 标题/数值 */
--ck-text-secondary: #8cb4d5; /* 正文/标签 */
--ck-text-muted: #4a7a9b;     /* 辅助/时间戳 */
```
- **设计原则**：每降一层亮度差约 15-20%，保证层次感
- **推广建议**：新建暗色页面应复用此变量体系或按同样分层方式定义

#### 决策 5：侧边栏插件导航与告警徽章联动
```javascript
// 在 _updateStats 中，遍历插件更新对应的侧边栏徽章
plugins.forEach(p => {
    const badge = document.querySelector(`.sidebar-badge[data-alarm="${p.id}"]`);
    badge.textContent = count;
    badge.classList.toggle('has-alarm', count > 0);  // 仅有告警时显示红色
});
```
- **对比 dashboard.html**：侧边栏模块列表硬编码 18 项，新增插件需改 HTML
- **cockpit 做法**：HTML 中用 `data-plugin="xxx"` 属性标记，JS 动态更新
- **推广建议**：所有与插件相关的导航/列表都应使用 `data-plugin` 属性而非硬编码

### 四、命名规范总结

| 类型 | 正确模式 | 错误模式 | 示例 |
|------|----------|----------|------|
| 页面文件 | `{page_name}.html` | `{page_name}_v4.html` | `cockpit.html` 非 `cockpit_v4.html` |
| CSS 变量 | `--{prefix}-{category}-{name}` | `--{generic-name}` | `--ck-bg-primary` 非 `--panel-bg` |
| JS 模块 | `PascalCase` 对象 | 散落的全局函数 | `CockpitDashboard.init()` 非 `initCockpit()` |
| JS 状态 | `PascalCase + State` | 多个 `let` 变量 | `CockpitState.plugins` 非 `let plugins` |
| DOM ID | `kebab-case` | `camelCase` | `plugin-status-chart` 非 `pluginStatusChart` |
| CSS 类名 | `kebab-case` | `camelCase` 或 `snake_case` | `stat-card-header` 非 `statCardHeader` |
| data 属性 | `data-{noun}` | 无属性，靠位置选择 | `data-plugin="indoor_fence"` 非 `.sidebar-item:nth-child(3)` |

### 五、可复用组件候选

以下区域在后续页面重构中可提取为独立组件：

| 组件 | 当前位置 | 复用场景 |
|------|----------|----------|
| 环形图（Canvas 2D） | `cockpit.js:_renderPluginStatus` | 任何需要比例展示的面板 |
| 告警卡片 | `cockpit.html:rt-alarm-card` | 室内/室外中心的告警列表 |
| 健康度瓦片 | `cockpit.html:health-tile` | 插件管理器的状态概览 |
| 电压等级进度条 | `cockpit.html:voltage-rate-item` | 设置页、统计报表 |
| 统计大数字卡片 | `cockpit.html:stat-card` | 任何需要 KPI 展示的页面 |
