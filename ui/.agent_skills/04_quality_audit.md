# 04 Quality Audit

本文件记录 UI 模块的通用质量规则。每次重构后 Claude 必须将新发现的通用规则追加到本文件。

---

## 1. 文件结构规则

| 编号 | 规则 | 触发条件 | 正确范式 |
|------|------|----------|----------|
| F-001 | 页面模板必须拆分 | 单文件 > 400 行 | 页面模板只保留布局骨架，内容区域 `{% include %}` 组件 |
| F-002 | CSS 不得内联到 HTML | `<style>` 块出现在 `.html` 文件中 | 提取为同名 `.css` 文件，通过 `{% block extra_css %}` 引入 |
| F-003 | JS 不得内联到 HTML | `<script>` 块超过 20 行出现在 `.html` 文件中 | 提取为同名 `.js` 文件，通过 `{% block extra_js %}` 引入 |
| F-004 | JS 脚本文件必须拆分 | 单文件 > 500 行 | 按职责拆分：数据层 / 渲染层 / 事件层 |
| F-005 | 消除代码重复 | 同一 HTML/JS 片段出现 >= 3 处 | 提取为 `components/` 模板或 JS 函数 |

**违反案例**：`dashboard.html` 将 551 行 CSS 内联在 `<style>` 块中（F-002），总计 914 行混合 CSS + HTML + JS。

## 2. 展示与逻辑分离规则

| 编号 | 规则 | 触发条件 | 正确范式 |
|------|------|----------|----------|
| R-001 | 组件不得调 API | `components/` 目录下的模板出现 `fetch` | 数据由页面层 `include` 时通过 Jinja2 变量传入 |
| R-002 | 状态集中定义 | 超过 2 个变量用 `let`/`var` 在函数外定义 | 定义单一状态对象（如 `const State = { ... }`） |
| R-003 | 定时器必须注册 | `setInterval` 返回值未存储 | ID 存入 `State.timers`，`beforeunload` 时统一 `clearInterval` |
| R-004 | Chart.js 先销毁再创建 | 对同一 `canvas` 多次 `new Chart()` | `if (State.charts.foo) State.charts.foo.destroy()` 后再创建 |
| R-005 | 内联 onclick 禁止用于业务逻辑 | HTML 中 `onclick="funcName()"` 调用非导航函数 | 使用 `addEventListener` 绑定，HTML 只做结构 |

**违反案例**：`dashboard.html:575-582` 使用 `onclick="openSettings()"` / `onclick="startAutoPatrol()"`（R-005）；`index.html:244` 使用 `alert('ROI编辑功能开发中...')`（H-002）。

## 3. 三态覆盖规则

所有数据驱动的 UI 区域必须显式处理以下三种状态：

| 状态 | 要求 | 审查方法 |
|------|------|----------|
| **loading** | 显示 spinner 或骨架屏，禁止空白 | 搜索 `spinner` 或 `loading` 关键字，每个数据区域至少一个 |
| **empty** | 显示图标 + 文字提示，禁止空容器 | 搜索 `暂无` 或 `empty`，每个列表/网格至少一个 |
| **error** | 显示错误提示，API 失败必须 catch 并渲染 | 搜索 `catch`，每个 `fetch` 必须有对应的错误 UI |

```javascript
// ✗ 反模式：仅处理成功路径（index.html:190-200）
fetch('/api/sites').then(r => r.json()).then(data => {
    data.forEach(site => { select.innerHTML += `<option>...</option>`; });
}).catch(err => console.error('加载站点失败:', err));  // 用户看不到任何反馈

// ✓ 正确范式：三态都有 UI 反馈
async function loadData(container) {
    container.innerHTML = '<div class="loading">加载中...</div>';
    try {
        const data = await fetch('/api/data').then(r => { if(!r.ok) throw r; return r.json(); });
        if (data.length === 0) { container.innerHTML = '<div class="empty">暂无数据</div>'; }
        else { container.innerHTML = renderList(data); }
    } catch(e) { container.innerHTML = '<div class="error">加载失败，<a onclick="loadData(this.parentNode)">重试</a></div>'; }
}
```

## 4. 样式规则

| 编号 | 规则 | 触发条件 | 正确范式 |
|------|------|----------|----------|
| S-001 | 颜色变量化 | 任意 `#xxx` 或 `rgb()` 出现在选择器内部 | 定义为 `:root { --prefix-name: #xxx }` 后引用 |
| S-002 | 选择器嵌套 <=3 层 | `.a .b .c .d { }` 出现 | 用 BEM 或扁平类名替代 |
| S-003 | Grid/Flex 布局优先 | `position: absolute` 用于非浮层目的 | 页面级用 `grid-template-areas`，组件级用 `flex` |
| S-004 | !important 受限 | 项目内 CSS 使用 `!important` | 仅在覆盖 Bootstrap 等外部框架时允许 |
| S-005 | 三档响应式 | 页面无 `@media` 规则 | 1400px（缩窄侧栏）、1024px（隐藏侧栏）、768px（单列） |
| S-006 | 禁止 inline style 做布局 | `style="min-height:480px"` 等出现在模板中 | 提取到 CSS 类，HTML 只有 class 引用 |
| S-007 | CSS 变量命名必须有前缀 | `:root` 中定义 `--color` 这种通用名 | 使用 `--ck-color`（cockpit）、`--ud-color`（unified）区分页面 |

**违反案例**：`dashboard.html` 定义 `--panel-bg` 等无前缀变量（S-007）；`index.html:22` 内联 `style="min-height: 480px; background: #1a1a1a"`（S-006 + S-001）。`dashboard.html` 在 CSS 中有 15 处硬编码颜色值如 `color: #888`、`color: #ccc`（S-001）。

## 5. 交互规则

| 编号 | 规则 | 触发条件 | 正确范式 |
|------|------|----------|----------|
| I-001 | 可点击必须有反馈 | `cursor: pointer` 但无 `:hover` 样式 | 添加 hover 背景变化或颜色变化 |
| I-002 | 异步按钮防重复 | 按钮点击发起 `fetch` 但不 disable | 点击后立即 `btn.disabled = true`，完成/失败后恢复 |
| I-003 | 导航用 `<a>` | `<div onclick="location.href=...">` | 改为 `<a href="...">` |
| I-004 | 折叠有过渡 | `display: none/block` 切换内容 | 使用 `max-height` + `transition` 或 `overflow: hidden` |
| I-005 | 零值徽章不闪烁 | 告警数为 0 时仍显示红色动画 | `badge.classList.toggle('has-alarm', count > 0)` |

## 6. 依赖规则

| 编号 | 规则 | 触发条件 | 正确范式 |
|------|------|----------|----------|
| D-001 | 禁止框架级依赖 | `import React` 或 `<script src="vue">` | 使用已有的 Vanilla JS + Bootstrap |
| D-002 | 图表库统一 | 引入 ECharts/D3/Highcharts | 统一使用已引入的 Chart.js 4.x |
| D-003 | CSS 框架统一 | 引入 Tailwind/Ant Design | 统一使用已引入的 Bootstrap 5.3 |

## 7. 代码卫生规则

| 编号 | 规则 | 触发条件 | 正确范式 |
|------|------|----------|----------|
| H-001 | 禁止 TODO/FIXME | grep 到 `TODO`/`FIXME`/`HACK` | 要么实现，要么开 issue 追踪后删除注释 |
| H-002 | 禁止 alert 占位 | `alert('xxx开发中')` | 按钮灰显 + `title="功能开发中"` 或直接不展示 |
| H-003 | console 限制 | `console.log` 在非 catch 块中出现 | 仅在 `catch` 中使用 `console.warn`/`console.error` |
| H-004 | 无死代码 | 定义了函数但从未调用 | 删除未使用的函数、变量、CSS 类 |
| H-005 | fetch 必须检查 status | `fetch(...).then(r => r.json())` 不检查 `r.ok` | `if (!r.ok) throw new Error(...)` 后再 `.json()` |

**违反案例**：`index.html:203` `// TODO: 初始化WebSocket视频流`（H-001）；`index.html:240` `// TODO: 实现任务状态轮询`（H-001）；`index.html:245` `alert('ROI编辑功能开发中...')`（H-002）；`index.html:193` `fetch('/api/sites').then(r => r.json())` 未检查 `r.ok`（H-005）。

## 8. base.html 继承规则

| 编号 | 规则 | 触发条件 | 正确范式 |
|------|------|----------|----------|
| B-001 | 不隐藏 content wrapper | 子页面 CSS 中 `body > .container-fluid { display: none }` | 改为重置边距：`margin:0; padding:0; width:100%; max-width:100%` |
| B-002 | 可隐藏导航和底栏 | 全屏页面需要隐去 base.html 的 navbar/footer | `body > .navbar { display: none !important }` 可以 |
| B-003 | block 使用规范 | 页面只用 `{% block content %}` 不用 `extra_css`/`extra_js` | 三个 block 都应使用：`extra_css` 引外部 CSS，`extra_js` 引外部 JS |

**教训来源**：cockpit 开发中，最初用 `display:none` 隐藏 `.container-fluid.mt-3`，导致 `{% block content %}` 内的所有内容不可见（因为该 div 就是 content 的容器）。

---

## 复盘记录

### [2026-04-03] UI 统一升级 — Phase 1+2 复盘

#### 核心问题
旧版 UI 回流、驾驶舱与插件状态无法实时同步、三套视觉体系并存（cockpit 深蓝 / base.html Bootstrap 浅蓝 / standalone 第三套）。

#### 原因
1. 旧版路由（`/home`, `/dashboard`, `/unified-dashboard`）仍可直达，未重定向
2. `MODULES` 字典仅注册 7 个插件，系统共 16 个，映射断裂
3. `_generateDefaultPlugins()` 在 API 失败时生成随机假数据，用户无法区分真假
4. `--ck-*` CSS 变量仅在 cockpit.css 定义，其他页面仍用 Bootstrap 默认色
5. `CockpitState` 为页面私有状态，无跨页面事件总线
6. WebSocket 基础设施已有但未启用，仍用 30s setInterval 轮询

#### 采用模式
| 模式 | 做法 |
|------|------|
| **旧路由 301 重定向** | `/home`、`/dashboard`、`/unified-dashboard` → `RedirectResponse(301)` → `/cockpit` |
| **全局 CSS 变量提升** | `--ck-*` 从 cockpit.css 移至 main.css，所有页面共享深蓝主题 |
| **AppState + EventBus** | main.js 中建立全局 `AppState` 对象，含 `on/emit/updatePlugin/addAlarm` |
| **WebSocket 断线重连** | `WSManager` 指数退避重连 + 轮询降级，UI 实时显示连接状态 |
| **聚合接口** | 新增 `/api/cockpit/overview` 一次返回全部数据 |
| **三态处理** | 全局 `renderPanel()` 函数 + `.panel-skeleton/.panel-error/.panel-empty` CSS 类 |
| **插件卡片可跳转** | 健康矩阵 `<div>` → `<a href="/outdoor?plugin=xxx">` |
| **body class 隔离** | cockpit 用 `body.cockpit-page` 选择器隐藏导航，不再用全局 `body > .navbar` |

#### 新增规则
| 编号 | 规则 |
|------|------|
| A-001 | mock 数据仅用于开发环境；生产环境 API 失败必须显示 error 态而非假数据 |
| A-002 | 旧版路由必须 301 重定向到新版入口，不得保留可直达旧页面的路由 |
| A-003 | CSS 变量必须定义在 main.css 全局级别，页面级 CSS 仅定义布局变量 |
| A-004 | WebSocket 状态必须在 UI 上有视觉指示（导航栏 + footer） |
| A-005 | 后端 MODULES 字典必须与前端 PLUGIN_META 一一对应，保持 16 个插件完整注册 |

---

### [2026-03-28] 主控驾驶舱重构 — 深度复盘

#### 一、原始结构主要缺陷

**缺陷 1：首页（index.html）功能定位错误**
- 设计为"单任务操作台"（选站点 → 选设备 → 执行 → 查结果），适合操作员而非管理者
- 16 个插件的状态完全不可见，用户必须逐个跳转到 `/outdoor` 或 `/indoor` 才能了解系统状态
- 任务模板下拉框只列出 3 种（主变/开关/表计），与实际 16 个插件脱节

**缺陷 2：dashboard.html 将 551 行 CSS 内联在 `<style>` 块**
- 总文件 914 行，混合 CSS (551行) + HTML (350行) + JS引用 (13行)
- CSS 中有 15 处硬编码颜色（`#888`、`#ccc`、`#666`、`#222` 等），未使用变量
- CSS 变量定义无前缀（`--panel-bg`），与其他页面的变量名存在碰撞风险
- 对应的 JS 文件 `dashboard.js` 与内联 CSS 分离，但共享状态方式不清晰

**缺陷 3：三个仪表盘页面各自独立，无数据共享**
- `index.html`：单任务操作，无概览
- `dashboard.html`：单模块视频监控，按"模块"维度组织
- `unified_dashboard.html`：按"设备/区域"维度组织
- 三个页面没有统一的插件聚合视图，用户不知道该去哪个页面

**缺陷 4：交互占位和死代码**
- `index.html` 包含 2 个 `TODO` 注释（视频初始化、任务轮询）
- `addROI()` 函数使用 `alert()` 作为占位
- `pollTaskStatus()` 函数体为空
- `dashboard.html` 的多个 `onclick` 引用未定义的函数（`openSettings`、`openTraining`、`startAutoPatrol`）

#### 二、采用的重构模式

| 模式 | 具体做法 | 效果 |
|------|----------|------|
| **新建而非改造** | 创建 `cockpit.html` + `cockpit.css` + `cockpit.js` 三文件，不修改任何旧页面 | 零回归风险，旧页面通过 `/home`、`/dashboard` 仍然可访问 |
| **CSS Grid 五区域布局** | `grid-template-areas: "header header header" "sidebar main right" "footer footer footer"` | 一个声明定义全局布局，响应式只需改 `grid-template` |
| **模块化对象模式** | `CockpitDashboard` 对象封装 `init/destroy/_loadPlugins/_renderX/_startPolling` | 所有状态在 `CockpitState` 中，所有定时器在 `timers` 中，页面卸载一行清理 |
| **CSS 变量体系化** | `:root` 中 42 个 `--ck-*` 变量，覆盖 5 层背景、3 层文字、5 种语义色 | 主题切换只需改变量值，零硬编码颜色 |
| **API 降级** | `_loadPlugins` catch 后调 `_generateDefaultPlugins()`，生成 16 个默认插件 | 无后端时页面仍完整渲染，开发/演示无障碍 |
| **三态覆盖** | 每个动态区域（健康矩阵、告警列表、图片网格）都有 loading/empty/error 三种 HTML | 任何数据异常场景下用户都看到有意义的 UI |

#### 三、禁止反模式（跨页面通用）

| 反模式 | 触发条件 | 为什么有害 | 正确做法 |
|--------|----------|------------|----------|
| **CSS 内联到 HTML** | `<style>` 块在 `.html` 模板中 | 文件膨胀、无法缓存、无法复用、难以搜索 | 提取为 `{page}.css`，`{% block extra_css %}` 引入 |
| **硬编码颜色** | CSS 中直接写 `#1a1a2e` 而非 `var(--x)` | 主题无法统一切换、全局搜索/替换困难 | 所有颜色走 CSS 变量 |
| **alert 占位** | `alert('xxx开发中')` | 用户以为是 bug，而非未完成功能 | 按钮 disabled + title 提示，或不显示 |
| **空函数** | `function pollTaskStatus(taskId) {}` | 调用者以为功能正常，实际无效 | 删除函数和调用处，或抛出 NotImplementedError |
| **内联 onclick 调未定义函数** | `onclick="openSettings()"` 但全局无此函数 | 点击报错，控制台红色异常 | addEventListener 绑定 + 编译时检查 |
| **display:none 隐藏内容父容器** | 子页面 CSS 隐藏 base.html 的 `.container-fluid` | 子内容也被隐藏 | 重置边距而非隐藏 |

#### 四、沉淀为通用规则

已写入本文件上方规则条目：
- F-002（CSS 不内联到 HTML）、F-003（JS 不内联到 HTML）
- S-001（颜色变量化）、S-006（禁止 inline style 做布局）、S-007（CSS 变量必须有前缀）
- R-005（禁止内联 onclick 调业务逻辑）
- H-005（fetch 必须检查 status）
- B-001 ~ B-003（base.html 继承规则）
