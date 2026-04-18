# 01 Architecture Rules

## 1. 分层原则

### 1.1 页面层（templates/pages/）
- 负责整体布局、数据请求调度、状态管理
- 每个页面对应一个独立的 `.html` 模板和可选的 `.js`/`.css` 文件
- 页面超过 400 行必须拆分为组件

### 1.2 组件层（templates/components/）
- 纯展示组件，接受 include 参数
- 不得直接调用 API 接口
- 不得持有全局状态
- 可被多个页面复用

### 1.3 脚本层（static/js/）
- `main.js` — 全局工具函数、API 封装、插件加载
- 页面专用脚本以页面名命名（如 `cockpit.js`、`dashboard.js`）
- 脚本内使用模块化对象模式（如 `CockpitDashboard`），避免全局污染

### 1.4 样式层（static/css/）
- `main.css` — 全局基础样式
- 页面专用样式以页面名命名（如 `cockpit.css`、`unified_dashboard.css`）
- 颜色、间距、圆角等使用 CSS 变量，不硬编码

## 2. 布局规则

### 2.1 CSS Grid 优先
- 页面级布局必须使用 CSS Grid
- 组件内部布局可使用 Flexbox
- 禁止使用绝对定位实现布局（仅用于浮层/覆盖层）

### 2.2 响应式断点
| 断点 | 宽度 | 行为 |
|------|------|------|
| Desktop | > 1400px | 完整三栏布局 |
| Laptop | 1024px - 1400px | 侧边栏缩窄 |
| Tablet | 768px - 1024px | 隐藏侧边栏 |
| Mobile | < 768px | 单列堆叠 |

## 3. 数据流规则

### 3.1 请求层
- 所有 API 调用通过 `fetch` 或 `main.js` 中的 `API` 对象
- 请求失败必须 catch 并显示用户可见反馈
- 轮询间隔：统计数据 10s、插件状态 30s、趋势图 60s

### 3.2 状态管理
- 页面级状态使用 JS 模块内的状态对象（如 `CockpitState`）
- 状态对象集中定义，不分散在各函数中
- 定时器 ID 必须注册到状态对象，页面卸载时统一清理

### 3.3 DOM 更新
- 使用 `innerHTML` 批量更新列表/网格内容
- 单个值更新使用 `textContent`
- 禁止在循环中频繁操作 DOM

## 4. 主题系统

### 4.1 CSS 变量命名规范
- 全局变量：`--ck-{category}-{name}`（cockpit 主题）
- 已有前缀：`--ud-*`（unified dashboard）、`--ck-*`（cockpit）
- 新页面应定义自己的前缀或复用已有变量

### 4.2 颜色语义
| 语义 | 变量 | 用途 |
|------|------|------|
| 主色 | `--ck-accent` | 高亮、链接、图标 |
| 成功 | `--ck-success` | 正常状态、在线 |
| 警告 | `--ck-warning` | 注意、告警 |
| 危险 | `--ck-danger` | 异常、严重告警 |
| 信息 | `--ck-info` | 提示、辅助信息 |

## 5. 路由规则
- 新增页面必须在 `apps/ui_server.py` 注册路由
- 路由名必须与模板文件名一致
- 导航栏入口在 `templates/base.html` 中维护
- 不得删除已有路由，只能新增或标记废弃

## 6. 依赖管理
- 仅允许使用已引入的 CDN 依赖（Bootstrap 5、Chart.js、Three.js）
- 引入新依赖需要在 PROJECT_CARD.md 中记录理由
- 禁止引入 jQuery、React、Vue 等框架级依赖
