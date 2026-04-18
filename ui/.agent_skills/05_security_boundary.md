# 05 Security Boundary

## 1. 前端安全规则

### 1.1 XSS 防护
- Jinja2 默认自动转义 HTML，不得使用 `| safe` 过滤器除非内容来源可信
- `innerHTML` 赋值前必须确保数据来源为内部 API，不含用户输入
- 动态生成 HTML 时，文本内容使用 `textContent` 而非 `innerHTML`

### 1.2 API 请求
- 所有 API 请求走相对路径（`/api/...`），不硬编码域名
- POST 请求必须设置 `Content-Type: application/json`
- 不得在 URL 参数中传递敏感信息

### 1.3 敏感信息
- 前端代码不得包含密码、token、密钥等敏感信息
- WebSocket 地址不得硬编码，从配置或页面变量获取
- 错误消息不得暴露后端堆栈或内部路径

### 1.4 第三方依赖
- CDN 引用应使用 `integrity` 属性（SRI）
- 不得引入未经审查的第三方脚本
- 不得使用 `eval()` 执行动态代码

## 2. 操作安全

### 2.1 危险操作
- 删除操作必须有二次确认
- 批量操作必须显示影响范围
- 插件启用/禁用需要明确提示

### 2.2 表单安全
- 表单提交按钮必须防重复提交（disabled + loading）
- 文件上传必须限制类型和大小
- 输入框长度必须有合理限制

## 3. 目录权限
- UI 模块仅允许修改 `ui/` 和 `apps/` 目录下的路由注册
- 不得修改 `plugins/` 目录下的核心逻辑
- 不得修改 `platform_core/` 目录
- 不得修改 `training/` 目录
