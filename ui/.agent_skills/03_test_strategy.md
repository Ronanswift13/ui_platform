# 03 Test Strategy

## 1. 测试层级

### 1.1 模板渲染测试
- 使用 Jinja2 的 `Environment` 直接渲染模板，验证无语法错误
- 检查关键 DOM 节点 ID 存在（如 `cockpit-app`、`plugin-status-chart`）
- 验证所有 `{% block %}` 正确继承

```python
# 示例：验证模板渲染
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('ui/templates'))
t = env.get_template('pages/cockpit.html')
html = t.render(request=None, active_tab='cockpit', version='4.0.0')
assert 'cockpit-app' in html
assert 'plugin-status-chart' in html
```

### 1.2 JavaScript 语法验证
- 检查花括号/方括号/圆括号配对
- 验证关键函数存在（如 `init`、`destroy`、`_loadPlugins`）
- 检查无未定义的全局变量引用

### 1.3 CSS 语法验证
- 检查花括号配对
- 验证关键选择器存在
- 检查无硬编码像素值（应使用变量或 rem）

### 1.4 路由集成测试
- 验证 FastAPI 路由注册成功
- 验证页面返回 200 状态码
- 验证关键路由不冲突

```python
# 示例：验证路由
from apps.ui_server import create_app
app = create_app()
routes = [r.path for r in app.routes if hasattr(r, 'path')]
assert '/cockpit' in routes
assert '/outdoor' in routes
```

### 1.5 预览验证
- 启动服务后截图确认视觉效果
- 检查控制台无 JS 错误（API 404 在静态预览中可忽略）
- 验证响应式布局在不同断点下正常

## 2. 测试命令

```bash
# 模板渲染测试
python3 -c "
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('ui/templates'))
for page in ['pages/cockpit.html', 'pages/dashboard.html', 'index.html']:
    t = env.get_template(page)
    html = t.render(request=None, active_tab='test', version='1.0')
    print(f'✓ {page}: {len(html)} chars')
"

# JS/CSS 语法检查
python3 -c "
for f in ['static/js/cockpit.js', 'static/css/cockpit.css']:
    content = open(f'ui/{f}').read()
    opens = content.count('{')
    closes = content.count('}')
    status = '✓' if opens == closes else '✗'
    print(f'{status} {f}: {opens} opens, {closes} closes')
"

# 路由验证
cd /path/to/DarkBreaker && python3 -c "
from apps.ui_server import create_app
app = create_app()
routes = [r.path for r in app.routes if hasattr(r, 'path')]
for r in ['/', '/cockpit', '/outdoor', '/indoor', '/dashboard']:
    print(f\"{'✓' if r in routes else '✗'} {r}\")
"
```

## 3. 验收检查清单
- [ ] 页面渲染无 Jinja2 错误
- [ ] JS 无语法错误
- [ ] CSS 花括号配对
- [ ] 路由注册正确
- [ ] loading 状态有 spinner 或提示
- [ ] empty 状态有占位 UI
- [ ] error 状态有 fallback
- [ ] 响应式断点 1400/1024/768 正常
- [ ] 无 TODO 或空按钮
- [ ] Chart.js 图表可渲染
