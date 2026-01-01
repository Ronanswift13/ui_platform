# 电压等级选择功能 - UI集成指南

## 📁 文件结构

```
ui_integration/
├── templates/
│   └── pages/
│       └── settings_voltage.html    # 电压等级设置页面模板
├── static/
│   └── js/
│       └── voltage_settings.js      # 前端交互脚本
├── platform_core/
│   └── voltage_api.py               # 后端API路由
└── INTEGRATION_GUIDE.md             # 本集成指南
```

---

## 🚀 快速集成步骤

### 步骤 1: 复制文件到项目目录

```bash
# 复制页面模板
cp ui_integration/templates/pages/settings_voltage.html \
   破夜绘明激光监测平台/ui/templates/pages/

# 复制 JavaScript 文件
cp ui_integration/static/js/voltage_settings.js \
   破夜绘明激光监测平台/ui/static/js/

# 复制 API 路由文件
cp ui_integration/platform_core/voltage_api.py \
   破夜绘明激光监测平台/platform_core/
```

### 步骤 2: 修改 UI 服务器 (apps/ui_server.py)

在 `apps/ui_server.py` 中添加电压设置页面路由：

```python
# 在 create_app() 函数中添加以下代码

# 导入电压管理器
from platform_core.voltage_api import voltage_manager

@app.get("/settings/voltage", response_class=HTMLResponse)
async def settings_voltage_page(request: Request):
    """电压等级设置页面"""
    current_level = voltage_manager.get_current_level()
    return templates.TemplateResponse(
        "pages/settings_voltage.html",
        {
            "request": request,
            "active_tab": "settings",
            "current_voltage_level": current_level,
        },
    )
```

### 步骤 3: 修改 API 服务器 (apps/api_server.py)

在 `apps/api_server.py` 中集成电压等级 API 路由：

```python
# 在文件顶部添加导入
from platform_core.voltage_api import integrate_voltage_routes

# 在 create_api_app() 函数中添加
def create_api_app():
    app = FastAPI(...)
    
    # ... 其他现有路由 ...
    
    # 集成电压等级管理路由
    integrate_voltage_routes(app)
    
    return app
```

### 步骤 4: 修改现有设置页面导航

在 `ui/templates/pages/settings.html` 中添加电压等级导航链接：

```html
<!-- 在设置导航列表中添加 -->
<div class="list-group list-group-flush">
    <a href="#general" class="list-group-item list-group-item-action active" data-bs-toggle="list">
        <i class="bi bi-sliders"></i> 通用设置
    </a>
    
    <!-- 新增：电压等级设置 -->
    <a href="/settings/voltage" class="list-group-item list-group-item-action">
        <i class="bi bi-lightning-charge"></i> 电压等级
    </a>
    
    <a href="#plugins" class="list-group-item list-group-item-action" data-bs-toggle="list">
        <i class="bi bi-puzzle"></i> 插件管理
    </a>
    <!-- ... 其他导航项 ... -->
</div>
```

### 步骤 5: 创建模型目录结构

```bash
# 创建模型目录
mkdir -p 破夜绘明激光监测平台/models/220kV/{transformer,switch,busbar,capacitor,meter}
mkdir -p 破夜绘明激光监测平台/models/500kV/{transformer,switch,busbar,capacitor,meter}

# 创建配置目录
mkdir -p 破夜绘明激光监测平台/configs
```

---

## 📝 完整代码修改示例

### apps/ui_server.py 完整修改

```python
# apps/ui_server.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# ... 其他导入 ...

# 新增导入
from platform_core.voltage_api import voltage_manager

def create_app():
    app = FastAPI(title="输变电监测平台")
    
    # 静态文件
    app.mount("/static", StaticFiles(directory="ui/static"), name="static")
    
    # 模板
    templates = Jinja2Templates(directory="ui/templates")
    
    # ... 现有路由 ...
    
    # ============== 新增：电压设置页面 ==============
    @app.get("/settings/voltage", response_class=HTMLResponse)
    async def settings_voltage_page(request: Request):
        """电压等级设置页面"""
        current_level = voltage_manager.get_current_level()
        return templates.TemplateResponse(
            "pages/settings_voltage.html",
            {
                "request": request,
                "active_tab": "settings",
                "current_voltage_level": current_level,
            },
        )
    
    return app
```

### apps/api_server.py 完整修改

```python
# apps/api_server.py

from fastapi import FastAPI
# ... 其他导入 ...

# 新增导入
from platform_core.voltage_api import integrate_voltage_routes

def create_api_app():
    app = FastAPI(
        title="输变电监测平台 API",
        version="1.0.0"
    )
    
    # ... 现有路由注册 ...
    
    # ============== 新增：电压等级管理 API ==============
    integrate_voltage_routes(app)
    
    return app
```

---

## 🔌 API 端点说明

集成后将提供以下 API 端点：

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/voltage/current` | 获取当前电压等级 |
| POST | `/api/voltage/set` | 设置电压等级 |
| GET | `/api/voltage/config/{type}` | 获取设备配置 |
| GET | `/api/voltage/models` | 获取所有模型路径 |
| GET | `/api/voltage/model-status` | 检查模型文件状态 |
| GET | `/api/voltage/detection-classes/{type}` | 获取检测类别 |
| GET | `/api/voltage/thermal-thresholds` | 获取热成像阈值 |
| GET | `/api/voltage/angle-reference/{type}` | 获取开关角度参考 |

### API 调用示例

```javascript
// 获取当前电压等级
fetch('/api/voltage/current')
    .then(res => res.json())
    .then(data => console.log(data.voltage_level));

// 设置电压等级
fetch('/api/voltage/set', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ level: '500kV' })
})
    .then(res => res.json())
    .then(data => console.log(data));

// 获取开关配置
fetch('/api/voltage/config/switch')
    .then(res => res.json())
    .then(data => console.log(data.config));
```

---

## 🎨 UI 功能说明

### 电压等级选择页面功能

1. **电压等级选择卡片**
   - 220kV 和 500kV 两个选项
   - 显示各等级的典型参数
   - 点击选择高亮显示

2. **模型库状态显示**
   - 显示当前电压等级对应的所有模型
   - 检查模型文件是否存在
   - 显示"已就绪"或"待训练"状态

3. **配置详情展示**
   - 显示检测类别列表
   - 显示各插件的设备参数

4. **操作按钮**
   - 应用设置：确认切换电压等级
   - 刷新状态：重新检查模型状态

5. **确认对话框**
   - 切换前显示确认对话框
   - 列出将执行的操作

---

## ⚙️ 配置文件

电压等级配置保存在 `configs/voltage_config.json`：

```json
{
    "current_voltage_level": "500kV",
    "updated_at": "2025-01-01T10:30:00"
}
```

---

## 🔧 故障排除

### 问题 1: API 路由未找到

**现象**: 访问 `/api/voltage/current` 返回 404

**解决**: 
1. 确认已在 `api_server.py` 中调用 `integrate_voltage_routes(app)`
2. 重启 API 服务器

### 问题 2: 页面模板未找到

**现象**: 访问 `/settings/voltage` 返回 500 错误

**解决**:
1. 确认 `settings_voltage.html` 已复制到 `ui/templates/pages/`
2. 检查文件路径是否正确

### 问题 3: JavaScript 加载失败

**现象**: 页面功能不响应

**解决**:
1. 确认 `voltage_settings.js` 已复制到 `ui/static/js/`
2. 检查浏览器控制台是否有 404 错误
3. 清除浏览器缓存

### 问题 4: 模型状态始终显示"待训练"

**现象**: 即使模型文件存在，状态也显示"待训练"

**解决**:
1. 检查模型文件路径是否与配置匹配
2. 确认文件扩展名为 `.onnx`
3. 检查文件权限

---

## 📋 验证清单

- [ ] `settings_voltage.html` 已复制到正确位置
- [ ] `voltage_settings.js` 已复制到正确位置
- [ ] `voltage_api.py` 已复制到正确位置
- [ ] `ui_server.py` 已添加页面路由
- [ ] `api_server.py` 已集成 API 路由
- [ ] `settings.html` 已添加导航链接
- [ ] 模型目录结构已创建
- [ ] 服务已重启
- [ ] 页面可正常访问
- [ ] API 可正常调用
- [ ] 电压切换功能正常

---

## 💡 后续扩展

1. **添加更多电压等级**: 可扩展支持 110kV、750kV、1000kV
2. **模型自动下载**: 集成模型下载功能
3. **配置备份/恢复**: 添加配置导入导出功能
4. **权限控制**: 限制电压切换操作仅管理员可用
