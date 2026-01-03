#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变电站电压等级管理系统 - 示例应用
==========================================

完整的 FastAPI 示例应用，演示如何集成电压等级管理功能。

运行方法:
    uvicorn example_app:app --reload --port 8000
    
或者:
    python example_app.py

访问:
    http://localhost:8000         - 主页
    http://localhost:8000/docs    - API文档
    http://localhost:8000/api/voltage/current - 当前电压等级

作者: 破夜绘明团队
日期: 2025
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# 导入电压等级管理模块
from platform_core.voltage_adapter_extended import (
    VoltageAdapterManager,
    get_all_voltage_categories,
    VOLTAGE_CONFIGS,
    PLUGIN_CAPABILITIES,
)
from platform_core.voltage_api_extended import integrate_voltage_routes


# =============================================================================
# 创建应用
# =============================================================================
app = FastAPI(
    title="变电站电压等级管理系统",
    description="支持特高压、超高压、高压、中压、低压变电站的完整管理系统",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 集成电压等级管理API路由
integrate_voltage_routes(app)

# 静态文件和模板
static_path = PROJECT_ROOT / "ui" / "static"
templates_path = PROJECT_ROOT / "ui" / "templates"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

if templates_path.exists():
    templates = Jinja2Templates(directory=str(templates_path))
else:
    templates = None

# 全局电压管理器
voltage_manager = VoltageAdapterManager()


# =============================================================================
# 页面路由
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """主页"""
    if templates:
        return templates.TemplateResponse(
            "pages/settings_voltage_extended.html",
            {"request": request}
        )
    
    # 如果没有模板，返回简单HTML
    return HTMLResponse(content=get_simple_html())


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """演示页面"""
    return HTMLResponse(content=get_demo_html())


@app.get("/compare", response_class=HTMLResponse)
async def compare_page():
    """电压等级对比页面"""
    return HTMLResponse(content=get_compare_html())


# =============================================================================
# 辅助函数 - 生成HTML
# =============================================================================

def get_simple_html():
    """生成简单的HTML页面"""
    categories = get_all_voltage_categories()
    
    categories_html = ""
    for cat in categories:
        levels_html = " ".join([
            f'<button class="btn btn-sm btn-outline-primary m-1" onclick="setLevel(\'{l}\')">{l}</button>'
            for l in cat["levels"]
        ])
        categories_html += f"""
        <div class="card mb-3">
            <div class="card-header">{cat["category"]} - {cat["description"]}</div>
            <div class="card-body">{levels_html}</div>
        </div>
        """
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>变电站电压等级管理系统</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container py-4">
            <h1>变电站电压等级管理系统</h1>
            <p class="lead">选择变电站电压等级以加载对应的AI模型和设备配置</p>
            
            <div class="row">
                <div class="col-md-6">
                    <h4>选择电压等级</h4>
                    {categories_html}
                </div>
                <div class="col-md-6">
                    <h4>当前配置</h4>
                    <div id="currentConfig" class="border p-3 rounded bg-light">
                        <p>当前电压等级: <strong id="currentLevel">未设置</strong></p>
                        <p>分类: <span id="currentCategory">-</span></p>
                        <div id="configDetails"></div>
                    </div>
                </div>
            </div>
            
            <div class="mt-4">
                <h4>API 端点</h4>
                <ul>
                    <li><a href="/docs">/docs</a> - Swagger API 文档</li>
                    <li><a href="/api/voltage/current">/api/voltage/current</a> - 当前电压等级</li>
                    <li><a href="/api/voltage/categories">/api/voltage/categories</a> - 所有电压分类</li>
                    <li><a href="/api/voltage/plugins/all">/api/voltage/plugins/all</a> - 所有插件</li>
                </ul>
            </div>
        </div>
        
        <script>
            // 初始化
            fetchCurrentLevel();
            
            async function fetchCurrentLevel() {{
                const res = await fetch('/api/voltage/current');
                const data = await res.json();
                if (data.voltage_level) {{
                    document.getElementById('currentLevel').textContent = data.voltage_level;
                    document.getElementById('currentCategory').textContent = data.category || '-';
                    fetchConfig();
                }}
            }}
            
            async function setLevel(level) {{
                const res = await fetch('/api/voltage/set', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{level: level}})
                }});
                const data = await res.json();
                if (data.success) {{
                    document.getElementById('currentLevel').textContent = data.voltage_level;
                    document.getElementById('currentCategory').textContent = data.category || '-';
                    fetchConfig();
                    alert('切换成功: ' + data.voltage_level);
                }}
            }}
            
            async function fetchConfig() {{
                const res = await fetch('/api/voltage/info');
                const data = await res.json();
                if (data.success) {{
                    const info = data.info;
                    let html = '<h6>支持的插件:</h6><ul>';
                    (info.supported_plugins || []).forEach(p => {{
                        html += '<li>' + p.name + '</li>';
                    }});
                    html += '</ul>';
                    
                    html += '<h6>热成像阈值:</h6>';
                    const th = info.thermal_thresholds || {{}};
                    html += '<span class="badge bg-success me-1">正常: ' + (th.normal || '-') + '°C</span>';
                    html += '<span class="badge bg-warning me-1">警告: ' + (th.warning || '-') + '°C</span>';
                    html += '<span class="badge bg-danger">告警: ' + (th.alarm || '-') + '°C</span>';
                    
                    document.getElementById('configDetails').innerHTML = html;
                }}
            }}
        </script>
    </body>
    </html>
    """


def get_demo_html():
    """生成演示HTML页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>电压等级管理 - 功能演示</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    </head>
    <body>
        <nav class="navbar navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="/">
                    <i class="bi bi-lightning-charge-fill"></i> 电压等级管理系统
                </a>
            </div>
        </nav>
        
        <div class="container py-4">
            <h2>功能演示</h2>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">快速切换测试</div>
                        <div class="card-body">
                            <div class="btn-group-vertical w-100">
                                <button class="btn btn-danger" onclick="quickSwitch('1000kV_AC')">
                                    <i class="bi bi-lightning-fill"></i> 切换到 1000kV 特高压
                                </button>
                                <button class="btn btn-warning" onclick="quickSwitch('500kV_AC')">
                                    <i class="bi bi-broadcast"></i> 切换到 500kV 超高压
                                </button>
                                <button class="btn btn-success" onclick="quickSwitch('220kV')">
                                    <i class="bi bi-building"></i> 切换到 220kV 高压
                                </button>
                                <button class="btn btn-info" onclick="quickSwitch('35kV')">
                                    <i class="bi bi-house-door"></i> 切换到 35kV 中压
                                </button>
                                <button class="btn btn-secondary" onclick="quickSwitch('10kV')">
                                    <i class="bi bi-plug"></i> 切换到 10kV 低压
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">API 响应结果</div>
                        <div class="card-body">
                            <pre id="apiResponse" class="bg-dark text-light p-3 rounded" style="max-height: 400px; overflow: auto;">
等待操作...
                            </pre>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">API 测试按钮</div>
                        <div class="card-body">
                            <button class="btn btn-outline-primary m-1" onclick="testApi('/api/voltage/current')">
                                获取当前等级
                            </button>
                            <button class="btn btn-outline-primary m-1" onclick="testApi('/api/voltage/categories')">
                                获取所有分类
                            </button>
                            <button class="btn btn-outline-primary m-1" onclick="testApi('/api/voltage/plugins')">
                                获取支持插件
                            </button>
                            <button class="btn btn-outline-primary m-1" onclick="testApi('/api/voltage/thermal-thresholds')">
                                获取热成像阈值
                            </button>
                            <button class="btn btn-outline-primary m-1" onclick="testApi('/api/voltage/model-status')">
                                获取模型状态
                            </button>
                            <button class="btn btn-outline-primary m-1" onclick="testApi('/api/voltage/detection-classes')">
                                获取检测类别
                            </button>
                            <button class="btn btn-outline-primary m-1" onclick="testApi('/api/voltage/info')">
                                获取完整信息
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function quickSwitch(level) {
                const res = await fetch('/api/voltage/set', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({level: level})
                });
                const data = await res.json();
                document.getElementById('apiResponse').textContent = JSON.stringify(data, null, 2);
            }
            
            async function testApi(url) {
                const res = await fetch(url);
                const data = await res.json();
                document.getElementById('apiResponse').textContent = JSON.stringify(data, null, 2);
            }
        </script>
    </body>
    </html>
    """


def get_compare_html():
    """生成对比HTML页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>电压等级对比</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container py-4">
            <h2>电压等级对比</h2>
            
            <div class="row mt-4">
                <div class="col-md-4">
                    <label>电压等级 1:</label>
                    <select id="level1" class="form-select">
                        <option value="1000kV_AC">1000kV 特高压交流</option>
                        <option value="500kV_AC" selected>500kV 超高压交流</option>
                        <option value="220kV">220kV 高压</option>
                        <option value="110kV">110kV 高压</option>
                        <option value="35kV">35kV 中压</option>
                        <option value="10kV">10kV 低压</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label>电压等级 2:</label>
                    <select id="level2" class="form-select">
                        <option value="1000kV_AC">1000kV 特高压交流</option>
                        <option value="500kV_AC">500kV 超高压交流</option>
                        <option value="220kV" selected>220kV 高压</option>
                        <option value="110kV">110kV 高压</option>
                        <option value="35kV">35kV 中压</option>
                        <option value="10kV">10kV 低压</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label>&nbsp;</label>
                    <button class="btn btn-primary d-block w-100" onclick="compare()">对比</button>
                </div>
            </div>
            
            <div id="compareResult" class="mt-4"></div>
        </div>
        
        <script>
            async function compare() {
                const level1 = document.getElementById('level1').value;
                const level2 = document.getElementById('level2').value;
                
                const res = await fetch(`/api/voltage/compare?level1=${level1}&level2=${level2}`);
                const data = await res.json();
                
                if (data.success) {
                    const c = data.comparison;
                    document.getElementById('compareResult').innerHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">${c.level1.voltage_level}</div>
                                    <div class="card-body">
                                        <p>分类: ${c.level1.category}</p>
                                        <p>支持插件数: ${c.level1.plugin_count}</p>
                                        <p>热成像阈值: 正常${c.level1.thermal_thresholds?.normal || '-'}°C / 
                                           告警${c.level1.thermal_thresholds?.alarm || '-'}°C</p>
                                        <p>特殊功能: ${c.level1.special_features?.join(', ') || '-'}</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">${c.level2.voltage_level}</div>
                                    <div class="card-body">
                                        <p>分类: ${c.level2.category}</p>
                                        <p>支持插件数: ${c.level2.plugin_count}</p>
                                        <p>热成像阈值: 正常${c.level2.thermal_thresholds?.normal || '-'}°C / 
                                           告警${c.level2.thermal_thresholds?.alarm || '-'}°C</p>
                                        <p>特殊功能: ${c.level2.special_features?.join(', ') || '-'}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="alert alert-info mt-3">
                            <strong>对比结果:</strong>
                            <br>分类相同: ${c.differences.category_same ? '是' : '否'}
                            <br>正常温度阈值差: ${c.differences.thermal_threshold_diff?.normal || 0}°C
                            <br>告警温度阈值差: ${c.differences.thermal_threshold_diff?.alarm || 0}°C
                        </div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """


# =============================================================================
# 启动应用
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("变电站电压等级管理系统 v2.0.0")
    print("=" * 60)
    print("\n支持的电压等级分类:")
    for cat in get_all_voltage_categories():
        print(f"  - {cat['category']}: {cat['description']}")
        print(f"    电压等级: {', '.join(cat['levels'])}")
    
    print("\n" + "=" * 60)
    print("启动服务器...")
    print("访问地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("演示页面: http://localhost:8000/demo")
    print("对比页面: http://localhost:8000/compare")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
