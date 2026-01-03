"""
UI 服务器

提供Web UI界面,集成API接口
"""


from __future__ import annotations
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 模板和静态文件目录
TEMPLATES_DIR = PROJECT_ROOT / "ui" / "templates"
STATIC_DIR = PROJECT_ROOT / "ui" / "static"


# 模块配置
MODULES = {
    "transformer": {
        "name": "主变自主巡视",
        "icon": "box",
        "description": "主变压器本体及附属设施的外观缺陷识别、状态识别和热成像分析",
        "plugin_id": "transformer_inspection",
    },
    "switch": {
        "name": "开关间隔巡视",
        "icon": "toggle-on",
        "description": "断路器、隔离开关、接地开关的分合位状态识别和逻辑校验",
        "plugin_id": "switch_inspection",
    },
    "busbar": {
        "name": "母线自主巡视",
        "icon": "diagram-3",
        "description": "绝缘子串、金具、导线连接点的远距小目标检测",
        "plugin_id": "busbar_inspection",
    },
    "capacitor": {
        "name": "电容器巡视",
        "icon": "battery-charging",
        "description": "电容器组的结构完整性检测和区域入侵检测",
        "plugin_id": "capacitor_inspection",
    },
    "meter": {
        "name": "表计读数",
        "icon": "speedometer2",
        "description": "站内全类型模拟表计和数字表计的任意角度读数识别",
        "plugin_id": "meter_reading",
    },
}


def create_app() -> FastAPI:
    """创建UI应用"""
    app = FastAPI(
        title="输变电监测平台",
        description="输变电激光星芒破夜绘明监测平台 Web UI",
        version="2.0.0",
    )

    # 挂载静态文件
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # 模板引擎
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # 集成API
    from apps.api_server import create_api_app
    api_app = create_api_app()

    # 挂载API路由 (去掉/api前缀因为已在api_server中定义)
    for route in api_app.routes:
        route_path = getattr(route, "path", None)
        if route_path and route_path.startswith("/api"):
            app.routes.append(route)

    # ============== 集成训练API路由 ==============
    try:
        from apps.training_api import router as training_router
        app.include_router(training_router)
        print("✓ 训练API路由已集成")
    except ImportError as e:
        print(f"✗ 训练API导入失败: {e}")

    # ============== 页面路由 ==============

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """首页"""
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "active_tab": "home",
                "version": "1.0.0",
            },
        )

    @app.get("/module/{module_id}", response_class=HTMLResponse)
    async def module_page(request: Request, module_id: str):
        """功能模块页面"""
        if module_id not in MODULES:
            return templates.TemplateResponse(
                "pages/404.html",
                {"request": request},
                status_code=404,
            )

        module = MODULES[module_id]

        # 检查插件状态并获取UI配置
        from platform_core.plugin_manager import PluginManager
        pm = PluginManager()

        # 尝试获取或加载插件
        plugin = pm.get_plugin(module["plugin_id"])
        if plugin is None:
            try:
                plugin = pm.load_plugin(module["plugin_id"])
            except Exception as e:
                print(f"加载插件 {module['plugin_id']} 失败: {e}")
                plugin = None

        plugin_status = "ready" if (plugin and plugin.status.value == "ready") else "placeholder"

        # 获取插件的UI配置
        ui_config = None
        if plugin and hasattr(plugin, 'get_ui_config'):
            try:
                ui_config = plugin.get_ui_config()
            except Exception as e:
                print(f"获取插件 {module['plugin_id']} 的UI配置失败: {e}")

        return templates.TemplateResponse(
            "pages/module_with_training.html",
            {
                "request": request,
                "active_tab": module_id,
                "module_id": module_id,
                "module_name": module["name"],
                "module_icon": module["icon"],
                "module_description": module["description"],
                "plugin_status": plugin_status,
                "ui_config": ui_config,
            },
        )

    @app.get("/replay", response_class=HTMLResponse)
    async def replay_page(request: Request):
        """回放页面"""
        return templates.TemplateResponse(
            "pages/replay.html",
            {
                "request": request,
                "active_tab": "replay",
            },
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        """设置页面"""
        return templates.TemplateResponse(
            "pages/settings.html",
            {
                "request": request,
                "active_tab": "settings",
            },
        )

    # ============== 电压等级设置页面 ==============
    @app.get("/settings/voltage", response_class=HTMLResponse)
    async def settings_voltage_page(request: Request):
        """电压等级设置页面"""
        return templates.TemplateResponse(
            "pages/settings_voltage_extended.html",
            {
                "request": request,
                "active_tab": "settings",
            },
        )

    # ============== 模型训练页面 ==============
    @app.get("/training", response_class=HTMLResponse)
    async def training_page(request: Request):
        """模型训练管理页面"""
        return templates.TemplateResponse(
            "pages/training_enhanced.html",
            {
                "request": request,
                "active_tab": "training",
                "version": "2.0.0",
            },
        )

    return app


def main():
    """主函数"""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="输变电监测平台 UI 服务器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="热重载")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
