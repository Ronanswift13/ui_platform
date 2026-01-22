"""
UI 服务器

提供Web UI界面,集成API接口
"""


from __future__ import annotations
import sys
from pathlib import Path

# 添加项目根目录到路径 (必须在其他导入之前)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# 高级插件集成 (在路径设置之后导入)
try:
    from apps.ui_integration import integrate_all_advanced_features
    ADVANCED_PLUGINS_AVAILABLE = True
except ImportError as e:
    ADVANCED_PLUGINS_AVAILABLE = False
    print(f"⚠ 高级插件模块未找到: {e}")

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
    # ========== 新增模块 ==========
    "bird": {
        "name": "鸟类监控",
        "icon": "feather",
        "description": "室外输电线路鸟类识别、风险评估和驱离告警",
        "plugin_id": "bird_monitoring",
    },
    "indoor_fence": {
        "name": "室内电子围栏",
        "icon": "shield-shaded",
        "description": "基于激光雷达的电子围栏、多人监测、黄线越界检测和授权校验",
        "plugin_id": "indoor_fence",
    },
}


def create_app() -> FastAPI:
    """创建UI应用"""
    app = FastAPI(
        title="输变电监测平台",
        description="输变电激光星芒破夜绘明监测平台 Web UI",
        version="3.0.0",
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
        from training.training_api import router as training_router
        app.include_router(training_router)
        print("✓ 训练API路由已集成 (增强版)")
    except ImportError as e:
        print(f"✗ 训练API导入失败: {e}")
        # 回退到旧版本
        try:
            from apps.training_api import router as training_router
            app.include_router(training_router)
            print("✓ 训练API路由已集成 (旧版)")
        except ImportError as e2:
            print(f"✗ 训练API导入完全失败: {e2}")

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
            "pages/module.html",
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
            "pages/settings_voltage.html",
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
            "pages/training.html",
            {
                "request": request,
                "active_tab": "training",
                "version": "3.0.0",
            },
        )

    # ============== 监测中心Dashboard页面 ==============
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page(request: Request):
        """监测中心Dashboard页面"""
        return templates.TemplateResponse(
            "pages/dashboard.html",
            {
                "request": request,
                "active_tab": "dashboard",
                "version": "3.0.0",
            },
        )

    # ============== 室外监测中心 (V3.5新增) ==============
    @app.get("/outdoor", response_class=HTMLResponse)
    async def outdoor_center_page(request: Request):
        """室外监测中心页面 - 融合基础巡视和高级监测"""
        return templates.TemplateResponse(
            "pages/outdoor_center.html",
            {
                "request": request,
                "active_tab": "outdoor",
                "version": "3.5.0",
            },
        )

    # ============== 室内监测中心 (V3.5新增) ==============
    @app.get("/indoor", response_class=HTMLResponse)
    async def indoor_center_page(request: Request):
        """室内监测中心页面 - 包含电子围栏、动物入侵、温度监测等"""
        return templates.TemplateResponse(
            "pages/indoor_center.html",
            {
                "request": request,
                "active_tab": "indoor",
                "version": "3.5.0",
            },
        )

    # ============== 数据导入页面 (V3.0新增) ==============
    @app.get("/data-import", response_class=HTMLResponse)
    async def data_import_page(request: Request):
        """数据导入页面"""
        return templates.TemplateResponse(
            "pages/data_import.html",
            {
                "request": request,
                "active_tab": "data_import",
                "version": "3.0.0",
            },
        )

    # ============== 统一仪表盘页面 (V3.0新增) ==============
    @app.get("/unified-dashboard", response_class=HTMLResponse)
    async def unified_dashboard_page(request: Request):
        """统一监控仪表盘页面"""
        return templates.TemplateResponse(
            "pages/unified_dashboard.html",
            {
                "request": request,
                "active_tab": "unified_dashboard",
                "version": "3.0.0",
            },
        )

    # ============== 中期迭代功能页面 (V3.0新增) ==============

    # 中期迭代模块配置
    ITERATION_MODULES = {
        "point-cloud": {
            "name": "3D点云处理",
            "icon": "box-seam",
            "description": "基于深度摄像头和毫米波雷达的3D点云处理，支持室内越线检测和多目标跟踪",
            "category": "ai",
            "features": ["深度图转点云", "点云下采样", "点云过滤", "多传感器融合"],
        },
        "temporal": {
            "name": "时序分析引擎",
            "icon": "clock-history",
            "description": "基于LSTM/Transformer的多帧时序分析，用于开关状态判定和异常检测",
            "category": "ai",
            "features": ["多帧状态融合", "异常动作检测", "状态转换分析", "置信度聚合"],
        },
        "bird-risk": {
            "name": "鸟类风险预测",
            "icon": "exclamation-triangle",
            "description": "升级版鸟类监测系统，融合YOLOv8-ViT检测和轨迹预测模型",
            "category": "ai",
            "features": ["鸟类物种分类", "轨迹预测", "风险评估", "智能驱鸟策略"],
        },
        "mlops": {
            "name": "MLOps流水线",
            "icon": "graph-up-arrow",
            "description": "完整的机器学习运维平台，支持模型训练、评估、部署和主动学习",
            "category": "platform",
            "features": ["模型注册", "训练调度", "主动学习", "Canary部署"],
        },
        "microservice": {
            "name": "微服务架构",
            "icon": "diagram-3",
            "description": "基于FastAPI/gRPC的微服务架构，支持服务注册、负载均衡和任务队列",
            "category": "platform",
            "features": ["服务注册发现", "API网关", "任务队列", "WebSocket通信"],
        },
        "monitoring": {
            "name": "系统监控",
            "icon": "activity",
            "description": "完整的可观测性平台，集成Prometheus指标、结构化日志和分布式追踪",
            "category": "platform",
            "features": ["Prometheus指标", "结构化日志", "分布式追踪", "健康检查"],
        },
    }

    @app.get("/iteration/{module_id}", response_class=HTMLResponse)
    async def iteration_page(request: Request, module_id: str):
        """中期迭代功能页面"""
        if module_id not in ITERATION_MODULES:
            return templates.TemplateResponse(
                "pages/404.html",
                {"request": request},
                status_code=404,
            )

        module = ITERATION_MODULES[module_id]

        return templates.TemplateResponse(
            "pages/iteration_module.html",
            {
                "request": request,
                "active_tab": "iteration",
                "module_id": module_id,
                "module_name": module["name"],
                "module_icon": module["icon"],
                "module_description": module["description"],
                "module_category": module["category"],
                "module_features": module["features"],
                "all_modules": ITERATION_MODULES,
                "version": "3.0.0",
            },
        )

    # ============== 集成高级插件 ==============
    if ADVANCED_PLUGINS_AVAILABLE:
        try:
            integrate_all_advanced_features(app, templates)
            print("✓ 高级插件功能已集成")
        except Exception as e:
            print(f"✗ 高级插件集成失败: {e}")

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
