#!/usr/bin/env python3
"""
训练调试平台 - 独立FastAPI应用

集成:
- 数据上传 API (来自 training/data_upload_api)
- 训练管理 API (来自 training/training_api)
- Agent Skills / Project Card 查看
- 插件注册表与训练映射
- 实时训练进度 WebSocket
"""
import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 导入数据上传和训练API路由
from training.data_upload_api import (
    integrate_data_upload_routes,
    record_manager,
    PLUGIN_CLASSES,
    SUPPORTED_FORMATS,
)
from training.training_api import (
    integrate_training_routes,
    training_manager,
    TrainingStatus,
)
from training.training_api_v2 import create_v2_router

# ============================================================
# 路径常量
# ============================================================
TRAINING_DIR = _project_root / "training"
STANDALONE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = STANDALONE_DIR / "templates"
STATIC_DIR = STANDALONE_DIR / "static"
SDK_STATIC_DIR = _project_root / "darkbreaker_sdk" / "standalone" / "static"
AGENT_SKILLS_DIR = _project_root / ".agent_skills"
PROJECT_CARD_PATH = _project_root / "PROJECT_CARD.md"
PLUGIN_MAPPING_PATH = TRAINING_DIR / "registry" / "plugin_training_mapping.json"
TRAINING_CONFIG_PATH = TRAINING_DIR / "configs" / "training_config.yaml"


# ============================================================
# Agent Skills / Project Card 读取工具
# ============================================================

def _read_agent_skills() -> list:
    """读取所有 agent skills 文件"""
    skills = []
    if not AGENT_SKILLS_DIR.exists():
        return skills
    for f in sorted(AGENT_SKILLS_DIR.glob("*.md")):
        try:
            content = f.read_text(encoding="utf-8")
            skills.append({
                "filename": f.name,
                "title": f.stem.replace("_", " ").title(),
                "content": content,
                "size": len(content),
            })
        except Exception:
            pass
    return skills


def _read_project_card() -> dict:
    """读取 PROJECT_CARD.md"""
    if not PROJECT_CARD_PATH.exists():
        return {"exists": False, "content": ""}
    try:
        content = PROJECT_CARD_PATH.read_text(encoding="utf-8")
        return {"exists": True, "content": content}
    except Exception:
        return {"exists": False, "content": ""}


def _read_plugin_mapping() -> dict:
    """读取插件训练映射注册表"""
    if not PLUGIN_MAPPING_PATH.exists():
        return {}
    try:
        return json.loads(PLUGIN_MAPPING_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_data_overview() -> dict:
    """扫描训练数据目录，汇总各插件/电压等级的数据量"""
    data_dir = TRAINING_DIR / "data"
    overview = {"raw": {}, "processed": {}, "augmented": {}}
    for stage in overview:
        stage_dir = data_dir / stage
        if not stage_dir.exists():
            continue
        for voltage_dir in stage_dir.iterdir():
            if not voltage_dir.is_dir():
                continue
            voltage = voltage_dir.name
            count = sum(1 for _ in voltage_dir.rglob("*") if _.is_file())
            overview[stage][voltage] = count
    return overview


# ============================================================
# FastAPI 应用构建
# ============================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="DarkBreaker 训练调试平台",
        description="独立运行的训练数据上传与模型训练调试界面",
        version="1.0.0",
    )

    # 模板引擎
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # 静态文件挂载
    if STATIC_DIR.exists():
        app.mount("/plugin-static", StaticFiles(directory=str(STATIC_DIR)), name="plugin-static")
    if SDK_STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(SDK_STATIC_DIR)), name="sdk-static")

    # 集成已有路由
    integrate_data_upload_routes(app)
    integrate_training_routes(app)

    # 集成 v2 统一训练路由
    try:
        v2_router = create_v2_router()
        app.include_router(v2_router)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"v2 路由加载失败 (非致命): {e}")

    # ── Dashboard ──────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse("training_dashboard.html", {
            "request": request,
            "title": "训练调试平台",
            "plugin_classes": PLUGIN_CLASSES,
            "supported_formats": SUPPORTED_FORMATS,
        })

    # ── Agent Skills API ───────────────────────────────────

    @app.get("/api/agent-skills")
    async def list_agent_skills():
        return JSONResponse({"success": True, "skills": _read_agent_skills()})

    @app.get("/api/agent-skills/{filename}")
    async def get_agent_skill(filename: str):
        skills = _read_agent_skills()
        for s in skills:
            if s["filename"] == filename:
                return JSONResponse({"success": True, "skill": s})
        return JSONResponse({"success": False, "message": "未找到"}, status_code=404)

    # ── Project Card API ───────────────────────────────────

    @app.get("/api/project-card")
    async def get_project_card():
        card = _read_project_card()
        return JSONResponse({"success": True, **card})

    # ── Plugin Training Mapping ────────────────────────────

    @app.get("/api/plugin-mapping")
    async def get_plugin_mapping():
        mapping = _read_plugin_mapping()
        return JSONResponse({"success": True, "mapping": mapping})

    # ── Data Overview ──────────────────────────────────────

    @app.get("/api/data-overview")
    async def data_overview():
        overview = _get_data_overview()
        return JSONResponse({"success": True, "overview": overview})

    # ── Training Progress WebSocket ────────────────────────

    @app.websocket("/ws/training")
    async def training_ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # 等待客户端心跳或请求
                data = await websocket.receive_text()
                msg = json.loads(data) if data else {}

                # 返回所有任务的实时状态
                tasks = []
                for t in training_manager.tasks.values():
                    tasks.append(t.to_dict())

                # 返回上传记录摘要
                uploads = [r.to_dict() for r in record_manager.list_all(20)]

                await websocket.send_json({
                    "type": "status",
                    "tasks": tasks,
                    "uploads": uploads,
                })
        except WebSocketDisconnect:
            pass

    return app


# ============================================================
# 入口
# ============================================================

def main():
    app = create_app()
    print("\n" + "=" * 60)
    print("  DarkBreaker 训练调试平台")
    print("  Web界面: http://localhost:8095")
    print("  API文档: http://localhost:8095/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8095)


if __name__ == "__main__":
    main()
