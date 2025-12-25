"""
REST API 服务器

提供HTTP API接口:
- 任务管理
- 插件管理
- 证据查询
- 健康检查
"""


from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============== 请求/响应模型 ==============

class TaskRunRequest(BaseModel):
    """任务运行请求"""
    site_id: str
    position_id: str
    device_id: str
    plugin_id: str
    task_template: Optional[str] = None
    config: dict[str, Any] = {}
    roi_ids: list[str] = []


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: str
    message: str = ""


class PluginResponse(BaseModel):
    """插件响应"""
    id: str
    name: str
    version: str
    status: str
    capabilities: list[str]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    plugins_loaded: int
    devices_connected: int


class EvidenceRunResponse(BaseModel):
    """证据运行记录响应"""
    run_id: str
    task_id: str
    plugin_id: str
    success: bool
    started_at: str
    completed_at: Optional[str]


# ============== API 应用 ==============

def create_api_app() -> FastAPI:
    """创建API应用"""
    app = FastAPI(
        title="输变电监测平台 API",
        description="输变电激光星芒破夜绘明监测平台 REST API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ============== 健康检查 ==============

    @app.get("/api/health", response_model=HealthResponse, tags=["系统"])
    async def health_check():
        """健康检查"""
        from platform_core.plugin_manager import PluginManager
        from platform_core.device_adapter import DeviceManager

        pm = PluginManager()
        dm = DeviceManager()

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            plugins_loaded=len(pm.list_plugins()),
            devices_connected=sum(1 for d in dm.list_devices() if d.is_connected),
        )

    @app.get("/api/status", tags=["系统"])
    async def get_status():
        """获取系统状态"""
        from platform_core.plugin_manager import PluginManager
        from platform_core.device_adapter import DeviceManager
        from platform_core.scheduler import TaskEngine

        pm = PluginManager()
        dm = DeviceManager()
        te = TaskEngine()

        return {
            "status": "running",
            "plugins": pm.registry.get_statistics(),
            "devices": dm.get_status_summary(),
            "running_tasks": len(te.get_running_tasks()),
        }

    # ============== 站点管理 ==============

    @app.get("/api/sites", tags=["站点"])
    async def list_sites():
        """获取站点列表"""
        # TODO: 从配置或数据库加载
        return [
            {"id": "site_001", "name": "示例变电站", "code": "S001"},
        ]

    @app.get("/api/sites/{site_id}", tags=["站点"])
    async def get_site(site_id: str):
        """获取站点详情"""
        return {
            "id": site_id,
            "name": "示例变电站",
            "positions": [],
        }

    # ============== 插件管理 ==============

    @app.get("/api/plugins", response_model=list[PluginResponse], tags=["插件"])
    async def list_plugins():
        """获取插件列表"""
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()
        manifests = pm.discover_plugins()

        return [
            PluginResponse(
                id=m.id,
                name=m.name,
                version=m.version,
                status="discovered",
                capabilities=[c.value for c in m.capabilities],
            )
            for m in manifests
        ]

    @app.get("/api/plugins/{plugin_id}", tags=["插件"])
    async def get_plugin(plugin_id: str):
        """获取插件详情"""
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()
        plugin = pm.get_plugin(plugin_id)

        if plugin is None:
            raise HTTPException(status_code=404, detail="插件不存在")

        return {
            "id": plugin.id,
            "name": plugin.name,
            "version": plugin.version,
            "code_hash": plugin.code_hash,
            "status": plugin.status.value,
            "capabilities": [c.value for c in plugin.manifest.capabilities],
        }

    @app.post("/api/plugins/{plugin_id}/load", tags=["插件"])
    async def load_plugin(plugin_id: str):
        """加载插件"""
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()
        try:
            plugin = pm.load_plugin(plugin_id)
            return {
                "status": "success",
                "plugin": {
                    "id": plugin.id,
                    "name": plugin.name,
                    "version": plugin.version,
                    "status": plugin.status.value,
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/plugins/{plugin_id}/unload", tags=["插件"])
    async def unload_plugin(plugin_id: str):
        """卸载插件"""
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()
        success = pm.unload_plugin(plugin_id)
        if not success:
            raise HTTPException(status_code=404, detail="插件不存在或无法卸载")
        return {"status": "success", "plugin_id": plugin_id}

    @app.post("/api/plugins/{plugin_id}/reload", tags=["插件"])
    async def reload_plugin(plugin_id: str):
        """重新加载插件"""
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()
        try:
            plugin = pm.reload_plugin(plugin_id)
            return {"status": "success", "plugin": plugin.id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/plugins/{plugin_id}/health", tags=["插件"])
    async def plugin_healthcheck(plugin_id: str):
        """插件健康检查"""
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()
        plugin = pm.get_plugin(plugin_id)

        if plugin is None:
            # 尝试加载插件
            try:
                plugin = pm.load_plugin(plugin_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"插件加载失败: {e}")

        health = plugin.healthcheck()
        return {
            "plugin_id": plugin_id,
            "healthy": health.healthy,
            "message": health.message,
            "last_check": health.last_check.isoformat(),
            "details": health.details,
        }

    @app.get("/api/plugins/health", tags=["插件"])
    async def all_plugins_healthcheck():
        """所有插件健康检查"""
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()

        # 先发现并加载所有插件
        manifests = pm.discover_plugins()
        for m in manifests:
            if pm.get_plugin(m.id) is None:
                try:
                    pm.load_plugin(m.id)
                except Exception:
                    pass

        results = pm.healthcheck_all()
        return {
            plugin_id: {
                "healthy": status.healthy,
                "message": status.message,
                "last_check": status.last_check.isoformat(),
            }
            for plugin_id, status in results.items()
        }

    @app.get("/api/plugins/{plugin_id}/template", tags=["插件"])
    async def get_plugin_template(plugin_id: str):
        """获取插件开发模板"""
        return {
            "plugin_id": plugin_id,
            "template_url": f"/static/templates/plugin_template.zip",
            "documentation_url": f"/docs/plugins/{plugin_id}",
        }

    # ============== 任务管理 ============== #

    @app.post("/api/tasks/run", response_model=TaskResponse, tags=["任务"])
    async def run_task(request: TaskRunRequest, background_tasks: BackgroundTasks):
        """运行任务"""
        from uuid import uuid4

        task_id = str(uuid4())

        # TODO: 实际任务执行逻辑
        # background_tasks.add_task(execute_task, task_id, request)

        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="任务已提交",
        )

    @app.get("/api/tasks/{task_id}", tags=["任务"])
    async def get_task(task_id: str):
        """获取任务状态"""
        from platform_core.scheduler import TaskEngine

        te = TaskEngine()
        result = te.get_task_result(task_id)

        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")

        return {
            "task_id": task_id,
            "status": "completed" if result.success else "failed",
            "run_id": result.run_id,
            "duration_ms": result.duration_ms,
        }

    @app.post("/api/tasks/{task_id}/cancel", tags=["任务"])
    async def cancel_task(task_id: str):
        """取消任务"""
        from platform_core.scheduler import TaskEngine

        te = TaskEngine()
        success = te.cancel_task(task_id)

        if not success:
            raise HTTPException(status_code=404, detail="任务不存在或无法取消")

        return {"status": "cancelled", "task_id": task_id}

    # ============== 证据管理 ==============

    @app.get("/api/evidence/runs", tags=["证据"])
    async def list_evidence_runs(
        plugin_id: Optional[str] = Query(None),
        limit: int = Query(50, le=100),
    ):
        """获取证据运行记录列表"""
        from platform_core.evidence import EvidenceManager

        em = EvidenceManager()
        runs = em.list_runs(plugin_id=plugin_id, limit=limit)

        return [
            {
                "run_id": r.run_id,
                "task_id": r.task_id,
                "plugin_id": r.plugin_id,
                "success": r.success,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in runs
        ]

    @app.get("/api/evidence/runs/{run_id}", tags=["证据"])
    async def get_evidence_run(run_id: str):
        """获取证据运行记录详情"""
        from platform_core.evidence import EvidenceManager

        em = EvidenceManager()
        meta = em.get_run(run_id)

        if meta is None:
            raise HTTPException(status_code=404, detail="记录不存在")

        files = em.get_run_files(run_id)

        return {
            "metadata": meta.to_dict(),
            "files": [str(f) for f in files],
        }

    @app.delete("/api/evidence/runs/{run_id}", tags=["证据"])
    async def delete_evidence_run(run_id: str):
        """删除证据运行记录"""
        from platform_core.evidence import EvidenceManager

        em = EvidenceManager()
        success = em.delete_run(run_id)

        if not success:
            raise HTTPException(status_code=404, detail="记录不存在")

        return {"status": "deleted", "run_id": run_id}

    # ============== 设备管理 ==============

    @app.get("/api/devices", tags=["设备"])
    async def list_devices():
        """获取设备列表"""
        from platform_core.device_adapter import DeviceManager

        dm = DeviceManager()
        devices = dm.list_devices()

        return [d.get_status_info() for d in devices]

    @app.post("/api/devices/{device_id}/connect", tags=["设备"])
    async def connect_device(device_id: str):
        """连接设备"""
        from platform_core.device_adapter import DeviceManager

        dm = DeviceManager()
        success = dm.connect_device(device_id)

        return {"status": "connected" if success else "failed", "device_id": device_id}

    @app.post("/api/devices/{device_id}/disconnect", tags=["设备"])
    async def disconnect_device(device_id: str):
        """断开设备"""
        from platform_core.device_adapter import DeviceManager

        dm = DeviceManager()
        success = dm.disconnect_device(device_id)

        return {"status": "disconnected" if success else "failed", "device_id": device_id}

    return app


def main():
    """主函数"""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="输变电监测平台 API 服务器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="热重载")
    args = parser.parse_args()

    app = create_api_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
