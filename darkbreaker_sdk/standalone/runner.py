"""
Standalone Plugin Runner

A FastAPI application that wraps any BasePlugin, providing:
- Web dashboard UI with plugin-specific templates
- REST API for detection, configuration and monitoring
- WebSocket streaming for real-time results
- Detection statistics tracking
- Platform-level services (inference, models, data, alarms, config)
- Voltage level selection and device type configuration

Usage:
    from darkbreaker_sdk.standalone import StandalonePluginRunner
    plugin = MyPlugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent / "templates",
        plugin_static_dir=Path(__file__).parent / "static",
    )
    runner.run(host="0.0.0.0", port=8081)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from darkbreaker_sdk.interfaces.base_plugin import BasePlugin, PluginContext
from darkbreaker_sdk.interfaces.lifecycle import PluginStatus
from darkbreaker_sdk.schemas.alarm import AlarmRule
from darkbreaker_sdk.schemas.common import ROI

logger = logging.getLogger(__name__)

SDK_DIR = Path(__file__).parent
TEMPLATES_DIR = SDK_DIR / "templates"
STATIC_DIR = SDK_DIR / "static"


class StandalonePluginRunner:
    """
    FastAPI server that wraps any BasePlugin for standalone operation.

    Provides a web dashboard, REST API, WebSocket streaming,
    and platform-level services (inference, models, data, alarms, config).
    Each plugin can provide its own templates and static files.
    """

    @staticmethod
    def _safe_get_name(plugin: Any) -> str:
        """Safely get a plugin's display name."""
        for attr in ('name', 'PLUGIN_NAME', 'PLUGIN_ID', 'id'):
            val = getattr(plugin, attr, None)
            if val and isinstance(val, str):
                return val
        return "Plugin"

    def __init__(
        self,
        plugin: BasePlugin,
        title: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        plugin_templates_dir: Optional[Union[str, Path]] = None,
        plugin_static_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.plugin = plugin
        self.title = title or f"{self._safe_get_name(plugin)} - Standalone"
        self.host = host
        self.port = port
        self.plugin_static_dir = Path(plugin_static_dir) if plugin_static_dir else None

        # Initialize platform services
        self._init_services()

        # Detection statistics
        self._stats: Dict[str, Any] = {
            "total_frames": 0,
            "total_detections": 0,
            "total_alarms": 0,
            "avg_inference_time_ms": 0.0,
            "total_inference_time_ms": 0.0,
            "last_detection_time": None,
            "fps": 0.0,
        }
        self._fps_timestamps: List[float] = []

        # WebSocket connections
        self._ws_clients: List[WebSocket] = []

        # Template directories - plugin-specific first, SDK defaults as fallback
        template_dirs: List[str] = []
        if plugin_templates_dir:
            template_dirs.append(str(plugin_templates_dir))
        template_dirs.append(str(TEMPLATES_DIR))
        self.templates = Jinja2Templates(directory=template_dirs)

        # Build FastAPI app
        self.app = self._create_app()

    def _init_services(self) -> None:
        """Initialize local platform services for standalone operation."""
        from darkbreaker_sdk.services import (
            LocalInferenceEngine,
            LocalModelRegistry,
            LocalDataManager,
            LocalAlarmManager,
            LocalConfigManager,
        )

        plugin_dir = None
        if hasattr(self.plugin, 'plugin_dir'):
            plugin_dir = str(self.plugin.plugin_dir)

        # Config manager
        self.config_manager = LocalConfigManager(plugin_dir=plugin_dir)
        if plugin_dir:
            self.config_manager.load_config()

        # Data manager
        data_dir = str(Path(plugin_dir or ".") / "data")
        self.data_manager = LocalDataManager(data_dir=data_dir)

        # Alarm manager
        self.alarm_manager = LocalAlarmManager(data_dir=data_dir)

        # Model registry
        model_dirs = []
        if plugin_dir:
            model_dirs.append(str(Path(plugin_dir) / "models"))
        model_dirs.append("models")  # global models dir
        self.model_registry = LocalModelRegistry(search_dirs=model_dirs)
        self.model_registry.scan()

        # Inference engine
        self.inference_engine = LocalInferenceEngine()

        logger.info("Platform services initialized for standalone mode")

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(title=self.title)

        # Mount static files - plugin-specific first, then SDK defaults
        if self.plugin_static_dir and self.plugin_static_dir.exists():
            app.mount(
                "/plugin-static",
                StaticFiles(directory=str(self.plugin_static_dir)),
                name="plugin_static",
            )
        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        # Register routes
        app.get("/", response_class=HTMLResponse)(self._dashboard)
        app.get("/api/status")(self._get_status)
        app.get("/api/health")(self._get_health)
        app.get("/api/config")(self._get_config)
        app.put("/api/config")(self._update_config)
        app.post("/api/detect")(self._detect)
        app.get("/api/stats")(self._get_stats)
        app.websocket("/ws/stream")(self._ws_stream)

        # Platform service routes
        app.get("/api/platform/info")(self._get_platform_info)
        app.get("/api/platform/voltage_levels")(self._get_voltage_levels)
        app.put("/api/settings/voltage_level")(self._set_voltage_level)
        app.get("/api/platform/device_types")(self._get_device_types)
        app.put("/api/settings/device_type")(self._set_device_type)
        app.get("/api/platform/models")(self._get_models)
        app.post("/api/platform/models/load")(self._load_model)
        app.get("/api/alarms")(self._get_alarms)
        app.get("/api/alarms/stats")(self._get_alarm_stats)
        app.post("/api/alarms/{alarm_id}/acknowledge")(self._acknowledge_alarm)
        app.get("/api/data/results")(self._get_results)

        # Register plugin-specific routes
        try:
            for route in self.plugin.get_standalone_routes():
                app.add_api_route(
                    path=route["path"],
                    endpoint=route["endpoint"],
                    methods=route.get("methods", ["GET"]),
                    summary=route.get("summary", ""),
                )
        except Exception as e:
            logger.warning(f"Failed to register plugin routes: {e}")

        return app

    def _get_plugin_id(self) -> str:
        """Get plugin ID from various possible attributes."""
        if hasattr(self.plugin, 'id'):
            return self.plugin.id
        if hasattr(self.plugin, 'PLUGIN_ID'):
            return self.plugin.PLUGIN_ID
        if hasattr(self.plugin, 'name'):
            return self.plugin.name
        return "unknown"

    def _get_plugin_name(self) -> str:
        """Get human-readable plugin name."""
        if hasattr(self.plugin, 'PLUGIN_NAME'):
            return self.plugin.PLUGIN_NAME
        if hasattr(self.plugin, 'name'):
            return self.plugin.name
        return self._get_plugin_id()

    def _get_plugin_version(self) -> str:
        """Get plugin version string."""
        if hasattr(self.plugin, 'version'):
            return self.plugin.version
        if hasattr(self.plugin, 'PLUGIN_VERSION'):
            return self.plugin.PLUGIN_VERSION
        return "1.0.0"

    def _get_template_context(self, request: Request) -> Dict[str, Any]:
        """Build common template context with platform info."""
        plugin_id = self._get_plugin_id()
        manifest = getattr(self.plugin, 'manifest', None)
        return {
            "request": request,
            "plugin_name": self._get_plugin_name(),
            "plugin_id": plugin_id,
            "plugin_version": self._get_plugin_version(),
            "title": self.title,
            "plugin_category": getattr(manifest, 'category', '') if manifest else '',
            "plugin_capabilities": [
                c.value if hasattr(c, 'value') else str(c)
                for c in (getattr(manifest, 'capabilities', []) if manifest else [])
            ],
            "voltage_levels": self.data_manager.list_voltage_levels(),
            "device_types": self.config_manager.list_device_types(),
            "current_voltage_level": self.config_manager.get_voltage_level() or "",
            "current_device_type": self.config_manager.get_device_type() or "",
            "models": [
                {"model_id": m.model_id, "path": m.path, "format": m.format}
                for m in self.model_registry.list_models()
            ],
            "inference_backend": self.inference_engine.backend.value,
            "inference_device": self.inference_engine.device,
        }

    async def _dashboard(self, request: Request) -> HTMLResponse:
        """Serve the plugin dashboard."""
        plugin_id = self._get_plugin_id()
        context = self._get_template_context(request)

        # Try plugin-specific template first, fall back to generic
        template_name = f"{plugin_id}.html"
        try:
            return self.templates.TemplateResponse(template_name, context)
        except Exception:
            # Fall back to generic dashboard
            return self.templates.TemplateResponse("plugin_dashboard.html", context)

    @staticmethod
    def _normalize_health(health: Any) -> Dict[str, Any]:
        """Normalize a healthcheck result to a consistent dict format.

        Handles both HealthStatus objects and plain dicts returned by plugins.
        """
        if isinstance(health, dict):
            # Plugin returns a raw dict (e.g. animal_detection)
            status_val = health.get("status", "healthy")
            is_healthy = status_val in ("healthy", "ok", True)
            return {
                "healthy": is_healthy,
                "message": health.get("message", status_val),
                "last_check": health.get("last_check"),
                "details": {k: v for k, v in health.items()
                            if k not in ("healthy", "message", "last_check", "status")},
            }
        # HealthStatus object
        return {
            "healthy": getattr(health, "healthy", True),
            "message": getattr(health, "message", "OK"),
            "last_check": (
                health.last_check.isoformat()
                if hasattr(health, "last_check") and hasattr(getattr(health, "last_check", None), "isoformat")
                else str(getattr(health, "last_check", None))
            ),
            "details": getattr(health, "details", {}),
        }

    async def _get_status(self) -> JSONResponse:
        """Return plugin status and statistics."""
        try:
            health = self.plugin.healthcheck()
            health_data = self._normalize_health(health)
        except Exception as e:
            health_data = {"healthy": True, "message": str(e), "last_check": None, "details": {}}

        # Get plugin-specific status data
        plugin_status: Dict[str, Any] = {}
        try:
            if hasattr(self.plugin, 'get_status'):
                result = self.plugin.get_status()
                if isinstance(result, dict):
                    plugin_status = result
        except Exception:
            pass

        # Safely get plugin status enum
        status_str = "ready"
        try:
            if hasattr(self.plugin, 'status'):
                s = self.plugin.status
                status_str = s.value if hasattr(s, 'value') else str(s)
        except Exception:
            pass

        return JSONResponse({
            "success": True,
            "plugin_id": self._get_plugin_id(),
            "plugin_name": self._get_plugin_name(),
            "plugin_version": self._get_plugin_version(),
            "status": status_str,
            "health": health_data,
            "stats": self._stats,
            **plugin_status,
        })

    async def _get_health(self) -> JSONResponse:
        """Lightweight health check endpoint."""
        try:
            health = self.plugin.healthcheck()
            normalized = self._normalize_health(health)
            return JSONResponse({
                "healthy": normalized["healthy"],
                "message": normalized["message"],
                "plugin_id": self._get_plugin_id(),
            })
        except Exception as e:
            return JSONResponse({"healthy": False, "message": str(e)}, status_code=500)

    async def _get_config(self) -> JSONResponse:
        """Return current plugin configuration."""
        config = {}
        schema = {}
        try:
            config = self.plugin._config if hasattr(self.plugin, '_config') else {}
            if hasattr(self.plugin, 'manifest') and self.plugin.manifest:
                schema = self.plugin.manifest.config_schema if hasattr(self.plugin.manifest, 'config_schema') else {}
        except Exception:
            pass
        return JSONResponse({"config": config, "config_schema": schema})

    async def _update_config(self, request: Request) -> JSONResponse:
        """Update plugin configuration."""
        body = await request.json()
        new_config = body.get("config", {})
        try:
            self.plugin.on_config_update(new_config)
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)
        config = self.plugin._config if hasattr(self.plugin, '_config') else {}
        return JSONResponse({"status": "updated", "config": config})

    async def _detect(self, file: UploadFile = File(...)) -> JSONResponse:
        """
        Run detection on an uploaded image.

        Reads the uploaded image, runs plugin.infer() and plugin.postprocess(),
        and returns JSON results with timing information.
        """
        try:
            import cv2
        except ImportError:
            return JSONResponse(
                {"error": "opencv-python (cv2) is required for image detection"},
                status_code=500,
            )

        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(
                {"error": "Could not decode uploaded image"},
                status_code=400,
            )

        # Create context
        context = PluginContext(
            task_id=f"standalone-{int(time.time())}",
            site_id="standalone",
            device_id="upload",
        )

        # Run inference
        start_time = time.time()
        results = self.plugin.infer(frame, [], context)
        inference_time_ms = (time.time() - start_time) * 1000

        # Post-process
        try:
            alarms = self.plugin.postprocess(results, [])
        except Exception:
            alarms = []

        # Update statistics
        self._update_stats(len(results), len(alarms), inference_time_ms)

        # Save results to data manager
        try:
            self.data_manager.save_result(self._get_plugin_id(), {
                "inference_time_ms": round(inference_time_ms, 2),
                "num_results": len(results),
                "num_alarms": len(alarms),
            })
        except Exception:
            pass

        # Register alarms with alarm manager
        for a in alarms:
            try:
                alarm_id = getattr(a, 'id', None) or getattr(a, 'alarm_id', f"alarm_{int(time.time()*1000)}")
                level = getattr(a, 'level', None)
                level_str = level.value if hasattr(level, 'value') else str(level) if level else "warning"
                message = getattr(a, 'message', None) or getattr(a, 'title', str(a))
                self.alarm_manager.create_alarm(
                    alarm_id=str(alarm_id),
                    level=level_str,
                    source_plugin=self._get_plugin_id(),
                    device_id="standalone",
                    alarm_type=getattr(a, 'alarm_type', "detection"),
                    message=str(message),
                )
            except Exception:
                pass

        # Serialize results
        results_json = []
        for r in results:
            if hasattr(r, 'model_dump'):
                results_json.append(r.model_dump(mode="json"))
            elif hasattr(r, 'dict'):
                results_json.append(r.dict())
            elif isinstance(r, dict):
                results_json.append(r)
            else:
                results_json.append(str(r))

        alarms_json = []
        for a in alarms:
            if hasattr(a, 'model_dump'):
                alarms_json.append(a.model_dump(mode="json"))
            elif hasattr(a, 'dict'):
                alarms_json.append(a.dict())
            elif isinstance(a, dict):
                alarms_json.append(a)
            else:
                alarms_json.append(str(a))

        response_data = {
            "success": True,
            "inference_time_ms": round(inference_time_ms, 2),
            "results": results_json,
            "alarms": alarms_json,
            "stats": self._stats,
        }

        # Broadcast to WebSocket clients
        await self._broadcast(response_data)

        return JSONResponse(response_data)

    async def _get_stats(self) -> JSONResponse:
        """Return detection statistics."""
        return JSONResponse({"stats": self._stats})

    async def _ws_stream(self, websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time detection streaming."""
        await websocket.accept()
        self._ws_clients.append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "get_stats":
                        await websocket.send_json({
                            "type": "stats",
                            "data": self._stats,
                        })
                    elif msg.get("type") == "get_status":
                        plugin_status = {}
                        try:
                            if hasattr(self.plugin, 'get_status'):
                                plugin_status = self.plugin.get_status()
                        except Exception:
                            pass
                        await websocket.send_json({
                            "type": "status",
                            "data": plugin_status,
                        })
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            if websocket in self._ws_clients:
                self._ws_clients.remove(websocket)

    async def _broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients."""
        disconnected = []
        for ws in self._ws_clients:
            try:
                await ws.send_json({"type": "detection", "data": data})
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if ws in self._ws_clients:
                self._ws_clients.remove(ws)

    def _update_stats(
        self, num_detections: int, num_alarms: int, inference_time_ms: float
    ) -> None:
        """Update detection statistics."""
        self._stats["total_frames"] += 1
        self._stats["total_detections"] += num_detections
        self._stats["total_alarms"] += num_alarms
        self._stats["total_inference_time_ms"] += inference_time_ms
        self._stats["avg_inference_time_ms"] = round(
            self._stats["total_inference_time_ms"] / self._stats["total_frames"], 2
        )
        self._stats["last_detection_time"] = time.time()

        # Calculate FPS (over last 10 seconds)
        now = time.time()
        self._fps_timestamps.append(now)
        self._fps_timestamps = [t for t in self._fps_timestamps if now - t < 10]
        if len(self._fps_timestamps) > 1:
            elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
            self._stats["fps"] = round(
                (len(self._fps_timestamps) - 1) / max(elapsed, 0.001), 1
            )

    # ==================== Platform Service API Endpoints ====================

    async def _get_platform_info(self) -> JSONResponse:
        """Return platform service information."""
        return JSONResponse({
            "mode": "standalone",
            "plugin_id": self._get_plugin_id(),
            "plugin_name": self._get_plugin_name(),
            "plugin_version": self._get_plugin_version(),
            "inference_backend": self.inference_engine.backend.value,
            "inference_device": self.inference_engine.device,
            "models_loaded": len(self.inference_engine.list_models()),
            "models_available": len(self.model_registry.list_models()),
            "voltage_level": self.config_manager.get_voltage_level(),
            "device_type": self.config_manager.get_device_type(),
            "alarm_stats": self.alarm_manager.get_alarm_stats(),
        })

    async def _get_voltage_levels(self) -> JSONResponse:
        """Return available voltage levels."""
        return JSONResponse({
            "voltage_levels": self.data_manager.list_voltage_levels(),
            "current": self.config_manager.get_voltage_level(),
        })

    async def _set_voltage_level(self, request: Request) -> JSONResponse:
        """Set active voltage level."""
        body = await request.json()
        level_id = body.get("voltage_level", "")
        self.config_manager.set_voltage_level(level_id)
        self.data_manager.set_voltage_level(level_id)
        return JSONResponse({"status": "ok", "voltage_level": level_id})

    async def _get_device_types(self) -> JSONResponse:
        """Return available device types."""
        return JSONResponse({
            "device_types": self.config_manager.list_device_types(),
            "current": self.config_manager.get_device_type(),
        })

    async def _set_device_type(self, request: Request) -> JSONResponse:
        """Set active device type."""
        body = await request.json()
        device_type = body.get("device_type", "")
        self.config_manager.set_device_type(device_type)
        return JSONResponse({"status": "ok", "device_type": device_type})

    async def _get_models(self) -> JSONResponse:
        """Return available and loaded models."""
        available = [
            {"model_id": m.model_id, "path": m.path, "format": m.format, "size_bytes": m.size_bytes}
            for m in self.model_registry.list_models()
        ]
        loaded = [
            {"model_id": m.model_id, "backend": m.backend.value, "loaded": m.loaded}
            for m in self.inference_engine.list_models()
        ]
        return JSONResponse({"available": available, "loaded": loaded})

    async def _load_model(self, request: Request) -> JSONResponse:
        """Load a model into the inference engine."""
        body = await request.json()
        model_id = body.get("model_id", "")
        entry = self.model_registry.get(model_id)
        if not entry:
            return JSONResponse({"error": f"Model {model_id} not found"}, status_code=404)
        input_size = tuple(body.get("input_size", [640, 640]))
        class_names = body.get("class_names", [])
        success = self.inference_engine.load_model(
            model_id=model_id,
            model_path=entry.path,
            input_size=input_size,
            class_names=class_names,
        )
        return JSONResponse({"status": "loaded" if success else "error", "model_id": model_id})

    async def _get_alarms(self, level: Optional[str] = None) -> JSONResponse:
        """Return active alarms."""
        return JSONResponse({
            "alarms": self.alarm_manager.get_active_alarms(level=level),
        })

    async def _get_alarm_stats(self) -> JSONResponse:
        """Return alarm statistics."""
        return JSONResponse(self.alarm_manager.get_alarm_stats())

    async def _acknowledge_alarm(self, alarm_id: str) -> JSONResponse:
        """Acknowledge an alarm."""
        success = self.alarm_manager.acknowledge(alarm_id)
        return JSONResponse({"status": "ok" if success else "not_found"})

    async def _get_results(self, limit: int = 100) -> JSONResponse:
        """Return recent detection results."""
        plugin_id = self._get_plugin_id()
        results = self.data_manager.get_results(plugin_id, limit=limit)
        return JSONResponse({"results": results, "count": len(results)})

    # ==================== Server Entry Point ====================

    def run(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Start the standalone server.

        Args:
            host: Override host (default: uses constructor value)
            port: Override port (default: uses constructor value)
        """
        import uvicorn

        run_host = host or self.host
        run_port = port or self.port

        plugin_id = self._get_plugin_id()
        plugin_version = self._get_plugin_version()

        print(f"\n{'='*60}")
        print(f"  DarkBreaker Plugin: {self._get_plugin_name()}")
        print(f"  ID: {plugin_id}  Version: {plugin_version}")
        print(f"  Dashboard: http://{run_host}:{run_port}")
        print(f"  API Docs:  http://{run_host}:{run_port}/docs")
        print(f"  Health:    http://{run_host}:{run_port}/api/health")
        print(f"{'='*60}\n")

        uvicorn.run(self.app, host=run_host, port=run_port)
