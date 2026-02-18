"""
Standalone Plugin Runner

A FastAPI application that wraps any BasePlugin, providing:
- Web dashboard UI
- REST API for detection and configuration
- WebSocket streaming for real-time results
- Detection statistics tracking
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any, Optional

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


SDK_DIR = Path(__file__).parent
TEMPLATES_DIR = SDK_DIR / "templates"
STATIC_DIR = SDK_DIR / "static"


class StandalonePluginRunner:
    """
    FastAPI server that wraps any BasePlugin for standalone operation.

    Provides a web dashboard, REST API, and WebSocket streaming.
    """

    def __init__(
        self,
        plugin: BasePlugin,
        title: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        plugin_templates_dir: str | Path | None = None,
    ) -> None:
        self.plugin = plugin
        self.title = title or f"{plugin.name} - Standalone"
        self.host = host
        self.port = port

        # Detection statistics
        self._stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_alarms": 0,
            "avg_inference_time_ms": 0.0,
            "total_inference_time_ms": 0.0,
            "last_detection_time": None,
            "fps": 0.0,
        }
        self._fps_timestamps: list[float] = []

        # WebSocket connections
        self._ws_clients: list[WebSocket] = []

        # Template directories - plugin-specific first, SDK defaults as fallback
        template_dirs = []
        if plugin_templates_dir:
            template_dirs.append(str(plugin_templates_dir))
        template_dirs.append(str(TEMPLATES_DIR))

        self.templates = Jinja2Templates(directory=template_dirs)

        # Build FastAPI app
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(title=self.title)

        # Mount static files
        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        # Register routes
        app.get("/", response_class=HTMLResponse)(self._dashboard)
        app.get("/api/status")(self._get_status)
        app.get("/api/config")(self._get_config)
        app.put("/api/config")(self._update_config)
        app.post("/api/detect")(self._detect)
        app.websocket("/ws/stream")(self._ws_stream)

        # Register plugin-specific routes
        for route in self.plugin.get_standalone_routes():
            app.add_api_route(
                path=route["path"],
                endpoint=route["endpoint"],
                methods=route.get("methods", ["GET"]),
                summary=route.get("summary", ""),
            )

        return app

    async def _dashboard(self, request: Request) -> HTMLResponse:
        """Serve the plugin dashboard."""
        return self.templates.TemplateResponse(
            "plugin_dashboard.html",
            {
                "request": request,
                "plugin_name": self.plugin.name,
                "plugin_id": self.plugin.id,
                "plugin_version": self.plugin.version,
                "title": self.title,
            },
        )

    async def _get_status(self) -> JSONResponse:
        """Return plugin status and statistics."""
        health = self.plugin.healthcheck()
        return JSONResponse({
            "plugin_id": self.plugin.id,
            "plugin_name": self.plugin.name,
            "plugin_version": self.plugin.version,
            "status": self.plugin.status.value,
            "health": {
                "healthy": health.healthy,
                "message": health.message,
                "last_check": health.last_check.isoformat(),
                "details": health.details,
            },
            "stats": self._stats,
        })

    async def _get_config(self) -> JSONResponse:
        """Return current plugin configuration."""
        return JSONResponse({
            "config": self.plugin._config,
            "config_schema": self.plugin.manifest.config_schema,
        })

    async def _update_config(self, request: Request) -> JSONResponse:
        """Update plugin configuration."""
        body = await request.json()
        new_config = body.get("config", {})
        self.plugin.on_config_update(new_config)
        return JSONResponse({
            "status": "updated",
            "config": self.plugin._config,
        })

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
        alarms = self.plugin.postprocess(results, [])

        # Update statistics
        self._update_stats(len(results), len(alarms), inference_time_ms)

        # Serialize results
        results_json = [r.model_dump(mode="json") for r in results]
        alarms_json = [a.model_dump(mode="json") for a in alarms]

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

    async def _ws_stream(self, websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time detection streaming."""
        await websocket.accept()
        self._ws_clients.append(websocket)
        try:
            while True:
                # Keep connection alive; handle incoming messages
                data = await websocket.receive_text()
                # Client can send commands via WebSocket
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "get_stats":
                        await websocket.send_json({
                            "type": "stats",
                            "data": self._stats,
                        })
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            self._ws_clients.remove(websocket)

    async def _broadcast(self, data: dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients."""
        disconnected = []
        for ws in self._ws_clients:
            try:
                await ws.send_json({"type": "detection", "data": data})
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
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

    def run(self) -> None:
        """Start the standalone server."""
        import uvicorn
        print(f"Starting {self.title}")
        print(f"  Plugin: {self.plugin.id} v{self.plugin.version}")
        print(f"  Dashboard: http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)
