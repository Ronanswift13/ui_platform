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

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import base64
import io

from platform_core.config import get_config

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


class CameraRegisterRequest(BaseModel):
    """摄像头注册请求"""
    camera_id: str
    camera_type: str = "rtsp"
    url: str = ""
    device_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 25
    buffer_size: int = 1
    auto_connect: bool = True


class CameraDetectRequest(BaseModel):
    """摄像头检测请求"""
    camera_id: str
    defect_types: list[str] = []
    state_types: list[str] = []
    thermal_enabled: bool = False
    thermal_threshold: float = 80.0
    thermal_min_temp: float = 20.0
    thermal_max_temp: float = 120.0
    algorithms: dict[str, Any] = {}


# ============== API 应用 ==============

def _get_label_cn(label: str) -> str:
    """获取标签的中文名称"""
    label_map = {
        "damage": "破损",
        "rust": "锈蚀",
        "oil_leak": "渗漏油",
        "foreign_object": "异物悬挂",
        "silica_gel_normal": "硅胶正常",
        "silica_gel_abnormal": "硅胶变色",
        "valve_open": "阀门开启",
        "valve_closed": "阀门关闭",
    }
    return label_map.get(label, label)


def _get_camera_device(camera_id: str):
    """获取摄像头设备"""
    from platform_core.device_adapter import DeviceManager
    from platform_core.device_adapter.camera import CameraDevice

    dm = DeviceManager()
    device = dm.get_device(camera_id)
    if device is None or not isinstance(device, CameraDevice):
        raise HTTPException(status_code=404, detail="摄像头不存在")
    return device


def _encode_jpeg(frame: Any, quality: int = 85) -> bytes:
    """编码帧为JPEG字节"""
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("图像编码失败")
    return buffer.tobytes()


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

    # 自动加载插件
    config = get_config()
    if config.plugin.auto_load:
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()
        manifests = pm.discover_plugins()
        for manifest in manifests:
            try:
                pm.load_plugin(manifest.id)
            except Exception as e:
                print(f"自动加载插件 {manifest.id} 失败: {e}")

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

    # ============== 摄像头管理 ==============

    @app.get("/api/cameras", tags=["设备"])
    async def list_cameras():
        """获取摄像头列表"""
        from platform_core.device_adapter import DeviceManager
        from platform_core.device_adapter.camera import CameraDevice

        dm = DeviceManager()
        cameras = []
        for device in dm.list_devices():
            if not isinstance(device, CameraDevice):
                continue
            config = device.camera_config
            cameras.append({
                "id": device.device_id,
                "type": "camera",
                "name": device.info.name if device.info else "Camera",
                "status": device.status.value,
                "is_connected": device.is_connected,
                "camera_type": config.camera_type.value,
                "width": config.width,
                "height": config.height,
                "fps": config.fps,
            })
        return cameras

    @app.post("/api/cameras/register", tags=["设备"])
    async def register_camera(request: CameraRegisterRequest):
        """注册摄像头设备"""
        from platform_core.device_adapter import DeviceManager

        dm = DeviceManager()
        if dm.get_device(request.camera_id):
            dm.remove_device(request.camera_id)

        config = {
            "camera_type": request.camera_type,
            "url": request.url,
            "device_index": request.device_index,
            "width": request.width,
            "height": request.height,
            "fps": request.fps,
            "buffer_size": request.buffer_size,
        }

        dm.create_device(request.camera_id, "camera", config)
        connected = False
        if request.auto_connect:
            connected = dm.connect_device(request.camera_id)

        return {
            "camera_id": request.camera_id,
            "status": "connected" if connected else "created",
            "connected": connected,
            "config": config,
        }

    @app.get("/api/cameras/{camera_id}/status", tags=["设备"])
    async def camera_status(camera_id: str):
        """获取摄像头状态"""
        device = _get_camera_device(camera_id)
        return device.get_status_info()

    @app.post("/api/cameras/{camera_id}/connect", tags=["设备"])
    async def connect_camera(camera_id: str):
        """连接摄像头"""
        from platform_core.device_adapter import DeviceManager

        dm = DeviceManager()
        success = dm.connect_device(camera_id)
        return {"status": "connected" if success else "failed", "camera_id": camera_id}

    @app.post("/api/cameras/{camera_id}/disconnect", tags=["设备"])
    async def disconnect_camera(camera_id: str):
        """断开摄像头"""
        from platform_core.device_adapter import DeviceManager

        dm = DeviceManager()
        success = dm.disconnect_device(camera_id)
        return {"status": "disconnected" if success else "failed", "camera_id": camera_id}

    @app.get("/api/cameras/{camera_id}/snapshot", tags=["设备"])
    async def snapshot_camera(camera_id: str):
        """抓拍摄像头图像"""
        device = _get_camera_device(camera_id)
        if not device.is_connected:
            raise HTTPException(status_code=503, detail="摄像头未连接")

        frame = device.capture()
        if frame is None:
            raise HTTPException(status_code=503, detail="摄像头无信号输入")

        encoded = _encode_jpeg(frame)
        return {
            "camera_id": camera_id,
            "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("utf-8"),
        }

    @app.get("/api/cameras/{camera_id}/stream", tags=["设备"])
    async def stream_camera(camera_id: str, fps: int = Query(10, ge=1, le=30)):
        """摄像头视频流 (MJPEG)"""
        device = _get_camera_device(camera_id)
        if not device.is_connected:
            raise HTTPException(status_code=503, detail="摄像头未连接")

        def frame_generator():
            import time

            while True:
                frame = device.capture()
                if frame is None:
                    time.sleep(0.2)
                    continue
                try:
                    encoded = _encode_jpeg(frame, quality=80)
                except Exception:
                    time.sleep(0.1)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    encoded +
                    b"\r\n"
                )
                time.sleep(max(1.0 / fps, 0.03))

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ============== 图像检测 ==============

    @app.post("/api/detect/{plugin_id}", tags=["检测"])
    async def detect_image(
        plugin_id: str,
        file: UploadFile = File(...),
        detection_types: str = Form("defect,state"),
    ):
        """
        执行图像检测

        Args:
            plugin_id: 插件ID
            file: 上传的图像文件
            detection_types: 检测类型，逗号分隔 (defect, state)
        """
        import cv2
        import numpy as np
        from uuid import uuid4
        from datetime import datetime
        from platform_core.plugin_manager import PluginManager
        from platform_core.plugin_manager.base import PluginContext
        from platform_core.schema.models import ROI, BoundingBox, ROIType

        # 获取插件
        pm = PluginManager()
        plugin = pm.get_plugin(plugin_id)
        if plugin is None:
            try:
                plugin = pm.load_plugin(plugin_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"插件加载失败: {e}")

        # 读取图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")

        h, w = frame.shape[:2]
        task_id = str(uuid4())

        # 创建上下文
        context = PluginContext(
            task_id=task_id,
            site_id="web_upload",
            device_id="manual",
            component_id="user_upload",
        )

        # 解析检测类型
        types = [t.strip() for t in detection_types.split(",")]

        # 创建全图ROI
        rois = []
        if "defect" in types:
            rois.append(ROI(
                id="full_defect",
                name="全图缺陷检测",
                component_id="transformer_body",
                roi_type=ROIType.DEFECT,
                bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            ))
        if "state" in types:
            rois.append(ROI(
                id="full_state",
                name="全图状态识别",
                component_id="transformer_body",
                roi_type=ROIType.STATE,
                bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            ))

        # 执行推理
        try:
            results = plugin.infer(frame, rois, context)

            # 后处理生成告警
            alarms = plugin.postprocess(results, [])

            # 在图像上绘制结果
            annotated = frame.copy()
            for result in results:
                bbox = result.bbox
                x1 = int(bbox.x * w)
                y1 = int(bbox.y * h)
                x2 = int((bbox.x + bbox.width) * w)
                y2 = int((bbox.y + bbox.height) * h)

                # 根据标签选择颜色
                if result.label in ["oil_leak", "damage"]:
                    color = (0, 0, 255)  # 红色 - 严重
                elif result.label in ["rust", "foreign_object", "silica_gel_abnormal"]:
                    color = (0, 165, 255)  # 橙色 - 警告
                else:
                    color = (0, 255, 0)  # 绿色 - 正常

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # 添加标签
                label_text = f"{result.label}: {result.confidence:.2f}"
                cv2.putText(annotated, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 编码结果图像为base64
            _, buffer = cv2.imencode('.jpg', annotated)
            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

            return {
                "success": True,
                "task_id": task_id,
                "plugin_id": plugin_id,
                "image_size": {"width": w, "height": h},
                "detection_types": types,
                "results": [
                    {
                        "roi_id": r.roi_id,
                        "label": r.label,
                        "label_cn": _get_label_cn(r.label),
                        "confidence": r.confidence,
                        "value": r.value,
                        "bbox": {
                            "x": r.bbox.x,
                            "y": r.bbox.y,
                            "width": r.bbox.width,
                            "height": r.bbox.height,
                        }
                    }
                    for r in results
                ],
                "alarms": [
                    {
                        "level": a.level.value,
                        "title": a.title,
                        "message": a.message,
                    }
                    for a in alarms
                ],
                "annotated_image": f"data:image/jpeg;base64,{img_base64}",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

    @app.post("/api/detect/{plugin_id}/camera", tags=["检测"])
    async def detect_camera(plugin_id: str, request: CameraDetectRequest):
        """
        执行摄像头检测

        Args:
            plugin_id: 插件ID
            request: 摄像头检测请求
        """
        import cv2
        from uuid import uuid4
        from datetime import datetime
        from platform_core.plugin_manager import PluginManager
        from platform_core.plugin_manager.base import PluginContext
        from platform_core.schema.models import ROI, BoundingBox, ROIType

        # 获取插件
        pm = PluginManager()
        plugin = pm.get_plugin(plugin_id)
        if plugin is None:
            try:
                plugin = pm.load_plugin(plugin_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"插件加载失败: {e}")

        # 获取摄像头
        camera = _get_camera_device(request.camera_id)
        if not camera.is_connected:
            raise HTTPException(status_code=503, detail="摄像头未连接")

        # 捕获图像
        frame = camera.capture()
        if frame is None:
            raise HTTPException(status_code=503, detail="摄像头无信号输入")

        h, w = frame.shape[:2]
        task_id = str(uuid4())

        # 运行配置
        runtime_config = {
            "recognition": {
                "defect_types": request.defect_types,
                "state_types": request.state_types,
            },
            "thermal": {
                "enabled": request.thermal_enabled,
                "temperature_threshold": request.thermal_threshold,
                "min_temp": request.thermal_min_temp,
                "max_temp": request.thermal_max_temp,
            },
            "algorithms": request.algorithms,
        }

        # 创建上下文
        context = PluginContext(
            task_id=task_id,
            site_id="camera_stream",
            device_id=request.camera_id,
            component_id="transformer_body",
            config=runtime_config,
        )

        # 创建ROI
        rois = []
        if request.defect_types:
            rois.append(ROI(
                id="full_defect",
                name="全图缺陷检测",
                component_id="transformer_body",
                roi_type=ROIType.DEFECT,
                bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            ))
        if request.state_types:
            rois.append(ROI(
                id="full_state",
                name="全图状态识别",
                component_id="transformer_body",
                roi_type=ROIType.STATE,
                bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            ))

        if not rois and not request.thermal_enabled:
            raise HTTPException(status_code=400, detail="未选择检测类型")

        # 执行推理
        try:
            results = plugin.infer(frame, rois, context) if rois else []

            # 后处理生成告警
            alarms = plugin.postprocess(results, [])

            # 在图像上绘制结果
            annotated = frame.copy()
            for result in results:
                bbox = result.bbox
                x1 = int(bbox.x * w)
                y1 = int(bbox.y * h)
                x2 = int((bbox.x + bbox.width) * w)
                y2 = int((bbox.y + bbox.height) * h)

                # 根据标签选择颜色
                if result.label in ["oil_leak", "damage"]:
                    color = (0, 0, 255)  # 红色 - 严重
                elif result.label in ["rust", "foreign_object", "silica_gel_abnormal"]:
                    color = (0, 165, 255)  # 橙色 - 警告
                else:
                    color = (0, 255, 0)  # 绿色 - 正常

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # 添加标签
                label_text = f"{result.label}: {result.confidence:.2f}"
                cv2.putText(annotated, label_text, (x1, max(0, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 编码原始和结果图像
            raw_base64 = base64.b64encode(_encode_jpeg(frame)).decode("utf-8")
            annotated_base64 = base64.b64encode(_encode_jpeg(annotated)).decode("utf-8")

            # 热成像分析
            thermal = None
            if request.thermal_enabled and hasattr(plugin, "analyze_thermal"):
                thermal = plugin.analyze_thermal(frame, runtime_config)

            return {
                "success": True,
                "task_id": task_id,
                "plugin_id": plugin_id,
                "camera_id": request.camera_id,
                "image_size": {"width": w, "height": h},
                "results": [
                    {
                        "roi_id": r.roi_id,
                        "label": r.label,
                        "label_cn": _get_label_cn(r.label),
                        "confidence": r.confidence,
                        "value": r.value,
                        "bbox": {
                            "x": r.bbox.x,
                            "y": r.bbox.y,
                            "width": r.bbox.width,
                            "height": r.bbox.height,
                        }
                    }
                    for r in results
                ],
                "alarms": [
                    {
                        "level": a.level.value,
                        "title": a.title,
                        "message": a.message,
                    }
                    for a in alarms
                ],
                "snapshot_image": f"data:image/jpeg;base64,{raw_base64}",
                "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
                "thermal": thermal,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

    @app.get("/api/detect/{plugin_id}/capabilities", tags=["检测"])
    async def get_detection_capabilities(plugin_id: str):
        """获取插件检测能力"""
        capabilities = {
            "transformer_inspection": {
                "name": "主变自主巡视",
                "detection_types": [
                    {"id": "defect", "name": "缺陷检测", "description": "渗漏油、锈蚀、破损、异物检测"},
                    {"id": "state", "name": "状态识别", "description": "呼吸器硅胶、阀门状态识别"},
                    {"id": "thermal", "name": "热成像", "description": "红外图像温度提取"},
                ],
                "defect_types": [
                    {"id": "oil_leak", "name": "渗漏油", "level": "error", "color": "#dc3545"},
                    {"id": "damage", "name": "破损", "level": "error", "color": "#dc3545"},
                    {"id": "rust", "name": "锈蚀", "level": "warning", "color": "#fd7e14"},
                    {"id": "foreign_object", "name": "异物悬挂", "level": "warning", "color": "#fd7e14"},
                ],
                "state_types": [
                    {"id": "silica_gel_normal", "name": "硅胶正常", "color": "#28a745"},
                    {"id": "silica_gel_abnormal", "name": "硅胶变色", "level": "warning", "color": "#fd7e14"},
                    {"id": "valve_open", "name": "阀门开启", "color": "#17a2b8"},
                    {"id": "valve_closed", "name": "阀门关闭", "color": "#6c757d"},
                ],
            }
        }

        if plugin_id not in capabilities:
            # 返回通用能力
            return {
                "name": plugin_id,
                "detection_types": [
                    {"id": "defect", "name": "缺陷检测", "description": "通用缺陷检测"},
                ],
                "defect_types": [],
                "state_types": [],
            }

        return capabilities[plugin_id]

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

    # ============== 集成增强路由 ==============
    try:
        from platform_core.api_routes import integrate_enhanced_routes
        integrate_enhanced_routes(app)
    except ImportError as e:
        print(f"[Warning] 增强路由模块未加载: {e}")

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
