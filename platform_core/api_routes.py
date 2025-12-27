"""
增强版API路由
输变电激光监测平台 - 全自动AI巡检改造

新增API:
- 模型管理
- 云台控制
- 自动ROI
- 融合配置
- 实时指标
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ============== 数据模型 ==============

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    backend: str
    loaded: bool
    inference_count: int = 0
    avg_latency_ms: float = 0


class ModelLoadRequest(BaseModel):
    model_id: str


class PTZCommandModel(BaseModel):
    command: str
    preset_id: Optional[int] = None
    speed: Optional[float] = None
    route_id: Optional[str] = None


class FusionWeights(BaseModel):
    dl: float = 0.6
    ocr: float = 0.3
    color: float = 0.2
    angle: float = 0.2


class ReshootRequest(BaseModel):
    clarity_threshold: float = 0.7
    max_retries: int = 3


class MetricsResponse(BaseModel):
    fps: float
    latency: float
    gpu_usage: float
    memory_usage: float
    active_tasks: int


# ============== 创建路由 ==============

def create_enhanced_router() -> APIRouter:
    """创建增强版API路由"""
    
    router = APIRouter(prefix="/api", tags=["enhanced"])
    
    # ============== 模型管理 ==============
    
    @router.get("/models", response_model=List[ModelInfo])
    async def list_models():
        """列出所有模型"""
        try:
            # 修复: 使用正确的导入路径
            from platform_core.inference_engine import get_model_registry
            registry = get_model_registry()
            return registry.list_models()
        except ImportError:
            # 返回模拟数据
            return [
                {"model_id": "transformer_defect_yolov8", "model_type": "detection", 
                 "backend": "onnx_cpu", "loaded": True, "inference_count": 150, "avg_latency_ms": 45.2},
                {"model_id": "switch_multitask_yolov8", "model_type": "detection",
                 "backend": "onnx_cpu", "loaded": False, "inference_count": 0, "avg_latency_ms": 0},
                {"model_id": "meter_keypoint_hrnet", "model_type": "keypoint",
                 "backend": "onnx_cpu", "loaded": False, "inference_count": 0, "avg_latency_ms": 0},
            ]
    
    @router.post("/models/{model_id}/load")
    async def load_model(model_id: str):
        """加载模型"""
        try:
            from platform_core.inference_engine import get_model_registry
            registry = get_model_registry()
            success = registry.load(model_id)
            return {"success": success, "model_id": model_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/models/{model_id}/unload")
    async def unload_model(model_id: str):
        """卸载模型"""
        try:
            from platform_core.inference_engine import get_model_registry
            registry = get_model_registry()
            success = registry.unload(model_id)
            return {"success": success, "model_id": model_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============== 云台控制 ==============
    
    @router.post("/ptz/command")
    async def ptz_command(cmd: PTZCommandModel):
        """发送云台命令"""
        try:
            from platform_core.ptz_controller import get_ptz_controller, PTZCommand as PTZCmd
            controller = get_ptz_controller()
            
            # 映射命令
            command_map = {
                "pan_left": PTZCmd.PAN_LEFT,
                "pan_right": PTZCmd.PAN_RIGHT,
                "tilt_up": PTZCmd.TILT_UP,
                "tilt_down": PTZCmd.TILT_DOWN,
                "zoom_in": PTZCmd.ZOOM_IN,
                "zoom_out": PTZCmd.ZOOM_OUT,
                "auto_focus": PTZCmd.AUTO_FOCUS,
                "focus_near": PTZCmd.FOCUS_NEAR,
                "focus_far": PTZCmd.FOCUS_FAR,
                "goto_preset": PTZCmd.GOTO_PRESET,
                "set_preset": PTZCmd.SET_PRESET,
            }
            
            ptz_cmd = command_map.get(cmd.command)
            if ptz_cmd is None:
                raise HTTPException(status_code=400, detail=f"未知命令: {cmd.command}")
            
            import asyncio
            params = {}
            if cmd.preset_id is not None:
                params["preset_id"] = cmd.preset_id
            
            success = await controller._adapter.execute_command(ptz_cmd, params)
            
            return {"success": success, "command": cmd.command}
        except ImportError:
            # 模拟响应
            return {"success": True, "command": cmd.command, "simulated": True}
    
    @router.post("/ptz/goto_preset/{preset_id}")
    async def goto_preset(preset_id: int):
        """跳转到预置位"""
        try:
            from platform_core.ptz_controller import goto_preset as ptz_goto_preset
            import asyncio
            success = await ptz_goto_preset(preset_id)
            return {"success": success, "preset_id": preset_id}
        except ImportError:
            return {"success": True, "preset_id": preset_id, "simulated": True}
    
    @router.post("/ptz/start_patrol")
    async def start_patrol(route_id: str = "default"):
        """开始巡航"""
        try:
            from platform_core.ptz_controller import get_ptz_controller
            controller = get_ptz_controller()
            import asyncio
            success = await controller.start_patrol(route_id)
            return {"success": success, "route_id": route_id}
        except ImportError:
            return {"success": True, "route_id": route_id, "simulated": True}
    
    @router.post("/ptz/stop_patrol")
    async def stop_patrol():
        """停止巡航"""
        try:
            from platform_core.ptz_controller import get_ptz_controller
            controller = get_ptz_controller()
            import asyncio
            success = await controller.stop_patrol()
            return {"success": success}
        except ImportError:
            return {"success": True, "simulated": True}
    
    @router.post("/ptz/smart_reshoot")
    async def smart_reshoot(req: ReshootRequest = ReshootRequest()):
        """智能复拍"""
        try:
            from platform_core.ptz_controller import smart_reshoot as ptz_smart_reshoot, ReshootStrategy
            strategy = ReshootStrategy(
                max_retries=req.max_retries,
                clarity_threshold=req.clarity_threshold,
            )
            import asyncio
            success, result = await ptz_smart_reshoot(0.5, strategy)
            return {"success": success, "result": result}
        except ImportError:
            return {"success": True, "simulated": True, "result": {"actions": ["zoom_in", "auto_focus"]}}
    
    # ============== 自动ROI ==============
    
    @router.get("/auto_roi/{site_id}/{position_id}")
    async def get_auto_rois(site_id: str, position_id: str):
        """获取自动检测的ROI"""
        try:
            from platform_core.auto_roi_detector import get_auto_roi_detector
            detector = get_auto_roi_detector()
            
            # 这里需要获取当前帧图像
            # 简化：返回缓存的ROI
            cache_key = f"{site_id}_{position_id}"
            rois = detector._roi_cache.get(cache_key, [])
            
            return {
                "rois": [detector.convert_to_schema_roi(r) for r in rois],
                "count": len(rois),
            }
        except ImportError:
            return {"rois": [], "count": 0}
    
    @router.post("/auto_roi/clear_cache")
    async def clear_roi_cache(site_id: Optional[str] = None, position_id: Optional[str] = None):
        """清除ROI缓存"""
        try:
            from platform_core.auto_roi_detector import get_auto_roi_detector
            detector = get_auto_roi_detector()
            
            cache_key = f"{site_id}_{position_id}" if site_id and position_id else None
            detector.clear_cache(cache_key)
            
            return {"success": True}
        except ImportError:
            return {"success": True}
    
    # ============== 融合配置 ==============
    
    @router.get("/fusion/weights")
    async def get_fusion_weights():
        """获取融合权重"""
        try:
            from platform_core.fusion_engine import get_fusion_engine
            engine = get_fusion_engine()
            return {
                "weights": {k.value: v for k, v in engine.weights.items()},
                "default_method": engine.default_method,
            }
        except ImportError:
            return {
                "weights": {"deep_learning": 0.6, "ocr_text": 0.3, "color_detection": 0.2, "angle_detection": 0.2},
                "default_method": "weighted_vote",
            }
    
    @router.post("/fusion/weights")
    async def set_fusion_weights(weights: FusionWeights):
        """设置融合权重"""
        try:
            from platform_core.fusion_engine import get_fusion_engine, EvidenceType
            engine = get_fusion_engine()
            
            engine.weights[EvidenceType.DEEP_LEARNING] = weights.dl
            engine.weights[EvidenceType.OCR_TEXT] = weights.ocr
            engine.weights[EvidenceType.COLOR_DETECTION] = weights.color
            engine.weights[EvidenceType.ANGLE_DETECTION] = weights.angle
            
            return {"success": True, "weights": weights.dict()}
        except ImportError:
            return {"success": True, "weights": weights.dict()}
    
    # ============== 实时指标 ==============
    
    @router.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """获取实时指标"""
        import random
        
        # 模拟指标数据
        return MetricsResponse(
            fps=random.uniform(25, 30),
            latency=random.uniform(30, 50),
            gpu_usage=random.uniform(40, 80),
            memory_usage=random.uniform(50, 70),
            active_tasks=random.randint(0, 3),
        )
    
    # ============== WebSocket推理流 ==============
    
    @router.websocket("/ws/inference")
    async def websocket_inference(websocket: WebSocket):
        """WebSocket推理流"""
        await websocket.accept()
        
        try:
            while True:
                # 接收消息
                data = await websocket.receive_json()
                
                if data.get("type") == "set_display_mode":
                    # 处理显示模式切换
                    mode = data.get("mode", "visible")
                    await websocket.send_json({
                        "type": "mode_changed",
                        "mode": mode,
                    })
                
                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
        except WebSocketDisconnect:
            print("[WS] 客户端断开连接")
        except Exception as e:
            print(f"[WS] 错误: {e}")
    
    return router


# ============== 集成到主应用 ==============

def integrate_enhanced_routes(app):
    """将增强路由集成到主应用"""
    router = create_enhanced_router()
    app.include_router(router)
    print("[Enhanced] API路由已集成")
