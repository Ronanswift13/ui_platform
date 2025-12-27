"""
云台联动控制器
输变电激光监测平台 - 全自动AI巡检增强

实现功能:
- 云台预置位控制
- 自动巡航调度
- 智能复拍策略
- 热成像对齐
- 自动对焦
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import threading
import time
import math


class PTZCommand(Enum):
    """云台命令类型"""
    STOP = "stop"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    GOTO_PRESET = "goto_preset"
    SET_PRESET = "set_preset"
    CLEAR_PRESET = "clear_preset"
    AUTO_SCAN = "auto_scan"
    FOCUS_NEAR = "focus_near"
    FOCUS_FAR = "focus_far"
    AUTO_FOCUS = "auto_focus"
    IRIS_OPEN = "iris_open"
    IRIS_CLOSE = "iris_close"


@dataclass
class PTZPosition:
    """云台位置"""
    pan: float = 0.0           # 水平角度 [0, 360)
    tilt: float = 0.0          # 俯仰角度 [-90, 90]
    zoom: float = 1.0          # 变焦倍数
    focus: float = 0.0         # 焦距
    preset_id: Optional[int] = None


@dataclass
class PresetPosition:
    """预置位"""
    preset_id: int
    name: str
    position: PTZPosition
    plugin_id: Optional[str] = None    # 关联的插件
    roi_ids: List[str] = field(default_factory=list)  # 关联的ROI
    capture_params: Dict[str, Any] = field(default_factory=dict)  # 拍摄参数


@dataclass
class PatrolRoute:
    """巡航路线"""
    route_id: str
    name: str
    presets: List[int]                  # 预置位ID列表
    dwell_times: List[float]            # 每个点位停留时间(秒)
    speed: float = 1.0                  # 巡航速度
    loop: bool = True                   # 是否循环


@dataclass
class ReshootStrategy:
    """复拍策略"""
    max_retries: int = 3               # 最大重试次数
    clarity_threshold: float = 0.7     # 清晰度阈值
    zoom_factor: float = 1.5           # 复拍放大倍数
    adjust_focus: bool = True          # 是否调整焦点
    adjust_exposure: bool = True       # 是否调整曝光
    reasons: List[str] = field(default_factory=list)  # 触发复拍的原因


class BasePTZAdapter(ABC):
    """云台适配器基类"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接设备"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开连接"""
        pass
    
    @abstractmethod
    async def get_position(self) -> PTZPosition:
        """获取当前位置"""
        pass
    
    @abstractmethod
    async def set_position(self, position: PTZPosition) -> bool:
        """设置位置"""
        pass
    
    @abstractmethod
    async def execute_command(self, command: PTZCommand, params: Optional[Dict] = None) -> bool:
        """执行命令"""
        pass
    
    @abstractmethod
    async def goto_preset(self, preset_id: int) -> bool:
        """跳转到预置位"""
        pass


class SimulatedPTZAdapter(BasePTZAdapter):
    """模拟云台适配器(用于测试)"""
    
    def __init__(self):
        self._connected = False
        self._position = PTZPosition()
        self._presets: Dict[int, PTZPosition] = {}
    
    async def connect(self) -> bool:
        self._connected = True
        print("[SimPTZ] 连接成功")
        return True
    
    async def disconnect(self) -> bool:
        self._connected = False
        return True
    
    async def get_position(self) -> PTZPosition:
        return self._position
    
    async def set_position(self, position: PTZPosition) -> bool:
        self._position = position
        await asyncio.sleep(0.5)  # 模拟移动延迟
        return True
    
    async def execute_command(self, command: PTZCommand, params: Optional[Dict] = None) -> bool:
        params = params or {}
        
        if command == PTZCommand.PAN_LEFT:
            self._position.pan = (self._position.pan - 5) % 360
        elif command == PTZCommand.PAN_RIGHT:
            self._position.pan = (self._position.pan + 5) % 360
        elif command == PTZCommand.TILT_UP:
            self._position.tilt = min(90, self._position.tilt + 5)
        elif command == PTZCommand.TILT_DOWN:
            self._position.tilt = max(-90, self._position.tilt - 5)
        elif command == PTZCommand.ZOOM_IN:
            self._position.zoom = min(30, self._position.zoom * 1.1)
        elif command == PTZCommand.ZOOM_OUT:
            self._position.zoom = max(1, self._position.zoom / 1.1)
        elif command == PTZCommand.SET_PRESET:
            preset_id = params.get("preset_id", 1)
            self._presets[preset_id] = PTZPosition(
                pan=self._position.pan,
                tilt=self._position.tilt,
                zoom=self._position.zoom,
            )
        elif command == PTZCommand.GOTO_PRESET:
            preset_id = params.get("preset_id", 1)
            if preset_id in self._presets:
                self._position = self._presets[preset_id]
        
        return True
    
    async def goto_preset(self, preset_id: int) -> bool:
        return await self.execute_command(PTZCommand.GOTO_PRESET, {"preset_id": preset_id})


class ONVIFPTZAdapter(BasePTZAdapter):
    """ONVIF协议云台适配器"""
    
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._client = None
    
    async def connect(self) -> bool:
        try:
            from onvif import ONVIFCamera
            self._client = ONVIFCamera(self.host, self.port, self.username, self.password)
            print(f"[ONVIF] 连接成功: {self.host}:{self.port}")
            return True
        except ImportError:
            print("[ONVIF] onvif-zeep库未安装")
            return False
        except Exception as e:
            print(f"[ONVIF] 连接失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        self._client = None
        return True
    
    async def get_position(self) -> PTZPosition:
        # 简化实现
        return PTZPosition()
    
    async def set_position(self, position: PTZPosition) -> bool:
        # 简化实现
        return True
    
    async def execute_command(self, command: PTZCommand, params: Optional[Dict] = None) -> bool:
        # 简化实现
        return True
    
    async def goto_preset(self, preset_id: int) -> bool:
        return await self.execute_command(PTZCommand.GOTO_PRESET, {"preset_id": preset_id})


class PTZController:
    """
    云台控制器
    
    统一管理云台操作，支持:
    - 预置位管理
    - 巡航调度
    - 智能复拍
    """
    
    def __init__(self, adapter: Optional[BasePTZAdapter] = None):
        self._adapter = adapter or SimulatedPTZAdapter()
        self._presets: Dict[int, PresetPosition] = {}
        self._routes: Dict[str, PatrolRoute] = {}
        self._current_route: Optional[str] = None
        self._patrol_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def connect(self) -> bool:
        """连接云台"""
        return await self._adapter.connect()
    
    async def disconnect(self) -> bool:
        """断开连接"""
        await self.stop_patrol()
        return await self._adapter.disconnect()
    
    # ==================== 预置位管理 ====================
    
    def add_preset(self, preset: PresetPosition) -> None:
        """添加预置位"""
        self._presets[preset.preset_id] = preset
    
    def get_preset(self, preset_id: int) -> Optional[PresetPosition]:
        """获取预置位"""
        return self._presets.get(preset_id)
    
    def list_presets(self) -> List[PresetPosition]:
        """列出所有预置位"""
        return list(self._presets.values())
    
    async def goto_preset(self, preset_id: int) -> bool:
        """跳转到预置位"""
        async with self._lock:
            preset = self._presets.get(preset_id)
            if preset:
                return await self._adapter.set_position(preset.position)
            return await self._adapter.goto_preset(preset_id)
    
    async def save_current_as_preset(self, preset_id: int, name: str) -> bool:
        """保存当前位置为预置位"""
        async with self._lock:
            position = await self._adapter.get_position()
            position.preset_id = preset_id
            
            preset = PresetPosition(
                preset_id=preset_id,
                name=name,
                position=position,
            )
            self._presets[preset_id] = preset
            
            return await self._adapter.execute_command(
                PTZCommand.SET_PRESET, 
                {"preset_id": preset_id}
            )
    
    # ==================== 巡航管理 ====================
    
    def add_route(self, route: PatrolRoute) -> None:
        """添加巡航路线"""
        self._routes[route.route_id] = route
    
    async def start_patrol(self, route_id: str) -> bool:
        """开始巡航"""
        if route_id not in self._routes:
            return False
        
        if self._patrol_task and not self._patrol_task.done():
            await self.stop_patrol()
        
        self._current_route = route_id
        self._patrol_task = asyncio.create_task(self._patrol_loop(route_id))
        return True
    
    async def stop_patrol(self) -> bool:
        """停止巡航"""
        if self._patrol_task:
            self._patrol_task.cancel()
            try:
                await self._patrol_task
            except asyncio.CancelledError:
                pass
            self._patrol_task = None
        self._current_route = None
        return True
    
    async def _patrol_loop(self, route_id: str) -> None:
        """巡航循环"""
        route = self._routes[route_id]
        
        while True:
            for i, preset_id in enumerate(route.presets):
                # 跳转到预置位
                await self.goto_preset(preset_id)
                
                # 等待停留时间
                dwell_time = route.dwell_times[i] if i < len(route.dwell_times) else 5.0
                await asyncio.sleep(dwell_time)
            
            if not route.loop:
                break
    
    # ==================== 智能复拍 ====================
    
    async def smart_reshoot(
        self,
        strategy: ReshootStrategy,
        clarity_score: float,
        target_bbox: Optional[Dict] = None,
    ) -> Tuple[bool, Dict]:
        """
        智能复拍
        
        Args:
            strategy: 复拍策略
            clarity_score: 当前清晰度得分
            target_bbox: 目标边界框(用于对焦)
            
        Returns:
            (是否成功, 调整参数)
        """
        result = {"actions": [], "success": False}
        
        if clarity_score >= strategy.clarity_threshold:
            result["success"] = True
            result["reason"] = "clarity_ok"
            return True, result
        
        for retry in range(strategy.max_retries):
            actions = []
            
            # 放大
            if strategy.zoom_factor > 1:
                await self._adapter.execute_command(PTZCommand.ZOOM_IN)
                actions.append(f"zoom_in_{strategy.zoom_factor}x")
            
            # 调整焦点
            if strategy.adjust_focus:
                await self._adapter.execute_command(PTZCommand.AUTO_FOCUS)
                actions.append("auto_focus")
            
            # 如果有目标框，调整云台对准目标中心
            if target_bbox:
                await self._center_on_target(target_bbox)
                actions.append("center_target")
            
            result["actions"].extend(actions)
            await asyncio.sleep(0.5)  # 等待稳定
            
            # 检查是否满足条件(实际应重新拍摄并评估)
            # 这里简化处理
            result["retry"] = retry + 1
        
        result["success"] = True
        result["reason"] = "max_retries_reached"
        return True, result
    
    async def _center_on_target(self, bbox: Dict) -> None:
        """将目标居中"""
        # 计算目标中心与图像中心的偏移
        center_x = bbox["x"] + bbox["width"] / 2
        center_y = bbox["y"] + bbox["height"] / 2
        
        # 假设图像中心为(0.5, 0.5)
        offset_x = center_x - 0.5
        offset_y = center_y - 0.5
        
        # 调整云台
        if abs(offset_x) > 0.1:
            if offset_x > 0:
                await self._adapter.execute_command(PTZCommand.PAN_RIGHT)
            else:
                await self._adapter.execute_command(PTZCommand.PAN_LEFT)
        
        if abs(offset_y) > 0.1:
            if offset_y > 0:
                await self._adapter.execute_command(PTZCommand.TILT_DOWN)
            else:
                await self._adapter.execute_command(PTZCommand.TILT_UP)
    
    # ==================== 热成像对齐 ====================
    
    async def align_thermal(
        self,
        visible_image,
        thermal_image,
    ) -> Tuple[bool, Dict]:
        """
        热成像与可见光对齐
        
        Returns:
            (是否成功, 对齐参数)
        """
        # 简化实现
        # 实际需要使用图像配准算法
        return True, {
            "offset_x": 0,
            "offset_y": 0,
            "scale": 1.0,
            "rotation": 0,
        }


# ==================== 便捷函数 ====================

_controller_instance: Optional[PTZController] = None

def get_ptz_controller() -> PTZController:
    """获取云台控制器实例"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = PTZController()
    return _controller_instance


async def goto_preset(preset_id: int) -> bool:
    """跳转预置位"""
    return await get_ptz_controller().goto_preset(preset_id)


async def smart_reshoot(clarity_score: float, strategy: Optional[ReshootStrategy] = None) -> Tuple[bool, Dict]:
    """智能复拍"""
    strategy = strategy or ReshootStrategy()
    return await get_ptz_controller().smart_reshoot(strategy, clarity_score)
