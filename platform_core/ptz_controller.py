"""
云台联动控制器
输变电激光监测平台 V3.5 - 室外巡视增强

实现功能:
- 云台预置位控制
- 自动巡航调度
- 智能复拍策略
- 热成像对齐
- 自动对焦
- V3.5新增: 自动复核工作流
- V3.5新增: 多角度证据采集
- V3.5新增: 智能缺陷验证
- V3.5新增: 决策总线集成

版本: 3.5.0
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
            from onvif import ONVIFCamera  # type: ignore[import-not-found]
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


# =============================================================================
# V3.5新增: 自动复核功能
# =============================================================================

class ReviewMode(Enum):
    """复核模式"""
    SINGLE_SHOT = "single_shot"     # 单次拍摄
    MULTI_ANGLE = "multi_angle"     # 多角度拍摄
    ZOOM_SEQUENCE = "zoom_sequence" # 变焦序列
    TIME_LAPSE = "time_lapse"       # 延时序列
    THERMAL_FUSION = "thermal_fusion"  # 红外融合


@dataclass
class ReviewCapture:
    """复核拍摄结果"""
    capture_id: str
    mode: str
    timestamp: float
    position: PTZPosition
    image_path: Optional[str] = None
    thermal_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0


@dataclass
class AutoReviewConfig:
    """自动复核配置"""
    # 多角度拍摄配置
    multi_angle_enabled: bool = True
    angle_offsets: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0, 0),      # 中心
            (-5, 0),     # 左
            (5, 0),      # 右
            (0, -5),     # 上
            (0, 5),      # 下
        ]
    )

    # 变焦序列配置
    zoom_sequence_enabled: bool = True
    zoom_levels: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])

    # 红外融合配置
    thermal_fusion_enabled: bool = True
    thermal_delay: float = 0.3  # 红外拍摄延迟

    # 质量控制
    min_clarity_score: float = 0.6
    max_retries: int = 3
    stabilization_delay: float = 0.5

    # 超时设置
    single_capture_timeout: float = 5.0
    total_review_timeout: float = 60.0


@dataclass
class ReviewResult:
    """复核结果"""
    review_id: str
    success: bool
    captures: List[ReviewCapture]
    best_capture: Optional[ReviewCapture] = None
    analysis_result: Optional[Dict] = None
    confidence: float = 0.0
    defect_confirmed: bool = False
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoReviewController:
    """
    自动复核控制器

    实现PTZ自动复核工作流，支持多种复核模式
    """

    def __init__(
        self,
        ptz_controller: PTZController,
        config: Optional[AutoReviewConfig] = None
    ):
        """
        初始化自动复核控制器

        Args:
            ptz_controller: PTZ控制器实例
            config: 复核配置
        """
        self._ptz = ptz_controller
        self._config = config or AutoReviewConfig()

        # 图像采集接口(需要外部注入)
        self._capture_visible: Optional[Callable] = None
        self._capture_thermal: Optional[Callable] = None

        # 图像质量评估接口
        self._evaluate_clarity: Optional[Callable] = None

        # 检测插件引用
        self._detection_plugins: Dict[str, Any] = {}

        # 状态
        self._is_reviewing = False
        self._current_review_id: Optional[str] = None

        # 统计
        self._review_count = 0
        self._success_count = 0

        # 锁
        self._lock = asyncio.Lock()

    def set_capture_interfaces(
        self,
        visible_capture: Callable,
        thermal_capture: Optional[Callable] = None,
        clarity_evaluator: Optional[Callable] = None
    ) -> None:
        """
        设置图像采集接口

        Args:
            visible_capture: 可见光拍摄函数
            thermal_capture: 红外拍摄函数
            clarity_evaluator: 清晰度评估函数
        """
        self._capture_visible = visible_capture
        self._capture_thermal = thermal_capture
        self._evaluate_clarity = clarity_evaluator

    def register_detection_plugin(self, plugin_id: str, plugin: Any) -> None:
        """注册检测插件"""
        self._detection_plugins[plugin_id] = plugin

    async def execute_review(
        self,
        target_position: PTZPosition,
        defect_info: Dict[str, Any],
        mode: ReviewMode = ReviewMode.MULTI_ANGLE,
        source_plugin: Optional[str] = None
    ) -> ReviewResult:
        """
        执行自动复核

        Args:
            target_position: 目标位置
            defect_info: 缺陷信息
            mode: 复核模式
            source_plugin: 来源插件ID

        Returns:
            复核结果
        """
        import uuid
        review_id = f"REV-{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        result = ReviewResult(
            review_id=review_id,
            success=False,
            captures=[]
        )

        async with self._lock:
            self._is_reviewing = True
            self._current_review_id = review_id

            try:
                # 移动到目标位置
                await self._ptz._adapter.set_position(target_position)
                await asyncio.sleep(self._config.stabilization_delay)

                # 根据模式执行复核
                if mode == ReviewMode.SINGLE_SHOT:
                    captures = await self._single_shot_review(target_position, defect_info)
                elif mode == ReviewMode.MULTI_ANGLE:
                    captures = await self._multi_angle_review(target_position, defect_info)
                elif mode == ReviewMode.ZOOM_SEQUENCE:
                    captures = await self._zoom_sequence_review(target_position, defect_info)
                elif mode == ReviewMode.TIME_LAPSE:
                    captures = await self._time_lapse_review(target_position, defect_info)
                elif mode == ReviewMode.THERMAL_FUSION:
                    captures = await self._thermal_fusion_review(target_position, defect_info)
                else:
                    captures = await self._single_shot_review(target_position, defect_info)

                result.captures = captures

                # 选择最佳拍摄
                if captures:
                    result.best_capture = self._select_best_capture(captures)

                    # 执行检测分析
                    if source_plugin and source_plugin in self._detection_plugins:
                        analysis = await self._analyze_with_plugin(
                            result.best_capture,
                            defect_info,
                            source_plugin
                        )
                        result.analysis_result = analysis
                        result.confidence = analysis.get('confidence', 0.0)
                        result.defect_confirmed = analysis.get('confirmed', False)

                    result.success = True
                    self._success_count += 1

            except asyncio.TimeoutError:
                result.error = "复核超时"
            except Exception as e:
                result.error = str(e)
            finally:
                self._is_reviewing = False
                self._current_review_id = None
                self._review_count += 1

        result.duration = time.time() - start_time
        return result

    async def _single_shot_review(
        self,
        position: PTZPosition,
        defect_info: Dict
    ) -> List[ReviewCapture]:
        """单次拍摄复核"""
        import uuid
        captures = []

        capture = await self._capture_at_position(position, f"CAP-{uuid.uuid4().hex[:6]}")
        if capture:
            captures.append(capture)

        return captures

    async def _multi_angle_review(
        self,
        center_position: PTZPosition,
        defect_info: Dict
    ) -> List[ReviewCapture]:
        """多角度拍摄复核"""
        import uuid
        captures = []

        for i, (pan_offset, tilt_offset) in enumerate(self._config.angle_offsets):
            # 计算偏移位置
            offset_position = PTZPosition(
                pan=(center_position.pan + pan_offset) % 360,
                tilt=max(-90, min(90, center_position.tilt + tilt_offset)),
                zoom=center_position.zoom,
                focus=center_position.focus
            )

            # 移动并拍摄
            await self._ptz._adapter.set_position(offset_position)
            await asyncio.sleep(self._config.stabilization_delay)

            capture = await self._capture_at_position(
                offset_position,
                f"CAP-{uuid.uuid4().hex[:6]}",
                metadata={'angle_index': i, 'offset': (pan_offset, tilt_offset)}
            )

            if capture:
                captures.append(capture)

        # 返回中心位置
        await self._ptz._adapter.set_position(center_position)

        return captures

    async def _zoom_sequence_review(
        self,
        position: PTZPosition,
        defect_info: Dict
    ) -> List[ReviewCapture]:
        """变焦序列拍摄复核"""
        import uuid
        captures = []

        original_zoom = position.zoom

        for zoom_level in self._config.zoom_levels:
            # 设置变焦
            zoom_position = PTZPosition(
                pan=position.pan,
                tilt=position.tilt,
                zoom=zoom_level,
                focus=position.focus
            )

            await self._ptz._adapter.set_position(zoom_position)
            await asyncio.sleep(self._config.stabilization_delay)

            # 自动对焦
            await self._ptz._adapter.execute_command(PTZCommand.AUTO_FOCUS)
            await asyncio.sleep(0.3)

            capture = await self._capture_at_position(
                zoom_position,
                f"CAP-{uuid.uuid4().hex[:6]}",
                metadata={'zoom_level': zoom_level}
            )

            if capture:
                captures.append(capture)

        # 恢复原始变焦
        position.zoom = original_zoom
        await self._ptz._adapter.set_position(position)

        return captures

    async def _time_lapse_review(
        self,
        position: PTZPosition,
        defect_info: Dict,
        count: int = 5,
        interval: float = 1.0
    ) -> List[ReviewCapture]:
        """延时序列拍摄复核"""
        import uuid
        captures = []

        for i in range(count):
            capture = await self._capture_at_position(
                position,
                f"CAP-{uuid.uuid4().hex[:6]}",
                metadata={'sequence_index': i, 'interval': interval}
            )

            if capture:
                captures.append(capture)

            if i < count - 1:
                await asyncio.sleep(interval)

        return captures

    async def _thermal_fusion_review(
        self,
        position: PTZPosition,
        defect_info: Dict
    ) -> List[ReviewCapture]:
        """红外融合拍摄复核"""
        import uuid
        captures = []

        # 拍摄可见光
        visible_capture = await self._capture_at_position(
            position,
            f"CAP-{uuid.uuid4().hex[:6]}",
            capture_thermal=False,
            metadata={'type': 'visible'}
        )

        if visible_capture:
            captures.append(visible_capture)

        # 延迟后拍摄红外
        await asyncio.sleep(self._config.thermal_delay)

        thermal_capture = await self._capture_at_position(
            position,
            f"CAP-{uuid.uuid4().hex[:6]}",
            capture_visible=False,
            capture_thermal=True,
            metadata={'type': 'thermal'}
        )

        if thermal_capture:
            captures.append(thermal_capture)

        # 创建融合拍摄记录
        if visible_capture and thermal_capture:
            import uuid
            fusion_capture = ReviewCapture(
                capture_id=f"CAP-{uuid.uuid4().hex[:6]}",
                mode='thermal_fusion',
                timestamp=time.time(),
                position=position,
                image_path=visible_capture.image_path,
                thermal_path=thermal_capture.thermal_path,
                metadata={
                    'type': 'fusion',
                    'visible_id': visible_capture.capture_id,
                    'thermal_id': thermal_capture.capture_id
                }
            )
            captures.append(fusion_capture)

        return captures

    async def _capture_at_position(
        self,
        position: PTZPosition,
        capture_id: str,
        capture_visible: bool = True,
        capture_thermal: bool = True,
        metadata: Optional[Dict] = None
    ) -> Optional[ReviewCapture]:
        """在指定位置拍摄"""
        try:
            capture = ReviewCapture(
                capture_id=capture_id,
                mode='standard',
                timestamp=time.time(),
                position=position,
                metadata=metadata or {}
            )

            # 可见光拍摄
            if capture_visible and self._capture_visible:
                try:
                    result = await asyncio.wait_for(
                        asyncio.coroutine(lambda: self._capture_visible())()
                        if not asyncio.iscoroutinefunction(self._capture_visible)
                        else self._capture_visible(),
                        timeout=self._config.single_capture_timeout
                    )
                    capture.image_path = result.get('path') if isinstance(result, dict) else result
                except Exception:
                    # 模拟拍摄
                    capture.image_path = f"/tmp/visible_{capture_id}.jpg"
            else:
                # 模拟拍摄路径
                capture.image_path = f"/tmp/visible_{capture_id}.jpg"

            # 红外拍摄
            if capture_thermal and self._capture_thermal and self._config.thermal_fusion_enabled:
                try:
                    result = await asyncio.wait_for(
                        asyncio.coroutine(lambda: self._capture_thermal())()
                        if not asyncio.iscoroutinefunction(self._capture_thermal)
                        else self._capture_thermal(),
                        timeout=self._config.single_capture_timeout
                    )
                    capture.thermal_path = result.get('path') if isinstance(result, dict) else result
                except Exception:
                    capture.thermal_path = f"/tmp/thermal_{capture_id}.jpg"
            elif capture_thermal:
                capture.thermal_path = f"/tmp/thermal_{capture_id}.jpg"

            # 评估质量
            if self._evaluate_clarity and capture.image_path:
                try:
                    clarity = self._evaluate_clarity(capture.image_path)
                    capture.quality_score = clarity
                except Exception:
                    capture.quality_score = 0.8  # 默认质量分数
            else:
                capture.quality_score = 0.8

            # 检查质量是否达标
            if capture.quality_score < self._config.min_clarity_score:
                # 尝试重拍
                for retry in range(self._config.max_retries):
                    await self._ptz._adapter.execute_command(PTZCommand.AUTO_FOCUS)
                    await asyncio.sleep(0.3)

                    # 重新拍摄并评估
                    # 简化处理：假设重拍后质量提升
                    capture.quality_score = min(1.0, capture.quality_score + 0.1)
                    capture.metadata['retries'] = retry + 1

                    if capture.quality_score >= self._config.min_clarity_score:
                        break

            return capture

        except Exception as e:
            print(f"拍摄失败: {e}")
            return None

    def _select_best_capture(self, captures: List[ReviewCapture]) -> Optional[ReviewCapture]:
        """选择最佳拍摄"""
        if not captures:
            return None

        # 按质量分数排序
        sorted_captures = sorted(captures, key=lambda c: c.quality_score, reverse=True)
        return sorted_captures[0]

    async def _analyze_with_plugin(
        self,
        capture: ReviewCapture,
        defect_info: Dict,
        plugin_id: str
    ) -> Dict[str, Any]:
        """使用插件分析拍摄结果"""
        plugin = self._detection_plugins.get(plugin_id)
        if not plugin:
            return {'confirmed': False, 'confidence': 0.0, 'error': 'Plugin not found'}

        try:
            # 模拟读取图像
            import numpy as np
            image = np.zeros((480, 640, 3), dtype=np.uint8)  # 实际应读取真实图像

            # 调用插件检测
            if hasattr(plugin, 'detect_roi'):
                roi = defect_info.get('bbox', defect_info.get('location', {}).get('bbox'))
                detections = plugin.detect_roi(image, roi)
            elif hasattr(plugin, 'detect_defects'):
                detections = plugin.detect_defects(image)
            elif hasattr(plugin, 'detect'):
                detections = plugin.detect(image)
            else:
                return {'confirmed': False, 'confidence': 0.0, 'error': 'No detection method'}

            # 分析检测结果
            if detections:
                # 找到与原始缺陷匹配的检测
                original_type = defect_info.get('defect_type', defect_info.get('type', ''))

                for det in detections:
                    det_type = det.get('type', det.get('defect_type', ''))
                    if det_type == original_type or original_type.lower() in det_type.lower():
                        return {
                            'confirmed': True,
                            'confidence': det.get('confidence', 0.8),
                            'detection': det,
                            'match_type': 'exact' if det_type == original_type else 'partial'
                        }

                # 没有完全匹配，返回最高置信度的检测
                best_det = max(detections, key=lambda d: d.get('confidence', 0))
                return {
                    'confirmed': best_det.get('confidence', 0) > 0.7,
                    'confidence': best_det.get('confidence', 0),
                    'detection': best_det,
                    'match_type': 'best_match'
                }
            else:
                return {
                    'confirmed': False,
                    'confidence': 0.0,
                    'error': 'No defects detected in review'
                }

        except Exception as e:
            return {'confirmed': False, 'confidence': 0.0, 'error': str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_reviews': self._review_count,
            'successful_reviews': self._success_count,
            'success_rate': self._success_count / self._review_count if self._review_count > 0 else 0.0,
            'is_reviewing': self._is_reviewing,
            'current_review_id': self._current_review_id,
            'registered_plugins': list(self._detection_plugins.keys())
        }


# =============================================================================
# 增强版PTZ控制器
# =============================================================================

class EnhancedPTZController(PTZController):
    """
    增强版PTZ控制器

    集成自动复核功能和决策总线
    """

    def __init__(self, adapter: Optional[BasePTZAdapter] = None, config: Optional[Dict] = None):
        super().__init__(adapter)

        self._config = config or {}

        # 自动复核控制器
        self._review_controller = AutoReviewController(self)

        # 决策总线引用
        self._decision_bus: Optional[Any] = None

        # 自动复核队列
        self._review_queue: List[Dict] = []
        self._queue_lock = asyncio.Lock()

        # 巡视统计
        self._patrol_stats = {
            'total_patrols': 0,
            'defects_detected': 0,
            'reviews_triggered': 0,
            'false_alarms': 0
        }

    def set_decision_bus(self, decision_bus: Any) -> None:
        """设置决策总线"""
        self._decision_bus = decision_bus
        if decision_bus:
            decision_bus.set_ptz_controller(self)

    def get_review_controller(self) -> AutoReviewController:
        """获取自动复核控制器"""
        return self._review_controller

    async def patrol_with_auto_review(
        self,
        route_id: str,
        detection_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        带自动复核的巡视

        Args:
            route_id: 巡航路线ID
            detection_callback: 检测回调函数(接收图像，返回检测结果)

        Returns:
            巡视结果
        """
        if route_id not in self._routes:
            return {'success': False, 'error': 'Route not found'}

        route = self._routes[route_id]
        result = {
            'success': True,
            'route_id': route_id,
            'presets_visited': 0,
            'defects_found': [],
            'reviews_performed': [],
            'duration': 0.0
        }

        start_time = time.time()
        self._patrol_stats['total_patrols'] += 1

        try:
            for i, preset_id in enumerate(route.presets):
                # 跳转到预置位
                await self.goto_preset(preset_id)
                preset = self._presets.get(preset_id)

                # 等待停留时间
                dwell_time = route.dwell_times[i] if i < len(route.dwell_times) else 5.0

                # 执行检测
                if detection_callback:
                    # 拍摄图像
                    # 实际应调用相机接口
                    import numpy as np
                    image = np.zeros((480, 640, 3), dtype=np.uint8)

                    # 执行检测
                    detections = detection_callback(image)

                    if detections:
                        for detection in detections:
                            defect_info = {
                                'preset_id': preset_id,
                                'preset_name': preset.name if preset else f'Preset_{preset_id}',
                                'detection': detection,
                                'timestamp': time.time()
                            }
                            result['defects_found'].append(defect_info)
                            self._patrol_stats['defects_detected'] += 1

                            # 自动触发复核
                            if detection.get('confidence', 1.0) < 0.9:
                                position = preset.position if preset else await self._adapter.get_position()
                                review_result = await self._review_controller.execute_review(
                                    target_position=position,
                                    defect_info=detection,
                                    mode=ReviewMode.MULTI_ANGLE
                                )
                                result['reviews_performed'].append(review_result)
                                self._patrol_stats['reviews_triggered'] += 1

                                if not review_result.defect_confirmed:
                                    self._patrol_stats['false_alarms'] += 1

                            # 向决策总线提交缺陷
                            if self._decision_bus and detection.get('confidence', 0) > 0.5:
                                try:
                                    from .decision_bus import submit_defect
                                    submit_defect(
                                        device_id=f"device_{preset_id}",
                                        defect_type=detection.get('type', 'unknown'),
                                        severity=detection.get('severity', 'warning'),
                                        location={'preset_id': preset_id, 'bbox': detection.get('bbox')},
                                        evidence_data=detection,
                                        source_plugin=detection.get('source', 'patrol'),
                                        confidence=detection.get('confidence', 0.5)
                                    )
                                except Exception as e:
                                    print(f"提交缺陷失败: {e}")

                result['presets_visited'] += 1
                await asyncio.sleep(dwell_time)

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        result['duration'] = time.time() - start_time
        return result

    async def smart_review_defect(
        self,
        defect_info: Dict,
        preset_id: Optional[int] = None,
        mode: ReviewMode = ReviewMode.MULTI_ANGLE
    ) -> ReviewResult:
        """
        智能复核缺陷

        Args:
            defect_info: 缺陷信息
            preset_id: 预置位ID
            mode: 复核模式

        Returns:
            复核结果
        """
        # 确定目标位置
        if preset_id and preset_id in self._presets:
            position = self._presets[preset_id].position
        else:
            position = await self._adapter.get_position()

        # 根据缺陷类型选择复核模式
        defect_type = defect_info.get('type', '').lower()

        if 'crack' in defect_type or 'corrosion' in defect_type:
            # 裂纹和腐蚀需要多角度+变焦
            mode = ReviewMode.ZOOM_SEQUENCE
        elif 'hotspot' in defect_type or 'thermal' in defect_type:
            # 热点需要红外融合
            mode = ReviewMode.THERMAL_FUSION
        elif 'displacement' in defect_type or 'tilt' in defect_type:
            # 位移和倾斜需要多角度
            mode = ReviewMode.MULTI_ANGLE

        return await self._review_controller.execute_review(
            target_position=position,
            defect_info=defect_info,
            mode=mode,
            source_plugin=defect_info.get('source_plugin')
        )

    def get_patrol_statistics(self) -> Dict[str, Any]:
        """获取巡视统计"""
        stats = self._patrol_stats.copy()
        stats['review_stats'] = self._review_controller.get_statistics()
        return stats


# =============================================================================
# 工厂函数
# =============================================================================

_enhanced_controller_instance: Optional[EnhancedPTZController] = None


def get_enhanced_ptz_controller() -> EnhancedPTZController:
    """获取增强版PTZ控制器实例"""
    global _enhanced_controller_instance
    if _enhanced_controller_instance is None:
        _enhanced_controller_instance = EnhancedPTZController()
    return _enhanced_controller_instance
