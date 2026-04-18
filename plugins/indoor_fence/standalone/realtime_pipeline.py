"""
Indoor Fence V3.0 - Realtime Multi-Person Pipeline
====================================================

实时多人监控管道，整合：
- 传感器数据采集 (Camera/LiDAR/UWB/IMU)
- 多传感器融合 (EKF)
- 多目标跟踪 (Hungarian + Kalman)
- 距离计算与告警
- 仿真渲染

支持仿真模式和实际设备模式。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.fusion.sensor_fusion_v3 import SensorFusionV3
from ..core.tracking.realtime_tracker import (
    DistanceLevel,
    RealtimeMultiPersonTracker,
    RealtimeTrackingResult,
)
from ..protocols import SensorType

logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    """管道运行模式"""
    SIMULATION = "simulation"  # 仿真模式
    HARDWARE = "hardware"      # 硬件模式
    HYBRID = "hybrid"          # 混合模式（部分硬件 + 仿真补充）


@dataclass
class PipelineConfig:
    """管道配置"""
    mode: PipelineMode = PipelineMode.SIMULATION
    fence_line_y: float = 3.0
    scene_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 6.0)
    update_rate: float = 10.0  # Hz
    enable_camera: bool = True
    enable_lidar: bool = True
    enable_uwb: bool = True
    enable_imu: bool = True
    show_distances: bool = True
    show_predictions: bool = True
    show_velocities: bool = True


@dataclass
class PipelineState:
    """管道状态"""
    is_running: bool = False
    frame_count: int = 0
    last_update_time: float = 0.0
    active_persons: int = 0
    total_alerts: int = 0
    fps: float = 0.0
    errors: List[str] = field(default_factory=list)


class RealtimeMultiPersonPipeline:
    """
    实时多人监控管道

    整合传感器采集、融合、跟踪、渲染的完整流程。
    支持仿真和硬件两种模式。
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_frame_callback: Optional[Callable[[np.ndarray], None]] = None,
        on_alert_callback: Optional[Callable[[List[Dict]], None]] = None,
        on_tracking_callback: Optional[Callable[[RealtimeTrackingResult], None]] = None,
    ):
        self._config = config or PipelineConfig()
        self._on_frame = on_frame_callback
        self._on_alert = on_alert_callback
        self._on_tracking = on_tracking_callback

        # 传感器融合引擎
        self._fusion = SensorFusionV3(
            max_association_distance=2.0,
            process_noise=0.05,
        )

        # 跟踪器
        self._tracker = RealtimeMultiPersonTracker(
            fence_line_y=self._config.fence_line_y,
            max_age=30,
            min_hits=3,
            prediction_horizon=10,
        )

        # 渲染器（延迟初始化）
        self._renderer = None

        # 仿真器（延迟初始化）
        self._simulator = None

        # 适配器（延迟初始化）
        self._camera_adapter = None
        self._lidar_adapter = None
        self._uwb_adapter = None
        self._imu_adapter = None

        # 状态
        self._state = PipelineState()
        self._lock = threading.Lock()

        # 运行线程
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # FPS 计算
        self._fps_timestamps: List[float] = []
        self._fps_window = 30

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def state(self) -> PipelineState:
        with self._lock:
            return PipelineState(
                is_running=self._state.is_running,
                frame_count=self._state.frame_count,
                last_update_time=self._state.last_update_time,
                active_persons=self._state.active_persons,
                total_alerts=self._state.total_alerts,
                fps=self._state.fps,
                errors=list(self._state.errors),
            )

    @property
    def tracker(self) -> RealtimeMultiPersonTracker:
        return self._tracker

    def initialize(self) -> bool:
        """初始化管道组件"""
        try:
            # 初始化渲染器
            from .simulation_renderer import SimulationRenderer
            self._renderer = SimulationRenderer(
                scene_bounds=self._config.scene_bounds,
                resolution=(640, 480),
                yellow_line_y=self._config.fence_line_y,
                show_distances=self._config.show_distances,
                show_predictions=self._config.show_predictions,
                show_velocities=self._config.show_velocities,
            )

            # 根据模式初始化
            if self._config.mode == PipelineMode.SIMULATION:
                self._init_simulation()
            elif self._config.mode == PipelineMode.HARDWARE:
                self._init_hardware()
            else:
                self._init_hybrid()

            logger.info(f"Pipeline initialized in {self._config.mode.value} mode")
            return True

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            self._state.errors.append(str(e))
            return False

    def _init_simulation(self) -> None:
        """初始化仿真模式（启用全部传感器）"""
        from ..adapters.simulator import Simulator, SimulatorConfig

        # 根据配置决定启用哪些传感器
        sensor_types = []
        if self._config.enable_camera:
            sensor_types.append(SensorType.CAMERA)
        if self._config.enable_lidar:
            sensor_types.append(SensorType.LIDAR)
        if self._config.enable_uwb:
            sensor_types.append(SensorType.UWB)
        if self._config.enable_imu:
            sensor_types.append(SensorType.IMU)

        # 至少保留 Camera 作为降级
        if not sensor_types:
            sensor_types = [SensorType.CAMERA]

        config = SimulatorConfig(
            scene_bounds=self._config.scene_bounds,
            num_persons=2,
            yellow_line_y=self._config.fence_line_y,
            sensor_types=sensor_types,
        )
        self._simulator = Simulator(config=config)

    def _init_hardware(self) -> None:
        """初始化硬件模式"""
        if self._config.enable_camera:
            try:
                from ..adapters.camera_adapter import CameraAdapter
                self._camera_adapter = CameraAdapter()
                self._camera_adapter.start()
            except Exception as e:
                logger.warning(f"Camera adapter init failed: {e}, using fallback")

        if self._config.enable_lidar:
            try:
                from ..adapters.lidar_adapter import LidarAdapter
                self._lidar_adapter = LidarAdapter()
                self._lidar_adapter.start()
            except Exception as e:
                logger.warning(f"LiDAR adapter init failed: {e}, using fallback")

    def _init_hybrid(self) -> None:
        """初始化混合模式"""
        self._init_hardware()
        self._init_simulation()

    def start(self) -> bool:
        """启动管道"""
        if self._state.is_running:
            return True

        if not self._renderer:
            if not self.initialize():
                return False

        self._stop_event.clear()
        self._state.is_running = True

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info("Pipeline started")
        return True

    def stop(self) -> None:
        """停止管道"""
        self._stop_event.set()
        self._state.is_running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        # 清理适配器
        if self._camera_adapter:
            try:
                self._camera_adapter.stop()
            except Exception:
                pass

        if self._lidar_adapter:
            try:
                self._lidar_adapter.stop()
            except Exception:
                pass

        logger.info("Pipeline stopped")

    def _run_loop(self) -> None:
        """主运行循环"""
        interval = 1.0 / self._config.update_rate

        while not self._stop_event.is_set():
            start_time = time.time()

            try:
                self._process_frame()
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                with self._lock:
                    self._state.errors.append(str(e))

            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(self) -> None:
        """处理单帧：传感器采集 → 多传感器融合 → 多目标跟踪 → 渲染"""
        current_time = time.time()

        # 1. 获取原始传感器数据
        sensor_data = self._get_sensor_data()

        # 2. 多传感器融合（EKF）
        if sensor_data:
            fusion_outputs = self._fusion.update(sensor_data)
            # 将融合输出转换为跟踪器检测格式
            detections = [
                {
                    "position": fo.position_3d,
                    "velocity": fo.velocity_3d,
                    "confidence": fo.confidence,
                    "sources": [s.value for s in fo.sources],
                }
                for fo in fusion_outputs
            ]
        else:
            # 降级：直接使用原始检测
            detections = self._get_detections()

        # 3. 更新跟踪器
        result = self._tracker.update(detections, current_time)

        # 4. 渲染帧
        frame = self._render_frame(result)

        # 5. 更新状态
        self._update_state(result, current_time)

        # 6. 回调
        if self._on_frame and frame is not None:
            self._on_frame(frame)

        if self._on_tracking:
            self._on_tracking(result)

        if self._on_alert and result.alerts:
            self._on_alert(result.alerts)

    def _get_sensor_data(self) -> Optional[Dict]:
        """获取多传感器原始数据（供融合引擎使用）"""
        if self._config.mode == PipelineMode.SIMULATION:
            if self._simulator:
                return self._simulator.step()
            return None
        elif self._config.mode == PipelineMode.HARDWARE:
            return self._get_hardware_sensor_data()
        else:
            # 混合模式：硬件优先，仿真补充
            hw_data = self._get_hardware_sensor_data()
            if hw_data:
                return hw_data
            if self._simulator:
                return self._simulator.step()
            return None

    def _get_hardware_sensor_data(self) -> Optional[Dict]:
        """从硬件适配器收集传感器数据"""
        from ..protocols import SensorData as SD
        result = {}
        ts = time.time()

        if self._camera_adapter:
            try:
                camera_dets = self._camera_adapter.get_detections()
                result[SensorType.CAMERA] = SD(
                    sensor_type=SensorType.CAMERA,
                    timestamp=ts,
                    data={"detections": [
                        {"position_2d": d.get("position", (0, 0))[:2]}
                        for d in camera_dets
                    ]},
                    confidence=0.85,
                )
            except Exception as e:
                logger.debug(f"Camera sensor read failed: {e}")

        if self._lidar_adapter:
            try:
                lidar_dets = self._lidar_adapter.get_detections()
                result[SensorType.LIDAR] = SD(
                    sensor_type=SensorType.LIDAR,
                    timestamp=ts,
                    data={"clusters": [
                        {"position_2d": d.get("position", (0, 0))[:2]}
                        for d in lidar_dets
                    ]},
                    confidence=0.9,
                )
            except Exception as e:
                logger.debug(f"LiDAR sensor read failed: {e}")

        if self._uwb_adapter:
            try:
                uwb_data = self._uwb_adapter.get_data()
                if uwb_data:
                    result[SensorType.UWB] = uwb_data
            except Exception as e:
                logger.debug(f"UWB sensor read failed: {e}")

        if self._imu_adapter:
            try:
                imu_data = self._imu_adapter.get_data()
                if imu_data:
                    result[SensorType.IMU] = imu_data
            except Exception as e:
                logger.debug(f"IMU sensor read failed: {e}")

        return result if result else None

    def _get_detections(self) -> List[Dict[str, Any]]:
        """获取检测数据"""
        detections = []

        if self._config.mode == PipelineMode.SIMULATION:
            detections = self._get_simulation_detections()
        elif self._config.mode == PipelineMode.HARDWARE:
            detections = self._get_hardware_detections()
        else:
            # 混合模式：优先硬件，仿真补充
            hw_dets = self._get_hardware_detections()
            if hw_dets:
                detections = hw_dets
            else:
                detections = self._get_simulation_detections()

        return detections

    def _get_simulation_detections(self) -> List[Dict[str, Any]]:
        """从仿真器获取检测"""
        if not self._simulator:
            return []

        sensor_data = self._simulator.step()

        # 从相机数据中提取检测
        if SensorType.CAMERA in sensor_data:
            camera_data = sensor_data[SensorType.CAMERA]
            detections_raw = camera_data.data.get("detections", [])
            return [
                {
                    "position": (d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0)),
                    "velocity": (d.get("vx", 0.0), d.get("vy", 0.0), 0.0),
                    "confidence": d.get("confidence", 0.9),
                }
                for d in detections_raw
            ]

        return []

    def _get_hardware_detections(self) -> List[Dict[str, Any]]:
        """从硬件适配器获取检测"""
        detections = []

        # 相机检测
        if self._camera_adapter:
            try:
                camera_dets = self._camera_adapter.get_detections()
                for det in camera_dets:
                    detections.append({
                        "position": det.get("position", (0, 0, 0)),
                        "velocity": det.get("velocity", (0, 0, 0)),
                        "confidence": det.get("confidence", 0.7),
                        "source": "camera",
                    })
            except Exception as e:
                logger.debug(f"Camera detection failed: {e}")

        # LiDAR 检测
        if self._lidar_adapter:
            try:
                lidar_dets = self._lidar_adapter.get_detections()
                for det in lidar_dets:
                    detections.append({
                        "position": det.get("position", (0, 0, 0)),
                        "velocity": det.get("velocity", (0, 0, 0)),
                        "confidence": det.get("confidence", 0.8),
                        "source": "lidar",
                    })
            except Exception as e:
                logger.debug(f"LiDAR detection failed: {e}")

        return detections

    def _render_frame(self, result: RealtimeTrackingResult) -> Optional[np.ndarray]:
        """渲染帧"""
        if not self._renderer:
            return None

        # 转换轨迹数据为渲染格式
        persons = []
        for track in result.tracks:
            if track.status.value in ("confirmed", "tentative"):
                persons.append({
                    "id": track.track_id,
                    "x": track.position[0],
                    "y": track.position[1],
                    "z": track.position[2],
                    "vx": track.velocity[0],
                    "vy": track.velocity[1],
                })

        # 设置告警
        self._renderer.set_alerts(result.alerts)

        # 渲染
        return self._renderer.render(persons, self._state.frame_count)

    def _update_state(self, result: RealtimeTrackingResult, current_time: float) -> None:
        """更新管道状态"""
        with self._lock:
            self._state.frame_count += 1
            self._state.last_update_time = current_time
            self._state.active_persons = result.stats.get("active_tracks", 0)
            self._state.total_alerts += len(result.alerts)

            # 计算 FPS
            self._fps_timestamps.append(current_time)
            if len(self._fps_timestamps) > self._fps_window:
                self._fps_timestamps = self._fps_timestamps[-self._fps_window:]

            if len(self._fps_timestamps) >= 2:
                dt = self._fps_timestamps[-1] - self._fps_timestamps[0]
                if dt > 0:
                    self._state.fps = (len(self._fps_timestamps) - 1) / dt

    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧（用于 HTTP 流）"""
        if not self._renderer:
            return None

        # 获取最新跟踪结果
        tracks = self._tracker.get_confirmed_tracks()
        persons = [
            {
                "id": t.track_id,
                "x": t.position[0],
                "y": t.position[1],
                "z": t.position[2],
                "vx": t.velocity[0],
                "vy": t.velocity[1],
            }
            for t in tracks
        ]

        return self._renderer.render(persons, self._state.frame_count)

    def get_tracking_data(self) -> Dict[str, Any]:
        """获取跟踪数据（用于 WebSocket）"""
        tracks = self._tracker.get_confirmed_tracks()

        return {
            "timestamp": time.time(),
            "frame_count": self._state.frame_count,
            "fps": round(self._state.fps, 1),
            "persons": [
                {
                    "id": t.track_id,
                    "x": round(t.position[0], 3),
                    "y": round(t.position[1], 3),
                    "z": round(t.position[2], 3),
                    "vx": round(t.velocity[0], 3),
                    "vy": round(t.velocity[1], 3),
                    "distance_to_fence": round(t.distance_to_fence, 3),
                    "distance_level": t.distance_level.value,
                    "confidence": round(t.confidence, 2),
                    "trajectory": [
                        {"x": round(p.x, 3), "y": round(p.y, 3), "t": round(p.timestamp, 3)}
                        for p in list(t.history)[-20:]  # 最近 20 个点
                    ],
                    "prediction": [
                        {"x": round(p[0], 3), "y": round(p[1], 3)}
                        for p in t.predicted_positions[:5]  # 预测 5 步
                    ],
                }
                for t in tracks
            ],
            "alerts": [
                {
                    "type": a.get("type"),
                    "level": a.get("level"),
                    "track_id": a.get("track_id"),
                    "message": a.get("message"),
                }
                for a in self._tracker._generate_alerts(self._tracker._compute_distances())
            ],
            "stats": {
                "active_persons": len(tracks),
                "total_alerts": self._state.total_alerts,
            },
        }

    def set_scenario(self, scenario_name: str) -> bool:
        """设置仿真场景"""
        if self._simulator and hasattr(self._simulator, "load_scenario"):
            try:
                self._simulator.load_scenario(scenario_name)
                if self._renderer:
                    self._renderer.scenario_name = scenario_name
                return True
            except Exception as e:
                logger.error(f"Failed to load scenario: {e}")
        return False

    def reset(self) -> None:
        """重置管道"""
        self._tracker.reset()
        self._fusion = SensorFusionV3(
            max_association_distance=2.0,
            process_noise=0.05,
        )
        if self._renderer:
            self._renderer.reset_trails()
        # 重新初始化仿真器
        if self._simulator and self._config.mode in (PipelineMode.SIMULATION, PipelineMode.HYBRID):
            self._init_simulation()

        with self._lock:
            self._state = PipelineState()
            self._fps_timestamps.clear()


__all__ = [
    'PipelineMode',
    'PipelineConfig',
    'PipelineState',
    'RealtimeMultiPersonPipeline',
]
