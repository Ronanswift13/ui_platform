"""
室内电子围栏插件 V2.1 - 多人安全监测系统
输变电激光星芒破夜绘明监测平台 (G组)

功能范围:
- 2D激光雷达电子围栏: 基于扫描雷达的精确测距
- 多人操作目标识别: 视觉-雷达融合多目标跟踪
- 黄线越界检测: 实时监测人员与警戒线距离
- 机柜授权校验: 验证人员操作授权状态

算法五层架构:
1. 传感器适配层 (Adapters)
2. 地面投影层 (Ground Projection)
3. 多目标跟踪与融合层 (Tracking & Fusion)
4. 区域逻辑与状态机层 (Zone Logic & State Machine)
5. 输出与控制层 (Output & Control)

性能指标:
- 检测延迟: <100ms
- 多目标容量: 10人/过道
- 定位精度: ±0.1m
- 告警响应: <50ms

"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import copy
import hashlib
import time
import threading
from collections import deque
import json
import sys
import importlib.util
import logging
import numpy as np

# =============================================================================
# 动态加载支持 - 确保子模块可被正确导入
# =============================================================================
def _ensure_package_registered():
    """确保包和子模块在sys.modules中正确注册，以支持PluginManager的动态加载"""
    plugin_dir = Path(__file__).parent
    package_name = "plugins.indoor_fence"

    # 如果包已正确注册且有__path__属性，则跳过
    if package_name in sys.modules:
        pkg = sys.modules[package_name]
        if hasattr(pkg, '__path__'):
            return

    # 注册包本身
    init_path = plugin_dir / "__init__.py"
    if init_path.exists():
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(plugin_dir)]
        )
        if spec:
            pkg_module = importlib.util.module_from_spec(spec)
            pkg_module.__path__ = [str(plugin_dir)]
            sys.modules[package_name] = pkg_module

    # 注册子包
    for subpkg in ["core", "adapters"]:
        subpkg_dir = plugin_dir / subpkg
        subpkg_name = f"{package_name}.{subpkg}"
        if subpkg_dir.is_dir() and subpkg_name not in sys.modules:
            subpkg_init = subpkg_dir / "__init__.py"
            if subpkg_init.exists():
                spec = importlib.util.spec_from_file_location(
                    subpkg_name,
                    subpkg_init,
                    submodule_search_locations=[str(subpkg_dir)]
                )
                if spec:
                    subpkg_module = importlib.util.module_from_spec(spec)
                    subpkg_module.__path__ = [str(subpkg_dir)]
                    sys.modules[subpkg_name] = subpkg_module
                    if spec.loader:
                        spec.loader.exec_module(subpkg_module)

_ensure_package_registered()

from darkbreaker_sdk.interfaces import (
    BasePlugin,
    HealthStatus,
    PluginContext,
    PluginManifest,
    PluginStatus,
)
from darkbreaker_sdk.schemas import (
    Alarm,
    AlarmLevel,
    AlarmRule,
    RecognitionResult,
    ROI,
    BoundingBox,
)

# 导入v2核心模块 (使用绝对导入以支持动态加载)
from plugins.indoor_fence.core import (
    Point2D, Polygon,
    ZoneConfiguration, ZoneConfigLoader, ZoneType, AlertLevel,
    PersonState, GlobalAlarmLevel, PersonStateResult, GlobalStateResult, StateMachine,
    Detection, Track, MultiTargetTracker,
    VisualDetection, LidarDetection, FusedDetection, SensorFusion, FusedTracker,
    IndoorFenceConfigManager,
)

# 导入适配器 (使用绝对导入以支持动态加载)
from plugins.indoor_fence.adapters import (
    BaseAdapter, AdapterStatus,
    CameraAdapter, CameraConfig, PersonDetection,
    LidarAdapter, LidarConfig, LidarScan, LidarCluster,
    LightAdapter, LightColor, LightConfig, LightState,
)


logger = logging.getLogger(__name__)


# =============================================================================
# 审计日志
# =============================================================================

class AuditLogger:
    """审计日志记录器"""

    def __init__(self, config: Dict):
        self.enabled = config.get("enabled", True)
        self.log_dir = Path(config.get("log_dir", "logs/indoor_fence"))
        self.log_level = config.get("log_level", "event")

        # 创建日志目录
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self._frame_logs: deque = deque(maxlen=1000)
        self._event_logs: deque = deque(maxlen=1000)

        # 当日文件
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        self._file_handle = None

    def log_frame(self, frame_data: Dict):
        """记录帧级数据"""
        if not self.enabled or self.log_level != "frame":
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "frame",
            "data": frame_data,
        }
        self._frame_logs.append(entry)
        self._write_to_file(entry)

    def log_event(self, event_type: str, event_data: Dict):
        """记录事件"""
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": event_data,
        }
        self._event_logs.append(entry)
        self._write_to_file(entry)

    def log_state_change(self, person_id: str, old_state: str, new_state: str, details: Dict = None):
        """记录状态变化"""
        self.log_event("state_change", {
            "person_id": person_id,
            "old_state": old_state,
            "new_state": new_state,
            "details": details or {},
            "message": f"人员{person_id}: {old_state} -> {new_state}",
        })

    def log_alarm(self, alarm_level: str, message: str, details: Dict = None):
        """记录告警"""
        self.log_event("alarm", {
            "level": alarm_level,
            "message": message,
            "details": details or {},
        })

    def log_violation(self, violation_type: str, details: Dict):
        """记录违规"""
        self.log_event("violation", {
            "violation_type": violation_type,
            **details,
        })

    def _write_to_file(self, entry: Dict):
        """写入文件"""
        try:
            # 检查日期是否变化
            today = datetime.now().strftime("%Y-%m-%d")
            if today != self._current_date:
                self._current_date = today
                if self._file_handle:
                    self._file_handle.close()
                    self._file_handle = None

            # 打开文件
            if self._file_handle is None:
                log_file = self.log_dir / f"audit_{self._current_date}.jsonl"
                self._file_handle = open(log_file, "a", encoding="utf-8")

            # 写入
            self._file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file_handle.flush()

        except Exception as e:
            pass  # 静默处理日志写入错误

    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """获取最近事件"""
        return list(self._event_logs)[-limit:]

    def close(self):
        """关闭日志"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


# =============================================================================
# 主插件类
# =============================================================================

class IndoorFencePlugin(BasePlugin):
    """
    室内电子围栏插件 V2.1

    实现基于视觉+2D激光雷达融合的安全监测系统
    使用五层算法架构
    """

    # 告警级别映射
    ALARM_LEVELS = {
        PersonState.NORMAL: AlarmLevel.INFO,
        PersonState.ON_LINE: AlarmLevel.WARNING,
        PersonState.CROSS_LINE: AlarmLevel.CRITICAL,
        PersonState.MISPLACED: AlarmLevel.WARNING,
        PersonState.HIGH_RISK: AlarmLevel.CRITICAL,
    }

    # 标签名称
    LABEL_NAMES = {
        "line_cross": "越线告警",
        "unauthorized": "未授权操作",
        "multi_person": "同柜多人",
        "safe": "安全状态",
        "on_line": "压线警告",
        "high_risk": "高风险",
    }

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        """初始化插件"""
        super().__init__(manifest, plugin_dir)
        self._config_manager = IndoorFenceConfigManager(plugin_dir, manifest.config_schema)
        self._config: Dict[str, Any] = {}
        self._config_revision = 0
        self._initialized = False

        # V2 核心组件
        self._zone_config: Optional[ZoneConfiguration] = None
        self._zone_config_path: Optional[Path] = None
        self._state_machine: Optional[StateMachine] = None
        self._fused_tracker: Optional[FusedTracker] = None

        # 适配器
        self._camera_adapter: Optional[CameraAdapter] = None
        self._lidar_adapter: Optional[LidarAdapter] = None
        self._light_adapter: Optional[LightAdapter] = None

        # 审计日志
        self._audit_logger: Optional[AuditLogger] = None

        # 运行时状态
        self._frame_count = 0
        self._last_process_time = 0.0
        self._inference_times: deque = deque(maxlen=200)
        self._exception_count = 0
        self._last_global_state: Optional[GlobalStateResult] = None
        self._person_state_cache: Dict[str, PersonState] = {}

        # 授权管理
        self._allow_list: List[int] = []

        # 告警计数
        self._alert_count = 0

        # 线程锁
        self._lock = threading.RLock()

        # 代码哈希
        self._code_hash = self._calculate_code_hash()

    def _calculate_code_hash(self) -> str:
        """计算代码版本hash"""
        h = hashlib.sha256()
        files_to_hash = ["plugin.py"]
        for fname in files_to_hash:
            fpath = self.plugin_dir / fname
            if fpath.exists():
                h.update(fpath.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"

    @property
    def code_hash(self) -> str:
        """返回代码版本hash"""
        return self._code_hash

    def init(self, config: dict[str, Any] | None = None) -> bool:
        """
        初始化插件

        Args:
            config: 运行时配置

        Returns:
            是否成功初始化
        """
        config = config or {}

        try:
            normalized = self._config_manager.build(update_config=config)
            with self._lock:
                self._apply_runtime_config(normalized)
                self._initialized = True
                self.status = PluginStatus.READY
                self._config_revision += 1

            logger.info(
                "[%s] 初始化成功: zones=%s cabinets=%s allow_list=%s",
                self.id,
                len(self._zone_config.zones) if self._zone_config else 0,
                len(self._zone_config.cabinets) if self._zone_config else 0,
                self._allow_list,
            )
            return True

        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            logger.exception("[%s] 初始化失败: %s", self.id, e)
            return False

    def _apply_runtime_config(self, config: Dict[str, Any]) -> None:
        """将配置应用到插件运行时组件。"""
        self._config = copy.deepcopy(config)

        # 1. 区域配置
        self._zone_config_path = self._config_manager.resolve_zone_config_path(self._config)
        loader = ZoneConfigLoader(self._zone_config_path) if self._zone_config_path else ZoneConfigLoader()
        self._zone_config = loader.load()

        # 2. 状态机与阈值
        self._state_machine = StateMachine(self._zone_config)
        safety_config = self._config.get("safety_zone", {})
        self._state_machine.on_line_threshold = float(safety_config.get("warning_distance_m", 0.3))
        self._state_machine.cross_line_threshold = float(safety_config.get("danger_distance_m", 0.1))

        # 3. 授权策略
        self._apply_authorization_policy()

        # 4. 融合跟踪
        fusion_config = self._config.get("fusion", {})
        camera_cfg = self._config.get("camera", {})
        lidar_cfg = self._config.get("lidar", {})
        tracker_config = self._config.get("tracking", {})
        self._fused_tracker = FusedTracker(
            fusion_config={
                "max_time_diff_ms": fusion_config.get("max_time_diff_ms", 100),
                "angle_match_threshold_deg": fusion_config.get("angle_match_threshold_deg", 10.0),
                "distance_match_threshold_m": fusion_config.get("distance_match_threshold_m", 0.5),
                "image_size": tuple(camera_cfg.get("resolution", [640, 480])),
                "camera_height": camera_cfg.get("height_m", 3.0),
                "camera_tilt_deg": camera_cfg.get("tilt_deg", 40.0),
                "lidar_height": lidar_cfg.get("install", {}).get("height_m", lidar_cfg.get("height_m", 2.5)),
                "lidar_tilt_deg": lidar_cfg.get("install", {}).get("tilt_deg", lidar_cfg.get("tilt_deg", 20.0)),
            },
            tracker_config={
                "max_age": tracker_config.get("max_age", 30),
                "min_hits": tracker_config.get("min_hits", 3),
                "distance_threshold": tracker_config.get("distance_threshold", 1.0),
                "use_kalman": tracker_config.get("use_kalman", True),
            },
        )

        # 5. 适配器
        self._init_adapters(self._config)

        # 6. 审计日志
        self._init_audit_logger(self._config.get("audit", {}))

    def _apply_authorization_policy(self) -> None:
        """应用授权与机柜人数策略。"""
        auth_config = self._config.get("cabinet_authorization", {})
        self._allow_list = list(auth_config.get("allow_list", []))

        if not self._zone_config:
            return

        require_auth = bool(auth_config.get("enabled", True))
        max_persons = int(auth_config.get("max_persons_per_cabinet", 1))
        for cabinet in self._zone_config.cabinets.values():
            cabinet.zone.require_authorization = require_auth
            cabinet.zone.max_occupancy = max_persons

    def _init_audit_logger(self, audit_config: Dict[str, Any]) -> None:
        """初始化审计日志。"""
        if self._audit_logger:
            self._audit_logger.close()
        self._audit_logger = AuditLogger(audit_config)

    def _disconnect_adapters(self) -> None:
        """断开全部适配器连接。"""
        if self._camera_adapter:
            self._camera_adapter.disconnect()
            self._camera_adapter = None
        if self._lidar_adapter:
            self._lidar_adapter.disconnect()
            self._lidar_adapter = None
        if self._light_adapter:
            self._light_adapter.all_off()
            self._light_adapter.disconnect()
            self._light_adapter = None

    def _init_adapters(self, config: Dict):
        """初始化适配器"""
        self._disconnect_adapters()

        # 相机适配器
        cam_config = config.get("camera", {})
        if cam_config.get("enabled", True):
            camera_config = CameraConfig(
                source=str(cam_config.get("source", "0")),
                resolution=tuple(cam_config.get("resolution", [640, 480])),
                fps=cam_config.get("fps", 30),
                confidence_threshold=cam_config.get("confidence_threshold", 0.5),
                model_path=cam_config.get("model_path", ""),
                nms_threshold=cam_config.get("nms_threshold", 0.45),
                tracking_enabled=config.get("tracking", {}).get("enabled", True),
                max_track_age=config.get("tracking", {}).get("max_age", 30),
                height_m=cam_config.get("height_m", 3.0),
                tilt_deg=cam_config.get("tilt_deg", 40.0),
                fx=cam_config.get("fx", 500.0),
                fy=cam_config.get("fy", 500.0),
                cx=cam_config.get("cx", 320.0),
                cy=cam_config.get("cy", 240.0),
            )
            camera_config.device_id = "camera_main"
            self._camera_adapter = CameraAdapter(camera_config)
            self._camera_adapter.connect()

        # 雷达适配器
        lidar_config = config.get("lidar", {})
        if lidar_config.get("enabled", True):
            lidar_cfg = LidarConfig(
                device_ip=lidar_config.get("device_ip", "192.168.1.100"),
                device_port=lidar_config.get("device_port", 2368),
                scan_rate_hz=lidar_config.get("scan_rate_hz", 10),
                angle_min_deg=lidar_config.get("angle_min_deg", -45),
                angle_max_deg=lidar_config.get("angle_max_deg", 45),
                angle_offset_deg=lidar_config.get("angle_offset_deg", 0.0),
                height_m=lidar_config.get("install", {}).get("height_m", lidar_config.get("height_m", 2.5)),
                tilt_deg=lidar_config.get("install", {}).get("tilt_deg", lidar_config.get("tilt_deg", 20.0)),
            )
            lidar_cfg.device_id = "lidar_main"
            self._lidar_adapter = LidarAdapter(lidar_cfg)
            self._lidar_adapter.connect()

        # 灯光适配器
        light_config = config.get("light", {})
        if light_config.get("enabled", True):
            light_cfg = LightConfig(
                output_type=light_config.get("output_type", "simulate"),
            )
            self._light_adapter = LightAdapter(light_cfg)
            self._light_adapter.connect()

    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: Optional[PluginContext],
    ) -> list[RecognitionResult]:
        """
        执行推理 - 实现五层算法架构
        """
        context = self._ensure_context(context)
        rois = rois or []
        default_bbox = rois[0].bbox if rois else BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        roi_entries: list[Optional[ROI]] = rois if rois else [None]

        if not self._initialized:
            return self._build_error_results(
                context=context,
                roi_entries=roi_entries,
                default_bbox=default_bbox,
                failure_reason="9000",
                error_message="插件未初始化",
            )

        start_time = time.time()
        self._frame_count += 1
        self.status = PluginStatus.RUNNING

        results: list[RecognitionResult] = []

        with self._lock:
            try:
                if not self._state_machine or not self._fused_tracker:
                    raise RuntimeError("核心组件未就绪，请检查初始化流程")

                # ===== 第1层: 传感器适配 =====
                visual_detections = self._get_visual_detections(frame)
                lidar_detections = self._get_lidar_detections()

                # ===== 第2-3层: 融合与跟踪 =====
                tracks = self._fused_tracker.update(visual_detections, lidar_detections)

                # 获取跟踪位置
                person_positions = {}
                person_metadata = {}

                for track in tracks:
                    person_positions[track.track_id] = track.position
                    person_metadata[track.track_id] = {
                        "confidence": track.confidence,
                        "velocity": track.velocity,
                        "last_detection": track.last_detection,
                    }

                # ===== 第4层: 状态机评估 =====
                # 设置授权
                for person_id in person_positions.keys():
                    # 根据allow_list设置授权
                    self._state_machine.set_authorization(person_id, self._allow_list)

                global_state = self._state_machine.evaluate_global(
                    person_positions,
                    person_metadata
                )

                # 检查多人违规
                violations = self._state_machine.check_multi_person_violation(
                    global_state.person_states
                )

                # 记录状态变化
                self._check_state_changes(global_state.person_states)

                # ===== 第5层: 输出与控制 =====
                # 更新灯光
                if self._light_adapter:
                    self._light_adapter.set_from_alarm_level(
                        global_state.light_color,
                        global_state.status_message
                    )

                # 记录告警
                if self._audit_logger and global_state.alarm_level != GlobalAlarmLevel.GREEN:
                    self._audit_logger.log_alarm(
                        global_state.light_color,
                        global_state.status_message,
                        {"active_alarms": global_state.active_alarms}
                    )

                # 记录违规
                if self._audit_logger:
                    for violation in violations:
                        self._audit_logger.log_violation(
                            violation["type"],
                            violation
                        )

                # 保存状态
                self._last_global_state = global_state
                self._last_process_time = time.time() - start_time
                self._inference_times.append(self._last_process_time)

                # 生成结果
                for roi in roi_entries:
                    roi_id = roi.id if roi else "full_frame"
                    roi_bbox = roi.bbox if roi else default_bbox

                    # 为每个人员生成结果
                    for ps in global_state.person_states:
                        result = self._create_result(roi, ps, context, default_bbox)
                        results.append(result)

                    # 添加多人违规告警
                    for violation in violations:
                        results.append(RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=context.component_id,
                            roi_id=roi_id if roi else f"cab_{violation.get('cabinet_id', 0)}",
                            bbox=default_bbox,
                            label="multi_person",
                            confidence=1.0,
                            value=violation.get("message", ""),
                            model_version=self.version,
                            code_version=self.code_hash,
                            metadata=violation
                        ))

                    # 如果没有检测到人
                    if not global_state.person_states:
                        results.append(RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=context.component_id,
                            roi_id=roi_id,
                            bbox=roi_bbox,
                            label="no_person",
                            confidence=1.0,
                            value="区域空闲",
                            model_version=self.version,
                            code_version=self.code_hash,
                            metadata={"message": "未检测到人员"}
                        ))

                self.status = PluginStatus.READY
                return results

            except Exception as e:
                self._exception_count += 1
                self.status = PluginStatus.ERROR
                logger.exception("[%s] 推理失败: %s", self.id, e)
                if self._audit_logger:
                    self._audit_logger.log_event("infer_error", {"error": str(e)})
                return self._build_error_results(
                    context=context,
                    roi_entries=roi_entries,
                    default_bbox=default_bbox,
                    failure_reason="9001",
                    error_message=str(e),
                )

    def _ensure_context(self, context: Optional[PluginContext]) -> PluginContext:
        """兜底运行上下文，兼容独立调试调用。"""
        if context is not None:
            return context
        return PluginContext(
            task_id=f"standalone-{int(time.time() * 1000)}",
            site_id="standalone",
            device_id=self.id,
            component_id="default",
        )

    def _build_error_results(
        self,
        context: PluginContext,
        roi_entries: List[Optional[ROI]],
        default_bbox: BoundingBox,
        failure_reason: str,
        error_message: str,
    ) -> list[RecognitionResult]:
        """构建统一错误返回。"""
        outputs: list[RecognitionResult] = []
        for roi in roi_entries:
            outputs.append(RecognitionResult(
                task_id=context.task_id,
                site_id=context.site_id,
                device_id=context.device_id,
                component_id=context.component_id,
                roi_id=roi.id if roi else "full_frame",
                bbox=roi.bbox if roi else default_bbox,
                label="error",
                confidence=0.0,
                model_version=self.version,
                code_version=self.code_hash,
                failure_reason=failure_reason,
                metadata={"error": error_message},
            ))
        return outputs

    def _get_visual_detections(self, frame: Optional[np.ndarray]) -> List[VisualDetection]:
        """获取视觉检测结果"""
        detections = []

        if self._camera_adapter and self._camera_adapter.is_connected:
            person_detections = self._camera_adapter.get_person_detections(frame)

            if person_detections:
                for pd in person_detections:
                    vis_det = VisualDetection(
                        detection_id=pd.detection_id,
                        bbox=pd.bbox,
                        foot_pixel=pd.foot_pixel,
                        confidence=pd.confidence,
                        timestamp=pd.timestamp,
                        track_id=pd.track_id,
                    )
                    detections.append(vis_det)

        return detections

    def _get_lidar_detections(self) -> List[LidarDetection]:
        """获取雷达检测结果"""
        detections = []

        if self._lidar_adapter and self._lidar_adapter.is_connected:
            scan = self._lidar_adapter.get_scan()

            if scan:
                clusters = self._lidar_adapter.get_clusters(scan)

                for cluster in clusters:
                    lid_det = LidarDetection(
                        cluster_id=cluster.cluster_id,
                        center_distance=cluster.center_distance,
                        center_angle=cluster.center_angle,
                        point_count=cluster.point_count,
                        confidence=cluster.confidence,
                    )
                    detections.append(lid_det)

        return detections

    def _check_state_changes(self, person_states: List[PersonStateResult]):
        """检查并记录状态变化"""
        if not self._audit_logger:
            return
        for ps in person_states:
            old_state = self._person_state_cache.get(ps.person_id)
            new_state = ps.state

            if old_state != new_state:
                self._audit_logger.log_state_change(
                    ps.person_id,
                    old_state.value if old_state else "none",
                    new_state.value,
                    {
                        "zone": ps.current_zone,
                        "cabinet": ps.current_cabinet,
                        "distance_to_line": ps.distance_to_yellow_line,
                        "authorized": ps.is_authorized,
                    }
                )
                self._person_state_cache[ps.person_id] = new_state

    def _create_result(
        self,
        roi: Optional[ROI],
        person_state: PersonStateResult,
        context: PluginContext,
        default_bbox: Optional[BoundingBox] = None,
    ) -> RecognitionResult:
        """创建识别结果"""
        state = person_state.state

        if state in (PersonState.CROSS_LINE, PersonState.HIGH_RISK):
            self._alert_count += 1

        # 状态标签映射
        label_map = {
            PersonState.NORMAL: "safe",
            PersonState.ON_LINE: "on_line",
            PersonState.CROSS_LINE: "line_cross",
            PersonState.MISPLACED: "unauthorized",
            PersonState.HIGH_RISK: "high_risk",
        }

        label_name_map = {
            PersonState.NORMAL: "安全状态",
            PersonState.ON_LINE: "压线警告",
            PersonState.CROSS_LINE: "越线危险",
            PersonState.MISPLACED: "未授权操作",
            PersonState.HIGH_RISK: "高风险区域",
        }

        # 计算边界框
        bbox = roi.bbox if roi else (default_bbox or BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0))

        return RecognitionResult(
            task_id=context.task_id,
            site_id=context.site_id,
            device_id=context.device_id,
            component_id=context.component_id,
            roi_id=roi.id if roi else "full_frame",
            label=label_map.get(state, "unknown"),
            confidence=1.0,
            value=f"机柜{person_state.current_cabinet or 0} · {person_state.distance_to_yellow_line:.2f}m",
            bbox=bbox,
            model_version=self.version,
            code_version=self.code_hash,
            metadata={
                "person_id": person_state.person_id,
                "state": state.value,
                "label_name": label_name_map.get(state, "未知"),
                "cabinet_id": person_state.current_cabinet,
                "zone_id": person_state.current_zone,
                "authorized": person_state.is_authorized,
                "distance_to_line_m": person_state.distance_to_yellow_line,
                "position": {"x": person_state.position.x, "y": person_state.position.y},
                "alert_level": person_state.alert_level.value,
                "alert_message": person_state.alert_message,
            }
        )

    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """后处理和告警生成"""
        alarms: list[Alarm] = []

        for result in results:
            if result.label == "line_cross":
                person_id = result.metadata.get("person_id", "人员")
                distance = result.metadata.get("distance_to_line_m")
                cabinet_id = result.metadata.get("cabinet_id")
                distance_text = f"，距离{distance:.2f}m" if distance is not None else ""
                cabinet_text = f"机柜{cabinet_id}" if cabinet_id is not None else "警戒区域"
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.CRITICAL,
                    title=self.LABEL_NAMES.get("line_cross", "越线告警"),
                    message=f"{person_id}已越过{cabinet_text}黄色警戒线{distance_text}",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))

            elif result.label == "high_risk":
                person_id = result.metadata.get("person_id", "人员")
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.CRITICAL,
                    title=self.LABEL_NAMES.get("high_risk", "高风险"),
                    message=f"{person_id}进入高风险区域！",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))

            elif result.label == "unauthorized":
                person_id = result.metadata.get("person_id", "人员")
                cabinet_id = result.metadata.get("cabinet_id")
                cabinet_text = f"机柜{cabinet_id}" if cabinet_id is not None else "未授权区域"
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=self.LABEL_NAMES.get("unauthorized", "未授权操作"),
                    message=f"{person_id}在{cabinet_text}进行操作",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))

            elif result.label == "on_line":
                person_id = result.metadata.get("person_id", "人员")
                distance = result.metadata.get("distance_to_line_m")
                distance_text = f"{distance:.2f}m" if isinstance(distance, (int, float)) else "未知距离"
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=self.LABEL_NAMES.get("on_line", "压线警告"),
                    message=f"{person_id}接近黄线，距离{distance_text}",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))

            elif result.label == "multi_person":
                cabinet_id = result.metadata.get("cabinet_id")
                person_count = result.metadata.get("current_count")
                cabinet_text = f"机柜{cabinet_id}" if cabinet_id is not None else "机柜区域"
                count_text = f"{person_count}人" if person_count is not None else "多人"
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=self.LABEL_NAMES.get("multi_person", "同柜多人"),
                    message=f"{cabinet_text}前检测到{count_text}同时操作",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))

        return alarms

    def healthcheck(self) -> HealthStatus:
        """健康检查"""
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
                details={"status": "not_initialized"},
            )

        # 检查适配器状态
        adapters_ok = True
        adapter_details: Dict[str, Any] = {}
        for name, adapter in (
            ("camera", self._camera_adapter),
            ("lidar", self._lidar_adapter),
            ("light", self._light_adapter),
        ):
            if not adapter:
                continue
            adapter_details[name] = {
                "status": adapter.status.value,
                "connected": adapter.is_connected,
                "simulated": adapter.is_simulated,
                "last_error": adapter.last_error,
                "stats": adapter.get_stats(),
            }
            if not adapter.is_connected and adapter.status not in (AdapterStatus.SIMULATED, AdapterStatus.SIMULATING):
                adapters_ok = False

        # 检查处理时间，目标值来自方案: <100ms
        latency_ok = self._last_process_time < 0.1
        inference_ms = [x * 1000.0 for x in self._inference_times]
        latency_summary = {
            "current_ms": round(self._last_process_time * 1000.0, 2),
            "window_size": len(inference_ms),
            "avg_ms": round(float(np.mean(inference_ms)), 2) if inference_ms else 0.0,
            "p95_ms": round(float(np.percentile(inference_ms, 95)), 2) if len(inference_ms) >= 5 else 0.0,
            "max_ms": round(float(np.max(inference_ms)), 2) if inference_ms else 0.0,
        }

        person_count = len(self._last_global_state.person_states) if self._last_global_state else 0
        cabinet_count = len(self._zone_config.cabinets) if self._zone_config else 0

        return HealthStatus(
            healthy=adapters_ok and latency_ok,
            message=f"室内监控就绪 V2.1，{cabinet_count}个机柜，{person_count}人跟踪中",
            details={
                "status": "ready",
                "version": self.version,
                "config_revision": self._config_revision,
                "frame_count": self._frame_count,
                "alert_count": self._alert_count,
                "exception_count": self._exception_count,
                "tracked_persons": person_count,
                "cabinet_count": cabinet_count,
                "allow_list": self._allow_list,
                "adapters": adapter_details,
                "latency": latency_summary,
                "last_process_time_ms": self._last_process_time * 1000.0,
                "queue_lengths": {
                    "inference_window": len(self._inference_times),
                    "audit_events": len(self._audit_logger._event_logs) if self._audit_logger else 0,
                },
                "last_alarm_level": self._last_global_state.light_color if self._last_global_state else "green",
                "zone_config_path": str(self._zone_config_path) if self._zone_config_path else "",
            }
        )

    def cleanup(self) -> None:
        """清理资源"""
        # 断开适配器
        self._disconnect_adapters()

        # 关闭审计日志
        if self._audit_logger:
            self._audit_logger.close()
            self._audit_logger = None

        # 重置状态
        self._person_state_cache.clear()

        self._initialized = False
        self.status = PluginStatus.UNLOADED
        logger.info("[%s] 插件已清理", self.id)

    # ==================== 扩展API接口 ====================

    def update_allow_list(self, cabinet_ids: List[int]):
        """更新授权机柜列表"""
        with self._lock:
            self._allow_list = cabinet_ids
            self._config.setdefault("cabinet_authorization", {})["allow_list"] = list(cabinet_ids)
        if self._audit_logger:
            self._audit_logger.log_event("config_change", {
                "key": "allow_list",
                "value": cabinet_ids,
            })
        logger.info("[%s] 授权列表已更新: %s", self.id, self._allow_list)

    def set_authorization(self, person_id: str, cabinet_ids: List[int]):
        """设置人员授权"""
        if self._state_machine:
            self._state_machine.set_authorization(person_id, cabinet_ids)
            if self._audit_logger:
                self._audit_logger.log_event("authorization", {
                    "person_id": person_id,
                    "cabinet_ids": cabinet_ids,
                    "message": f"设置授权: {person_id} -> 机柜{cabinet_ids}",
                })

    def clear_authorization(self, person_id: str):
        """清除人员授权"""
        if self._state_machine:
            self._state_machine.clear_authorization(person_id)
            if self._audit_logger:
                self._audit_logger.log_event("authorization", {
                    "person_id": person_id,
                    "action": "clear",
                    "message": f"清除授权: {person_id}",
                })

    def get_cabinet_status(self) -> Dict[int, Dict]:
        """获取所有机柜状态"""
        if not self._zone_config:
            return {}

        return {
            cab_id: {
                "id": cab_id,
                "name": cab.name,
                "status": cab.status,
                "occupants": cab.current_occupants,
                "authorized": cab.is_authorized,
            }
            for cab_id, cab in self._zone_config.cabinets.items()
        }

    def get_tracked_persons(self) -> List[Dict]:
        """获取跟踪人员列表"""
        if not self._last_global_state:
            return []

        return [
            {
                "person_id": ps.person_id,
                "state": ps.state.value,
                "position": {"x": ps.position.x, "y": ps.position.y},
                "cabinet_id": ps.current_cabinet,
                "authorized": ps.is_authorized,
                "distance_to_line_m": ps.distance_to_yellow_line,
            }
            for ps in self._last_global_state.person_states
        ]

    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """获取最近事件"""
        if self._audit_logger:
            return self._audit_logger.get_recent_events(limit)
        return []

    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置"""
        return {
            "detection_types": [
                {"id": "person_detection", "name": "人员检测", "default": True},
                {"id": "line_check", "name": "越线检测", "default": True},
                {"id": "auth_check", "name": "授权校验", "default": True},
                {"id": "multi_person", "name": "多人检测", "default": True},
            ],
            "cabinet_count": len(self._zone_config.cabinets) if self._zone_config else 0,
            "allow_list": self._allow_list,
            "yellow_line_distance_m": self._config.get("safety_zone", {}).get("yellow_line_distance_m", 0.5),
            "lidar_enabled": self._config.get("lidar", {}).get("enabled", False),
            "camera_enabled": self._config.get("camera", {}).get("enabled", False),
            "version": self.version,
            "config_revision": self._config_revision,
        }

    def get_zone_config(self) -> Dict:
        """获取区域配置"""
        if not self._zone_config:
            return {}

        return {
            "id": self._zone_config.config_id,
            "name": self._zone_config.config_name,
            "version": self._zone_config.version,
            "path": str(self._zone_config_path) if self._zone_config_path else "",
            "zones": [
                {
                    "id": z.zone_id,
                    "name": z.name,
                    "type": z.zone_type.value,
                    "vertices": [(v.x, v.y) for v in z.polygon.vertices],
                }
                for z in self._zone_config.zones.values()
            ],
            "cabinets": [
                {
                    "id": c.cabinet_id,
                    "name": c.name,
                    "x_start": c.x_start,
                    "x_end": c.x_end,
                }
                for c in self._zone_config.cabinets.values()
            ],
            "yellow_lines": [
                {
                    "id": yl.line_id,
                    "name": yl.name,
                    "start": (yl.line.start.x, yl.line.start.y),
                    "end": (yl.line.end.x, yl.line.end.y),
                    "cabinets": yl.associated_cabinets,
                    "warning_distance": yl.warning_distance,
                    "danger_distance": yl.danger_distance,
                }
                for yl in self._zone_config.yellow_lines.values()
            ],
        }

    def get_runtime_config(self) -> Dict[str, Any]:
        """获取当前运行配置快照。"""
        return copy.deepcopy(self._config)

    def update_config(self, new_config: Dict[str, Any], merge: bool = True) -> Dict[str, Any]:
        """
        更新插件配置。

        - merge=True: 在当前配置基础上增量更新
        - merge=False: 以默认配置为基线替换
        """
        with self._lock:
            previous_config = copy.deepcopy(self._config)
            previous_revision = self._config_revision
            try:
                normalized = self._config_manager.build(
                    update_config=new_config,
                    base_config=previous_config if merge else None,
                )
                self._apply_runtime_config(normalized)
                self._config_revision = previous_revision + 1
                self.status = PluginStatus.READY
                if self._audit_logger:
                    self._audit_logger.log_event("config_update", {
                        "merge": merge,
                        "revision": self._config_revision,
                    })
                logger.info("[%s] 配置已更新, revision=%s", self.id, self._config_revision)
                return self.get_runtime_config()
            except Exception as e:
                self._exception_count += 1
                self.status = PluginStatus.ERROR
                self._last_error = str(e)
                logger.exception("[%s] 配置更新失败，执行回滚: %s", self.id, e)
                if previous_config:
                    try:
                        self._apply_runtime_config(previous_config)
                        self.status = PluginStatus.READY
                        self._config_revision = previous_revision
                    except Exception as rollback_error:
                        logger.exception("[%s] 回滚失败: %s", self.id, rollback_error)
                raise ValueError(f"配置更新失败: {e}") from e

    def on_config_update(self, new_config: dict[str, Any]) -> None:
        """SDK 配置更新回调。"""
        self.update_config(new_config, merge=True)

    def update_zone_config(self, zone_config_data: Dict[str, Any], persist: Optional[bool] = None) -> Dict[str, Any]:
        """更新区域配置并按需落盘。"""
        if not isinstance(zone_config_data, dict):
            raise ValueError("zone_config_data 必须是字典")

        with self._lock:
            loader = ZoneConfigLoader()
            try:
                parsed_config = loader._parse_config(zone_config_data)
            except Exception as e:
                raise ValueError(f"区域配置解析失败: {e}") from e

            self._zone_config = parsed_config
            self._state_machine = StateMachine(self._zone_config)
            safety_cfg = self._config.get("safety_zone", {})
            self._state_machine.on_line_threshold = float(safety_cfg.get("warning_distance_m", 0.3))
            self._state_machine.cross_line_threshold = float(safety_cfg.get("danger_distance_m", 0.1))
            self._apply_authorization_policy()

            persist_flag = (
                self._config.get("zone_config", {}).get("persist_on_update", True)
                if persist is None else bool(persist)
            )
            if persist_flag:
                target_path = self._zone_config_path or self._config_manager.resolve_zone_config_path(self._config)
                if target_path is None:
                    target_path = self._config_manager.default_zone_config_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not loader.save(parsed_config, target_path):
                    raise ValueError("区域配置持久化失败")
                self._zone_config_path = target_path
                try:
                    rel_path = target_path.relative_to(self.plugin_dir)
                    self._config.setdefault("zone_config", {})["path"] = str(rel_path)
                except ValueError:
                    self._config.setdefault("zone_config", {})["path"] = str(target_path)

            self._config_revision += 1
            if self._audit_logger:
                self._audit_logger.log_event("zone_config_update", {
                    "revision": self._config_revision,
                    "persist": persist_flag,
                })
            return self.get_zone_config()

    def update_config_key(self, key: str, value: Any) -> Dict[str, Any]:
        """按点路径更新单个配置字段。"""
        patched = IndoorFenceConfigManager.patch_by_key(self.get_runtime_config(), key, value)
        return self.update_config(patched, merge=False)

    def get_status(self) -> Dict[str, Any]:
        """提供给 standalone runner 的额外状态信息。"""
        return {
            "tracked_persons": self.get_tracked_persons(),
            "cabinet_status": self.get_cabinet_status(),
            "allow_list": self._allow_list,
            "zone_config": self.get_zone_config(),
            "config_revision": self._config_revision,
        }

    def get_standalone_routes(self) -> list:
        """注册电子围栏专用 standalone API。"""

        async def _get_plugin_config():
            return {
                "config": self.get_runtime_config(),
                "config_schema": self.manifest.config_schema,
                "revision": self._config_revision,
            }

        async def _patch_plugin_config(payload: Dict[str, Any]):
            payload = payload or {}
            cfg = payload.get("config", payload)
            merge = bool(payload.get("merge", True))
            return {
                "status": "updated",
                "config": self.update_config(cfg, merge=merge),
                "revision": self._config_revision,
            }

        async def _update_config_key(payload: Dict[str, Any]):
            payload = payload or {}
            return {
                "status": "updated",
                "config": self.update_config_key(payload.get("key", ""), payload.get("value")),
                "revision": self._config_revision,
            }

        async def _get_zone():
            return self.get_zone_config()

        async def _put_zone(payload: Dict[str, Any]):
            payload = payload or {}
            zone_data = payload.get("zone_config", payload)
            persist = payload.get("persist", None)
            return {
                "status": "updated",
                "zone_config": self.update_zone_config(zone_data, persist=persist),
                "revision": self._config_revision,
            }

        async def _get_events(limit: int = 100):
            return self.get_recent_events(limit=limit)

        return [
            {
                "path": "/api/indoor-fence/config",
                "endpoint": _get_plugin_config,
                "methods": ["GET"],
                "summary": "获取电子围栏配置",
            },
            {
                "path": "/api/indoor-fence/config",
                "endpoint": _patch_plugin_config,
                "methods": ["PATCH", "PUT"],
                "summary": "更新电子围栏配置",
            },
            {
                "path": "/api/indoor-fence/config/key",
                "endpoint": _update_config_key,
                "methods": ["PUT"],
                "summary": "更新单个配置字段",
            },
            {
                "path": "/api/indoor-fence/zones",
                "endpoint": _get_zone,
                "methods": ["GET"],
                "summary": "获取区域配置",
            },
            {
                "path": "/api/indoor-fence/zones",
                "endpoint": _put_zone,
                "methods": ["PUT"],
                "summary": "更新区域配置",
            },
            {
                "path": "/api/indoor-fence/events",
                "endpoint": _get_events,
                "methods": ["GET"],
                "summary": "获取最近事件",
            },
        ]

    @classmethod
    def create_standalone(cls, config=None):
        """Create plugin instance for standalone operation."""
        plugin_dir = Path(__file__).resolve().parent
        manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
        instance = cls(manifest, plugin_dir)
        if config is None:
            from darkbreaker_sdk.utils import load_plugin_config
            config = load_plugin_config(plugin_dir / "configs" / "default.yaml")
        instance.init(config)
        return instance


# Alias for standalone imports
Plugin = IndoorFencePlugin
