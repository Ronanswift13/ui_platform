"""
室内电子围栏插件 V2.0 - 多人安全监测系统
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
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import threading
from collections import deque
import json
import sys
import importlib.util
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

from platform_core.plugin_manager.base import (
    BasePlugin,
    HealthStatus,
    PluginContext,
    PluginManifest,
    PluginStatus,
)
from platform_core.schema.models import (
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
)

# 导入适配器 (使用绝对导入以支持动态加载)
from plugins.indoor_fence.adapters import (
    BaseAdapter, AdapterStatus,
    CameraAdapter, CameraConfig, PersonDetection,
    LidarAdapter, LidarConfig, LidarScan, LidarCluster,
    LightAdapter, LightColor, LightConfig, LightState,
)


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
    室内电子围栏插件 V2.0

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
        self._config: Dict[str, Any] = {}
        self._initialized = False

        # V2 核心组件
        self._zone_config: Optional[ZoneConfiguration] = None
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
            self._config = config

            # 1. 加载区域配置
            zone_config_path = config.get("zone_config_path")
            if zone_config_path:
                loader = ZoneConfigLoader(Path(zone_config_path))
                self._zone_config = loader.load()
            else:
                loader = ZoneConfigLoader()
                self._zone_config = loader.load()

            print(f"[{self.id}] 区域配置已加载: "
                  f"{len(self._zone_config.zones)}个区域, "
                  f"{len(self._zone_config.cabinets)}个机柜")

            # 2. 初始化状态机
            self._state_machine = StateMachine(self._zone_config)

            # 配置阈值
            safety_config = config.get("safety_zone", {})
            self._state_machine.on_line_threshold = safety_config.get("warning_distance_m", 0.3)
            self._state_machine.cross_line_threshold = safety_config.get("danger_distance_m", 0.0)

            # 3. 初始化融合跟踪器
            fusion_config = {
                "max_time_diff_ms": config.get("fusion", {}).get("max_time_diff_ms", 100),
                "distance_match_threshold_m": config.get("fusion", {}).get("distance_match_threshold_m", 0.5),
            }
            tracker_config = {
                "max_age": config.get("tracking", {}).get("max_age", 30),
                "min_hits": config.get("tracking", {}).get("min_hits", 3),
                "distance_threshold": config.get("tracking", {}).get("distance_threshold", 1.0),
                "use_kalman": config.get("tracking", {}).get("use_kalman", True),
            }
            self._fused_tracker = FusedTracker(
                fusion_config=fusion_config,
                tracker_config=tracker_config,
            )

            # 4. 初始化适配器
            self._init_adapters(config)

            # 5. 初始化审计日志
            audit_config = config.get("audit", {"enabled": True, "log_level": "event"})
            self._audit_logger = AuditLogger(audit_config)

            # 6. 设置授权列表
            auth_config = config.get("cabinet_authorization", {})
            self._allow_list = auth_config.get("allow_list", [])

            self._initialized = True
            self.status = PluginStatus.READY

            print(f"[{self.id}] 插件初始化成功 (V2.0)")
            print(f"[{self.id}] 授权机柜: {self._allow_list}")

            return True

        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            print(f"[{self.id}] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_adapters(self, config: Dict):
        """初始化适配器"""
        # 相机适配器
        cam_config = config.get("camera", {})
        if cam_config.get("enabled", True):
            camera_config = CameraConfig(
                source=str(cam_config.get("source", "0")),
                resolution=tuple(cam_config.get("resolution", [640, 480])),
                fps=cam_config.get("fps", 30),
                confidence_threshold=cam_config.get("confidence_threshold", 0.5),
                model_path=cam_config.get("model_path", ""),
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
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """
        执行推理 - 实现五层算法架构
        """
        if not self._initialized:
            return [RecognitionResult(
                task_id=context.task_id,
                site_id=context.site_id,
                device_id=context.device_id,
                component_id=context.component_id,
                roi_id=roi.id,
                bbox=roi.bbox,
                label="error",
                confidence=0.0,
                model_version=self.version,
                code_version=self.code_hash,
                failure_reason="9000",
                metadata={"error": "插件未初始化"}
            ) for roi in rois]

        start_time = time.time()
        self._frame_count += 1
        self.status = PluginStatus.RUNNING

        results: list[RecognitionResult] = []

        with self._lock:
            try:
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
                if global_state.alarm_level != GlobalAlarmLevel.GREEN:
                    self._audit_logger.log_alarm(
                        global_state.light_color,
                        global_state.status_message,
                        {"active_alarms": global_state.active_alarms}
                    )

                # 记录违规
                for violation in violations:
                    self._audit_logger.log_violation(
                        violation["type"],
                        violation
                    )

                # 保存状态
                self._last_global_state = global_state
                self._last_process_time = time.time() - start_time

                # 生成结果
                default_bbox = rois[0].bbox if rois else BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)

                for roi in rois:
                    # 为每个人员生成结果
                    for ps in global_state.person_states:
                        result = self._create_result(roi, ps, context)
                        results.append(result)

                    # 添加多人违规告警
                    for violation in violations:
                        results.append(RecognitionResult(
                            task_id=context.task_id,
                            site_id=context.site_id,
                            device_id=context.device_id,
                            component_id=context.component_id,
                            roi_id=f"cab_{violation.get('cabinet_id', 0)}",
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
                            roi_id=roi.id,
                            bbox=roi.bbox,
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
                self.status = PluginStatus.ERROR
                import traceback
                traceback.print_exc()
                return [RecognitionResult(
                    task_id=context.task_id,
                    site_id=context.site_id,
                    device_id=context.device_id,
                    component_id=context.component_id,
                    roi_id=roi.id,
                    bbox=roi.bbox,
                    label="error",
                    confidence=0.0,
                    model_version=self.version,
                    code_version=self.code_hash,
                    failure_reason="9001",
                    metadata={"error": str(e)}
                ) for roi in rois]

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
        roi: ROI,
        person_state: PersonStateResult,
        context: PluginContext,
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
        bbox = roi.bbox

        return RecognitionResult(
            task_id=context.task_id,
            site_id=context.site_id,
            device_id=context.device_id,
            component_id=context.component_id,
            roi_id=roi.id,
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
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=self.LABEL_NAMES.get("on_line", "压线警告"),
                    message=f"{person_id}接近黄线，距离{distance:.2f}m",
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
        adapter_details = {}

        if self._camera_adapter:
            adapter_details["camera"] = self._camera_adapter.status.value
            if not self._camera_adapter.is_connected:
                adapters_ok = False

        if self._lidar_adapter:
            adapter_details["lidar"] = self._lidar_adapter.status.value
            if not self._lidar_adapter.is_connected:
                adapters_ok = False

        if self._light_adapter:
            adapter_details["light"] = self._light_adapter.status.value

        # 检查处理时间
        latency_ok = self._last_process_time < 0.2  # < 200ms

        person_count = len(self._last_global_state.person_states) if self._last_global_state else 0
        cabinet_count = len(self._zone_config.cabinets) if self._zone_config else 0

        return HealthStatus(
            healthy=adapters_ok and latency_ok,
            message=f"室内监控就绪 V2.0，{cabinet_count}个机柜，{person_count}人跟踪中",
            details={
                "status": "ready",
                "version": "2.0.0",
                "frame_count": self._frame_count,
                "alert_count": self._alert_count,
                "tracked_persons": person_count,
                "cabinet_count": cabinet_count,
                "allow_list": self._allow_list,
                "adapters": adapter_details,
                "last_process_time_ms": self._last_process_time * 1000,
                "last_alarm_level": self._last_global_state.light_color if self._last_global_state else "green",
            }
        )

    def cleanup(self) -> None:
        """清理资源"""
        # 断开适配器
        if self._camera_adapter:
            self._camera_adapter.disconnect()
        if self._lidar_adapter:
            self._lidar_adapter.disconnect()
        if self._light_adapter:
            self._light_adapter.all_off()
            self._light_adapter.disconnect()

        # 关闭审计日志
        if self._audit_logger:
            self._audit_logger.close()

        # 重置状态
        self._person_state_cache.clear()

        self._initialized = False
        self.status = PluginStatus.UNLOADED
        print(f"[{self.id}] 插件已清理")

    # ==================== 扩展API接口 ====================

    def update_allow_list(self, cabinet_ids: List[int]):
        """更新授权机柜列表"""
        self._allow_list = cabinet_ids
        if self._audit_logger:
            self._audit_logger.log_event("config_change", {
                "key": "allow_list",
                "value": cabinet_ids,
            })
        print(f"[{self.id}] 授权列表已更新: {self._allow_list}")

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
            "version": "2.0.0",
        }

    def get_zone_config(self) -> Dict:
        """获取区域配置"""
        if not self._zone_config:
            return {}

        return {
            "id": self._zone_config.config_id,
            "name": self._zone_config.config_name,
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
        }
