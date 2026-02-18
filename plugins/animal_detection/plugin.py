#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测插件 V1.0
===========================

完整闭环: 检测 → 热成像校验 → 跟踪 → 驱离 → 统计
实现 BasePlugin 接口: init / infer / postprocess / healthcheck

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# === 确保包路径可用 ===
_plugin_dir = Path(__file__).resolve().parent
_project_root = _plugin_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# === 平台基础设施导入 ===
try:
    from platform_core.plugin_manager.base import (
        BasePlugin, HealthStatus, PluginContext, PluginManifest, PluginStatus,
    )
    from platform_core.schema.models import (
        Alarm, AlarmLevel, AlarmRule, RecognitionResult, ROI, BoundingBox as PlatformBBox, PluginOutput,
    )
    PLATFORM_AVAILABLE = True
except ImportError:
    PLATFORM_AVAILABLE = False
    # 降级兼容: 定义最小占位
    class PluginStatus:
        UNLOADED = "unloaded"; LOADING = "loading"; READY = "ready"
        RUNNING = "running"; ERROR = "error"

# === 核心模块导入 ===
from plugins.animal_detection.core import (
    YOLOv8Detector,
    AnimalONNXEngine,
    ThermalValidator,
    AnimalTracker,
    DeterrentController,
    StatisticsCollector,
    AnimalDetectionResult,
    AnimalEvent,
    EventType,
    build_intrusion_event,
    EvidenceItem,
    AnimalClass,
    BoundingBox,
    CLASS_NAMES_CN,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 审计日志
# =============================================================================

class AnimalAuditLogger:
    """审计日志记录器"""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._events: List[Dict] = []
        self._lock = threading.Lock()

    def log_event(self, event: AnimalEvent):
        """记录事件到审计日志"""
        if not self.enabled:
            return

        entry = {
            "logged_at": datetime.now().isoformat(),
            "trace_id": event.trace_id,
            "event_type": event.type,
            "location": event.location,
            "confidence": event.confidence,
            "risk_level": event.risk_level,
            "detection_count": event.total_detections,
            "suggestion": event.suggestion,
            "deterrent_active": event.deterrent_active,
            "classes": [d.animal_class for d in event.detections],
        }

        with self._lock:
            self._events.append(entry)

            # 写入日志文件
            log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"审计日志写入失败: {e}")

    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        with self._lock:
            return self._events[-limit:]

    def close(self):
        pass


# =============================================================================
# 证据管理
# =============================================================================

class EvidenceManager:
    """证据截图管理"""

    def __init__(self, evidence_dir: str):
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def save_detection_frame(
        self,
        frame: np.ndarray,
        detections: List[AnimalDetectionResult],
        trace_id: str,
    ) -> List[EvidenceItem]:
        """保存检测帧截图（标注检测框）"""
        evidence_items = []
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 保存原始帧
        raw_path = self.evidence_dir / f"raw_{trace_id[:8]}_{now_str}.jpg"
        cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        evidence_items.append(EvidenceItem(
            type="raw_image",
            path=str(raw_path),
            description="原始检测帧",
        ))

        # 保存标注帧
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        for det in detections:
            if det.bbox is None:
                continue
            x1 = int(det.bbox.x * w)
            y1 = int(det.bbox.y * h)
            x2 = int((det.bbox.x + det.bbox.width) * w)
            y2 = int((det.bbox.y + det.bbox.height) * h)

            # 绘制框和标签
            color = (0, 0, 255) if det.confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES_CN.get(det.animal_class, det.animal_class)} {det.confidence:.2f}"
            if det.track_id:
                label += f" [{det.track_id}]"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ann_path = self.evidence_dir / f"annotated_{trace_id[:8]}_{now_str}.jpg"
        cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        evidence_items.append(EvidenceItem(
            type="annotated_image",
            path=str(ann_path),
            description=f"标注帧: 检测到{len(detections)}个目标",
        ))

        return evidence_items


# =============================================================================
# 主插件类
# =============================================================================

class AnimalDetectionPlugin:
    """
    室内动物入侵检测插件

    实现完整闭环:
    1. YOLOv8 检测
    2. 热成像校验 (可选)
    3. 多目标跟踪
    4. 声光驱离
    5. 统计与审计
    """

    PLUGIN_ID = "animal_detection"
    PLUGIN_NAME = "室内动物入侵检测插件"
    PLUGIN_VERSION = "1.0.0"

    def __init__(
        self,
        manifest: Optional[Any] = None,
        plugin_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # 与 PluginManager.load_plugin(plugin_class(manifest, plugin_dir)) 对齐
        self.manifest = manifest
        self.plugin_dir = Path(plugin_dir) if plugin_dir is not None else Path(__file__).parent

        self.id = getattr(manifest, "id", self.PLUGIN_ID)
        self.name = getattr(manifest, "name", self.PLUGIN_NAME)
        self.version = getattr(manifest, "version", self.PLUGIN_VERSION)
        self.status = PluginStatus.UNLOADED
        self.code_hash = self._compute_code_hash()
        self._last_error = ""

        self._config: Dict[str, Any] = config.copy() if isinstance(config, dict) else {}
        self._initialized = False
        self._lock = threading.Lock()
        self._model_ready = False

        # 核心组件
        self._detector: Optional[YOLOv8Detector] = None
        self._onnx_engine: Optional[AnimalONNXEngine] = None
        self._thermal_validator: Optional[ThermalValidator] = None
        self._tracker: Optional[AnimalTracker] = None
        self._deterrent: Optional[DeterrentController] = None
        self._statistics: Optional[StatisticsCollector] = None
        self._audit_logger: Optional[AnimalAuditLogger] = None
        self._evidence_manager: Optional[EvidenceManager] = None
        self._model_backend: str = "none"
        self._model_mode: str = "unknown"
        self._model_path: str = ""
        self._last_inference_backend_status: Dict[str, Any] = {}

        # 运行时状态
        self._frame_count: int = 0
        self._last_process_time: float = 0
        self._last_event: Optional[AnimalEvent] = None
        self._inference_count: int = 0
        self._error_count: int = 0
        self._last_inference_time: Optional[datetime] = None

        # 仿真模式
        self._simulation_mode: bool = False

    def set_status(self, status: Any, message: str = ""):
        """兼容 PluginManager 的状态设置接口。"""
        if isinstance(status, str):
            try:
                status = PluginStatus(status)
            except Exception:
                pass
        self.status = status
        if message:
            self._last_error = message

    # ==================== 生命周期接口 ====================

    def init(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化插件

        Args:
            config: 插件配置字典

        Returns:
            是否初始化成功
        """
        self.status = PluginStatus.LOADING
        if config:
            self._config = self._merge_dict(self._config, config)
        if not self._config:
            self._config = self._load_default_config()
        self._config = self._normalize_config(self._config)

        try:
            plugin_dir = self.plugin_dir

            # 仿真模式检查 (默认关闭)
            sim_config = self._config.get("simulation", {})
            self._simulation_mode = sim_config.get("enabled", False)
            if self._simulation_mode:
                logger.warning("⚠️ 动物检测插件运行在仿真模式（非真实推理）")

            # 1. 初始化模型推理后端 (ONNX 引擎优先, 兼容旧检测器)
            model_config = self._config.get("model", {})
            default_model_path = plugin_dir / "models" / "animal_yolov8n.onnx"
            model_path = self._resolve_model_path(
                model_config.get("path", str(default_model_path))
            )
            self._model_path = model_path
            inference_config = self._config.get("inference", {})

            if not self._simulation_mode:
                self._initialize_model_backend(
                    model_path=model_path,
                    model_config=model_config,
                    inference_config=inference_config,
                )
                if not self._model_ready:
                    # 非演示要求：无模型时不输出假数据，进入降级空跑模式
                    self._last_error = f"model_load_failed:{model_path}"
                    logger.error("模型加载失败，进入降级模式（不生成模拟检测数据）")
            else:
                self._model_ready = False
                self._model_backend = "simulation"
                logger.info("仿真模式: 跳过模型加载")

            # 2. 初始化热成像校验器
            thermal_config = self._config.get("thermal", {})
            self._thermal_validator = ThermalValidator(
                heat_diff_threshold_c=thermal_config.get("heat_diff_threshold_c", 2.0),
                background_update_rate=thermal_config.get("background_update_rate", 0.01),
                enabled=thermal_config.get("enabled", False),
            )

            # 3. 初始化跟踪器
            tracking_config = self._config.get("tracking", {})
            self._tracker = AnimalTracker(
                max_age=tracking_config.get("max_age", 30),
                min_hits=tracking_config.get("min_hits", 3),
                iou_threshold=tracking_config.get("iou_threshold", 0.3),
            )

            # 4. 初始化驱离控制器
            deterrent_config = self._config.get("deterrent", {})
            self._deterrent = DeterrentController(
                enabled=deterrent_config.get("enabled", False),
                stay_time_threshold_s=deterrent_config.get("stay_time_threshold_s", 30.0),
                cooldown_s=deterrent_config.get("cooldown_s", 60.0),
                audio_enabled=deterrent_config.get("audio_enabled", True),
                light_enabled=deterrent_config.get("light_enabled", True),
            )

            # 5. 初始化统计模块
            stats_config = self._config.get("statistics", {})
            self._statistics = StatisticsCollector(
                history_retention_days=stats_config.get("history_retention_days", 90),
                report_interval_hours=stats_config.get("report_interval_hours", 24),
            )

            # 6. 初始化审计日志
            log_dir = self._config.get("log_dir", str(plugin_dir / "logs"))
            self._audit_logger = AnimalAuditLogger(log_dir)

            # 7. 初始化证据管理
            evidence_dir = self._config.get("evidence_dir", str(plugin_dir / "evidence"))
            self._evidence_manager = EvidenceManager(evidence_dir)

            self._initialized = True
            self.status = PluginStatus.READY
            logger.info(
                f"动物检测插件初始化完成 | "
                f"模型: {model_path} | "
                f"模型可用: {'是' if self._model_ready else '否'} | "
                f"后端: {self._model_backend} | "
                f"模式: {self._model_mode} | "
                f"热成像: {'启用' if thermal_config.get('enabled') else '关闭'} | "
                f"跟踪: {'启用' if tracking_config.get('enabled', True) else '关闭'} | "
                f"驱离: {'启用' if deterrent_config.get('enabled') else '关闭'} | "
                f"仿真: {'是' if self._simulation_mode else '否'}"
            )
            return True

        except Exception as e:
            logger.error(f"插件初始化失败: {e}", exc_info=True)
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            self._error_count += 1
            return False

    def infer(
        self,
        frame: np.ndarray,
        rois: Optional[List[Any]] = None,
        context: Optional[Any] = None,
        thermal_frame: Optional[np.ndarray] = None,
        camera_id: str = "",
        location: str = "",
    ) -> AnimalEvent:
        """
        执行推理 - 完整流水线

        Args:
            frame: BGR图像帧
            rois: ROI列表 (可选)
            context: 插件上下文 (可选)
            thermal_frame: 热成像帧 (可选)
            camera_id: 摄像头ID
            location: 位置描述

        Returns:
            统一事件
        """
        if not self._initialized:
            return AnimalEvent(
                type=EventType.INTRUSION_CLEARED,
                location=location,
                value={"error": "plugin_not_initialized"},
            )

        start_time = time.time()
        self._frame_count += 1
        self.status = PluginStatus.RUNNING

        with self._lock:
            try:
                # === Step 1: 检测推理 (ONNX/兼容后端) ===
                detections = self._run_detection(frame)

                # === Step 2: 热成像校验 ===
                if detections and self._thermal_validator:
                    detections = self._thermal_validator.validate_detections(
                        detections, thermal_frame
                    )
                    # 如果热成像启用，过滤不通过的
                    if self._thermal_validator.enabled and thermal_frame is not None:
                        detections = self._thermal_validator.filter_non_thermal(detections)

                # === Step 3: 多目标跟踪 ===
                if self._tracker and detections:
                    detections = self._tracker.update(detections)
                elif self._tracker:
                    self._tracker.update([])  # 空帧也要更新tracker

                # === Step 4: 构建事件 ===
                event = build_intrusion_event(
                    detections=detections,
                    camera_id=camera_id,
                    location=location,
                    site_id=self._config.get("site_id", ""),
                    evidence=[],
                )
                backend_trace_id = self._last_inference_backend_status.get("trace_id")
                if backend_trace_id:
                    event.trace_id = backend_trace_id

                if detections and self._evidence_manager:
                    event.evidence = self._evidence_manager.save_detection_frame(
                        frame,
                        detections,
                        event.trace_id,
                    )

                # 更新证据trace_id
                for ev in event.evidence:
                    ev.description += f" [trace={event.trace_id[:8]}]"

                # === Step 5: 驱离评估 ===
                if self._deterrent and self._deterrent.enabled and self._tracker:
                    overstayed = self._tracker.get_overstayed_tracks(
                        self._deterrent.stay_time_threshold_s
                    )
                    if overstayed:
                        actions = self._deterrent.evaluate_and_trigger(overstayed)
                        if actions:
                            event.deterrent_active = True
                            event.deterrent_result = f"triggered_{len(actions)}_actions"

                    # 评估之前的驱离结果
                    all_tracks = self._tracker.get_all_tracks()
                    self._deterrent.evaluate_results(all_tracks)

                # === Step 6: 记录统计 ===
                if self._statistics:
                    for det in detections:
                        if det.track_id:
                            track = self._tracker.get_track(det.track_id) if self._tracker else None
                            was_deterred = track.deterrent_count > 0 if track else False
                            self._statistics.record_invasion(
                                detection=det,
                                camera_id=camera_id,
                                location=location,
                                was_deterred=was_deterred,
                            )

                # === Step 7: 审计日志 ===
                if self._audit_logger:
                    self._audit_logger.log_event(event)

                # 更新状态
                self._last_process_time = time.time() - start_time
                self._last_event = event
                self._inference_count += 1
                self._last_inference_time = datetime.now()
                self.status = PluginStatus.READY

                return event

            except Exception as e:
                logger.error(f"推理失败: {e}", exc_info=True)
                self._error_count += 1
                self.status = PluginStatus.ERROR
                return AnimalEvent(
                    type=EventType.INTRUSION_CLEARED,
                    location=location,
                    value={"error": str(e)},
                )

    def postprocess(
        self, event: AnimalEvent, rules: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        后处理 - 生成告警

        Args:
            event: 推理产出的事件
            rules: 告警规则列表

        Returns:
            告警列表
        """
        alarms = []

        if event.type != EventType.INTRUSION_DETECTED:
            return alarms

        for det in event.detections:
            # 高风险动物告警
            from .core.event_schema import ANIMAL_RISK_MAP, RiskLevel
            risk = ANIMAL_RISK_MAP.get(det.animal_class, RiskLevel.MEDIUM)

            if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                alarms.append({
                    "level": "critical" if risk == RiskLevel.CRITICAL else "error",
                    "title": f"室内动物入侵: {CLASS_NAMES_CN.get(det.animal_class, det.animal_class)}",
                    "message": (
                        f"在{event.location}检测到{CLASS_NAMES_CN.get(det.animal_class, '')}入侵 | "
                        f"置信度: {det.confidence:.2f} | "
                        f"停留: {det.stay_duration_s:.0f}秒 | "
                        f"建议: {event.suggestion}"
                    ),
                    "site_id": event.site_id,
                    "device_id": event.source_camera_id,
                    "trace_id": event.trace_id,
                    "detection": det.to_dict(),
                    "evidence": [e.to_dict() for e in event.evidence],
                })
            elif risk == RiskLevel.MEDIUM:
                alarms.append({
                    "level": "warning",
                    "title": f"室内动物入侵: {CLASS_NAMES_CN.get(det.animal_class, det.animal_class)}",
                    "message": (
                        f"在{event.location}检测到动物活动 | "
                        f"置信度: {det.confidence:.2f}"
                    ),
                    "site_id": event.site_id,
                    "device_id": event.source_camera_id,
                    "trace_id": event.trace_id,
                })

        return alarms

    def healthcheck(self) -> Dict[str, Any]:
        """健康检查"""
        status = "healthy"
        issues = []

        if not self._initialized:
            status = "error"
            issues.append("插件未初始化")

        if not self._simulation_mode and not self._model_ready:
            if status != "error":
                status = "degraded"
            issues.append("模型未就绪（真实推理不可用）")
            if self._last_error:
                issues.append(self._last_error)

        if self._error_count > 10:
            if status != "error":
                status = "degraded"
            issues.append(f"累计错误数: {self._error_count}")

        return {
            "status": status,
            "issues": issues,
            "plugin_id": self.id,
            "version": self.version,
            "simulation_mode": self._simulation_mode,
            "model_ready": self._model_ready,
            "model_backend": self._model_backend,
            "model_mode": self._model_mode,
            "model_path": self._model_path,
            "frame_count": self._frame_count,
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "last_process_time_ms": round(self._last_process_time * 1000, 2),
            "detector": self._detector.get_stats() if self._detector else None,
            "onnx_engine": self._onnx_engine.get_health_info() if self._onnx_engine else None,
            "thermal": self._thermal_validator.get_stats() if self._thermal_validator else None,
            "tracker": self._tracker.get_stats() if self._tracker else None,
            "deterrent": self._deterrent.get_stats() if self._deterrent else None,
            "statistics": self._statistics.get_stats() if self._statistics else None,
        }

    def cleanup(self):
        """清理资源"""
        if self._detector:
            self._detector.unload()
        if self._audit_logger:
            self._audit_logger.close()
        if self._tracker:
            self._tracker.reset()

        self._onnx_engine = None
        self._model_backend = "none"
        self._model_mode = "unknown"
        self._last_inference_backend_status = {}
        self._initialized = False
        self.status = PluginStatus.UNLOADED
        logger.info(f"[{self.id}] 插件已清理")

    # ==================== 扩展API ====================

    def get_statistics(self, period: str = "daily") -> Dict[str, Any]:
        """获取统计报告"""
        if not self._statistics:
            return {}
        if period == "weekly":
            return self._statistics.get_weekly_summary()
        elif period == "trend":
            return self._statistics.get_trend_data()
        return self._statistics.get_daily_summary()

    def export_report(self, format: str = "json",
                      start_time: float = None, end_time: float = None) -> str:
        """导出报告"""
        if not self._statistics:
            return ""
        if format == "csv":
            return self._statistics.export_csv(start_time, end_time)
        return self._statistics.export_json(start_time, end_time)

    def get_active_tracks(self) -> List[Dict]:
        """获取活跃跟踪"""
        if not self._tracker:
            return []
        return [
            {
                "track_id": t.track_id,
                "animal_class": t.animal_class,
                "confidence": t.confidence,
                "stay_duration_s": t.stay_duration_s,
                "state": t.state,
                "trajectory_length": len(t.trajectory),
                "deterrent_count": t.deterrent_count,
            }
            for t in self._tracker.get_active_tracks()
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
                {"id": "yolo_detection", "name": "YOLOv8动物检测", "default": True},
                {"id": "thermal_validation", "name": "热成像校验", "default": False},
                {"id": "tracking", "name": "多目标跟踪", "default": True},
                {"id": "deterrent", "name": "声光驱离", "default": False},
            ],
            "animal_classes": [
                {"id": cls.value, "name": CLASS_NAMES_CN.get(cls.value, cls.value)}
                for cls in AnimalClass
            ],
            "simulation_mode": self._simulation_mode,
            "version": self.version,
        }

    def detect(
        self,
        frame: np.ndarray,
        thermal_frame: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        兼容旧调用方：返回字典结构而非事件对象。
        """
        event = self.infer(
            frame=frame,
            context=context,
            thermal_frame=thermal_frame,
            camera_id=(context or {}).get("camera_id", ""),
            location=(context or {}).get("location", ""),
        )
        alarms = self.postprocess(event)
        return {
            "plugin_id": self.id,
            "plugin_version": self.version,
            "timestamp": datetime.now().isoformat(),
            "status": "alarm" if event.total_detections > 0 else "online",
            "detections": [d.to_dict() for d in event.detections],
            "alarmCount": len(alarms),
            "trace_id": event.trace_id,
            "suggestion": event.suggestion,
        }

    def get_detections(self) -> Dict[str, Any]:
        """
        室内中心 API 直接消费的数据接口。
        """
        if self._last_event is None:
            base_status = "online"
            if not self._model_ready and not self._simulation_mode:
                base_status = "degraded"
            return {
                "status": base_status,
                "detections": [],
                "animals": [],
                "totalCount": 0,
                "alarmCount": 0,
                "todayCount": self._statistics.get_today_count() if self._statistics else 0,
                "deterrentActive": False,
                "simulation_mode": self._simulation_mode,
                "model_info": self._build_model_info(),
            }

        detections: List[Dict[str, Any]] = []
        for idx, det in enumerate(self._last_event.detections):
            bbox = det.bbox.to_dict() if det.bbox else {"x": 0.5, "y": 0.5, "width": 0.0, "height": 0.0}
            # indoor_center.js 以中心点渲染 bbox
            center_bbox = {
                "x": max(0.0, min(1.0, bbox["x"] + bbox["width"] / 2)),
                "y": max(0.0, min(1.0, bbox["y"] + bbox["height"] / 2)),
                "width": bbox["width"],
                "height": bbox["height"],
            }
            detections.append(
                {
                    "id": det.track_id or det.detection_id or f"animal-{idx}",
                    "type": det.animal_class,
                    "confidence": round(det.confidence, 4),
                    "bbox": center_bbox,
                    "stay_duration_s": round(det.stay_duration_s, 1),
                    "track_id": det.track_id,
                    "is_thermal_validated": det.is_thermal_validated,
                }
            )

        alarm_count = len(
            [
                d
                for d in self._last_event.detections
                if d.animal_class in ("mouse", "snake", "rat", "weasel")
            ]
        )
        status = "alarm" if detections else "online"
        if not self._model_ready and not self._simulation_mode:
            status = "degraded"

        return {
            "timestamp": int(self._last_event.timestamp * 1000),
            "status": status,
            "detections": detections,
            "animals": detections,
            "totalCount": len(detections),
            "alarmCount": alarm_count,
            "todayCount": self._statistics.get_today_count() if self._statistics else 0,
            "deterrentActive": bool(self._last_event.deterrent_active),
            "trace_id": self._last_event.trace_id,
            "risk_level": self._last_event.risk_level,
            "suggestion": self._last_event.suggestion,
            "simulation_mode": self._simulation_mode,
            "model_info": self._build_model_info(),
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """室内中心控制面板能力描述。"""
        return {
            "name": self.name,
            "description": "室内动物入侵检测、热成像校验与声光驱离联动",
            "controls": [
                {"type": "slider", "id": "confidence_threshold", "label": "检测阈值", "min": 0, "max": 1, "step": 0.05, "default": self._config.get("inference", {}).get("confidence_threshold", 0.55)},
                {"type": "switch", "id": "thermal_enabled", "label": "热成像校验", "default": self._config.get("thermal", {}).get("enabled", False)},
                {"type": "switch", "id": "deterrent_enabled", "label": "自动驱离", "default": self._config.get("deterrent", {}).get("enabled", False)},
            ],
            "operations": [
                {"id": "activate_deterrent", "label": "启动驱离", "icon": "volume-up"},
                {"id": "view_history", "label": "查看历史", "icon": "clock-history"},
            ],
        }

    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """室内中心命令执行接口。"""
        operation = command.get("operation") or command.get("command")
        params = command.get("params") or {}

        if operation == "activate_deterrent":
            if self._deterrent is None:
                return {"success": False, "message": "驱离控制器不可用"}
            self._deterrent.enabled = True
            return {"success": True, "message": "已启用驱离控制", "enabled": True}

        if operation == "view_history":
            limit = int(params.get("limit", 50))
            return {"success": True, "events": self.get_recent_events(limit), "count": len(self.get_recent_events(limit))}

        if operation == "set_threshold":
            val = params.get("confidence_threshold")
            if val is None:
                return {"success": False, "message": "缺少 confidence_threshold"}
            try:
                val = float(val)
            except Exception:
                return {"success": False, "message": "confidence_threshold 非法"}
            self._config.setdefault("inference", {})["confidence_threshold"] = max(0.0, min(1.0, val))
            if self._detector:
                self._detector.confidence_threshold = self._config["inference"]["confidence_threshold"]
            if self._onnx_engine:
                self._onnx_engine.conf_threshold = self._config["inference"]["confidence_threshold"]
            return {"success": True, "confidence_threshold": self._config["inference"]["confidence_threshold"]}

        return {"success": False, "message": f"不支持的操作: {operation}"}

    # ==================== 内部方法 ====================

    def _initialize_model_backend(
        self,
        model_path: str,
        model_config: Dict[str, Any],
        inference_config: Dict[str, Any],
    ) -> None:
        """
        初始化模型推理后端:
        1) 优先使用 AnimalONNXEngine (新引擎)
        2) 失败时回退到 YOLOv8Detector (兼容旧逻辑)
        """
        self._model_ready = False
        self._model_backend = "none"
        self._model_mode = "unknown"
        self._detector = None
        self._onnx_engine = None

        conf_threshold = float(inference_config.get("confidence_threshold", 0.55))
        nms_threshold = float(inference_config.get("nms_threshold", 0.4))
        inference_device = self._config.get("inference_device", model_config.get("device", "cpu"))

        class_mapping_path = Path(model_path).with_name("class_mapping.json")
        mapping_arg = str(class_mapping_path) if class_mapping_path.is_file() else None

        self._onnx_engine = AnimalONNXEngine(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            class_mapping_path=mapping_arg,
            device=inference_device,
        )
        if self._onnx_engine.load():
            self._model_ready = True
            self._model_backend = "onnx_engine"
            self._model_mode = self._onnx_engine.model_mode or "unknown"
            logger.info(
                "ONNX 推理引擎加载成功 | model=%s | mode=%s",
                model_path,
                self._model_mode,
            )
            return

        logger.warning("ONNX 推理引擎加载失败, 尝试兼容后端 YOLOv8Detector")
        self._detector = YOLOv8Detector(
            model_path=model_path,
            device=inference_device,
            input_size=tuple(model_config.get("input_size", [640, 640])),
            confidence_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_detections=inference_config.get("max_detections", 50),
            multi_scale=inference_config.get("multi_scale", False),
        )
        if self._detector.load():
            self._model_ready = True
            self._model_backend = "legacy_detector"
            self._model_mode = "legacy"
            logger.warning("兼容后端加载成功: YOLOv8Detector")
            return

        self._model_backend = "none"
        self._model_mode = "unavailable"

    def _run_detection(self, frame: np.ndarray) -> List[AnimalDetectionResult]:
        """
        执行检测并统一为 AnimalDetectionResult 列表.
        """
        self._last_inference_backend_status = {}
        if self._simulation_mode:
            return []
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []
        if not self._model_ready:
            return []

        if self._model_backend == "onnx_engine" and self._onnx_engine:
            result = self._onnx_engine.infer(frame)
            self._last_inference_backend_status = {
                "backend": "onnx_engine",
                "status": result.status,
                "trace_id": result.trace_id,
                "model_name": result.model_name,
                "model_mode": result.model_mode,
                "inference_ms": round(result.inference_ms, 2),
                "error_message": result.error_message,
            }
            if result.status != "ok":
                return []
            detections: List[AnimalDetectionResult] = []
            for det in result.detections:
                converted = self._convert_onnx_detection(det)
                if converted is not None:
                    detections.append(converted)
            return detections

        if self._detector:
            detections = self._detector.detect(frame)
            self._last_inference_backend_status = {
                "backend": "legacy_detector",
                "status": "ok",
                "trace_id": "",
                "model_name": os.path.basename(self._model_path),
                "model_mode": "legacy",
                "inference_ms": 0.0,
                "error_message": "",
            }
            return detections

        return []

    def _convert_onnx_detection(self, detection: Any) -> Optional[AnimalDetectionResult]:
        """将 ONNX 引擎 Detection 转为平台事件检测结构."""
        if not hasattr(detection, "bbox_normalized"):
            return None
        x1, y1, x2, y2 = detection.bbox_normalized
        x1 = max(0.0, min(1.0, float(x1)))
        y1 = max(0.0, min(1.0, float(y1)))
        x2 = max(0.0, min(1.0, float(x2)))
        y2 = max(0.0, min(1.0, float(y2)))
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        return AnimalDetectionResult(
            animal_class=self._map_external_class(getattr(detection, "class_name", "")),
            confidence=float(getattr(detection, "confidence", 0.0)),
            bbox=BoundingBox(x=x1, y=y1, width=width, height=height),
        )

    @staticmethod
    def _map_external_class(class_name: str) -> str:
        """
        统一类别命名:
        - 新引擎 7 类: rat/cat/snake/bird/dog/poultry/other_wildlife
        - 平台事件枚举: mouse/cat/snake/bird/dog/poultry/insect/other
        """
        name = (class_name or "").strip().lower()
        mapping = {
            "rat": AnimalClass.MOUSE.value,
            "mouse": AnimalClass.MOUSE.value,
            "cat": AnimalClass.CAT.value,
            "snake": AnimalClass.SNAKE.value,
            "bird": AnimalClass.BIRD.value,
            "dog": AnimalClass.DOG.value,
            "poultry": AnimalClass.POULTRY.value,
            "insect": AnimalClass.INSECT.value,
            "other_wildlife": AnimalClass.OTHER.value,
            "other": AnimalClass.OTHER.value,
        }
        return mapping.get(name, AnimalClass.OTHER.value)

    def _resolve_model_path(self, model_path: str) -> str:
        """
        将模型路径标准化为绝对路径.
        相对路径默认按插件目录解析.
        """
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            project_path = (_project_root / path)
            plugin_path = (self.plugin_dir / path)
            if project_path.exists() and not plugin_path.exists():
                path = project_path
            else:
                path = plugin_path
        return str(path)

    def _build_model_info(self) -> Dict[str, Any]:
        """构建前端可消费的模型状态描述."""
        info: Dict[str, Any] = {
            "loaded": self._model_ready,
            "backend": self._model_backend,
            "model_mode": self._model_mode,
            "model_path": self._model_path,
        }
        if self._onnx_engine:
            info["onnx"] = self._onnx_engine.get_health_info()
        if self._last_inference_backend_status:
            info["last_inference"] = self._last_inference_backend_status
        if not self._model_ready and self._last_error:
            info["reason"] = self._last_error
        return info

    def _load_default_config(self) -> Dict[str, Any]:
        """优先加载 default.yaml，其次 default.json。"""
        config_dir = self.plugin_dir / "configs"

        yaml_path = config_dir / "default.yaml"
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"default.yaml 读取失败: {e}")

        json_path = config_dir / "default.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception as e:
                logger.warning(f"default.json 读取失败: {e}")

        return {}

    def _normalize_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        兼容两套配置结构:
        1) 旧版: detection / thermal_fusion
        2) 新版: inference / thermal / tracking
        """
        out = dict(cfg or {})
        old_detection = out.get("detection", {})
        old_thermal = out.get("thermal_fusion", {})

        inference = dict(out.get("inference", {}))
        inference.setdefault("confidence_threshold", old_detection.get("confidence_threshold", 0.55))
        inference.setdefault("nms_threshold", old_detection.get("nms_threshold", 0.4))
        inference.setdefault("max_detections", old_detection.get("max_detections", 100))
        inference.setdefault("multi_scale", old_detection.get("multi_scale", False))

        thermal = dict(out.get("thermal", {}))
        thermal.setdefault("enabled", old_thermal.get("enabled", False))
        thermal.setdefault("heat_diff_threshold_c", old_thermal.get("alive_temp_diff", 2.0))
        thermal.setdefault("background_update_rate", old_thermal.get("background_update_rate", 0.01))

        tracking = dict(out.get("tracking", {}))
        tracking.setdefault("enabled", old_detection.get("tracking_enabled", True))
        tracking.setdefault("max_age", old_detection.get("max_track_age", 30))
        tracking.setdefault("min_hits", old_detection.get("min_confirm_frames", 3))
        tracking.setdefault("iou_threshold", old_detection.get("iou_threshold", 0.3))

        deterrent = dict(out.get("deterrent", {}))
        deterrent.setdefault("enabled", deterrent.get("enabled", False))
        deterrent.setdefault("stay_time_threshold_s", deterrent.get("stay_time_threshold_s", 2.0))
        deterrent.setdefault("cooldown_s", deterrent.get("cooldown_s", deterrent.get("cooldown_seconds", 60.0)))
        deterrent.setdefault("audio_enabled", deterrent.get("audio_enabled", deterrent.get("ultrasonic_enabled", True)))
        deterrent.setdefault("light_enabled", deterrent.get("light_enabled", deterrent.get("strobe_enabled", True)))

        simulation = dict(out.get("simulation", {}))
        simulation.setdefault("enabled", False)

        out["inference"] = inference
        out["thermal"] = thermal
        out["tracking"] = tracking
        out["deterrent"] = deterrent
        out["simulation"] = simulation
        out.setdefault("inference_device", out.get("model", {}).get("device", "cpu"))
        out.setdefault("statistics", {"history_retention_days": 90, "report_interval_hours": 24})
        return out

    def _merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base or {})
        for k, v in (override or {}).items():
            if isinstance(result.get(k), dict) and isinstance(v, dict):
                result[k] = self._merge_dict(result[k], v)
            else:
                result[k] = v
        return result

    def _compute_code_hash(self) -> str:
        """计算插件代码哈希"""
        try:
            code_file = Path(__file__)
            return hashlib.md5(code_file.read_bytes()).hexdigest()[:12]
        except Exception:
            return "unknown"
