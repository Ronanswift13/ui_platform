"""
热成像分析插件 - 完整实现
版本: 1.0.0

基于规则的热成像异常检测，包装 EnhancedThermalAnalyzer。
algorithm_stage: baseline, 无真实深度学习模型。
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from darkbreaker_sdk.interfaces import HealthStatus
from darkbreaker_sdk.schemas import Alarm, AlarmLevel, BoundingBox, RecognitionResult
from platform_core.visual_output_protocol import build_visual_meta

logger = logging.getLogger(__name__)


class PluginStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class ThermalPlugin:
    PLUGIN_ID = "thermal"
    PLUGIN_NAME = "热成像分析"
    PLUGIN_VERSION = "1.0.0"

    # ------------------------------------------------------------------ init
    def __init__(self, manifest=None, plugin_dir=None, config=None):
        self.manifest = manifest
        self.plugin_dir = plugin_dir if plugin_dir else Path(__file__).parent
        self._status = PluginStatus.UNLOADED
        self._last_error = ""
        self.config: dict = config if isinstance(config, dict) else {}
        self._analyzer = None
        self._is_initialized = False

    @classmethod
    def create_standalone(cls, config=None):
        """Create plugin instance for standalone operation."""
        plugin_dir = Path(__file__).resolve().parent
        instance = cls()
        if config is None:
            from darkbreaker_sdk.utils import load_plugin_config

            config = load_plugin_config(plugin_dir / "configs" / "default.yaml")
        instance.init(config)
        return instance

    # -------------------------------------------------------------- properties
    @property
    def id(self) -> str:
        if self.manifest and hasattr(self.manifest, "id"):
            return self.manifest.id
        return self.PLUGIN_ID

    @property
    def name(self) -> str:
        if self.manifest and hasattr(self.manifest, "name"):
            return self.manifest.name
        return self.PLUGIN_NAME

    @property
    def version(self) -> str:
        if self.manifest and hasattr(self.manifest, "version"):
            return self.manifest.version
        return self.PLUGIN_VERSION

    @property
    def code_hash(self) -> str:
        h = hashlib.sha256()
        p = Path(self.plugin_dir) / "plugin.py"
        if p.exists():
            h.update(p.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"

    @property
    def status(self) -> str:
        return self._status.value

    @property
    def plugin_info(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": "热成像异常检测 - 热点识别、温度梯度分析、趋势预测",
            "status": self.status,
            "capabilities": [
                "thermal_analysis",
                "hotspot_detection",
                "temperature_trend_analysis",
            ],
        }

    # -------------------------------------------------------------- lifecycle
    def init(self, config=None) -> bool:
        try:
            self._status = PluginStatus.LOADING
            if config:
                self.config = self._merge(self.config, config)
            if not self.config:
                self.config = self._load_default_config()

            from .enhanced_thermal_analyzer import EnhancedThermalAnalyzer, ThermalConfig

            thresholds = self.config.get("thresholds", {})
            trend_cfg = self.config.get("trend", {})

            tc = ThermalConfig(
                use_deep_learning=False,
                enable_fusion=False,
                temp_warning=thresholds.get("temp_warning", 60.0),
                temp_serious=thresholds.get("temp_serious", 80.0),
                temp_critical=thresholds.get("temp_critical", 100.0),
                gradient_threshold=thresholds.get("gradient_threshold", 15.0),
                history_length=trend_cfg.get("history_length", 100),
                trend_window=trend_cfg.get("trend_window", 10),
                trend_threshold=trend_cfg.get("trend_threshold", 5.0),
            )
            self._analyzer = EnhancedThermalAnalyzer(tc)

            self._is_initialized = True
            self._status = PluginStatus.READY
            logger.info("[%s] 初始化成功 (rule-based)", self.PLUGIN_NAME)
            return True
        except Exception as e:
            self._last_error = str(e)
            self._status = PluginStatus.ERROR
            logger.error("[%s] 初始化失败: %s", self.PLUGIN_NAME, e)
            return False

    def set_status(self, status, message=""):
        if isinstance(status, PluginStatus):
            self._status = status
        elif isinstance(status, str):
            try:
                self._status = PluginStatus(status)
            except ValueError:
                pass
        if message:
            self._last_error = message

    def shutdown(self):
        self._analyzer = None
        self._is_initialized = False
        self._status = PluginStatus.UNLOADED

    # -------------------------------------------------------------- inference
    def infer(self, frame, rois, context) -> list:
        """Run thermal analysis on a single frame.

        Returns list[RecognitionResult].
        """
        if frame is None:
            return []

        task_id = getattr(context, "task_id", "")
        site_id = getattr(context, "site_id", "")
        device_id = getattr(context, "device_id", "")

        # ----- quality gate -----
        quality_gate_status = "pass"
        quality_cfg = self.config.get("quality", {})
        min_res = quality_cfg.get("min_resolution", [64, 64])
        valid_range = quality_cfg.get("valid_temp_range", [-20.0, 200.0])

        if frame.ndim < 2 or frame.shape[0] < min_res[0] or frame.shape[1] < min_res[1]:
            quality_gate_status = "hard_fail"

        frame_min = float(np.min(frame))
        frame_max = float(np.max(frame))
        if frame_min < valid_range[0] or frame_max > valid_range[1]:
            quality_gate_status = "soft_fail"

        # ----- analyze -----
        result = self._analyzer.analyze(frame)

        base_meta = build_visual_meta(
            plugin_name="thermal",
            task_type="thermal_analysis",
            modality="thermal",
            runtime_mode="rule_based",
            algorithm_stage="baseline",
            quality_gate_status=quality_gate_status,
            evidence_hint="thermal_heatmap",
            extra={
                "global_max_temp": result.global_max,
                "global_min_temp": result.global_min,
                "global_avg_temp": result.global_avg,
                "overall_health": result.overall_health,
            },
        )

        results: List[RecognitionResult] = []

        h, w = frame.shape[:2]

        for anomaly in result.anomalies:
            # 映射标签
            atype = anomaly.anomaly_type.value  # hotspot, gradient, ...
            label_map = {
                "hotspot": "hotspot",
                "gradient": "gradient_anomaly",
                "trend": "trend_anomaly",
                "distribution": "gradient_anomaly",
                "contact": "overheating",
                "overload": "overheating",
            }
            label = label_map.get(atype, "hotspot")

            # 归一化 bbox
            x1, y1, x2, y2 = anomaly.region.bbox
            bx = max(0.0, min(1.0, x1 / w))
            by = max(0.0, min(1.0, y1 / h))
            bw = max(0.0, min(1.0, (x2 - x1) / w))
            bh = max(0.0, min(1.0, (y2 - y1) / h))

            meta = {
                **base_meta,
                "anomaly_type": atype,
                "max_temp": anomaly.max_temp,
                "min_temp": anomaly.min_temp,
                "avg_temp": anomaly.avg_temp,
                "temp_gradient": anomaly.temp_gradient,
                "severity": anomaly.severity.name,
            }

            results.append(
                RecognitionResult(
                    task_id=task_id,
                    site_id=site_id,
                    device_id=device_id,
                    component_id="",
                    roi_id="",
                    bbox=BoundingBox(x=bx, y=by, width=bw, height=bh),
                    label=label,
                    confidence=anomaly.confidence,
                    model_version=self.version,
                    code_version=self.code_hash,
                    metadata=meta,
                )
            )

        # 无异常时返回 normal
        if not results:
            results.append(
                RecognitionResult(
                    task_id=task_id,
                    site_id=site_id,
                    device_id=device_id,
                    component_id="",
                    roi_id="",
                    bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
                    label="normal",
                    confidence=0.95,
                    model_version=self.version,
                    code_version=self.code_hash,
                    metadata=base_meta,
                )
            )

        return results

    # -------------------------------------------------------------- postprocess
    def postprocess(self, results, rules) -> list:
        """Generate alarms from recognition results."""
        alarms: List[Alarm] = []
        alarm_labels = {"hotspot", "overheating", "gradient_anomaly", "trend_anomaly"}

        for r in results:
            if r.label in alarm_labels and r.confidence > 0.5:
                if r.label == "overheating":
                    level = AlarmLevel.ERROR
                else:
                    level = AlarmLevel.WARNING

                alarms.append(
                    Alarm(
                        task_id=r.task_id,
                        result_id=None,
                        level=level,
                        title=f"热成像异常: {r.label}",
                        message=(
                            f"检测到{r.label}异常, 置信度{r.confidence:.0%}"
                        ),
                        site_id=r.site_id,
                        device_id=r.device_id,
                        component_id="",
                    )
                )

        return alarms

    # -------------------------------------------------------------- health
    def healthcheck(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._is_initialized,
            message="OK" if self._is_initialized else (self._last_error or "未初始化"),
            details={
                "plugin_id": self.PLUGIN_ID,
                "version": self.PLUGIN_VERSION,
                "runtime_mode": "rule_based",
                "model_status": "placeholder",
                "analyzer_ready": self._analyzer is not None,
            },
        )

    # -------------------------------------------------------------- ui
    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "detection_types": [
                {
                    "id": "heatmap",
                    "name": "热力图",
                    "icon": "thermometer-half",
                    "enabled": True,
                },
                {
                    "id": "hotspot",
                    "name": "热点检测",
                    "icon": "geo-alt",
                    "enabled": True,
                    "capabilities": [
                        {"label": "高温热点", "tags": ["hotspot"], "level": "warning"},
                    ],
                },
                {
                    "id": "trend",
                    "name": "趋势预测",
                    "icon": "graph-up",
                    "enabled": True,
                },
            ],
            "parameters": [
                {
                    "id": "temp_warning",
                    "name": "预警温度",
                    "type": "slider",
                    "min": 30,
                    "max": 100,
                    "default": 60,
                },
                {
                    "id": "temp_critical",
                    "name": "报警温度",
                    "type": "slider",
                    "min": 40,
                    "max": 150,
                    "default": 100,
                },
            ],
        }

    # -------------------------------------------------------------- helpers
    def _load_default_config(self) -> dict:
        try:
            import yaml

            p = Path(self.plugin_dir) / "configs" / "default.yaml"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}

    @staticmethod
    def _merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = ThermalPlugin._merge(result[k], v)
            else:
                result[k] = v
        return result


Plugin = ThermalPlugin
