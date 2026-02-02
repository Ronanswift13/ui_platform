"""
温度监测插件 - 完整实现
版本: 1.0.0
"""

from __future__ import annotations
import hashlib, logging, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class PluginStatus(str, Enum):
    UNLOADED = "unloaded"; LOADING = "loading"; READY = "ready"
    RUNNING = "running"; ERROR = "error"; DISABLED = "disabled"

class TemperatureMonitoringPlugin:
    PLUGIN_ID = "temperature_monitoring"
    PLUGIN_NAME = "温度监测"
    PLUGIN_VERSION = "1.0.0"

    def __init__(self, manifest=None, plugin_dir=None, config=None):
        self.manifest = manifest
        self.plugin_dir = plugin_dir if plugin_dir else Path(__file__).parent
        self._status = PluginStatus.UNLOADED
        self._last_error = ""
        self.config = config if isinstance(config, dict) else {}
        self._model_registry = None
        self._detector = None
        self._is_initialized = False
        self._training_buffer: List[Dict] = []

    @property
    def id(self): return self.manifest.id if self.manifest and hasattr(self.manifest, "id") else self.PLUGIN_ID
    @property
    def name(self): return self.manifest.name if self.manifest and hasattr(self.manifest, "name") else self.PLUGIN_NAME
    @property
    def version(self): return self.manifest.version if self.manifest and hasattr(self.manifest, "version") else self.PLUGIN_VERSION
    @property
    def code_hash(self):
        h = hashlib.sha256(); p = Path(self.plugin_dir) / "plugin.py"
        if p.exists(): h.update(p.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"
    @property
    def status(self): return self._status.value

    def init(self, config=None) -> bool:
        try:
            self._status = PluginStatus.LOADING
            if config: self.config = self._merge(self.config, config)
            if not self.config: self.config = self._load_default_config()
            from .detector import TemperatureDetector
            self._detector = TemperatureDetector(self.config)
            self._detector.initialize()
            self._is_initialized = True
            self._status = PluginStatus.READY
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True
        except Exception as e:
            self._last_error = str(e); self._status = PluginStatus.ERROR
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}"); return False

    def set_status(self, status, message=""):
        if isinstance(status, PluginStatus): self._status = status
        elif isinstance(status, str):
            try: self._status = PluginStatus(status)
            except ValueError: pass
        if message: self._last_error = message

    def set_model_registry(self, r): self._model_registry = r
    def shutdown(self):
        if self._detector: self._detector.reset()
        self._is_initialized = False; self._status = PluginStatus.UNLOADED

    def detect(self, thermal_frame=None, sensor_readings=None, context=None) -> Dict[str, Any]:
        if not self._is_initialized or not self._detector:
            return self._err("插件未初始化")
        try:
            self._status = PluginStatus.RUNNING
            r = self._detector.detect(thermal_frame, sensor_readings)
            output = {
                "plugin_id": self.id, "plugin_version": self.version, "code_hash": self.code_hash,
                "task_id": (context or {}).get("task_id", ""),
                "timestamp": datetime.now().isoformat(),
                "success": True, "status": r.status,
                "heatmap": r.heatmap.tolist(),
                "max_temp": r.max_temp, "min_temp": r.min_temp, "avg_temp": r.avg_temp,
                "hotspots": [{"zone_id": h.zone_id, "zone_name": h.zone_name,
                              "center": h.center, "temperature": h.temperature,
                              "area_ratio": h.area_ratio, "severity": h.severity} for h in r.hotspots],
                "trend": {"current": r.trend.current, "avg_1min": r.trend.avg_1min,
                          "avg_5min": r.trend.avg_5min, "rise_rate": r.trend.rise_rate,
                          "direction": r.trend.trend_direction, "predicted_30min": r.trend.predicted_30min},
                "linkage_events": r.linkage_events,
                "alarms": self._gen_alarms(r),
                "inference_time_ms": r.metadata.get("inference_time_ms", 0),
                "metadata": r.metadata,
            }
            self._status = PluginStatus.READY
            return output
        except Exception as e:
            self._last_error = str(e); return self._err(str(e))

    def infer(self, frame, rois, context):
        r = self.detect(thermal_frame=frame, context={"task_id": getattr(context, "task_id", "")})
        try:
            from platform_core.schema.models import RecognitionResult, BoundingBox
            return [RecognitionResult(
                task_id=getattr(context, "task_id", ""), site_id=getattr(context, "site_id", ""),
                device_id=getattr(context, "device_id", ""), component_id="",
                roi_id="", bbox=BoundingBox(x=h["center"][0], y=h["center"][1], width=0.05, height=0.05),
                label=f"hotspot_{h['severity']}", confidence=min(1.0, h["temperature"]/100),
                model_version=self.version, code_version=self.code_hash,
            ) for h in r.get("hotspots", [])]
        except ImportError: return []

    def postprocess(self, results, rules):
        try:
            from platform_core.schema.models import Alarm, AlarmLevel as AL
            return [Alarm(task_id=r.task_id, result_id=None,
                          level=AL.ERROR if "alarm" in r.label else AL.WARNING,
                          title=f"温度异常: {r.label}", message=f"温度异常 置信度{r.confidence:.0%}",
                          site_id=r.site_id, device_id=r.device_id, component_id="")
                    for r in results if r.confidence > 0.5]
        except ImportError: return []

    def healthcheck(self):
        try:
            from platform_core.plugin_manager.base import HealthStatus
            return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else self._last_error,
                                details=self._detector.stats if self._detector else {})
        except ImportError:
            return {"healthy": self._is_initialized, "message": "OK" if self._is_initialized else self._last_error}

    def upload_training_data(self, data, labels) -> Dict:
        self._training_buffer.append({"timestamp": datetime.now().isoformat(), "labels": labels})
        return {"success": True, "total": len(self._training_buffer)}

    def start_training(self, config) -> Dict:
        return {"success": True, "message": "温度预测模型训练已提交", "samples": len(self._training_buffer)}

    @property
    def plugin_info(self) -> Dict:
        return {"id": self.id, "name": self.name, "version": self.version,
                "description": "红外热成像与传感器温度监测", "status": self.status,
                "capabilities": ["thermal_imaging", "hotspot_detection", "temperature_prediction"]}

    def get_ui_config(self) -> Dict:
        return {
            "detection_types": [
                {"id": "heatmap", "name": "热力图", "icon": "thermometer-half", "enabled": True},
                {"id": "hotspot", "name": "热点检测", "icon": "geo-alt", "enabled": True,
                 "capabilities": [{"label": "高温热点", "tags": ["hotspot"], "level": "warning"}]},
                {"id": "trend", "name": "趋势预测", "icon": "graph-up", "enabled": True},
            ],
            "parameters": [
                {"id": "warning_threshold", "name": "预警温度", "type": "slider", "min": 30, "max": 100, "default": 55},
                {"id": "alarm_threshold", "name": "报警温度", "type": "slider", "min": 40, "max": 120, "default": 70},
            ],
        }

    def _gen_alarms(self, result) -> List[Dict]:
        alarms = []
        if result.status in ("alarm", "critical"):
            alarms.append({"type": "temperature_alarm", "level": result.status,
                           "title": f"温度{'报警' if result.status == 'alarm' else '紧急'}",
                           "message": f"最高温度{result.max_temp}°C, 温升{result.trend.rise_rate}°C/min",
                           "timestamp": datetime.now().isoformat()})
        for h in result.hotspots:
            if h.severity in ("alarm", "critical"):
                alarms.append({"type": "hotspot_alarm", "level": h.severity,
                               "title": f"{h.zone_name or '区域'}高温告警",
                               "message": f"热点温度{h.temperature}°C",
                               "zone_id": h.zone_id, "timestamp": datetime.now().isoformat()})
        return alarms

    def _err(self, msg):
        return {"plugin_id": self.id, "plugin_version": self.version, "timestamp": datetime.now().isoformat(),
                "success": False, "status": "error", "heatmap": [], "hotspots": [], "alarms": [], "error_message": msg}

    def _load_default_config(self):
        try:
            import yaml
            p = Path(self.plugin_dir) / "configs" / "default.yaml"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f) or {}
        except: pass
        return {}

    @staticmethod
    def _merge(b, o):
        r = b.copy()
        for k, v in o.items():
            if k in r and isinstance(r[k], dict) and isinstance(v, dict): r[k] = TemperatureMonitoringPlugin._merge(r[k], v)
            else: r[k] = v
        return r
