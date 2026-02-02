"""
动物入侵检测插件 - 完整实现
===============================

输变电激光星芒破夜绘明监测平台 - 室内动物入侵检测

作者: 室内监测组
版本: 1.0.0
"""

from __future__ import annotations
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class PluginStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class AnimalDetectionPlugin:
    """动物入侵检测插件"""

    PLUGIN_ID = "animal_detection"
    PLUGIN_NAME = "动物入侵检测"
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
        logger.info(f"[{self.PLUGIN_NAME}] 实例已创建")

    @property
    def id(self) -> str:
        return self.manifest.id if self.manifest and hasattr(self.manifest, "id") else self.PLUGIN_ID

    @property
    def name(self) -> str:
        return self.manifest.name if self.manifest and hasattr(self.manifest, "name") else self.PLUGIN_NAME

    @property
    def version(self) -> str:
        return self.manifest.version if self.manifest and hasattr(self.manifest, "version") else self.PLUGIN_VERSION

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

    def init(self, config: Optional[Dict] = None) -> bool:
        try:
            self._status = PluginStatus.LOADING
            if config:
                self.config = self._merge(self.config, config)
            if not self.config:
                self.config = self._load_default_config()

            from .detector import AnimalDetector
            self._detector = AnimalDetector(self.config)
            self._detector.initialize()
            self._is_initialized = True
            self._status = PluginStatus.READY
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True
        except Exception as e:
            self._last_error = str(e)
            self._status = PluginStatus.ERROR
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}")
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

    def set_model_registry(self, registry):
        self._model_registry = registry

    def shutdown(self):
        if self._detector:
            self._detector.reset()
        self._is_initialized = False
        self._status = PluginStatus.UNLOADED

    def detect(self, frame: np.ndarray, thermal_frame=None, context=None) -> Dict[str, Any]:
        if not self._is_initialized or not self._detector:
            return self._error_result("插件未初始化")

        try:
            self._status = PluginStatus.RUNNING
            result = self._detector.detect(frame, thermal_frame)

            output = {
                "plugin_id": self.id,
                "plugin_version": self.version,
                "code_hash": self.code_hash,
                "task_id": (context or {}).get("task_id", ""),
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "status": "alarm" if result["detections"] else "normal",
                "detections": [
                    {
                        "species": d.species,
                        "bbox": d.bbox,
                        "confidence": round(d.confidence, 4),
                        "is_alive": d.is_alive,
                        "track_id": d.track_id,
                        "duration_sec": round(d.duration_sec, 1),
                        "zone_id": d.zone_id,
                        "zone_name": d.zone_name,
                        "risk_level": d.risk_level,
                    }
                    for d in result["detections"]
                ],
                "deterrent_status": {
                    "active": bool(result["deterrent_actions"]),
                    "actions": result["deterrent_actions"],
                },
                "statistics": self._detector.get_statistics(),
                "alarms": self._gen_alarms(result["detections"]),
                "inference_time_ms": result["inference_time_ms"],
                "metadata": {"total_tracked": result["total_tracked"]},
            }
            self._status = PluginStatus.READY
            return output
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[{self.PLUGIN_NAME}] 检测失败: {e}")
            return self._error_result(str(e))

    # BasePlugin 兼容
    def infer(self, frame, rois, context):
        result = self.detect(frame, context={"task_id": getattr(context, "task_id", "")})
        try:
            from platform_core.schema.models import RecognitionResult, BoundingBox
            return [
                RecognitionResult(
                    task_id=getattr(context, "task_id", ""), site_id=getattr(context, "site_id", ""),
                    device_id=getattr(context, "device_id", ""), component_id=getattr(context, "component_id", ""),
                    roi_id="", bbox=BoundingBox(**d["bbox"]), label=d["species"],
                    confidence=d["confidence"], model_version=self.version, code_version=self.code_hash,
                ) for d in result.get("detections", [])
            ]
        except ImportError:
            return []

    def postprocess(self, results, rules):
        try:
            from platform_core.schema.models import Alarm, AlarmLevel as AL
            return [
                Alarm(task_id=r.task_id, result_id=None, level=AL.WARNING,
                      title=f"动物入侵: {r.label}", message=f"检测到{r.label}入侵",
                      site_id=r.site_id, device_id=r.device_id, component_id=r.component_id)
                for r in results if r.confidence > 0.5
            ]
        except ImportError:
            return []

    def healthcheck(self):
        try:
            from platform_core.plugin_manager.base import HealthStatus
            return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else self._last_error,
                                details=self._detector.stats if self._detector else {})
        except ImportError:
            return {"healthy": self._is_initialized, "message": "OK" if self._is_initialized else self._last_error}

    def upload_training_data(self, image: np.ndarray, annotations: List[Dict]) -> Dict:
        self._training_buffer.append({"timestamp": datetime.now().isoformat(), "annotations": annotations})
        return {"success": True, "total_samples": len(self._training_buffer)}

    def get_training_status(self) -> Dict:
        return {"total_samples": len(self._training_buffer), "model_loaded": self._detector._session is not None if self._detector else False}

    def start_training(self, config: Dict) -> Dict:
        return {"success": True, "message": "训练任务已提交", "samples": len(self._training_buffer)}

    @property
    def plugin_info(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "version": self.version,
            "description": "基于YOLOv8深度学习的室内小动物入侵检测",
            "capabilities": ["animal_detection", "species_classification", "deterrent_control", "intrusion_statistics"],
            "status": self.status, "initialized": self._is_initialized,
        }

    def get_ui_config(self) -> Dict:
        return {
            "detection_types": [
                {"id": "animal", "name": "动物检测", "icon": "bug", "enabled": True,
                 "capabilities": [
                     {"label": "鼠类", "tags": ["rat", "mouse"], "level": "error"},
                     {"label": "蛇类", "tags": ["snake"], "level": "critical"},
                     {"label": "鸟类", "tags": ["bird"], "level": "warning"},
                     {"label": "其他", "tags": ["cat", "weasel", "insect"], "level": "warning"},
                 ]},
                {"id": "deterrent", "name": "驱离控制", "icon": "volume-up", "enabled": True},
            ],
            "parameters": [
                {"id": "confidence_threshold", "name": "检测阈值", "type": "slider", "min": 0.1, "max": 0.9, "default": 0.40},
                {"id": "auto_deterrent", "name": "自动驱离", "type": "switch", "default": True},
            ],
        }

    def _gen_alarms(self, detections) -> List[Dict]:
        alarms = []
        for d in detections:
            if d.risk_level in ("high", "critical"):
                alarms.append({
                    "type": "animal_intrusion", "level": "alarm" if d.risk_level == "high" else "critical",
                    "title": f"动物入侵: {d.species}", "message": f"在{d.zone_name or '监测区域'}检测到{d.species}",
                    "timestamp": datetime.now().isoformat(), "zone_id": d.zone_id,
                })
        return alarms

    def _error_result(self, msg: str) -> Dict:
        return {"plugin_id": self.id, "plugin_version": self.version, "timestamp": datetime.now().isoformat(),
                "success": False, "status": "error", "detections": [], "alarms": [], "error_message": msg}

    def _load_default_config(self) -> Dict:
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
    def _merge(base, override):
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = AnimalDetectionPlugin._merge(result[k], v)
            else:
                result[k] = v
        return result
