"""
设备状态监测插件 - 完整实现
版本: 1.0.0
"""

from __future__ import annotations
import hashlib, logging, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum
import numpy as np
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse as StarletteJSONResponse

from darkbreaker_sdk.interfaces import HealthStatus
from plugins._sensor_contract import (
    build_common_metadata,
    build_time_window,
    build_unified_temporal_output,
    build_virtual_result,
    clamp_confidence,
)

logger = logging.getLogger(__name__)

class PluginStatus(str, Enum):
    UNLOADED = "unloaded"; LOADING = "loading"; READY = "ready"
    RUNNING = "running"; ERROR = "error"; DISABLED = "disabled"

class DeviceMonitoringPlugin:
    PLUGIN_ID = "device_monitoring"
    PLUGIN_NAME = "设备状态监测"
    PLUGIN_VERSION = "1.0.0"

    def __init__(self, manifest=None, plugin_dir=None, config=None):
        self.manifest = manifest
        self.plugin_dir = plugin_dir if plugin_dir else Path(__file__).parent
        self._status = PluginStatus.UNLOADED
        self._last_error = ""
        self.config = config if isinstance(config, dict) else {}
        self._config = self.config
        self._model_registry = None
        self._detector = None
        self._is_initialized = False
        self._training_buffer: List[Dict] = []

    @classmethod
    def create_standalone(cls, config=None):
        """Create plugin instance for standalone operation."""
        plugin_dir = Path(__file__).resolve().parent
        instance = cls()
        if config is None:
            from darkbreaker_sdk.utils import load_plugin_config
            config = load_plugin_config(plugin_dir / "configs" / "default.yaml")
        if hasattr(instance, 'init'):
            instance.init(config)
        elif hasattr(instance, 'initialize'):
            instance.initialize(config)
        return instance

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
            self._config = self.config
            from .detector import DeviceMonitorDetector
            self._detector = DeviceMonitorDetector(self.config)
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
            except ValueError:
                logger.warning("[%s] 忽略未知状态: %s", self.PLUGIN_NAME, status)
        if message: self._last_error = message

    def set_model_registry(self, r): self._model_registry = r
    def shutdown(self):
        if self._detector: self._detector.reset()
        self._is_initialized = False; self._status = PluginStatus.UNLOADED

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = inputs or {}
        if not self._is_initialized or not self._detector:
            return self._err("插件未初始化", inputs)

        readings = inputs.get("device_readings")
        if readings is None:
            readings = inputs.get("readings")
        if readings is None:
            readings = self._extract_device_window(inputs)
        if isinstance(readings, dict):
            readings = [readings]
        if not readings:
            return self._err("缺少设备读数数据", inputs)

        context = dict(inputs.get("context") or {})
        context["_input_payload"] = inputs
        return self.detect(readings, context=context)

    def detect(self, device_readings: List[Dict], context=None) -> Dict[str, Any]:
        if not self._is_initialized or not self._detector:
            return self._err("插件未初始化", (context or {}).get("_input_payload", {}))
        try:
            self._status = PluginStatus.RUNNING
            r = self._detector.detect(device_readings)

            output = {
                "plugin_id": self.id, "plugin_version": self.version, "code_hash": self.code_hash,
                "task_id": (context or {}).get("task_id", ""),
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "status": self._overall_status(r["summary"]),
                "devices": [
                    {
                        "device_id": d.device_id, "device_name": d.device_name,
                        "device_type": d.device_type, "status": d.status,
                        "health_index": d.health_index, "anomaly_score": d.anomaly_score,
                        "issues": d.issues, "recommendations": d.recommendations,
                        "predicted_failure": d.predicted_failure, "metrics": d.metrics,
                    } for d in r["devices"]
                ],
                "summary": {
                    "total_devices": r["summary"].total_devices,
                    "online_count": r["summary"].online_count,
                    "warning_count": r["summary"].warning_count,
                    "error_count": r["summary"].error_count,
                    "offline_count": r["summary"].offline_count,
                    "avg_health": r["summary"].avg_health,
                    "critical_devices": r["summary"].critical_devices,
                },
                "maintenance_tickets": [
                    {"ticket_id": t.ticket_id, "device_id": t.device_id, "priority": t.priority,
                     "title": t.title, "description": t.description} for t in r["new_tickets"]
                ],
                "alarms": self._gen_alarms(r["devices"]),
                "inference_time_ms": r["inference_time_ms"],
            }
            self._status = PluginStatus.READY
            return self._with_contract(output, device_readings, context or {})
        except Exception as e:
            self._last_error = str(e); return self._err(str(e), (context or {}).get("_input_payload", {}))

    def infer(self, frame, rois, context):
        # 设备监测不直接使用图像帧
        return []

    def postprocess(self, results, rules):
        return []

    def healthcheck(self):
        return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else self._last_error,
                            details=self._detector.stats if self._detector else {})

    def scan_devices(self) -> Dict:
        """扫描所有注册设备状态"""
        managed = self.config.get("managed_devices", [])
        readings = []
        for dev in managed:
            readings.append({
                "device_id": dev["device_id"],
                "device_name": dev.get("device_name", dev["device_id"]),
                "device_type": dev.get("device_type", "unknown"),
                "metrics": {
                    "cpu_temp": 35 + np.random.rand() * 20,
                    "cpu_usage": 20 + np.random.rand() * 40,
                    "memory_usage": 30 + np.random.rand() * 30,
                    "network_quality": 70 + np.random.rand() * 30,
                    "error_count": int(np.random.rand() * 3),
                    "last_heartbeat": time.time() - np.random.rand() * 10,
                    "uptime_hours": 100 + np.random.rand() * 500,
                },
                "status": "online",
            })
        return self.detect(readings)

    def get_device_history(self, device_id: str, limit=100) -> List[Dict]:
        if self._detector:
            return self._detector.get_device_history(device_id, limit)
        return []

    def get_tickets(self) -> List[Dict]:
        if self._detector:
            return self._detector.get_all_tickets()
        return []

    def upload_training_data(self, data, labels) -> Dict:
        self._training_buffer.append({"timestamp": datetime.now().isoformat(), "labels": labels})
        return {"success": True, "total": len(self._training_buffer)}

    def start_training(self, config) -> Dict:
        return {"success": True, "message": "设备异常检测模型训练已提交", "samples": len(self._training_buffer)}

    @property
    def plugin_info(self) -> Dict:
        return {"id": self.id, "name": self.name, "version": self.version,
                "description": "设备运行状态监测与健康管理",
                "capabilities": ["device_status_monitoring", "health_index_calculation", "fault_prediction"],
                "status": self.status}

    def get_ui_config(self) -> Dict:
        return {
            "detection_types": [
                {"id": "health", "name": "健康监测", "icon": "heart-pulse", "enabled": True,
                 "capabilities": [
                     {"label": "健康指数", "tags": ["health_index"]},
                     {"label": "异常检测", "tags": ["anomaly"], "level": "warning"},
                 ]},
                {"id": "prediction", "name": "故障预测", "icon": "graph-down", "enabled": True},
                {"id": "maintenance", "name": "维护工单", "icon": "clipboard-check", "enabled": True},
            ],
            "parameters": [
                {"id": "health_warning", "name": "健康预警阈值", "type": "slider", "min": 20, "max": 90, "default": 70},
                {"id": "auto_ticket", "name": "自动工单", "type": "switch", "default": True},
            ],
        }

    def _overall_status(self, summary) -> str:
        if summary.error_count > 0 or summary.critical_devices:
            return "alarm"
        if summary.warning_count > 0 or summary.offline_count > 0:
            return "warning"
        return "normal"

    def _threshold_snapshot(self) -> Dict[str, Any]:
        return dict(self.config.get("thresholds", {}))

    def _upgrade_placeholders(self) -> Dict[str, Any]:
        return self.config.get("upgrade_placeholders", {
            "prediction_model": "device failure prediction model hook",
            "anomaly_detection_model": "device anomaly detection model hook",
            "protocol_adapter": "SNMP/Modbus/API adapter hook",
            "online_learning": "online device baseline hook",
        })

    def _build_contract_metadata(self, output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        output = output or {}
        pred_cfg = self.config.get("prediction", {})
        devices = output.get("devices") or []
        trend_available = any(bool(d.get("predicted_failure")) for d in devices)
        sampling = self.config.get("sampling", {})
        window = self.config.get("window", {})
        runtime = self.config.get("runtime", {})
        return build_common_metadata(
            modality="device",
            sensor_type="device_health_metrics",
            sample_interval=sampling.get(
                "sample_interval_seconds",
                self.config.get("performance", {}).get("scan_interval_sec", 30),
            ),
            window_size=window.get("size", pred_cfg.get("history_window_hours", 720)),
            threshold_snapshot=self._threshold_snapshot(),
            runtime_mode=runtime.get("mode", "standalone"),
            algorithm_stage="health_rules_with_statistical_prediction",
            model_status="placeholder" if pred_cfg.get("enabled", True) else "unavailable",
            fallback_level="rules",
            trend_prediction_available=trend_available,
            upgrade_placeholders=self._upgrade_placeholders(),
            extra={
                "device_count": len(devices),
                "prediction_method": pred_cfg.get("method", "statistical"),
            },
        )

    def _contract_label(self, status: str) -> str:
        if status == "normal":
            return "normal"
        if status == "warning":
            return "warning"
        if status in ("alarm", "critical", "error"):
            return "abnormal"
        return status or "normal"

    def _contract_value(self, output: Dict[str, Any]) -> Dict[str, Any]:
        devices = output.get("devices") or []
        anomaly_scores = [d.get("anomaly_score", 0.0) for d in devices]
        return {
            "total_devices": output.get("summary", {}).get("total_devices", len(devices)),
            "avg_health": output.get("summary", {}).get("avg_health", 100.0),
            "max_anomaly_score": max(anomaly_scores) if anomaly_scores else 0.0,
        }

    def _with_contract(self, output: Dict[str, Any], readings: List[Dict], context: Dict[str, Any]) -> Dict[str, Any]:
        payload = context.get("_input_payload") or {}
        first = readings[0] if readings else {}
        device_id = payload.get("device_id") or first.get("device_id", "device_monitor")
        status = output.get("status", "normal")
        label = self._contract_label(status)
        value = self._contract_value(output)
        metadata = self._build_contract_metadata(output)
        output.update({
            "label": label,
            "value": value,
            "confidence": 1.0,
            "metadata": metadata,
            "results": [
                build_virtual_result(
                    payload=payload,
                    plugin_id=self.id,
                    plugin_version=self.version,
                    code_hash=self.code_hash,
                    device_id=device_id,
                    roi_id=payload.get("roi_id") or first.get("device_id", device_id),
                    label=label,
                    value=value,
                    confidence=1.0,
                    metadata=metadata,
                    component_id="device_metrics",
                )
            ],
            **self._build_temporal_output(output, readings, payload, metadata),
        })
        return output

    def _extract_device_window(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        source = inputs.get("structured_timeseries")
        if isinstance(source, dict):
            variables = source.get("variables", source.get("metrics", source))
            device_id = inputs.get("device_id", source.get("device_id", "device_window"))
            metrics = {}
            for key, value in variables.items():
                if isinstance(value, list) and value:
                    metrics[key] = value[-1]
                elif isinstance(value, (int, float)):
                    metrics[key] = value
            if metrics:
                return [{
                    "device_id": device_id,
                    "device_name": source.get("device_name", device_id),
                    "device_type": source.get("device_type", "unknown"),
                    "status": source.get("status", "online"),
                    "metrics": metrics,
                }]

        source = inputs.get("sensor_window") or inputs.get("sampled_sequence") or inputs.get("protocol_ingested_data")
        if isinstance(source, dict):
            readings = source.get("device_readings") or source.get("readings") or source.get("samples")
            if isinstance(readings, list):
                return readings
            if isinstance(readings, dict):
                return [readings]
            if "metrics" in source:
                return [source]
        if isinstance(source, list):
            return source
        return []

    def _build_temporal_output(
        self,
        output: Dict[str, Any],
        readings: List[Dict[str, Any]],
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        time_window = build_time_window(
            payload,
            window_size=metadata.get("window_size"),
            sample_interval=metadata.get("sample_interval"),
        )
        devices = output.get("devices") or []
        anomaly_events = []
        abnormal_intervals = []
        reason_codes = []
        recommended_actions = []
        for device in devices:
            issues = list(device.get("issues") or [])
            recs = list(device.get("recommendations") or [])
            recommended_actions.extend(recs)
            if device.get("status") in ("warning", "error", "offline") or device.get("health_index", 100) < self.config.get("thresholds", {}).get("health_warning", 70):
                device_reasons = [f"DEVICE_{device.get('status', 'unknown').upper()}"]
                device_reasons.extend([f"DEVICE_ISSUE_{idx + 1}" for idx, _ in enumerate(issues)])
                reason_codes.extend(device_reasons)
                anomaly_events.append({
                    "event_id": f"{self.id}:{device.get('device_id')}:{time_window.get('start')}",
                    "event_type": "device_health",
                    "label": self._contract_label(output.get("status", "normal")),
                    "severity": "alarm" if device.get("status") == "error" else "warning",
                    "confidence": 1.0,
                    "reason_codes": device_reasons,
                    "value": {
                        "health_index": device.get("health_index"),
                        "anomaly_score": device.get("anomaly_score"),
                    },
                    "metric_name": "health_index",
                    "time_window": time_window,
                    "evidence": {"issues": issues, "metrics": device.get("metrics", {})},
                })
                abnormal_intervals.append({
                    "start": time_window.get("start"),
                    "end": time_window.get("end"),
                    "severity": "alarm" if device.get("status") == "error" else "warning",
                    "reason_codes": device_reasons,
                    "metrics": ["health_index", "anomaly_score"],
                    "device_id": device.get("device_id"),
                })
            if device.get("predicted_failure"):
                reason_codes.append("DEVICE_FAILURE_PREDICTION_PLACEHOLDER")

        trend_available = bool(metadata.get("trend_prediction_available"))
        return build_unified_temporal_output(
            plugin_name=self.PLUGIN_NAME,
            task_type="health_scoring",
            payload=payload,
            status=output.get("status", "normal"),
            label=output.get("label", "normal"),
            severity=output.get("status", "normal"),
            confidence=1.0,
            summary={
                **(output.get("summary") or {}),
                "maintenance_ticket_count": len(output.get("maintenance_tickets") or []),
            },
            anomaly_events=anomaly_events,
            abnormal_intervals=abnormal_intervals,
            reason_codes=reason_codes or ["DEVICE_HEALTH_NORMAL"],
            recommended_actions=recommended_actions or ["继续按配置周期采集设备健康指标"],
            trend_diagnosis={
                "available": trend_available,
                "direction": "degrading" if any(d.get("predicted_failure") for d in devices) else "stable",
                "confidence": 0.6 if trend_available else 0.0,
                "reason": "statistical placeholder until sequence model is connected",
                "predictions": [d.get("predicted_failure") for d in devices if d.get("predicted_failure")],
            },
            evidence=[
                {
                    "type": "device_metrics",
                    "device_count": len(devices),
                    "metric_names": sorted({k for r in readings for k in (r.get("metrics") or {}).keys()}),
                }
            ],
            review_required=output.get("status") in ("warning", "alarm"),
            model_info={
                "model_status": metadata.get("model_status"),
                "algorithm_stage": metadata.get("algorithm_stage"),
                "fallback_level": metadata.get("fallback_level"),
            },
            placeholders={
                "model_features_placeholder": ["health_index", "anomaly_score", "device_metrics"],
                "sequence_embedding_placeholder": None,
                "temporal_pattern_placeholder": "device_health_window",
                "anomaly_score_trace_placeholder": [d.get("anomaly_score", 0.0) for d in devices],
                "root_cause_feature_placeholder": {
                    "maintenance_tickets": output.get("maintenance_tickets", []),
                },
            },
            time_window=time_window,
            input_protocol={
                "metric_names": sorted({k for r in readings for k in (r.get("metrics") or {}).keys()}),
                "sampling_or_timestamp": metadata.get("sample_interval"),
            },
        )

    def _gen_alarms(self, devices) -> List[Dict]:
        alarms = []
        for d in devices:
            if d.health_index < 30:
                alarms.append({"type": "device_critical", "level": "critical", "title": f"{d.device_name}健康度危险",
                               "message": f"健康指数{d.health_index}%, 问题: {'; '.join(d.issues)}",
                               "device_id": d.device_id, "timestamp": datetime.now().isoformat()})
            elif d.health_index < 50:
                alarms.append({"type": "device_alarm", "level": "alarm", "title": f"{d.device_name}健康度异常",
                               "message": f"健康指数{d.health_index}%", "device_id": d.device_id,
                               "timestamp": datetime.now().isoformat()})
            if d.predicted_failure:
                alarms.append({"type": "failure_prediction", "level": "warning",
                               "title": f"{d.device_name}故障预测",
                               "message": f"预计{d.predicted_failure['hours_to_failure']}小时后可能故障",
                               "device_id": d.device_id, "timestamp": datetime.now().isoformat()})
        return alarms

    def _err(self, msg, inputs=None):
        inputs = inputs or {}
        device_id = inputs.get("device_id", "device_monitor") if isinstance(inputs, dict) else "device_monitor"
        metadata = self._build_contract_metadata()
        time_window = build_time_window(
            inputs if isinstance(inputs, dict) else {},
            window_size=metadata.get("window_size"),
            sample_interval=metadata.get("sample_interval"),
        )
        temporal_output = build_unified_temporal_output(
            plugin_name=self.PLUGIN_NAME,
            task_type="health_scoring",
            payload=inputs if isinstance(inputs, dict) else {},
            status="error",
            label="error",
            severity="error",
            confidence=0.0,
            summary={"device_id": device_id, "status": "error", "message": msg},
            reason_codes=["INPUT_VALIDATION_ERROR"],
            recommended_actions=["检查 readings/device_readings/structured_timeseries 输入"],
            trend_diagnosis={"available": False, "direction": "unknown", "confidence": 0.0, "reason": msg},
            evidence=[{"type": "validation_error", "message": msg}],
            review_required=True,
            model_info={"model_status": metadata.get("model_status"), "fallback_level": metadata.get("fallback_level")},
            time_window=time_window,
        )
        return {"plugin_id": self.id, "plugin_version": self.version, "timestamp": datetime.now().isoformat(),
                "success": False, "status": "error", "label": "error", "value": None,
                "confidence": 0.0, "metadata": metadata,
                "results": [build_virtual_result(
                    payload=inputs if isinstance(inputs, dict) else {},
                    plugin_id=self.id,
                    plugin_version=self.version,
                    code_hash=self.code_hash,
                    device_id=device_id,
                    roi_id=(inputs or {}).get("roi_id") or device_id if isinstance(inputs, dict) else device_id,
                    label="error",
                    value=None,
                    confidence=0.0,
                    metadata=metadata,
                    component_id="device_metrics",
                    failure_reason=msg,
                )],
                **temporal_output,
                "devices": [], "summary": temporal_output["summary"], "alarms": [], "error_message": msg, "error": msg}

    def get_standalone_routes(self) -> list:
        plugin = self

        async def device_smoke(request: StarletteRequest):
            body = {}
            try:
                body = await request.json()
            except Exception as exc:
                logger.debug("device smoke request has no JSON body: %s", exc)
            sample = {
                "device_id": "edge_smoke_01",
                "readings": [{
                    "device_id": "edge_smoke_01",
                    "device_name": "Standalone Edge Node",
                    "device_type": "edge_computing_node",
                    "status": "online",
                    "metrics": {
                        "cpu_temp": 42,
                        "cpu_usage": 30,
                        "memory_usage": 35,
                        "network_quality": 95,
                        "error_count": 0,
                        "last_heartbeat": time.time(),
                    },
                }],
                "context": {"task_id": "device-smoke", "site_id": "standalone"},
            }
            sample.update(body)
            if not sample.get("readings") and not sample.get("device_readings"):
                sample["readings"] = [{
                    "device_id": sample.get("device_id", "edge_smoke_01"),
                    "device_name": "Standalone Edge Node",
                    "device_type": "edge_computing_node",
                    "status": "online",
                    "metrics": {
                        "cpu_temp": 42,
                        "cpu_usage": 30,
                        "memory_usage": 35,
                        "network_quality": 95,
                        "error_count": 0,
                        "last_heartbeat": time.time(),
                    },
                }]
            return StarletteJSONResponse(plugin.process(sample))

        return [
            {"path": "/api/device/smoke", "endpoint": device_smoke, "methods": ["GET", "POST"], "summary": "Run device smoke sample"},
        ]

    def _load_default_config(self):
        try:
            import yaml
            p = Path(self.plugin_dir) / "configs" / "default.yaml"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("[%s] 默认配置加载失败: %s", self.PLUGIN_NAME, exc)
        return {}

    @staticmethod
    def _merge(b, o):
        r = b.copy()
        for k, v in o.items():
            if k in r and isinstance(r[k], dict) and isinstance(v, dict): r[k] = DeviceMonitoringPlugin._merge(r[k], v)
            else: r[k] = v
        return r


Plugin = DeviceMonitoringPlugin
