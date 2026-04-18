"""
表计读数插件 - V3.1
输变电激光星芒破夜绘明监测平台 (E组)

功能范围:
- 任意角度读数: 关键点检测+透视矫正
- 自动量程识别: 刻度检测
- 失败兜底: 重试+人工复核标记

V3.1更新:
- 严格按算法合约输出 metadata
- MeterType 枚举传递 (不再传字符串)
- 统一 reading_status 三态
"""

from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import logging
import sys
import time
import numpy as np

from darkbreaker_sdk.interfaces import (
    BasePlugin, HealthStatus, PluginContext, PluginManifest, PluginStatus,
)
from darkbreaker_sdk.schemas import (
    Alarm, AlarmLevel, AlarmRule, RecognitionResult, ROI, BoundingBox,
)
from platform_core.visual_output_protocol import build_visual_meta

logger = logging.getLogger(__name__)


def _load_detector_module():
    detector_path = Path(__file__).parent / "detector_enhanced.py"
    module_name = "plugins.meter_reading.detector_enhanced"
    if not detector_path.exists():
        detector_path = Path(__file__).parent / "detector.py"
        module_name = "plugins.meter_reading.detector"

    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    spec = importlib.util.spec_from_file_location(module_name, detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    sys.modules.setdefault(detector_path.stem, module)
    return module

def _load_detector_class():
    module = _load_detector_module()
    if hasattr(module, 'MeterReadingDetectorEnhanced'):
        return module.MeterReadingDetectorEnhanced
    return module.MeterReadingDetector


_MeterReadingDetector = None
_MeterReadingDetectorModule = None


def get_detector_module():
    global _MeterReadingDetectorModule
    if _MeterReadingDetectorModule is None:
        _MeterReadingDetectorModule = _load_detector_module()
    return _MeterReadingDetectorModule


def get_detector_class():
    global _MeterReadingDetector
    if _MeterReadingDetector is None:
        _MeterReadingDetector = _load_detector_class()
    return _MeterReadingDetector


def _get_meter_type_enum(type_str: str):
    """将字符串转换为 MeterType 枚举"""
    MeterType = get_detector_module().MeterType
    try:
        return MeterType(type_str)
    except ValueError:
        return MeterType.PRESSURE_GAUGE


class MeterReadingPlugin(BasePlugin):
    """
    表计读数插件 V3.1

    实现任意角度读数、自动量程识别和失败兜底
    """

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

    METER_NAMES = {
        "pressure_gauge": "压强表",
        "temperature_gauge": "温度表",
        "oil_level_gauge": "油位表",
        "sf6_density_gauge": "SF6密度表",
        "digital_display": "数字显示屏",
        "led_indicator": "LED指示灯",
        "ammeter": "电流表",
        "voltmeter": "电压表",
        "seven_segment": "七段码显示",
    }

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector = None
        self._initialized = False
        self._last_inference_time = None
        self._inference_count = 0
        self._success_count = 0
        self._manual_review_count = 0
        self._config = {}

        self.confidence_threshold = 0.6
        self.retry_count = 3

    def init(self, config: dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self._config = config

            inference_config = config.get("inference", {})
            self.confidence_threshold = inference_config.get("confidence_threshold", 0.6)

            fallback_config = config.get("fallback", {})
            self.retry_count = fallback_config.get("retry_count", 3)

            MeterReadingDetector = get_detector_class()
            self._detector = MeterReadingDetector(config)
            self._detector.initialize()

            self.status = PluginStatus.READY
            self._initialized = True

            logger.info("[%s] 插件初始化成功", self.id)
            return True

        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            logger.warning("[%s] 初始化失败: %s", self.id, e)
            return False

    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """执行推理"""
        if not self._initialized or self._detector is None:
            return []

        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1

        results: list[RecognitionResult] = []

        for roi in rois:
            try:
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    continue

                meter_type_str = self._get_meter_type(roi)
                meter_type_enum = _get_meter_type_enum(meter_type_str)
                roi_id = f"{context.device_id}_{roi.id}"

                reading = self._detector.read_meter(
                    roi_image, meter_type_enum, None, roi_id,
                )

                if reading.status.value == "success" and reading.value is not None:
                    self._success_count += 1
                if reading.need_manual_review:
                    self._manual_review_count += 1

                if reading.value is not None:
                    label = f"{meter_type_str}_reading"
                    value = reading.value
                else:
                    label = f"{meter_type_str}_failed"
                    value = None

                reading_meta = dict(reading.metadata) if reading.metadata else {}
                reading_meta["unit"] = reading.unit
                reading_meta["meter_type"] = reading.meter_type.value
                reading_meta["reading_status"] = reading.status.value
                reading_meta["need_manual_review"] = reading.need_manual_review
                if "review_status" not in reading_meta:
                    reading_meta["review_status"] = (
                        "clear" if reading.status.value == "success"
                        else "manual_review_required"
                        if reading.status.value == "need_manual_review"
                        else "failed"
                    )
                if "failure_reason" not in reading_meta:
                    reading_meta["failure_reason"] = reading.failure_reason or ""
                if "pipeline_stage" not in reading_meta:
                    reading_meta["pipeline_stage"] = "unknown"
                if "timestamp_ms" not in reading_meta:
                    reading_meta["timestamp_ms"] = int(time.time() * 1000)

                metadata = build_visual_meta(
                    plugin_name="meter_reading",
                    task_type="ocr",
                    modality="meter",
                    runtime_mode="traditional_fallback",
                    algorithm_stage="baseline",
                    quality_gate_status="pass",
                    review_required=reading.need_manual_review,
                    reason_codes=(
                        [reading.failure_reason]
                        if reading.failure_reason else []
                    ),
                    evidence_hint="visual_frame",
                    extra=reading_meta,
                )

                result = RecognitionResult(
                    task_id=context.task_id,
                    site_id=context.site_id,
                    device_id=context.device_id,
                    component_id=context.component_id,
                    roi_id=roi.id,
                    bbox=roi.bbox,
                    label=label,
                    value=value,
                    confidence=reading.confidence,
                    model_version=self.version,
                    code_version=self.code_hash,
                    metadata=metadata,
                )
                results.append(result)

            except Exception as e:
                logger.warning("[%s] 处理ROI %s 时出错: %s", self.id, roi.id, e)
                continue

        self.status = PluginStatus.READY
        logger.info("[%s] 读数完成, 共 %d 个结果", self.id, len(results))

        return results

    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """后处理和告警生成"""
        alarms: list[Alarm] = []

        for result in results:
            metadata = result.metadata or {}

            if "_failed" in result.label:
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.INFO,
                    title="表计读数失败",
                    message=f"无法读取 {result.roi_id} 的表计读数",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)

            elif metadata.get("need_manual_review", False):
                meter_name = self._get_meter_name(result.label)
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.INFO,
                    title="表计读数需人工复核",
                    message=f"{meter_name} 读数 {result.value} {metadata.get('unit', '')}, 置信度较低, 请人工确认",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)

            for rule in rules:
                if self._check_rule(result, rule):
                    alarm = Alarm(
                        task_id=result.task_id,
                        result_id=None,
                        level=AlarmLevel.WARNING,
                        title="表计读数超限",
                        message=f"{self._get_meter_name(result.label)} 读数 {result.value} 超出正常范围",
                        site_id=result.site_id,
                        device_id=result.device_id,
                        component_id=result.component_id,
                    )
                    alarms.append(alarm)

        return alarms

    def healthcheck(self) -> HealthStatus:
        """健康检查"""
        if not self._initialized:
            return HealthStatus(healthy=False, message="插件未初始化")

        if self._detector is None:
            return HealthStatus(healthy=False, message="检测器未就绪")

        success_rate = self._success_count / self._inference_count if self._inference_count > 0 else 0

        return HealthStatus(
            healthy=True,
            message="插件运行正常",
            details={
                "status": self.status.value,
                "inference_count": self._inference_count,
                "success_count": self._success_count,
                "success_rate": round(success_rate, 3),
                "manual_review_count": self._manual_review_count,
                "last_inference": self._last_inference_time.isoformat() if self._last_inference_time else None,
            }
        )

    def get_ui_config(self) -> dict[str, Any]:
        """获取UI配置"""
        return {
            "detection_types": [
                {
                    "id": "reading",
                    "name": "表计读数",
                    "icon": "speedometer2",
                    "description": "指针式、数字式表计的任意角度读数识别",
                    "enabled": True,
                    "capabilities": [
                        {"label": "指针式读数", "tags": ["pointer_reading"]},
                        {"label": "数字式读数", "tags": ["digital_reading"]},
                        {"label": "七段码识别", "tags": ["seven_segment"]},
                        {"label": "LED显示读数", "tags": ["led_reading"]},
                    ]
                },
                {
                    "id": "quality",
                    "name": "读数质量评估",
                    "icon": "image",
                    "description": "关键点检测、透视矫正、失败兜底",
                    "enabled": True,
                    "capabilities": [
                        {"label": "关键点检测", "tags": ["keypoint_detection"]},
                        {"label": "透视矫正", "tags": ["perspective_correction"]},
                        {"label": "量程识别", "tags": ["range_detection"]},
                        {"label": "人工复核标记", "tags": ["manual_review"]},
                    ]
                }
            ],
            "parameters": [
                {
                    "name": "confidence_threshold",
                    "label": "置信度阈值",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": self.confidence_threshold,
                    "description": "读数结果的最小置信度"
                },
                {
                    "name": "retry_count",
                    "label": "重试次数",
                    "type": "number",
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "default": self.retry_count,
                    "description": "读数失败时的重试次数"
                },
                {
                    "name": "manual_review_threshold",
                    "label": "人工复核阈值",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.5,
                    "description": "低于此置信度的结果需要人工复核"
                }
            ]
        }

    # ==================== Standalone Routes ====================

    def get_standalone_routes(self) -> list[dict]:
        """返回独立运行模式的自定义 API 路由"""
        from fastapi import Body

        if not hasattr(self, '_simulator'):
            from plugins.meter_reading.standalone.meter_simulator import MeterSimulator
            from plugins.meter_reading.standalone.video_stream import VideoStreamService
            scenarios_dir = self.plugin_dir / "configs" / "scenarios"
            self._simulator = MeterSimulator(scenarios_dir)
            self._stream_service = VideoStreamService(
                simulator=self._simulator,
                detector=self._detector,
            )

        simulator = self._simulator
        stream_svc = self._stream_service

        async def simulator_scenarios():
            return {
                "scenarios": simulator.list_scenarios(),
                "active": simulator.get_active_scenario_id(),
            }

        async def simulator_load(payload: dict = Body(...)):
            scenario_id = payload.get("scenario")
            if not scenario_id:
                return {"error": "missing scenario id"}
            ok = simulator.load_scenario(scenario_id)
            if not ok:
                return {"error": f"scenario '{scenario_id}' not found"}
            state = simulator.get_state()
            return {"status": "loaded", **state}

        async def simulator_status():
            return simulator.get_state()

        async def stream_start(payload: dict = Body(...)):
            source = payload.get("source", "simulated")
            return stream_svc.start(source)

        async def stream_stop():
            return stream_svc.stop()

        async def stream_mjpeg():
            from starlette.responses import StreamingResponse
            return StreamingResponse(
                stream_svc.generate_mjpeg(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        async def stream_stats():
            return stream_svc.get_stats()

        async def stream_snapshot():
            from starlette.responses import Response
            jpeg = stream_svc.get_snapshot_jpeg()
            if jpeg is None:
                return {"error": "no frame available"}
            return Response(content=jpeg, media_type="image/jpeg")

        async def stream_readings(limit: int = 20):
            return stream_svc.get_recent_readings(limit)

        async def events(limit: int = 50):
            return stream_svc.get_events(limit)

        return [
            {"path": "/api/meter-reading/simulator/scenarios", "endpoint": simulator_scenarios, "methods": ["GET"], "summary": "List simulation scenarios"},
            {"path": "/api/meter-reading/simulator/load", "endpoint": simulator_load, "methods": ["POST"], "summary": "Load simulation scenario"},
            {"path": "/api/meter-reading/simulator/status", "endpoint": simulator_status, "methods": ["GET"], "summary": "Get simulator status"},
            {"path": "/api/meter-reading/stream/start", "endpoint": stream_start, "methods": ["POST"], "summary": "Start video stream"},
            {"path": "/api/meter-reading/stream/stop", "endpoint": stream_stop, "methods": ["POST"], "summary": "Stop video stream"},
            {"path": "/api/meter-reading/stream", "endpoint": stream_mjpeg, "methods": ["GET"], "summary": "MJPEG video stream"},
            {"path": "/api/meter-reading/stream/stats", "endpoint": stream_stats, "methods": ["GET"], "summary": "Stream statistics"},
            {"path": "/api/meter-reading/snapshot", "endpoint": stream_snapshot, "methods": ["GET"], "summary": "Single frame snapshot"},
            {"path": "/api/meter-reading/stream/readings", "endpoint": stream_readings, "methods": ["GET"], "summary": "Recent reading results"},
            {"path": "/api/meter-reading/events", "endpoint": events, "methods": ["GET"], "summary": "Event log"},
        ]

    # ==================== 辅助方法 ====================

    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """从帧中提取ROI区域"""
        h, w = frame.shape[:2]
        x1, y1 = int(bbox.x * w), int(bbox.y * h)
        x2, y2 = int((bbox.x + bbox.width) * w), int((bbox.y + bbox.height) * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _get_meter_type(self, roi: ROI) -> str:
        """获取表计类型字符串"""
        roi_type = getattr(roi, 'roi_type', None)
        if roi_type is not None:
            return roi_type.value if hasattr(roi_type, 'value') else str(roi_type)

        name = getattr(roi, 'name', '').lower()
        for mtype in self.METER_NAMES.keys():
            if mtype in name:
                return mtype

        return "pressure_gauge"

    def _get_meter_name(self, label: str) -> str:
        """获取表计中文名称"""
        for key, name in self.METER_NAMES.items():
            if key in label:
                return name
        return "表计"

    def _check_rule(self, result: RecognitionResult, rule: AlarmRule) -> bool:
        """检查告警规则"""
        if result.value is None:
            return False

        min_value = getattr(rule, "min_value", None)
        if min_value is not None and result.value < min_value:
            return True
        max_value = getattr(rule, "max_value", None)
        if max_value is not None and result.value > max_value:
            return True

        return False


Plugin = MeterReadingPlugin
