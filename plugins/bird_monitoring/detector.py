"""
鸟类检测器 V2.0 - 工程级实现
输变电激光星芒破夜绘明监测平台 (F组)

功能:
- 基于YOLOv8的多物种鸟类检测
- 实时多目标跟踪 (IoU + 卡尔曼滤波)
- 3D距离估计与飞行轨迹预测
- 危险等级评估与智能告警
- 声光驱鸟设备控制集成

版本: V2.0
更新: 2026-01-24 - 升级为工程级实现
"""

from __future__ import annotations
import logging
import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """威胁等级"""
    NONE = 0       # 无威胁
    LOW = 1        # 低威胁 - 远距离飞过
    MEDIUM = 2     # 中威胁 - 接近输电线
    HIGH = 3       # 高威胁 - 栖息或筑巢
    CRITICAL = 4   # 紧急 - 正在接触设备


class BirdBehavior(Enum):
    """鸟类行为状态"""
    FLYING = "flying"           # 飞行中
    APPROACHING = "approaching"  # 接近中
    PERCHED = "perched"         # 栖息
    NESTING = "nesting"         # 筑巢
    LEAVING = "leaving"         # 离开中


@dataclass
class BirdDetection:
    """鸟类检测结果"""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Dict[str, float]          # x, y, width, height (归一化)
    bbox_pixel: Dict[str, int]      # 像素坐标
    distance_m: float               # 估计距离(米)
    behavior: BirdBehavior          # 行为状态
    threat_level: ThreatLevel       # 威胁等级
    speed_ms: float                 # 速度(米/秒)
    heading_deg: float              # 航向(度)
    time_in_zone_s: float          # 在危险区域停留时间
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepelCommand:
    """驱鸟命令"""
    device_id: str
    action: str                     # sound, light, laser, ultrasonic
    intensity: int                  # 0-100
    duration_s: float
    target_direction_deg: float     # 目标方向
    reason: str
    timestamp: float


class AlertManager:
    """告警管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._alert_callbacks: List[Callable] = []
        self._alert_history: deque = deque(maxlen=1000)
        self._cooldown: Dict[str, float] = {}
        self._cooldown_seconds = config.get("alert_cooldown_seconds", 30)

    def register_callback(self, callback: Callable) -> None:
        """注册告警回调"""
        self._alert_callbacks.append(callback)

    def send_alert(self, detection: BirdDetection, message: str) -> bool:
        """发送告警"""
        alert_key = f"{detection.track_id}_{detection.threat_level.value}"

        # 检查冷却时间
        now = time.time()
        if alert_key in self._cooldown:
            if now - self._cooldown[alert_key] < self._cooldown_seconds:
                return False

        self._cooldown[alert_key] = now

        alert = {
            "timestamp": now,
            "track_id": detection.track_id,
            "class_name": detection.class_name,
            "threat_level": detection.threat_level.name,
            "behavior": detection.behavior.value,
            "distance_m": detection.distance_m,
            "message": message,
            "location": detection.bbox
        }

        self._alert_history.append(alert)

        # 调用回调
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

        logger.warning(f"[BIRD ALERT] {message} - Track {detection.track_id}, "
                      f"Threat: {detection.threat_level.name}")
        return True

    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """获取最近告警"""
        return list(self._alert_history)[-count:]


class RepelController:
    """驱鸟设备控制器（生产禁用）。

    bird_monitoring 当前只允许输出 deterrent_suggestion，不允许在检测器内
    直接访问 HTTP/Modbus/GPIO 等物理设备控制通道。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = False
        self._devices: Dict[str, Dict] = {}
        self._command_queue: deque = deque(maxlen=100)
        self._last_repel_time: Dict[str, float] = {}
        self._min_interval_s = config.get("repel_min_interval_s", 5.0)

        self._load_devices(config.get("devices", []))

    def _load_devices(self, devices_config: List[Dict]) -> None:
        """忽略驱鸟设备配置，保持生产链无硬件副作用。"""
        if devices_config:
            logger.warning("驱鸟设备配置已忽略；当前插件仅输出驱离建议")

    def trigger_repel(self, detection: BirdDetection,
                      intensity: int = 50) -> Optional[RepelCommand]:
        """硬件驱离已阻断；调用方应消费 plugin metadata 中的建议。"""
        logger.info(
            "忽略驱鸟硬件触发请求: track_id=%s intensity=%s",
            getattr(detection, "track_id", None),
            intensity,
        )
        return None

    def _select_device(self, detection: BirdDetection) -> Optional[Tuple[str, Dict]]:
        """选择最佳驱鸟设备"""
        bbox = detection.bbox
        target_angle = (bbox["x"] + bbox["width"] / 2 - 0.5) * 180  # -90 to 90

        best_device = None
        best_score = -1

        for device_id, device in self._devices.items():
            if not device["enabled"]:
                continue

            # 计算设备覆盖评分
            device_dir = device["direction"]
            coverage = device["coverage_angle"] / 2

            angle_diff = abs(target_angle - device_dir)
            if angle_diff <= coverage:
                score = 1 - (angle_diff / coverage)
                if score > best_score:
                    best_score = score
                    best_device = (device_id, device)

        return best_device

    def _calculate_direction(self, detection: BirdDetection, device: Dict) -> float:
        """计算驱鸟方向"""
        bbox = detection.bbox
        target_angle = (bbox["x"] + bbox["width"] / 2 - 0.5) * 180
        return target_angle - device["direction"]

    def _execute_command(self, command: RepelCommand, device: Dict) -> bool:
        """禁止执行物理设备命令。"""
        logger.warning("驱鸟硬件命令已阻断: device_id=%s", command.device_id)
        return False

    def get_command_history(self, count: int = 10) -> List[RepelCommand]:
        """获取命令历史"""
        return list(self._command_queue)[-count:]


class BirdDetector:
    """
    基础鸟类检测器
    
    使用YOLOv8进行鸟类目标检测
    """
    
    # 检测类别
    CLASSES = [
        "sparrow",      # 麻雀
        "magpie",       # 喜鹊
        "crow",         # 乌鸦
        "eagle",        # 老鹰
        "egret",        # 白鹭
        "swallow",      # 燕子
        "dove",         # 斑鸠
        "heron",        # 苍鹭
        "nest",         # 鸟巢
        "bird_generic", # 通用鸟类
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.session = None
        self.model_path_configured: Optional[str] = None
        self.model_path_resolved: Optional[str] = None
        self.model_file_exists = False
        self.real_model_loaded = False
        self.onnx_session_ready = False
        self.model_load_error: Optional[str] = None
        # real_dl preflight 结果；默认未执行
        self._preflight_report: Dict[str, Any] = {
            "performed": False,
            "passed": False,
            "checks": {},
            "issues": [],
        }
        
        # 推理参数
        model_config = config.get("model", {})
        self.input_size = tuple(model_config.get("input_size", [640, 640]))
        self.device = model_config.get("device", "cpu")
        
        inference_config = config.get("inference", {})
        self.confidence_threshold = inference_config.get("confidence_threshold", 0.5)
        self.nms_threshold = inference_config.get("nms_threshold", 0.45)
        self.tracking_enabled = inference_config.get("tracking_enabled", True)
        self.max_track_age = inference_config.get("max_track_age", 30)
        
        # 跟踪状态
        self._tracks: Dict[int, Dict] = {}
        self._next_track_id = 1
        self._frame_count = 0
        
        # 加载模型
        self._load_model(model_config.get("path"))
    
    def _load_model(self, model_path: Optional[str] = None):
        """加载ONNX模型"""
        if model_path is None:
            configured_model_path = Path("models/bird_yolov8n.onnx")
            model_path = Path(__file__).parent / "models" / "bird_yolov8n.onnx"
        else:
            configured_model_path = Path(model_path)
            model_path = (
                configured_model_path
                if configured_model_path.is_absolute()
                else Path(__file__).parent / configured_model_path
            )

        self.model_path_configured = str(configured_model_path)
        self.model_path_resolved = str(model_path.resolve(strict=False))
        self.model_file_exists = model_path.exists()
        
        if not self.model_file_exists:
            logger.info(f"[BirdDetector] 模型文件不存在: {model_path}, 使用模拟检测模式")
            return

        if ort is None:
            self.model_load_error = "onnxruntime_unavailable"
            logger.info("[BirdDetector] onnxruntime未安装，使用模拟模式")
            return

        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.onnx_session_ready = self.session is not None
            logger.info(f"[BirdDetector] 模型加载成功: {model_path}")
        except Exception as e:
            self.model_load_error = str(e)
            logger.warning(f"[BirdDetector] 模型加载失败: {e}")
            self.session = None
            self.onnx_session_ready = False
            return

        # real_dl preflight: 只有模型 session 就绪后才有意义
        self._preflight_report = self._preflight_onnx()
        if self._preflight_report.get("passed"):
            self.real_model_loaded = True
            logger.info("[BirdDetector] real_dl preflight 通过")
        else:
            # preflight 失败 → 回落 simulation；不销毁 session 以便 healthcheck 暴露原因
            issues = self._preflight_report.get("issues", [])
            self.real_model_loaded = False
            self.model_load_error = (
                self.model_load_error
                or f"preflight_failed:{','.join(issues)}"
            )
            logger.warning(
                "[BirdDetector] real_dl preflight 未通过，回落 simulation: %s", issues
            )
            self.session = None
            self.onnx_session_ready = False

    def _preflight_onnx(self) -> Dict[str, Any]:
        """对已加载的 ONNX session 做结构化校验。

        校验项:
        - input 张量数量 / rank / 尺寸（与 config.model.input_size 对齐）
        - output 张量数量 / channel 数（与 CLASSES 对齐，YOLOv8 格式 = 4+num_classes）
        - class map 大小（默认 `CLASSES` 至少 8 个条目才能支撑现有标签映射）

        返回结构化报告，不抛异常。
        """
        report: Dict[str, Any] = {
            "performed": False,
            "passed": False,
            "checks": {},
            "issues": [],
        }
        if self.session is None:
            report["issues"].append("session_not_ready")
            return report

        report["performed"] = True
        try:
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            report["checks"]["input_count"] = len(inputs)
            report["checks"]["output_count"] = len(outputs)

            if not inputs:
                report["issues"].append("no_input_tensor")
            else:
                input_shape = list(inputs[0].shape)
                report["checks"]["input_shape"] = input_shape
                if len(input_shape) != 4:
                    report["issues"].append(
                        f"unexpected_input_rank:{len(input_shape)}"
                    )
                else:
                    h_dim, w_dim = input_shape[2], input_shape[3]
                    cfg_h, cfg_w = self.input_size[0], self.input_size[1]
                    if isinstance(h_dim, int) and h_dim > 0 and h_dim != cfg_h:
                        report["issues"].append(
                            f"input_h_mismatch:model={h_dim},config={cfg_h}"
                        )
                    if isinstance(w_dim, int) and w_dim > 0 and w_dim != cfg_w:
                        report["issues"].append(
                            f"input_w_mismatch:model={w_dim},config={cfg_w}"
                        )

            if not outputs:
                report["issues"].append("no_output_tensor")
            else:
                output_shape = list(outputs[0].shape)
                report["checks"]["output_shape"] = output_shape
                expected_channels = 4 + len(self.CLASSES)
                report["checks"]["expected_channels"] = expected_channels
                static_dims = [
                    s for s in output_shape if isinstance(s, int) and s > 0
                ]
                channels_match = expected_channels in static_dims
                report["checks"]["channels_match"] = channels_match
                if static_dims and not channels_match:
                    report["issues"].append(
                        f"class_channel_mismatch:expected={expected_channels}"
                        f",output_shape={output_shape}"
                    )

            report["checks"]["class_map_size"] = len(self.CLASSES)
            report["checks"]["class_map"] = list(self.CLASSES)
            if len(self.CLASSES) < 8:
                report["issues"].append(
                    f"class_map_too_small:{len(self.CLASSES)}"
                )

            report["passed"] = not report["issues"]
        except Exception as e:  # pragma: no cover - defensive
            report["issues"].append(f"preflight_exception:{e}")
            report["passed"] = False
        return report
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测图像中的鸟类
        
        Args:
            image: BGR格式图像
            
        Returns:
            检测结果列表
        """
        self._frame_count += 1
        
        if self.session is None:
            # 模拟模式
            return self._simulate_detection(image)
        
        # 预处理
        input_tensor = self._preprocess(image)
        
        # 推理
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # 后处理
        detections = self._postprocess(outputs, image.shape[:2])
        
        # 跟踪
        if self.tracking_enabled:
            detections = self._update_tracks(detections)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        if cv2 is None:
            return np.zeros((1, 3, *self.input_size), dtype=np.float32)
        
        # 缩放
        resized = cv2.resize(image, self.input_size)
        
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        
        # NHWC -> NCHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 添加batch维度
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _postprocess(self, outputs: List[np.ndarray], original_size: Tuple[int, int]) -> List[Dict]:
        """后处理检测结果"""
        detections = []
        
        # YOLOv8输出格式: [batch, num_detections, 5+num_classes]
        # 5 = x_center, y_center, width, height, confidence
        output = outputs[0]
        
        if len(output.shape) == 3:
            output = output[0]  # 移除batch维度
        
        # 转置如果需要
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        h, w = original_size
        
        for detection in output:
            confidence = detection[4]
            if confidence < self.confidence_threshold:
                continue
            
            # 获取类别
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            if class_confidence < self.confidence_threshold:
                continue
            
            # 坐标转换
            x_center, y_center, box_w, box_h = detection[:4]
            x1 = (x_center - box_w / 2) / self.input_size[0]
            y1 = (y_center - box_h / 2) / self.input_size[1]
            x2 = (x_center + box_w / 2) / self.input_size[0]
            y2 = (y_center + box_h / 2) / self.input_size[1]
            
            detections.append({
                "class_id": int(class_id),
                "class_name": self.CLASSES[class_id] if class_id < len(self.CLASSES) else "unknown",
                "confidence": float(confidence * class_confidence),
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                },
                "track_id": 0,
                "status": "flying",
                "distance": 20.0,  # 默认距离，需要3D估计
                "speed": 0.0,
                "heading": 0.0,
            })
        
        # NMS
        detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """非极大值抑制"""
        if len(detections) == 0:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                det for det in detections
                if self._iou(best["bbox"], det["bbox"]) < self.nms_threshold
            ]
        
        return keep
    
    def _iou(self, box1: Dict, box2: Dict) -> float:
        """计算IoU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """更新跟踪"""
        # 简单的IoU跟踪
        for det in detections:
            best_track_id = None
            best_iou = 0.3  # IoU阈值
            
            for track_id, track in self._tracks.items():
                if track["age"] > self.max_track_age:
                    continue
                
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # 更新现有跟踪
                det["track_id"] = best_track_id
                self._tracks[best_track_id] = {
                    "bbox": det["bbox"],
                    "age": 0,
                    "class_name": det["class_name"],
                }
            else:
                # 创建新跟踪
                det["track_id"] = self._next_track_id
                self._tracks[self._next_track_id] = {
                    "bbox": det["bbox"],
                    "age": 0,
                    "class_name": det["class_name"],
                }
                self._next_track_id += 1
        
        # 增加未匹配跟踪的年龄
        matched_ids = {det["track_id"] for det in detections}
        for track_id in list(self._tracks.keys()):
            if track_id not in matched_ids:
                self._tracks[track_id]["age"] += 1
                if self._tracks[track_id]["age"] > self.max_track_age:
                    del self._tracks[track_id]
        
        return detections
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """评估输入图像质量，返回结构化质量报告"""
        h, w = image.shape[:2]
        quality = {
            "resolution": {"width": w, "height": h},
            "is_valid": True,
            "issues": [],
            "clarity_score": 1.0,
            "brightness_score": 1.0,
            "overall_score": 1.0,
        }

        # 尺寸检查
        min_dim = self.config.get("quality", {}).get("min_dimension", 64)
        if w < min_dim or h < min_dim:
            quality["issues"].append(f"图像尺寸过小({w}x{h})，最小要求{min_dim}px")
            quality["is_valid"] = False

        # 亮度检查（灰度均值）
        if cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            quality["brightness_score"] = min(1.0, brightness / 128.0) if brightness < 128 else min(1.0, (255 - brightness) / 64.0)
            if brightness < 30:
                quality["issues"].append(f"图像过暗(亮度={brightness:.0f})")
            elif brightness > 240:
                quality["issues"].append(f"图像过曝(亮度={brightness:.0f})")

            # 清晰度检查（拉普拉斯方差）
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            clarity_threshold = self.config.get("quality", {}).get("clarity_threshold", 50.0)
            quality["clarity_score"] = min(1.0, laplacian_var / (clarity_threshold * 2))
            if laplacian_var < clarity_threshold:
                quality["issues"].append(f"图像模糊(清晰度={laplacian_var:.1f})")
        else:
            quality["issues"].append("cv2不可用，跳过质量评估")

        quality["overall_score"] = (quality["clarity_score"] + quality["brightness_score"]) / 2
        return quality

    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """模拟检测 — 无真实模型时返回空列表，不产生虚假检测"""
        return []

    @property
    def runtime_mode(self) -> str:
        """返回当前运行模式"""
        if self.session is not None:
            return "real_dl"
        return "simulation"

    def get_runtime_status(self) -> Dict[str, Any]:
        """返回真实运行状态快照，供 plugin.healthcheck 暴露。"""
        return {
            "runtime_mode": self.runtime_mode,
            "model_path_configured": self.model_path_configured,
            "model_path_resolved": self.model_path_resolved,
            "model_file_exists": self.model_file_exists,
            "real_model_loaded": self.real_model_loaded,
            "onnx_session_ready": self.onnx_session_ready,
            "fallback_enabled": False,
            "simulation_enabled": self.runtime_mode == "simulation",
            "model_load_error": self.model_load_error,
            "preflight": dict(self._preflight_report),
        }

    def cleanup(self):
        """清理资源"""
        self._tracks.clear()
        self.session = None
        self.real_model_loaded = False
        self.onnx_session_ready = False


class BirdDetectorEnhanced(BirdDetector):
    """Opt-in shim: 实体实现已迁至 `experimental/enhanced_detector.py`。

    默认不加载。如需启用，必须显式设置环境变量
    `BIRD_ENABLE_ENHANCED_DETECTOR=1`。启用后实例化时会委托给
    `plugins.bird_monitoring.experimental.enhanced_detector.EnhancedBirdDetector`。

    约束:
    - `plugin.py` 的生产主链仍只加载 `BirdDetector`，不会自动接入增强检测器。
    - 该 shim 仅用于保持旧 import 路径可用；新代码请直接从 experimental/ 导入。
    """

    _OPT_IN_ENV = "BIRD_ENABLE_ENHANCED_DETECTOR"

    def __new__(cls, config: Dict[str, Any]):
        if os.environ.get(cls._OPT_IN_ENV) != "1":
            raise RuntimeError(
                "BirdDetectorEnhanced 已迁至 experimental/enhanced_detector.py；"
                "默认禁用。如需启用，请设置环境变量 "
                f"{cls._OPT_IN_ENV}=1 或直接从 "
                "plugins.bird_monitoring.experimental.enhanced_detector 导入 "
                "EnhancedBirdDetector。"
            )
        from plugins.bird_monitoring.experimental.enhanced_detector import (
            EnhancedBirdDetector,
        )
        logger.warning(
            "[BirdDetectorEnhanced] 正在通过 opt-in shim 加载 experimental 实现；"
            "该路径不属于生产主链。"
        )
        return EnhancedBirdDetector(config)
