"""
ONNX 推理引擎 — onnx_inference.py
===================================
职责:
  - 加载 animal_yolov8n.onnx 模型
  - 图像预处理 (resize, normalize, CHW 转换)
  - 执行推理, 解析 YOLOv8 输出
  - 后处理: NMS, 置信度过滤, 类别映射
  - 支持 COCO 预训练模型 (80类) 和自训练模型 (7类) 两种模式

集成位置:
  plugins/animal_detection/core/onnx_inference.py
  由 plugin.py 的 _load_model() / infer() 调用

硬约束:
  - 无模型时返回空列表 + degraded 状态, 不造假数据
  - 推理耗时计入日志审计
  - 每次推理附带 trace_id
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("animal_detection.onnx_inference")

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """单个检测结果."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (像素坐标)
    bbox_normalized: Tuple[float, float, float, float]  # x1, y1, x2, y2 (归一化)
    trace_id: str = ""
    inference_ms: float = 0.0
    source_model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": round(self.bbox[0], 1),
                "y1": round(self.bbox[1], 1),
                "x2": round(self.bbox[2], 1),
                "y2": round(self.bbox[3], 1),
            },
            "bbox_normalized": {
                "x1": round(self.bbox_normalized[0], 4),
                "y1": round(self.bbox_normalized[1], 4),
                "x2": round(self.bbox_normalized[2], 4),
                "y2": round(self.bbox_normalized[3], 4),
            },
            "trace_id": self.trace_id,
            "inference_ms": round(self.inference_ms, 1),
            "source_model": self.source_model,
        }


@dataclass
class InferenceResult:
    """推理结果聚合."""
    detections: List[Detection] = field(default_factory=list)
    status: str = "ok"  # ok / degraded / error
    trace_id: str = ""
    inference_ms: float = 0.0
    model_name: str = ""
    model_mode: str = ""  # coco_pretrained / custom_trained
    frame_shape: Tuple[int, int] = (0, 0)
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "status": self.status,
            "trace_id": self.trace_id,
            "inference_ms": round(self.inference_ms, 1),
            "model_name": self.model_name,
            "model_mode": self.model_mode,
            "frame_shape": list(self.frame_shape),
            "detection_count": len(self.detections),
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# 类别定义
# ---------------------------------------------------------------------------
TARGET_CLASSES = ["rat", "cat", "snake", "bird", "dog", "poultry", "other_wildlife"]

# COCO 80 类中与目标对应的映射 (COCO class_id -> target class_id)
COCO_CLASS_MAPPING = {
    14: 3,   # bird -> bird
    15: 1,   # cat -> cat
    16: 4,   # dog -> dog
}

# COCO 80 类名称 (用于日志)
COCO_NAMES = {
    14: "bird", 15: "cat", 16: "dog",
    17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}


# ---------------------------------------------------------------------------
# 核心推理引擎
# ---------------------------------------------------------------------------

class AnimalONNXEngine:
    """
    动物检测 ONNX 推理引擎.

    自动检测模型类型:
      - COCO 预训练 (80类输出) → 提取动物类别 + 映射
      - 自训练模型 (7类输出) → 直接使用

    使用:
        engine = AnimalONNXEngine(
            model_path="models/animal_yolov8n.onnx",
            conf_threshold=0.55,
            nms_threshold=0.4,
        )
        result = engine.infer(frame)  # frame: np.ndarray (H, W, 3) BGR/RGB
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.55,
        nms_threshold: float = 0.4,
        class_mapping_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.session = None
        self.input_name = ""
        self.input_shape = (1, 3, 640, 640)
        self.output_names: List[str] = []
        self.model_mode = "unknown"
        self.num_classes = 0
        self._loaded = False

        # 类别映射
        self.class_mapping: Optional[Dict[int, int]] = None
        if class_mapping_path and os.path.isfile(class_mapping_path):
            with open(class_mapping_path, "r") as f:
                mapping_cfg = json.load(f)
            if mapping_cfg.get("mode") == "coco_pretrained":
                self.class_mapping = {
                    int(k): v["target_id"]
                    for k, v in mapping_cfg.get("coco_to_animal", {}).items()
                }

    def load(self) -> bool:
        """加载 ONNX 模型. 失败返回 False, 不抛异常."""
        if not os.path.isfile(self.model_path):
            logger.error("模型文件不存在: %s", self.model_path)
            return False

        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    logger.warning("CUDA 不可用, 回退到 CPU")

            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers,
            )

            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.input_shape = tuple(inp.shape) if all(isinstance(d, int) for d in inp.shape) \
                else (1, 3, 640, 640)
            self.output_names = [o.name for o in self.session.get_outputs()]

            # 检测模型类型: 看输出维度判断是 80 类还是 7 类
            # YOLOv8 输出: [1, num_classes+4, num_detections]
            out_shape = self.session.get_outputs()[0].shape
            if out_shape and len(out_shape) == 3:
                feat_dim = out_shape[1] if isinstance(out_shape[1], int) else 84
                self.num_classes = feat_dim - 4
                if self.num_classes == 80:
                    self.model_mode = "coco_pretrained"
                    if self.class_mapping is None:
                        self.class_mapping = COCO_CLASS_MAPPING.copy()
                    logger.info("检测到 COCO 预训练模型 (80类), 将映射动物类别")
                elif self.num_classes == len(TARGET_CLASSES):
                    self.model_mode = "custom_trained"
                    logger.info("检测到自训练模型 (%d类)", self.num_classes)
                else:
                    self.model_mode = "custom_trained"
                    logger.warning("模型输出 %d 类, 与预期 %d 不一致",
                                   self.num_classes, len(TARGET_CLASSES))

            self._loaded = True
            logger.info("ONNX 模型加载成功: %s (mode=%s, classes=%d)",
                        self.model_path, self.model_mode, self.num_classes)
            return True

        except ImportError:
            logger.error("onnxruntime 未安装, 请执行: pip install onnxruntime")
            return False
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.session is not None

    def infer(self, frame: np.ndarray, trace_id: Optional[str] = None) -> InferenceResult:
        """
        对单帧执行推理.

        参数:
          frame: np.ndarray, shape (H, W, 3), BGR 或 RGB
          trace_id: 追踪 ID, 为 None 时自动生成

        返回:
          InferenceResult (含检测列表, 状态, 耗时等)
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())[:12]

        result = InferenceResult(
            trace_id=trace_id,
            model_name=os.path.basename(self.model_path),
            model_mode=self.model_mode,
        )

        # --- 前置检查 ---
        if not self.is_loaded:
            result.status = "degraded"
            result.error_message = "模型未加载"
            return result

        if frame is None or frame.size == 0:
            result.status = "error"
            result.error_message = "输入帧为空"
            return result

        orig_h, orig_w = frame.shape[:2]
        result.frame_shape = (orig_h, orig_w)

        t_start = time.perf_counter()

        try:
            # --- 预处理 ---
            input_tensor = self._preprocess(frame)

            # --- 推理 ---
            t_infer = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: input_tensor})
            t_infer_end = time.perf_counter()

            # --- 后处理 ---
            detections = self._postprocess(
                outputs[0], orig_w, orig_h, trace_id,
                (t_infer_end - t_infer) * 1000,
            )

            result.detections = detections
            result.status = "ok"

        except Exception as e:
            logger.error("推理异常 (trace=%s): %s", trace_id, e, exc_info=True)
            result.status = "error"
            result.error_message = str(e)

        result.inference_ms = (time.perf_counter() - t_start) * 1000
        return result

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """图像预处理: resize + normalize + CHW + batch."""
        _, _, target_h, target_w = self.input_shape

        # Letterbox resize (保持宽高比)
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        try:
            import cv2
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized = np.array(pil_img)

        # Letterbox padding
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # Normalize + CHW
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # add batch dim
        return blob

    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_w: int,
        orig_h: int,
        trace_id: str,
        infer_ms: float,
    ) -> List[Detection]:
        """
        YOLOv8 输出后处理.
        raw_output shape: [1, num_classes+4, num_detections]
        """
        # Squeeze batch dim
        output = raw_output[0]  # [num_classes+4, num_detections]
        output = output.T  # [num_detections, num_classes+4]

        # 提取 bbox 和 class scores
        boxes_xywh = output[:, :4]  # x_center, y_center, w, h
        class_scores = output[:, 4:]  # [num_detections, num_classes]

        # 获取最高置信度和对应类别
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # 置信度过滤
        mask = max_scores >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh -> xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # NMS
        keep_indices = self._nms(boxes_xyxy, max_scores, self.nms_threshold)
        boxes_xyxy = boxes_xyxy[keep_indices]
        max_scores = max_scores[keep_indices]
        class_ids = class_ids[keep_indices]

        # 缩放回原始图片坐标
        _, _, target_h, target_w = self.input_shape
        scale = min(target_w / orig_w, target_h / orig_h)
        pad_left = (target_w - int(orig_w * scale)) / 2
        pad_top = (target_h - int(orig_h * scale)) / 2

        detections = []
        for i in range(len(boxes_xyxy)):
            raw_cls_id = int(class_ids[i])
            conf = float(max_scores[i])

            # 类别映射
            if self.model_mode == "coco_pretrained" and self.class_mapping:
                if raw_cls_id not in self.class_mapping:
                    continue  # 非动物类别, 跳过
                target_cls_id = self.class_mapping[raw_cls_id]
            else:
                target_cls_id = raw_cls_id
                if target_cls_id >= len(TARGET_CLASSES):
                    continue

            # 坐标还原
            x1 = (boxes_xyxy[i, 0] - pad_left) / scale
            y1 = (boxes_xyxy[i, 1] - pad_top) / scale
            x2 = (boxes_xyxy[i, 2] - pad_left) / scale
            y2 = (boxes_xyxy[i, 3] - pad_top) / scale

            # 边界裁剪
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            det = Detection(
                class_id=target_cls_id,
                class_name=TARGET_CLASSES[target_cls_id],
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                bbox_normalized=(
                    x1 / orig_w, y1 / orig_h,
                    x2 / orig_w, y2 / orig_h,
                ),
                trace_id=trace_id,
                inference_ms=infer_ms,
                source_model=os.path.basename(self.model_path),
            )
            detections.append(det)

        # 按置信度降序排列
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression (纯 numpy 实现)."""
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def get_health_info(self) -> Dict[str, Any]:
        """返回引擎健康状态信息, 供插件健康度机制使用."""
        return {
            "loaded": self._loaded,
            "model_path": self.model_path,
            "model_mode": self.model_mode,
            "num_classes": self.num_classes,
            "conf_threshold": self.conf_threshold,
            "nms_threshold": self.nms_threshold,
            "device": self.device,
            "coverage": "full" if self.model_mode == "custom_trained" else "partial",
            "covered_classes": (
                TARGET_CLASSES if self.model_mode == "custom_trained"
                else [TARGET_CLASSES[v] for v in (self.class_mapping or {}).values()]
            ),
        }
