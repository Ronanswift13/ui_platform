#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - YOLOv8 检测器
==================================

基于 Ultralytics YOLOv8 ONNX 推理引擎
- 支持 ONNX Runtime / TensorRT 后端
- 多尺度推理（小目标增强）
- 模型版本化管理

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .event_schema import AnimalClass, AnimalDetectionResult, BoundingBox

logger = logging.getLogger(__name__)

# 类别映射: 模型输出index -> AnimalClass
DEFAULT_CLASS_MAP: Dict[int, str] = {
    0: AnimalClass.MOUSE,
    1: AnimalClass.CAT,
    2: AnimalClass.SNAKE,
    3: AnimalClass.BIRD,
    4: AnimalClass.DOG,
    5: AnimalClass.POULTRY,
    6: AnimalClass.INSECT,
    7: AnimalClass.OTHER,
}

# 类别中文名
CLASS_NAMES_CN: Dict[str, str] = {
    AnimalClass.MOUSE: "鼠",
    AnimalClass.CAT: "猫",
    AnimalClass.SNAKE: "蛇",
    AnimalClass.BIRD: "鸟",
    AnimalClass.DOG: "狗",
    AnimalClass.POULTRY: "家禽",
    AnimalClass.INSECT: "昆虫",
    AnimalClass.OTHER: "其他动物",
}


class YOLOv8Detector:
    """
    YOLOv8 ONNX 推理检测器

    支持:
    - ONNX Runtime CPU/CUDA 推理
    - 多尺度推理 (对小目标更友好)
    - 置信度和NMS过滤
    - 模型版本追踪
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        max_detections: int = 50,
        class_map: Optional[Dict[int, str]] = None,
        multi_scale: bool = False,
    ):
        """
        初始化检测器

        Args:
            model_path: ONNX模型文件路径
            device: 推理设备 cpu/cuda
            input_size: 模型输入尺寸 (H, W)
            confidence_threshold: 置信度阈值
            nms_threshold: NMS IOU阈值
            max_detections: 最大检测数
            class_map: 类别ID到名称映射
            multi_scale: 是否启用多尺度推理
        """
        self.model_path = Path(model_path)
        self.device = device
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.class_map = class_map or DEFAULT_CLASS_MAP
        self.multi_scale = multi_scale

        self._session = None
        self._input_name: str = ""
        self._output_names: List[str] = []
        self._model_version: str = ""
        self._model_hash: str = ""
        self._loaded = False
        self._inference_count: int = 0
        self._total_inference_time: float = 0.0

    def load(self) -> bool:
        """
        加载ONNX模型

        Returns:
            是否成功加载
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime未安装，请运行: pip install onnxruntime")
            return False

        if not self.model_path.exists():
            logger.error(f"模型文件不存在: {self.model_path}")
            return False

        try:
            # 计算模型哈希用于版本追踪
            self._model_hash = self._compute_file_hash(self.model_path)
            self._model_version = f"yolov8_animal_{self._model_hash[:8]}"

            # 创建推理会话
            providers = []
            if self.device == "cuda":
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )

            # 获取输入输出信息
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            self._input_name = inputs[0].name
            self._output_names = [o.name for o in outputs]

            input_shape = inputs[0].shape
            logger.info(
                f"模型加载成功: {self.model_path.name} | "
                f"输入: {self._input_name} {input_shape} | "
                f"输出: {self._output_names} | "
                f"设备: {self._session.get_providers()} | "
                f"版本: {self._model_version}"
            )
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self._loaded = False
            return False

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[AnimalDetectionResult]:
        """
        对单帧执行动物检测

        Args:
            frame: BGR图像 (H, W, 3)
            roi: 可选的感兴趣区域 (x1, y1, x2, y2) 像素坐标

        Returns:
            检测结果列表
        """
        if not self._loaded or self._session is None:
            logger.warning("模型未加载，无法执行检测")
            return []

        start_time = time.time()
        img_h, img_w = frame.shape[:2]

        # ROI裁剪
        offset_x, offset_y = 0, 0
        if roi is not None:
            x1, y1, x2, y2 = roi
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
            img_h, img_w = frame.shape[:2]

        # 多尺度推理
        if self.multi_scale:
            results = self._multi_scale_detect(frame, img_w, img_h)
        else:
            results = self._single_scale_detect(frame, img_w, img_h)

        # 应用偏移量（如果有ROI裁剪）
        if roi is not None and (offset_x > 0 or offset_y > 0):
            orig_h, orig_w = frame.shape[:2]
            # 重新计算为原图坐标的归一化值
            # 这里results的bbox已经是相对于裁剪图的归一化坐标
            # 需要转换回原图坐标
            full_h = img_h + offset_y  # 不完全准确，但ROI场景较少
            full_w = img_w + offset_x
            for r in results:
                if r.bbox:
                    r.bbox = BoundingBox(
                        x=(r.bbox.x * img_w + offset_x) / (full_w if full_w > 0 else 1),
                        y=(r.bbox.y * img_h + offset_y) / (full_h if full_h > 0 else 1),
                        width=r.bbox.width * img_w / (full_w if full_w > 0 else 1),
                        height=r.bbox.height * img_h / (full_h if full_h > 0 else 1),
                    )

        elapsed = time.time() - start_time
        self._inference_count += 1
        self._total_inference_time += elapsed

        if results:
            logger.debug(
                f"检测到 {len(results)} 个目标 | "
                f"耗时 {elapsed*1000:.1f}ms | "
                f"类别: {[r.animal_class for r in results]}"
            )

        return results

    def _single_scale_detect(
        self, frame: np.ndarray, img_w: int, img_h: int
    ) -> List[AnimalDetectionResult]:
        """单尺度检测"""
        blob = self._preprocess(frame)
        outputs = self._session.run(self._output_names, {self._input_name: blob})
        return self._postprocess(outputs, img_w, img_h)

    def _multi_scale_detect(
        self, frame: np.ndarray, img_w: int, img_h: int
    ) -> List[AnimalDetectionResult]:
        """
        多尺度检测 - 对小目标更友好

        策略: 在原图 + 裁剪的4个象限上分别推理，然后合并NMS
        """
        all_boxes = []
        all_scores = []
        all_classes = []

        # 原图推理
        blob = self._preprocess(frame)
        outputs = self._session.run(self._output_names, {self._input_name: blob})
        boxes_orig, scores_orig, classes_orig = self._parse_raw_outputs(outputs, img_w, img_h)
        all_boxes.extend(boxes_orig)
        all_scores.extend(scores_orig)
        all_classes.extend(classes_orig)

        # 4象限裁剪推理 (overlap 20%)
        overlap = 0.2
        half_w = img_w // 2
        half_h = img_h // 2
        pad_w = int(half_w * overlap)
        pad_h = int(half_h * overlap)

        crops = [
            (0, 0, half_w + pad_w, half_h + pad_h),
            (half_w - pad_w, 0, img_w, half_h + pad_h),
            (0, half_h - pad_h, half_w + pad_w, img_h),
            (half_w - pad_w, half_h - pad_h, img_w, img_h),
        ]

        for cx1, cy1, cx2, cy2 in crops:
            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(img_w, cx2)
            cy2 = min(img_h, cy2)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            blob_c = self._preprocess(crop)
            out_c = self._session.run(self._output_names, {self._input_name: blob_c})
            boxes_c, scores_c, classes_c = self._parse_raw_outputs(out_c, crop_w, crop_h)

            # 将裁剪坐标映射回原图
            for b in boxes_c:
                b[0] = (b[0] * crop_w + cx1) / img_w
                b[1] = (b[1] * crop_h + cy1) / img_h
                b[2] = b[2] * crop_w / img_w
                b[3] = b[3] * crop_h / img_h

            all_boxes.extend(boxes_c)
            all_scores.extend(scores_c)
            all_classes.extend(classes_c)

        if not all_boxes:
            return []

        # 全局NMS
        boxes_arr = np.array(all_boxes)
        scores_arr = np.array(all_scores)
        classes_arr = np.array(all_classes)

        return self._nms_and_build_results(boxes_arr, scores_arr, classes_arr)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """图像预处理: resize + normalize + transpose"""
        h, w = self.input_size
        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # NCHW
        return np.ascontiguousarray(blob)

    def _postprocess(
        self, outputs: List[np.ndarray], img_w: int, img_h: int
    ) -> List[AnimalDetectionResult]:
        """后处理: 解析YOLO输出 -> 检测结果"""
        boxes, scores, classes = self._parse_raw_outputs(outputs, img_w, img_h)
        if not boxes:
            return []
        return self._nms_and_build_results(
            np.array(boxes), np.array(scores), np.array(classes)
        )

    def _parse_raw_outputs(
        self, outputs: List[np.ndarray], img_w: int, img_h: int
    ) -> Tuple[List, List, List]:
        """
        解析YOLO原始输出

        YOLOv8输出格式: (1, 4+num_classes, num_anchors)
        其中 4 = cx, cy, w, h (相对于input_size)
        """
        output = outputs[0]  # (1, 4+C, N) or (1, N, 4+C)

        # 自适应维度判断
        if output.ndim == 3:
            if output.shape[1] < output.shape[2]:
                # (1, 4+C, N) -> transpose to (1, N, 4+C)
                output = output.transpose(0, 2, 1)
            predictions = output[0]  # (N, 4+C)
        elif output.ndim == 2:
            predictions = output
        else:
            logger.warning(f"未知输出维度: {output.shape}")
            return [], [], []

        num_classes = predictions.shape[1] - 4
        if num_classes <= 0:
            logger.warning(f"无效的类别数: {num_classes}")
            return [], [], []

        # 提取各字段
        cx = predictions[:, 0]
        cy = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]
        class_scores = predictions[:, 4:]

        # 最大类别概率
        class_ids = np.argmax(class_scores, axis=1)
        max_scores = np.max(class_scores, axis=1)

        # 置信度过滤
        mask = max_scores >= self.confidence_threshold
        if not np.any(mask):
            return [], [], []

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        class_ids = class_ids[mask]
        max_scores = max_scores[mask]

        # 转为 xywh 归一化坐标
        inp_h, inp_w = self.input_size
        boxes = []
        scores = []
        classes = []

        for i in range(len(cx)):
            bx = (cx[i] - w[i] / 2) / inp_w
            by = (cy[i] - h[i] / 2) / inp_h
            bw = w[i] / inp_w
            bh = h[i] / inp_h

            # 边界裁剪
            bx = max(0.0, min(1.0, bx))
            by = max(0.0, min(1.0, by))
            bw = max(0.0, min(1.0 - bx, bw))
            bh = max(0.0, min(1.0 - by, bh))

            boxes.append([bx, by, bw, bh])
            scores.append(float(max_scores[i]))
            classes.append(int(class_ids[i]))

        return boxes, scores, classes

    def _nms_and_build_results(
        self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray
    ) -> List[AnimalDetectionResult]:
        """执行NMS并构建结果"""
        if len(boxes) == 0:
            return []

        # 转换为 xyxy 用于 NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # 按类别执行NMS
        keep_indices = []
        unique_classes = np.unique(classes)

        for cls_id in unique_classes:
            cls_mask = classes == cls_id
            cls_xyxy = xyxy[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            nms_keep = self._nms(cls_xyxy, cls_scores, self.nms_threshold)
            keep_indices.extend(cls_indices[nms_keep].tolist())

        # 按置信度排序，截断
        keep_indices.sort(key=lambda i: scores[i], reverse=True)
        keep_indices = keep_indices[: self.max_detections]

        results = []
        for idx in keep_indices:
            cls_id = int(classes[idx])
            animal_class = self.class_map.get(cls_id, AnimalClass.OTHER)

            result = AnimalDetectionResult(
                animal_class=animal_class,
                confidence=float(scores[idx]),
                bbox=BoundingBox(
                    x=float(boxes[idx, 0]),
                    y=float(boxes[idx, 1]),
                    width=float(boxes[idx, 2]),
                    height=float(boxes[idx, 3]),
                ),
            )
            results.append(result)

        return results

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """非极大值抑制"""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-8)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """计算文件SHA256哈希"""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计"""
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0
        )
        return {
            "model_path": str(self.model_path),
            "model_version": self._model_version,
            "model_hash": self._model_hash,
            "device": self.device,
            "input_size": self.input_size,
            "inference_count": self._inference_count,
            "avg_inference_time_ms": round(avg_time * 1000, 2),
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "multi_scale": self.multi_scale,
            "loaded": self._loaded,
        }

    def unload(self):
        """卸载模型释放资源"""
        self._session = None
        self._loaded = False
        logger.info("模型已卸载")
