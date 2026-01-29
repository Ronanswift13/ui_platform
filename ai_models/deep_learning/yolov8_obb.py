"""
YOLOv8-OBB 旋转框目标检测模块
输变电激光星芒破夜绘明监测平台

功能:
- 旋转边界框 (Oriented Bounding Box) 检测
- 倾斜目标精确定位
- 多角度目标检测
- 电容器/绝缘子等倾斜设备检测

技术实现:
- YOLOv8-OBB架构
- 旋转框回归 (θ预测)
- 角度分类与回归混合
- 旋转NMS
"""

from __future__ import annotations
import logging
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入依赖
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class OBBDetectionTask(Enum):
    """OBB检测任务类型"""
    CAPACITOR_DEFECT = "capacitor_defect"       # 电容器缺陷
    INSULATOR_DEFECT = "insulator_defect"       # 绝缘子缺陷
    TRANSFORMER_COMPONENT = "transformer_component"  # 变压器组件
    TILTED_OBJECT = "tilted_object"             # 通用倾斜目标
    GENERAL = "general"


@dataclass
class OBBConfig:
    """OBB检测配置"""
    # 模型配置
    model_path: Optional[str] = None
    task: OBBDetectionTask = OBBDetectionTask.GENERAL

    # 输入配置
    input_size: Tuple[int, int] = (640, 640)

    # 检测配置
    num_classes: int = 10
    class_names: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 300

    # 角度配置
    angle_range: Tuple[float, float] = (-90.0, 90.0)  # 角度范围
    use_angle_classification: bool = False  # 使用角度分类 vs 回归

    # 倾斜检测配置
    tilt_warning_threshold: float = 5.0    # 警告阈值 (度)
    tilt_critical_threshold: float = 15.0  # 严重阈值 (度)

    # 推理配置
    device: str = "cpu"
    fp16: bool = False


@dataclass
class OrientedBox:
    """旋转边界框"""
    cx: float           # 中心x (归一化)
    cy: float           # 中心y (归一化)
    width: float        # 宽度 (归一化)
    height: float       # 高度 (归一化)
    angle: float        # 旋转角度 (度, -90~90)

    def to_corners(self, img_w: int = 1, img_h: int = 1) -> np.ndarray:
        """转换为四个角点坐标"""
        cx_px = self.cx * img_w
        cy_px = self.cy * img_h
        w_px = self.width * img_w
        h_px = self.height * img_h

        # 计算旋转后的四个角点
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 四个角点相对于中心的偏移
        corners_rel = np.array([
            [-w_px/2, -h_px/2],
            [w_px/2, -h_px/2],
            [w_px/2, h_px/2],
            [-w_px/2, h_px/2]
        ])

        # 旋转
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        corners_rot = np.dot(corners_rel, rotation_matrix.T)

        # 平移到中心位置
        corners = corners_rot + np.array([cx_px, cy_px])

        return corners

    def to_axis_aligned_bbox(self) -> Dict[str, float]:
        """转换为轴对齐边界框"""
        corners = self.to_corners()
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])

        return {
            "x": float(x_min),
            "y": float(y_min),
            "width": float(x_max - x_min),
            "height": float(y_max - y_min)
        }

    def get_tilt_status(
        self,
        warning_threshold: float = 5.0,
        critical_threshold: float = 15.0,
        reference_angle: float = 0.0
    ) -> Dict[str, Any]:
        """获取倾斜状态"""
        tilt_angle = abs(self.angle - reference_angle)

        if tilt_angle >= critical_threshold:
            status = "critical"
            level = "error"
        elif tilt_angle >= warning_threshold:
            status = "warning"
            level = "warning"
        else:
            status = "normal"
            level = "normal"

        return {
            "status": status,
            "level": level,
            "tilt_angle": tilt_angle,
            "reference_angle": reference_angle,
            "actual_angle": self.angle
        }


@dataclass
class OBBDetection:
    """OBB检测结果"""
    obb: OrientedBox
    confidence: float
    class_id: int
    class_name: str
    tilt_status: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OBBDetectionResult:
    """OBB检测输出"""
    detections: List[OBBDetection]
    inference_time_ms: float
    preprocess_time_ms: float
    postprocess_time_ms: float
    model_version: str
    input_size: Tuple[int, int]


class YOLOv8OBBDetector:
    """
    YOLOv8-OBB 旋转框目标检测器

    用于检测倾斜目标，如电容器、绝缘子等
    """

    # 预定义任务类别
    TASK_CLASSES = {
        OBBDetectionTask.CAPACITOR_DEFECT: [
            "capacitor_normal", "capacitor_tilt_warning",
            "capacitor_tilt_critical", "capacitor_collapse",
            "capacitor_missing", "fuse_normal", "fuse_damaged"
        ],
        OBBDetectionTask.INSULATOR_DEFECT: [
            "insulator_normal", "insulator_cracked",
            "insulator_flashover", "insulator_contaminated"
        ],
        OBBDetectionTask.TRANSFORMER_COMPONENT: [
            "bushing", "radiator", "conservator",
            "tap_changer", "oil_gauge"
        ],
    }

    def __init__(self, config: OBBConfig):
        self.config = config
        self._session = None
        self._initialized = False
        self._inference_count = 0

        # 设置类别
        if not config.class_names and config.task in self.TASK_CLASSES:
            self.config.class_names = self.TASK_CLASSES[config.task]
            self.config.num_classes = len(self.config.class_names)

        # 版本信息
        self._model_version = "yolov8_obb_v1.0"

        logger.info(f"[YOLOv8-OBB] 初始化检测器: {config.task.value}")

    def load(self, model_path: Optional[str] = None) -> bool:
        """加载模型"""
        path = model_path or self.config.model_path

        if path is None:
            logger.warning("[YOLOv8-OBB] 未指定模型路径，使用模拟模式")
            self._initialized = True
            return True

        if not ONNX_AVAILABLE:
            logger.warning("[YOLOv8-OBB] onnxruntime未安装，使用模拟模式")
            self._initialized = True
            return True

        try:
            from pathlib import Path
            if not Path(path).exists():
                logger.warning(f"[YOLOv8-OBB] 模型文件不存在: {path}, 使用模拟模式")
                self._initialized = True
                return True

            providers = ['CPUExecutionProvider']
            if 'cuda' in self.config.device.lower():
                providers.insert(0, 'CUDAExecutionProvider')

            self._session = ort.InferenceSession(path, providers=providers)
            self._initialized = True
            logger.info(f"[YOLOv8-OBB] 模型加载成功: {path}")
            return True

        except Exception as e:
            logger.error(f"[YOLOv8-OBB] 模型加载失败: {e}")
            self._initialized = True  # 回退到模拟模式
            return False

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None
    ) -> OBBDetectionResult:
        """
        执行OBB检测

        Args:
            image: 输入图像 (BGR格式)
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值

        Returns:
            OBB检测结果
        """
        if not self._initialized:
            self.load()

        conf_thresh = confidence_threshold or self.config.confidence_threshold
        nms_thresh = nms_threshold or self.config.nms_threshold

        start_time = time.perf_counter()

        # 预处理
        preprocess_start = time.perf_counter()
        input_tensor, scale_info = self._preprocess(image)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000

        # 推理
        infer_start = time.perf_counter()
        if self._session is not None:
            raw_output = self._infer_onnx(input_tensor)
        else:
            raw_output = self._simulate_inference(image)
        inference_time = (time.perf_counter() - infer_start) * 1000

        # 后处理
        postprocess_start = time.perf_counter()
        detections = self._postprocess(
            raw_output, scale_info, conf_thresh, nms_thresh
        )
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000

        self._inference_count += 1

        return OBBDetectionResult(
            detections=detections,
            inference_time_ms=inference_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time,
            model_version=self._model_version,
            input_size=self.config.input_size
        )

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """预处理图像"""
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.config.input_size

        # 保持比例缩放
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        if CV2_AVAILABLE:
            resized = cv2.resize(image, (new_w, new_h))
        else:
            resized = image

        # 创建画布 (letterbox)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # BGR to RGB
        rgb = canvas[:, :, ::-1]

        # HWC to CHW
        chw = rgb.transpose(2, 0, 1)

        # Normalize
        normalized = chw.astype(np.float32) / 255.0

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        scale_info = {
            "orig_size": (orig_h, orig_w),
            "new_size": (new_h, new_w),
            "scale": scale,
            "pad": (pad_h, pad_w),
        }

        return batched, scale_info

    def _infer_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """ONNX推理"""
        if self._session is None:
            raise RuntimeError("ONNX session未初始化")

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})
        return outputs[0]

    def _simulate_inference(self, image: np.ndarray) -> np.ndarray:
        """模拟OBB推理输出"""
        # OBB输出格式: [batch, 5+nc, num_boxes]
        # 5: cx, cy, w, h, angle
        num_boxes = 8400
        num_classes = self.config.num_classes
        output = np.random.randn(1, 5 + num_classes, num_boxes).astype(np.float32)

        # 生成一些模拟检测
        for i in range(min(5, num_boxes)):
            output[0, 0, i] = np.random.uniform(0.2, 0.8)  # cx
            output[0, 1, i] = np.random.uniform(0.2, 0.8)  # cy
            output[0, 2, i] = np.random.uniform(0.05, 0.2)  # w
            output[0, 3, i] = np.random.uniform(0.1, 0.3)   # h
            output[0, 4, i] = np.random.uniform(-45, 45)    # angle
            class_idx = np.random.randint(0, num_classes)
            output[0, 5 + class_idx, i] = np.random.uniform(0.6, 0.95)  # class score

        return output

    def _postprocess(
        self,
        output: np.ndarray,
        scale_info: Dict,
        conf_thresh: float,
        nms_thresh: float
    ) -> List[OBBDetection]:
        """后处理OBB输出"""
        detections = []

        # 输出格式: [batch, 5+nc, num_boxes]
        if len(output.shape) == 3:
            output = output[0]  # 移除batch维度

        # 转置: [5+nc, num_boxes] -> [num_boxes, 5+nc]
        if output.shape[0] < output.shape[1]:
            output = output.T

        orig_h, orig_w = scale_info["orig_size"]
        target_h, target_w = self.config.input_size
        pad_h, pad_w = scale_info["pad"]
        scale = scale_info["scale"]

        for row in output:
            # 解析: cx, cy, w, h, angle, class_scores...
            cx, cy, w, h, angle = row[:5]
            class_scores = row[5:]

            # 获取最大类别
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < conf_thresh:
                continue

            # 转换坐标
            cx_orig = (cx - pad_w) / scale / orig_w
            cy_orig = (cy - pad_h) / scale / orig_h
            w_orig = w / scale / orig_w
            h_orig = h / scale / orig_h

            # 角度归一化到 [-90, 90]
            angle = float(angle)
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180

            obb = OrientedBox(
                cx=float(np.clip(cx_orig, 0, 1)),
                cy=float(np.clip(cy_orig, 0, 1)),
                width=float(np.clip(w_orig, 0, 1)),
                height=float(np.clip(h_orig, 0, 1)),
                angle=angle
            )

            class_name = (
                self.config.class_names[class_id]
                if class_id < len(self.config.class_names)
                else f"class_{class_id}"
            )

            # 计算倾斜状态
            tilt_status = obb.get_tilt_status(
                warning_threshold=self.config.tilt_warning_threshold,
                critical_threshold=self.config.tilt_critical_threshold
            )

            detection = OBBDetection(
                obb=obb,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                tilt_status=tilt_status
            )
            detections.append(detection)

        # 旋转NMS
        detections = self._rotated_nms(detections, nms_thresh)

        # 限制最大检测数
        detections = detections[:self.config.max_detections]

        return detections

    def _rotated_nms(
        self,
        detections: List[OBBDetection],
        threshold: float
    ) -> List[OBBDetection]:
        """旋转NMS"""
        if not detections:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # 使用旋转IoU过滤
            detections = [
                d for d in detections
                if self._rotated_iou(best.obb, d.obb) < threshold
            ]

        return keep

    def _rotated_iou(self, obb1: OrientedBox, obb2: OrientedBox) -> float:
        """计算旋转IoU"""
        if not CV2_AVAILABLE:
            # 简化计算: 使用轴对齐bbox的IoU
            bbox1 = obb1.to_axis_aligned_bbox()
            bbox2 = obb2.to_axis_aligned_bbox()
            return self._axis_aligned_iou(bbox1, bbox2)

        try:
            # 获取角点
            corners1 = obb1.to_corners()
            corners2 = obb2.to_corners()

            # 使用cv2计算旋转矩形的交集
            rect1 = cv2.minAreaRect(corners1.astype(np.float32))
            rect2 = cv2.minAreaRect(corners2.astype(np.float32))

            int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)[1]

            if int_pts is None:
                return 0.0

            # 计算交集面积
            int_area = cv2.contourArea(int_pts)

            # 计算各自面积
            area1 = obb1.width * obb1.height
            area2 = obb2.width * obb2.height

            union = area1 + area2 - int_area

            return int_area / (union + 1e-8)

        except Exception:
            # 回退到轴对齐IoU
            bbox1 = obb1.to_axis_aligned_bbox()
            bbox2 = obb2.to_axis_aligned_bbox()
            return self._axis_aligned_iou(bbox1, bbox2)

    def _axis_aligned_iou(self, box1: Dict, box2: Dict) -> float:
        """计算轴对齐IoU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - inter

        return inter / (union + 1e-8)

    def detect_with_thermal(
        self,
        visible_image: np.ndarray,
        thermal_image: np.ndarray,
        thermal_weight: float = 0.4
    ) -> OBBDetectionResult:
        """
        融合热成像的OBB检测

        Args:
            visible_image: 可见光图像
            thermal_image: 红外热像
            thermal_weight: 热像权重

        Returns:
            融合检测结果
        """
        # 可见光检测
        visible_result = self.detect(visible_image)

        # 热成像检测
        thermal_result = self.detect(thermal_image)

        # 融合检测结果
        fused_detections = self._fuse_obb_detections(
            visible_result.detections,
            thermal_result.detections,
            thermal_weight
        )

        return OBBDetectionResult(
            detections=fused_detections,
            inference_time_ms=visible_result.inference_time_ms + thermal_result.inference_time_ms,
            preprocess_time_ms=visible_result.preprocess_time_ms + thermal_result.preprocess_time_ms,
            postprocess_time_ms=visible_result.postprocess_time_ms,
            model_version=self._model_version + "_thermal_fusion",
            input_size=self.config.input_size
        )

    def _fuse_obb_detections(
        self,
        visible_dets: List[OBBDetection],
        thermal_dets: List[OBBDetection],
        thermal_weight: float
    ) -> List[OBBDetection]:
        """融合OBB检测结果"""
        fused = []
        used_thermal = set()

        for v_det in visible_dets:
            best_match = None
            best_iou = 0.3  # 匹配阈值

            for i, t_det in enumerate(thermal_dets):
                if i in used_thermal:
                    continue
                iou = self._rotated_iou(v_det.obb, t_det.obb)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (i, t_det)

            if best_match is not None:
                i, t_det = best_match
                # 融合置信度
                fused_conf = (
                    v_det.confidence * (1 - thermal_weight) +
                    t_det.confidence * thermal_weight
                )

                fused_det = OBBDetection(
                    obb=v_det.obb,
                    confidence=fused_conf,
                    class_id=v_det.class_id,
                    class_name=v_det.class_name,
                    tilt_status=v_det.tilt_status,
                    metadata={
                        "visible_conf": v_det.confidence,
                        "thermal_conf": t_det.confidence,
                        "fusion": "matched"
                    }
                )
                fused.append(fused_det)
                used_thermal.add(i)
            else:
                v_det.metadata["fusion"] = "visible_only"
                fused.append(v_det)

        # 添加未匹配的热成像检测
        for i, t_det in enumerate(thermal_dets):
            if i not in used_thermal:
                t_det.confidence *= thermal_weight
                t_det.metadata["fusion"] = "thermal_only"
                fused.append(t_det)

        return self._rotated_nms(fused, self.config.nms_threshold)

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "model_version": self._model_version,
            "task": self.config.task.value,
            "num_classes": self.config.num_classes,
            "class_names": self.config.class_names,
            "input_size": self.config.input_size,
            "tilt_thresholds": {
                "warning": self.config.tilt_warning_threshold,
                "critical": self.config.tilt_critical_threshold
            },
            "inference_count": self._inference_count
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_capacitor_obb_detector(
    model_path: Optional[str] = None
) -> YOLOv8OBBDetector:
    """创建电容器OBB检测器"""
    config = OBBConfig(
        model_path=model_path,
        task=OBBDetectionTask.CAPACITOR_DEFECT,
        tilt_warning_threshold=5.0,
        tilt_critical_threshold=15.0
    )
    detector = YOLOv8OBBDetector(config)
    detector.load()
    return detector


def create_insulator_obb_detector(
    model_path: Optional[str] = None
) -> YOLOv8OBBDetector:
    """创建绝缘子OBB检测器"""
    config = OBBConfig(
        model_path=model_path,
        task=OBBDetectionTask.INSULATOR_DEFECT,
        tilt_warning_threshold=3.0,
        tilt_critical_threshold=10.0
    )
    detector = YOLOv8OBBDetector(config)
    detector.load()
    return detector


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "OBBDetectionTask",
    "OBBConfig",
    "OrientedBox",
    "OBBDetection",
    "OBBDetectionResult",
    "YOLOv8OBBDetector",
    "create_capacitor_obb_detector",
    "create_insulator_obb_detector"
]
