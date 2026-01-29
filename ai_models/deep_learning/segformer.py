"""
SegFormer 语义分割模块
输变电激光星芒破夜绘明监测平台

功能:
- 变压器组件语义分割
- 渗漏油区域分割
- 锈蚀区域识别
- 设备边界提取
- 异常区域定位

技术实现:
- SegFormer架构 (基于Transformer的语义分割)
- 多尺度特征融合
- 轻量级解码器
- 支持多类别分割
"""

from __future__ import annotations
import logging
import time
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


class SegmentationTask(Enum):
    """分割任务类型"""
    TRANSFORMER_COMPONENT = "transformer_component"  # 变压器组件
    OIL_LEAK = "oil_leak"                           # 渗漏油
    RUST_CORROSION = "rust_corrosion"               # 锈蚀
    EQUIPMENT_BOUNDARY = "equipment_boundary"        # 设备边界
    THERMAL_ANOMALY = "thermal_anomaly"             # 热异常
    GENERAL = "general"


class SegmentationClass(Enum):
    """分割类别"""
    BACKGROUND = 0              # 背景
    TRANSFORMER_BODY = 1        # 变压器本体
    BUSHING = 2                 # 套管
    RADIATOR = 3                # 散热器
    CONSERVATOR = 4             # 储油柜
    OIL_LEAK = 5                # 渗漏油
    RUST = 6                    # 锈蚀
    DAMAGE = 7                  # 损伤
    FOREIGN_OBJECT = 8          # 异物


@dataclass
class SegFormerConfig:
    """SegFormer配置"""
    # 模型配置
    model_path: Optional[str] = None
    task: SegmentationTask = SegmentationTask.GENERAL
    model_size: str = "b0"  # b0, b1, b2, b3, b4, b5

    # 输入配置
    input_size: Tuple[int, int] = (512, 512)
    num_classes: int = 9

    # 类别配置
    class_names: List[str] = field(default_factory=lambda: [
        "background", "transformer_body", "bushing",
        "radiator", "conservator", "oil_leak",
        "rust", "damage", "foreign_object"
    ])

    # 后处理配置
    confidence_threshold: float = 0.5
    min_area: int = 100  # 最小区域面积
    apply_crf: bool = False  # 是否应用CRF后处理

    # 推理配置
    device: str = "cpu"
    fp16: bool = False


@dataclass
class SegmentationMask:
    """分割掩码"""
    mask: np.ndarray                    # 分割掩码 (H, W)
    class_id: int
    class_name: str
    area: int                           # 像素面积
    bbox: Dict[str, float]              # 边界框 (归一化)
    centroid: Tuple[float, float]       # 中心点 (归一化)
    contour: Optional[np.ndarray] = None  # 轮廓点
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    """分割结果"""
    full_mask: np.ndarray               # 完整分割图 (H, W)
    class_masks: List[SegmentationMask]  # 各类别掩码
    probabilities: Optional[np.ndarray] = None  # 概率图 (C, H, W)
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    model_version: str = ""
    input_size: Tuple[int, int] = (512, 512)


class SegFormerEncoder:
    """
    SegFormer编码器 (模拟实现)

    基于Mix Transformer (MiT) 的分层特征提取
    """

    def __init__(self, model_size: str = "b0"):
        self.model_size = model_size

        # 不同尺寸的通道配置
        self.channels_config = {
            "b0": [32, 64, 160, 256],
            "b1": [64, 128, 320, 512],
            "b2": [64, 128, 320, 512],
            "b3": [64, 128, 320, 512],
            "b4": [64, 128, 320, 512],
            "b5": [64, 128, 320, 512],
        }

        self.channels = self.channels_config.get(model_size, [32, 64, 160, 256])

    def encode(self, image: np.ndarray) -> List[np.ndarray]:
        """
        提取多尺度特征

        Args:
            image: 输入图像 (H, W, C)

        Returns:
            多尺度特征列表
        """
        h, w = image.shape[:2]

        # 模拟4个尺度的特征图
        features = []
        for i, channels in enumerate(self.channels):
            scale = 2 ** (i + 2)  # 4, 8, 16, 32
            fh, fw = h // scale, w // scale
            # 生成模拟特征
            feat = np.random.randn(channels, fh, fw).astype(np.float32) * 0.1
            features.append(feat)

        return features


class SegFormerDecoder:
    """
    SegFormer解码器

    轻量级MLP解码器，融合多尺度特征
    """

    def __init__(self, num_classes: int = 9, embed_dim: int = 256):
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def decode(
        self,
        features: List[np.ndarray],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        解码特征生成分割图

        Args:
            features: 多尺度特征列表
            target_size: 目标尺寸 (H, W)

        Returns:
            分割概率图 (num_classes, H, W)
        """
        h, w = target_size

        # 模拟解码过程
        # 实际应进行特征上采样和融合
        output = np.random.randn(self.num_classes, h, w).astype(np.float32)

        # Softmax归一化
        exp_output = np.exp(output - np.max(output, axis=0, keepdims=True))
        probabilities = exp_output / np.sum(exp_output, axis=0, keepdims=True)

        return probabilities


class SegFormerSegmenter:
    """
    SegFormer语义分割器

    用于变压器组件分割和缺陷区域识别
    """

    # 预定义任务类别映射
    TASK_CLASSES = {
        SegmentationTask.TRANSFORMER_COMPONENT: [
            "background", "transformer_body", "bushing",
            "radiator", "conservator", "tap_changer"
        ],
        SegmentationTask.OIL_LEAK: [
            "background", "normal_surface", "oil_leak",
            "oil_stain", "wet_area"
        ],
        SegmentationTask.RUST_CORROSION: [
            "background", "normal_metal", "light_rust",
            "heavy_rust", "corrosion"
        ],
    }

    # 类别颜色映射 (用于可视化)
    CLASS_COLORS = {
        0: (0, 0, 0),         # 背景 - 黑色
        1: (0, 128, 0),       # 变压器本体 - 绿色
        2: (255, 0, 0),       # 套管 - 红色
        3: (0, 0, 255),       # 散热器 - 蓝色
        4: (255, 255, 0),     # 储油柜 - 黄色
        5: (255, 0, 255),     # 渗漏油 - 紫色
        6: (255, 128, 0),     # 锈蚀 - 橙色
        7: (128, 0, 128),     # 损伤 - 深紫
        8: (0, 255, 255),     # 异物 - 青色
    }

    def __init__(self, config: SegFormerConfig):
        self.config = config
        self._session = None
        self._initialized = False
        self._inference_count = 0

        # 设置类别
        if not config.class_names and config.task in self.TASK_CLASSES:
            self.config.class_names = self.TASK_CLASSES[config.task]
            self.config.num_classes = len(self.config.class_names)

        # 编码器和解码器
        self._encoder = SegFormerEncoder(config.model_size)
        self._decoder = SegFormerDecoder(config.num_classes)

        # 版本信息
        self._model_version = f"segformer_{config.model_size}_v1.0"

        logger.info(f"[SegFormer] 初始化分割器: {config.task.value}")

    def load(self, model_path: Optional[str] = None) -> bool:
        """加载模型"""
        path = model_path or self.config.model_path

        if path is None:
            logger.warning("[SegFormer] 未指定模型路径，使用模拟模式")
            self._initialized = True
            return True

        if not ONNX_AVAILABLE:
            logger.warning("[SegFormer] onnxruntime未安装，使用模拟模式")
            self._initialized = True
            return True

        try:
            from pathlib import Path
            if not Path(path).exists():
                logger.warning(f"[SegFormer] 模型文件不存在: {path}, 使用模拟模式")
                self._initialized = True
                return True

            providers = ['CPUExecutionProvider']
            if 'cuda' in self.config.device.lower():
                providers.insert(0, 'CUDAExecutionProvider')

            self._session = ort.InferenceSession(path, providers=providers)
            self._initialized = True
            logger.info(f"[SegFormer] 模型加载成功: {path}")
            return True

        except Exception as e:
            logger.error(f"[SegFormer] 模型加载失败: {e}")
            self._initialized = True
            return False

    def segment(
        self,
        image: np.ndarray,
        return_probabilities: bool = False
    ) -> SegmentationResult:
        """
        执行语义分割

        Args:
            image: 输入图像 (BGR格式)
            return_probabilities: 是否返回概率图

        Returns:
            分割结果
        """
        if not self._initialized:
            self.load()

        start_time = time.perf_counter()

        # 预处理
        preprocess_start = time.perf_counter()
        input_tensor, scale_info = self._preprocess(image)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000

        # 推理
        infer_start = time.perf_counter()
        if self._session is not None:
            probabilities = self._infer_onnx(input_tensor)
        else:
            probabilities = self._simulate_inference(image)
        inference_time = (time.perf_counter() - infer_start) * 1000

        # 后处理
        postprocess_start = time.perf_counter()
        full_mask, class_masks = self._postprocess(
            probabilities, scale_info
        )
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000

        self._inference_count += 1

        return SegmentationResult(
            full_mask=full_mask,
            class_masks=class_masks,
            probabilities=probabilities if return_probabilities else None,
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

        # 调整大小
        if CV2_AVAILABLE:
            resized = cv2.resize(image, (target_w, target_h))
        else:
            resized = image

        # BGR to RGB
        rgb = resized[:, :, ::-1] if len(resized.shape) == 3 else resized

        # HWC to CHW
        if len(rgb.shape) == 3:
            chw = rgb.transpose(2, 0, 1)
        else:
            chw = rgb[np.newaxis, :]

        # Normalize (ImageNet标准化)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        normalized = (chw.astype(np.float32) / 255.0 - mean) / std

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        scale_info = {
            "orig_size": (orig_h, orig_w),
            "target_size": (target_h, target_w)
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
        """模拟分割推理"""
        h, w = self.config.input_size

        # 提取特征
        features = self._encoder.encode(image)

        # 解码
        probabilities = self._decoder.decode(features, (h, w))

        # 模拟一些有意义的分割
        # 假设中心区域是变压器本体
        mask = np.zeros((h, w), dtype=np.int32)

        # 添加一些模拟区域
        cx, cy = w // 2, h // 2

        # 变压器本体 (中心大区域)
        mask[cy-100:cy+100, cx-80:cx+80] = 1

        # 套管 (顶部)
        mask[cy-150:cy-100, cx-30:cx+30] = 2

        # 散热器 (侧面)
        mask[cy-50:cy+50, cx+80:cx+120] = 3

        # 转换为概率图
        probabilities = np.zeros((self.config.num_classes, h, w), dtype=np.float32)
        for c in range(self.config.num_classes):
            probabilities[c] = (mask == c).astype(np.float32) * 0.9

        # 添加一些噪声
        probabilities += np.random.randn(*probabilities.shape).astype(np.float32) * 0.05
        probabilities = np.clip(probabilities, 0, 1)

        # 归一化
        sum_prob = np.sum(probabilities, axis=0, keepdims=True) + 1e-8
        probabilities = probabilities / sum_prob

        return probabilities[np.newaxis, :]  # 添加batch维度

    def _postprocess(
        self,
        probabilities: np.ndarray,
        scale_info: Dict
    ) -> Tuple[np.ndarray, List[SegmentationMask]]:
        """后处理"""
        # 移除batch维度
        if len(probabilities.shape) == 4:
            probabilities = probabilities[0]

        # 获取类别预测
        full_mask = np.argmax(probabilities, axis=0).astype(np.uint8)

        # 调整到原始大小
        orig_h, orig_w = scale_info["orig_size"]
        if CV2_AVAILABLE and (full_mask.shape[0] != orig_h or full_mask.shape[1] != orig_w):
            full_mask = cv2.resize(
                full_mask, (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

        # 提取各类别掩码
        class_masks = []
        for class_id in range(1, self.config.num_classes):  # 跳过背景
            binary_mask = (full_mask == class_id).astype(np.uint8)
            area = int(np.sum(binary_mask))

            if area < self.config.min_area:
                continue

            # 计算边界框
            if CV2_AVAILABLE:
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if not contours:
                    continue

                # 取最大轮廓
                contour = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(contour)

                bbox = {
                    "x": x / orig_w,
                    "y": y / orig_h,
                    "width": bw / orig_w,
                    "height": bh / orig_h
                }

                # 计算中心点
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"] / orig_w
                    cy = M["m01"] / M["m00"] / orig_h
                else:
                    cx = (x + bw / 2) / orig_w
                    cy = (y + bh / 2) / orig_h
            else:
                # 简单计算
                ys, xs = np.where(binary_mask)
                if len(xs) == 0:
                    continue

                bbox = {
                    "x": float(np.min(xs)) / orig_w,
                    "y": float(np.min(ys)) / orig_h,
                    "width": float(np.max(xs) - np.min(xs)) / orig_w,
                    "height": float(np.max(ys) - np.min(ys)) / orig_h
                }
                cx = float(np.mean(xs)) / orig_w
                cy = float(np.mean(ys)) / orig_h
                contour = None

            # 计算平均置信度
            class_prob = probabilities[class_id]
            if class_prob.shape != (orig_h, orig_w):
                class_prob_resized = cv2.resize(class_prob, (orig_w, orig_h)) if CV2_AVAILABLE else class_prob
            else:
                class_prob_resized = class_prob
            confidence = float(np.mean(class_prob_resized[binary_mask > 0]))

            class_name = (
                self.config.class_names[class_id]
                if class_id < len(self.config.class_names)
                else f"class_{class_id}"
            )

            class_masks.append(SegmentationMask(
                mask=binary_mask,
                class_id=class_id,
                class_name=class_name,
                area=area,
                bbox=bbox,
                centroid=(cx, cy),
                contour=contour,
                confidence=confidence
            ))

        return full_mask, class_masks

    def segment_oil_leak(self, image: np.ndarray) -> List[SegmentationMask]:
        """
        专门的渗漏油分割

        Args:
            image: 输入图像

        Returns:
            渗漏油区域列表
        """
        result = self.segment(image)

        # 过滤出渗漏油相关类别
        oil_classes = {"oil_leak", "oil_stain", "wet_area"}
        oil_masks = [
            mask for mask in result.class_masks
            if mask.class_name in oil_classes
        ]

        return oil_masks

    def segment_rust(self, image: np.ndarray) -> List[SegmentationMask]:
        """
        专门的锈蚀分割

        Args:
            image: 输入图像

        Returns:
            锈蚀区域列表
        """
        result = self.segment(image)

        # 过滤出锈蚀相关类别
        rust_classes = {"rust", "light_rust", "heavy_rust", "corrosion"}
        rust_masks = [
            mask for mask in result.class_masks
            if mask.class_name in rust_classes
        ]

        return rust_masks

    def visualize(
        self,
        image: np.ndarray,
        result: SegmentationResult,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        可视化分割结果

        Args:
            image: 原始图像
            result: 分割结果
            alpha: 透明度

        Returns:
            可视化图像
        """
        if not CV2_AVAILABLE:
            return image

        h, w = image.shape[:2]
        mask = result.full_mask

        # 调整掩码大小
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 创建彩色掩码
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in self.CLASS_COLORS.items():
            colored_mask[mask == class_id] = color

        # 融合
        output = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        return output

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "model_version": self._model_version,
            "task": self.config.task.value,
            "model_size": self.config.model_size,
            "num_classes": self.config.num_classes,
            "class_names": self.config.class_names,
            "input_size": self.config.input_size,
            "inference_count": self._inference_count
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_transformer_segmenter(
    model_path: Optional[str] = None,
    model_size: str = "b0"
) -> SegFormerSegmenter:
    """创建变压器分割器"""
    config = SegFormerConfig(
        model_path=model_path,
        task=SegmentationTask.TRANSFORMER_COMPONENT,
        model_size=model_size
    )
    segmenter = SegFormerSegmenter(config)
    segmenter.load()
    return segmenter


def create_oil_leak_segmenter(
    model_path: Optional[str] = None
) -> SegFormerSegmenter:
    """创建渗漏油分割器"""
    config = SegFormerConfig(
        model_path=model_path,
        task=SegmentationTask.OIL_LEAK,
        model_size="b1"
    )
    segmenter = SegFormerSegmenter(config)
    segmenter.load()
    return segmenter


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "SegmentationTask",
    "SegmentationClass",
    "SegFormerConfig",
    "SegmentationMask",
    "SegmentationResult",
    "SegFormerSegmenter",
    "create_transformer_segmenter",
    "create_oil_leak_segmenter"
]
