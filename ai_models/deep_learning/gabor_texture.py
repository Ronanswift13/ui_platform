"""
Gabor纹理分析模块
输变电激光星芒破夜绘明监测平台

功能:
- 表面纹理特征提取
- 锈蚀/腐蚀检测
- 渗漏油痕迹识别
- 涂层剥落检测
- 异常纹理定位

技术实现:
- Gabor滤波器组
- 多尺度多方向分析
- 纹理特征向量提取
- 异常检测和定位
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class TextureAnomalyType(Enum):
    """纹理异常类型"""
    RUST = "rust"                       # 锈蚀
    CORROSION = "corrosion"             # 腐蚀
    OIL_STAIN = "oil_stain"             # 油污
    PAINT_PEELING = "paint_peeling"     # 涂层剥落
    CRACK = "crack"                     # 裂纹
    SCRATCH = "scratch"                 # 划痕
    DEFORMATION = "deformation"         # 变形
    CONTAMINATION = "contamination"     # 污染
    NORMAL = "normal"                   # 正常


@dataclass
class GaborFilterConfig:
    """Gabor滤波器配置"""
    # 尺度配置 (波长)
    wavelengths: List[float] = field(default_factory=lambda: [4.0, 8.0, 16.0, 32.0])

    # 方向配置 (角度数)
    num_orientations: int = 8

    # 其他参数
    sigma_ratio: float = 0.5    # sigma = wavelength * sigma_ratio
    gamma: float = 0.5          # 空间纵横比
    psi: float = 0.0            # 相位偏移


@dataclass
class TextureFeature:
    """纹理特征"""
    region_bbox: Dict[str, float]       # 区域边界框
    feature_vector: np.ndarray          # 特征向量
    mean_response: float                # 平均响应
    max_response: float                 # 最大响应
    energy: float                       # 能量
    entropy: float                      # 熵
    dominant_orientation: float         # 主方向 (度)
    dominant_scale: float               # 主尺度
    uniformity: float                   # 均匀性
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextureAnomaly:
    """纹理异常"""
    anomaly_type: TextureAnomalyType
    bbox: Dict[str, float]
    confidence: float
    severity: float                     # 严重程度 [0, 1]
    feature: TextureFeature
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextureAnalysisResult:
    """纹理分析结果"""
    anomalies: List[TextureAnomaly]
    global_features: TextureFeature
    response_maps: Optional[Dict[str, np.ndarray]] = None  # 响应图
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GaborFilterBank:
    """
    Gabor滤波器组

    生成多尺度多方向的Gabor滤波器
    """

    def __init__(self, config: GaborFilterConfig):
        self.config = config
        self._filters: List[Tuple[np.ndarray, Dict]] = []
        self._build_filters()

    def _build_filters(self):
        """构建滤波器组"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV不可用，Gabor滤波器将使用简化实现")
            return

        self._filters = []

        for wavelength in self.config.wavelengths:
            sigma = wavelength * self.config.sigma_ratio

            for i in range(self.config.num_orientations):
                theta = i * np.pi / self.config.num_orientations

                # 计算核大小
                kernel_size = int(6 * sigma) | 1  # 确保为奇数

                # 创建Gabor核
                kernel = cv2.getGaborKernel(
                    ksize=(kernel_size, kernel_size),
                    sigma=sigma,
                    theta=theta,
                    lambd=wavelength,
                    gamma=self.config.gamma,
                    psi=self.config.psi,
                    ktype=cv2.CV_32F
                )

                # 归一化
                kernel = kernel / np.sum(np.abs(kernel))

                self._filters.append((kernel, {
                    "wavelength": wavelength,
                    "orientation": theta * 180 / np.pi,
                    "sigma": sigma
                }))

        logger.info(f"[GaborFilterBank] 构建了 {len(self._filters)} 个滤波器")

    def filter(self, image: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """
        应用滤波器组

        Args:
            image: 输入图像 (灰度)

        Returns:
            滤波响应列表 [(response, params), ...]
        """
        if not CV2_AVAILABLE:
            return self._filter_simple(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        responses = []
        for kernel, params in self._filters:
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append((response, params))

        return responses

    def _filter_simple(self, image: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """简化的滤波实现 (无OpenCV时)"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float32)

        responses = []

        for wavelength in self.config.wavelengths:
            for i in range(self.config.num_orientations):
                theta = i * np.pi / self.config.num_orientations

                # 简化的响应 (梯度近似)
                response = self._simple_gabor_response(gray, wavelength, theta)

                responses.append((response, {
                    "wavelength": wavelength,
                    "orientation": theta * 180 / np.pi
                }))

        return responses

    def _simple_gabor_response(
        self,
        image: np.ndarray,
        wavelength: float,
        theta: float
    ) -> np.ndarray:
        """简化的Gabor响应计算"""
        # 使用梯度近似纹理响应
        dx = np.cos(theta)
        dy = np.sin(theta)

        # 计算方向梯度
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * dx
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) * dy

        # 简单卷积 (边界处理)
        h, w = image.shape
        response = np.zeros_like(image, dtype=np.float32)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                patch = image[y-1:y+2, x-1:x+2]
                gx = np.sum(patch * kernel_x)
                gy = np.sum(patch * kernel_y)
                response[y, x] = np.sqrt(gx**2 + gy**2)

        return response


class GaborTextureAnalyzer:
    """
    Gabor纹理分析器

    用于检测表面纹理异常，如锈蚀、油污、涂层剥落等
    """

    # 异常检测阈值
    ANOMALY_THRESHOLDS = {
        TextureAnomalyType.RUST: {"energy_min": 0.3, "entropy_max": 4.0},
        TextureAnomalyType.OIL_STAIN: {"uniformity_min": 0.2, "energy_max": 0.5},
        TextureAnomalyType.PAINT_PEELING: {"entropy_min": 3.0, "uniformity_max": 0.3},
        TextureAnomalyType.CRACK: {"orientation_variance_min": 0.5},
    }

    # 颜色特征范围 (HSV)
    COLOR_RANGES = {
        TextureAnomalyType.RUST: {"h_range": (0, 30), "s_min": 50},
        TextureAnomalyType.OIL_STAIN: {"v_max": 80, "s_min": 20},
    }

    def __init__(
        self,
        filter_config: Optional[GaborFilterConfig] = None,
        analysis_window_size: int = 64,
        window_stride: int = 32
    ):
        self.filter_config = filter_config or GaborFilterConfig()
        self.window_size = analysis_window_size
        self.window_stride = window_stride

        # 初始化滤波器组
        self.filter_bank = GaborFilterBank(self.filter_config)

    def analyze(
        self,
        image: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        return_response_maps: bool = False
    ) -> TextureAnalysisResult:
        """
        执行纹理分析

        Args:
            image: 输入图像 (BGR)
            roi_mask: ROI掩码
            return_response_maps: 是否返回响应图

        Returns:
            纹理分析结果
        """
        import time
        start_time = time.perf_counter()

        h, w = image.shape[:2]

        # 应用Gabor滤波器
        responses = self.filter_bank.filter(image)

        # 计算全局特征
        global_features = self._compute_global_features(responses, (h, w))

        # 滑动窗口分析
        anomalies = self._analyze_windows(image, responses, roi_mask)

        # 合并相邻异常
        anomalies = self._merge_anomalies(anomalies)

        # 准备响应图
        response_maps = None
        if return_response_maps:
            response_maps = self._prepare_response_maps(responses)

        processing_time = (time.perf_counter() - start_time) * 1000

        return TextureAnalysisResult(
            anomalies=anomalies,
            global_features=global_features,
            response_maps=response_maps,
            processing_time_ms=processing_time
        )

    def _compute_global_features(
        self,
        responses: List[Tuple[np.ndarray, Dict]],
        image_size: Tuple[int, int]
    ) -> TextureFeature:
        """计算全局纹理特征"""
        h, w = image_size

        # 聚合所有响应
        all_responses = np.stack([r[0] for r in responses], axis=0)

        # 计算特征
        mean_response = float(np.mean(all_responses))
        max_response = float(np.max(all_responses))
        energy = float(np.mean(all_responses ** 2))

        # 熵计算
        hist, _ = np.histogram(all_responses.flatten(), bins=256, range=(0, all_responses.max() + 1e-6))
        hist = hist / (hist.sum() + 1e-8)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-8)))

        # 主方向和主尺度
        max_idx = np.unravel_index(np.argmax(np.mean(all_responses, axis=(1, 2))), all_responses.shape[:1])
        dominant_response = responses[max_idx[0]]
        dominant_orientation = dominant_response[1].get("orientation", 0)
        dominant_scale = dominant_response[1].get("wavelength", 8)

        # 均匀性
        uniformity = 1.0 - np.std(all_responses) / (np.mean(all_responses) + 1e-8)
        uniformity = float(np.clip(uniformity, 0, 1))

        # 特征向量
        feature_vector = np.array([
            mean_response, max_response, energy, entropy,
            dominant_orientation, dominant_scale, uniformity
        ])

        return TextureFeature(
            region_bbox={"x": 0, "y": 0, "width": 1, "height": 1},
            feature_vector=feature_vector,
            mean_response=mean_response,
            max_response=max_response,
            energy=energy,
            entropy=entropy,
            dominant_orientation=dominant_orientation,
            dominant_scale=dominant_scale,
            uniformity=uniformity
        )

    def _analyze_windows(
        self,
        image: np.ndarray,
        responses: List[Tuple[np.ndarray, Dict]],
        roi_mask: Optional[np.ndarray]
    ) -> List[TextureAnomaly]:
        """滑动窗口分析"""
        h, w = image.shape[:2]
        anomalies = []

        for y in range(0, h - self.window_size, self.window_stride):
            for x in range(0, w - self.window_size, self.window_stride):
                # 检查ROI
                if roi_mask is not None:
                    window_mask = roi_mask[y:y+self.window_size, x:x+self.window_size]
                    if np.mean(window_mask) < 0.5:
                        continue

                # 提取窗口特征
                window_feature = self._compute_window_features(
                    responses, x, y, self.window_size
                )

                # 提取颜色特征
                window_image = image[y:y+self.window_size, x:x+self.window_size]
                color_features = self._extract_color_features(window_image)

                # 检测异常
                anomaly = self._detect_anomaly(
                    window_feature, color_features,
                    x / w, y / h,
                    self.window_size / w, self.window_size / h
                )

                if anomaly is not None:
                    anomalies.append(anomaly)

        return anomalies

    def _compute_window_features(
        self,
        responses: List[Tuple[np.ndarray, Dict]],
        x: int, y: int, size: int
    ) -> TextureFeature:
        """计算窗口纹理特征"""
        window_responses = []
        for response, params in responses:
            window = response[y:y+size, x:x+size]
            window_responses.append(window)

        all_responses = np.stack(window_responses, axis=0)

        mean_response = float(np.mean(all_responses))
        max_response = float(np.max(all_responses))
        energy = float(np.mean(all_responses ** 2))

        # 简化熵计算
        std_val = np.std(all_responses)
        entropy = float(np.log2(std_val + 1e-8) + 1) if std_val > 0 else 0

        # 均匀性
        uniformity = 1.0 - std_val / (mean_response + 1e-8)
        uniformity = float(np.clip(uniformity, 0, 1))

        # 主方向
        response_means = [np.mean(w) for w in window_responses]
        max_idx = np.argmax(response_means)
        dominant_orientation = responses[max_idx][1].get("orientation", 0)
        dominant_scale = responses[max_idx][1].get("wavelength", 8)

        feature_vector = np.array([
            mean_response, max_response, energy, entropy,
            dominant_orientation, dominant_scale, uniformity
        ])

        return TextureFeature(
            region_bbox={"x": 0, "y": 0, "width": 1, "height": 1},
            feature_vector=feature_vector,
            mean_response=mean_response,
            max_response=max_response,
            energy=energy,
            entropy=entropy,
            dominant_orientation=dominant_orientation,
            dominant_scale=dominant_scale,
            uniformity=uniformity
        )

    def _extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取颜色特征"""
        if not CV2_AVAILABLE or len(image.shape) != 3:
            return {"h_mean": 0, "s_mean": 0, "v_mean": 128}

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        return {
            "h_mean": float(np.mean(h)),
            "h_std": float(np.std(h)),
            "s_mean": float(np.mean(s)),
            "s_std": float(np.std(s)),
            "v_mean": float(np.mean(v)),
            "v_std": float(np.std(v))
        }

    def _detect_anomaly(
        self,
        feature: TextureFeature,
        color_features: Dict[str, float],
        x: float, y: float, w: float, h: float
    ) -> Optional[TextureAnomaly]:
        """检测异常"""
        detected_type = TextureAnomalyType.NORMAL
        confidence = 0.0
        severity = 0.0

        # 锈蚀检测
        if (feature.energy > 0.3 and
            color_features.get("h_mean", 0) < 30 and
            color_features.get("s_mean", 0) > 50):
            detected_type = TextureAnomalyType.RUST
            confidence = min(0.9, 0.5 + feature.energy)
            severity = min(1.0, feature.energy * 2)

        # 油污检测
        elif (feature.uniformity > 0.6 and
              color_features.get("v_mean", 128) < 80):
            detected_type = TextureAnomalyType.OIL_STAIN
            confidence = min(0.85, 0.4 + feature.uniformity)
            severity = min(1.0, (1 - feature.uniformity) * 2)

        # 涂层剥落检测
        elif feature.entropy > 3.5 and feature.uniformity < 0.3:
            detected_type = TextureAnomalyType.PAINT_PEELING
            confidence = min(0.8, 0.3 + feature.entropy / 10)
            severity = min(1.0, feature.entropy / 5)

        # 裂纹检测 (高能量、特定方向)
        elif feature.energy > 0.4 and feature.max_response > 0.6:
            detected_type = TextureAnomalyType.CRACK
            confidence = min(0.75, 0.3 + feature.max_response)
            severity = min(1.0, feature.max_response)

        if detected_type != TextureAnomalyType.NORMAL and confidence > 0.5:
            return TextureAnomaly(
                anomaly_type=detected_type,
                bbox={"x": x, "y": y, "width": w, "height": h},
                confidence=confidence,
                severity=severity,
                feature=feature,
                description=f"检测到{detected_type.value}异常",
                metadata={
                    "color_features": color_features
                }
            )

        return None

    def _merge_anomalies(
        self,
        anomalies: List[TextureAnomaly],
        iou_threshold: float = 0.5
    ) -> List[TextureAnomaly]:
        """合并相邻异常"""
        if len(anomalies) <= 1:
            return anomalies

        # 按类型分组
        by_type: Dict[TextureAnomalyType, List[TextureAnomaly]] = {}
        for anomaly in anomalies:
            if anomaly.anomaly_type not in by_type:
                by_type[anomaly.anomaly_type] = []
            by_type[anomaly.anomaly_type].append(anomaly)

        merged = []
        for anomaly_type, type_anomalies in by_type.items():
            # NMS式合并
            type_anomalies = sorted(type_anomalies, key=lambda a: a.confidence, reverse=True)

            keep = []
            while type_anomalies:
                best = type_anomalies.pop(0)
                keep.append(best)

                type_anomalies = [
                    a for a in type_anomalies
                    if self._compute_iou(best.bbox, a.bbox) < iou_threshold
                ]

            merged.extend(keep)

        return merged

    def _compute_iou(self, box1: Dict, box2: Dict) -> float:
        """计算IoU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - inter

        return inter / (union + 1e-8)

    def _prepare_response_maps(
        self,
        responses: List[Tuple[np.ndarray, Dict]]
    ) -> Dict[str, np.ndarray]:
        """准备响应图"""
        maps = {}

        # 按尺度聚合
        for response, params in responses:
            scale_key = f"scale_{params.get('wavelength', 0):.0f}"
            if scale_key not in maps:
                maps[scale_key] = response.copy()
            else:
                maps[scale_key] = np.maximum(maps[scale_key], response)

        # 总响应图
        all_responses = np.stack([r[0] for r in responses], axis=0)
        maps["combined"] = np.max(all_responses, axis=0)

        return maps

    def analyze_defect_region(
        self,
        image: np.ndarray,
        bbox: Dict[str, float]
    ) -> TextureFeature:
        """
        分析特定缺陷区域的纹理

        Args:
            image: 输入图像
            bbox: 边界框

        Returns:
            纹理特征
        """
        h, w = image.shape[:2]
        x = int(bbox["x"] * w)
        y = int(bbox["y"] * h)
        bw = int(bbox["width"] * w)
        bh = int(bbox["height"] * h)

        # 裁剪区域
        roi = image[y:y+bh, x:x+bw]

        # 应用滤波器
        responses = self.filter_bank.filter(roi)

        # 计算特征
        feature = self._compute_global_features(responses, roi.shape[:2])
        feature.region_bbox = bbox

        return feature


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "TextureAnomalyType",
    "GaborFilterConfig",
    "TextureFeature",
    "TextureAnomaly",
    "TextureAnalysisResult",
    "GaborFilterBank",
    "GaborTextureAnalyzer"
]
