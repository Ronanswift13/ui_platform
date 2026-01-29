"""
红外-可见光图像配准模块
输变电激光星芒破夜绘明监测平台

功能:
- 红外与可见光图像配准 (Registration)
- 特征点匹配与变换估计
- 多模态图像融合
- 热点区域映射

技术实现:
- 基于特征点的配准 (SIFT/ORB/AKAZE)
- 基于深度学习的配准 (可选)
- 仿射变换/透视变换
- 热成像与可见光融合策略
"""

from __future__ import annotations
import logging
import time
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


class RegistrationMethod(Enum):
    """配准方法"""
    FEATURE_BASED = "feature_based"     # 基于特征点
    INTENSITY_BASED = "intensity_based" # 基于灰度
    DEEP_LEARNING = "deep_learning"     # 基于深度学习
    HYBRID = "hybrid"                   # 混合方法


class FeatureDetectorType(Enum):
    """特征检测器类型"""
    SIFT = "sift"
    ORB = "orb"
    AKAZE = "akaze"
    BRISK = "brisk"


class FusionMethod(Enum):
    """融合方法"""
    WEIGHTED_AVERAGE = "weighted_average"   # 加权平均
    LAPLACIAN_PYRAMID = "laplacian_pyramid" # 拉普拉斯金字塔
    WAVELET = "wavelet"                     # 小波变换
    PCA = "pca"                             # 主成分分析
    GUIDED_FILTER = "guided_filter"         # 引导滤波


@dataclass
class RegistrationConfig:
    """配准配置"""
    method: RegistrationMethod = RegistrationMethod.FEATURE_BASED
    feature_detector: FeatureDetectorType = FeatureDetectorType.AKAZE
    min_match_count: int = 10
    ransac_reproj_threshold: float = 5.0
    use_homography: bool = True  # True: 透视变换, False: 仿射变换
    max_features: int = 1000
    match_ratio: float = 0.75  # Lowe's ratio test


@dataclass
class FusionConfig:
    """融合配置"""
    method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE
    thermal_weight: float = 0.4
    visible_weight: float = 0.6
    pyramid_levels: int = 4
    enhance_thermal: bool = True


@dataclass
class RegistrationResult:
    """配准结果"""
    success: bool
    transform_matrix: Optional[np.ndarray] = None
    matched_points: int = 0
    inliers: int = 0
    reprojection_error: float = 0.0
    processing_time_ms: float = 0.0
    method_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """融合结果"""
    fused_image: Optional[np.ndarray] = None
    thermal_aligned: Optional[np.ndarray] = None
    hotspot_mask: Optional[np.ndarray] = None
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThermalVisibleRegistration:
    """
    红外-可见光图像配准

    实现红外热像与可见光图像的精确配准，
    支持多种配准方法和融合策略。
    """

    def __init__(
        self,
        registration_config: Optional[RegistrationConfig] = None,
        fusion_config: Optional[FusionConfig] = None
    ):
        self.reg_config = registration_config or RegistrationConfig()
        self.fusion_config = fusion_config or FusionConfig()

        # 特征检测器和匹配器
        self._detector = None
        self._matcher = None
        self._init_feature_detector()

        # 缓存
        self._last_transform: Optional[np.ndarray] = None
        self._calibration_matrix: Optional[np.ndarray] = None

    def _init_feature_detector(self):
        """初始化特征检测器"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV不可用，配准功能受限")
            return

        try:
            detector_type = self.reg_config.feature_detector

            if detector_type == FeatureDetectorType.SIFT:
                self._detector = cv2.SIFT_create(nfeatures=self.reg_config.max_features)
                self._matcher = cv2.BFMatcher(cv2.NORM_L2)
            elif detector_type == FeatureDetectorType.ORB:
                self._detector = cv2.ORB_create(nfeatures=self.reg_config.max_features)
                self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            elif detector_type == FeatureDetectorType.AKAZE:
                self._detector = cv2.AKAZE_create()
                self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            elif detector_type == FeatureDetectorType.BRISK:
                self._detector = cv2.BRISK_create()
                self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

            logger.info(f"特征检测器初始化成功: {detector_type.value}")

        except Exception as e:
            logger.error(f"特征检测器初始化失败: {e}")

    def register(
        self,
        thermal_image: np.ndarray,
        visible_image: np.ndarray,
        use_cache: bool = True
    ) -> RegistrationResult:
        """
        执行红外-可见光配准

        Args:
            thermal_image: 红外热像 (单通道或三通道)
            visible_image: 可见光图像 (BGR)
            use_cache: 是否使用缓存的变换矩阵

        Returns:
            配准结果
        """
        start_time = time.perf_counter()

        if not CV2_AVAILABLE:
            return RegistrationResult(
                success=False,
                metadata={"error": "OpenCV不可用"}
            )

        # 预处理图像
        thermal_gray = self._preprocess_thermal(thermal_image)
        visible_gray = self._preprocess_visible(visible_image)

        # 根据配置选择配准方法
        if self.reg_config.method == RegistrationMethod.FEATURE_BASED:
            result = self._register_feature_based(thermal_gray, visible_gray)
        elif self.reg_config.method == RegistrationMethod.INTENSITY_BASED:
            result = self._register_intensity_based(thermal_gray, visible_gray)
        else:
            result = self._register_feature_based(thermal_gray, visible_gray)

        # 更新缓存
        if result.success and result.transform_matrix is not None:
            self._last_transform = result.transform_matrix

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _preprocess_thermal(self, image: np.ndarray) -> np.ndarray:
        """预处理热像"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def _preprocess_visible(self, image: np.ndarray) -> np.ndarray:
        """预处理可见光图像"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        return gray

    def _register_feature_based(
        self,
        thermal_gray: np.ndarray,
        visible_gray: np.ndarray
    ) -> RegistrationResult:
        """基于特征点的配准"""
        if self._detector is None or self._matcher is None:
            return RegistrationResult(
                success=False,
                metadata={"error": "特征检测器未初始化"}
            )

        try:
            # 检测特征点
            kp1, des1 = self._detector.detectAndCompute(thermal_gray, None)
            kp2, des2 = self._detector.detectAndCompute(visible_gray, None)

            if des1 is None or des2 is None:
                return RegistrationResult(
                    success=False,
                    matched_points=0,
                    metadata={"error": "无法提取特征点"}
                )

            # 特征匹配 (KNN + Lowe's ratio test)
            matches = self._matcher.knnMatch(des1, des2, k=2)

            # 筛选好的匹配
            good_matches = []
            for m_list in matches:
                if len(m_list) >= 2:
                    m, n = m_list[0], m_list[1]
                    if m.distance < self.reg_config.match_ratio * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.reg_config.min_match_count:
                return RegistrationResult(
                    success=False,
                    matched_points=len(good_matches),
                    metadata={"error": f"匹配点不足: {len(good_matches)} < {self.reg_config.min_match_count}"}
                )

            # 提取匹配点坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算变换矩阵
            if self.reg_config.use_homography:
                M, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    cv2.RANSAC,
                    self.reg_config.ransac_reproj_threshold
                )
            else:
                M, mask = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.reg_config.ransac_reproj_threshold
                )

            if M is None:
                return RegistrationResult(
                    success=False,
                    matched_points=len(good_matches),
                    metadata={"error": "无法估计变换矩阵"}
                )

            inliers = int(mask.sum()) if mask is not None else 0

            # 计算重投影误差
            reproj_error = self._compute_reprojection_error(src_pts, dst_pts, M, mask)

            return RegistrationResult(
                success=True,
                transform_matrix=M,
                matched_points=len(good_matches),
                inliers=inliers,
                reprojection_error=reproj_error,
                method_used="feature_based"
            )

        except Exception as e:
            logger.error(f"特征配准失败: {e}")
            return RegistrationResult(
                success=False,
                metadata={"error": str(e)}
            )

    def _register_intensity_based(
        self,
        thermal_gray: np.ndarray,
        visible_gray: np.ndarray
    ) -> RegistrationResult:
        """基于灰度的配准 (ECC算法)"""
        try:
            # 调整尺寸一致
            h1, w1 = thermal_gray.shape
            h2, w2 = visible_gray.shape

            if (h1, w1) != (h2, w2):
                visible_gray = cv2.resize(visible_gray, (w1, h1))

            # 初始变换矩阵
            warp_mode = cv2.MOTION_HOMOGRAPHY if self.reg_config.use_homography else cv2.MOTION_AFFINE

            if self.reg_config.use_homography:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            # ECC配准
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

            try:
                _, warp_matrix = cv2.findTransformECC(
                    thermal_gray, visible_gray,
                    warp_matrix, warp_mode, criteria
                )

                return RegistrationResult(
                    success=True,
                    transform_matrix=warp_matrix,
                    method_used="intensity_based_ecc"
                )

            except cv2.error:
                return RegistrationResult(
                    success=False,
                    metadata={"error": "ECC配准收敛失败"}
                )

        except Exception as e:
            logger.error(f"灰度配准失败: {e}")
            return RegistrationResult(
                success=False,
                metadata={"error": str(e)}
            )

    def _compute_reprojection_error(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        M: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> float:
        """计算重投影误差"""
        if M.shape[0] == 3:
            # 透视变换
            src_pts_h = np.concatenate([src_pts, np.ones((len(src_pts), 1, 1))], axis=2)
            projected = np.matmul(M, src_pts_h.reshape(-1, 3).T).T
            projected = projected[:, :2] / projected[:, 2:3]
        else:
            # 仿射变换
            src_pts_flat = src_pts.reshape(-1, 2)
            ones = np.ones((len(src_pts_flat), 1))
            src_h = np.hstack([src_pts_flat, ones])
            projected = np.matmul(M, src_h.T).T

        dst_pts_flat = dst_pts.reshape(-1, 2)
        errors = np.linalg.norm(projected - dst_pts_flat, axis=1)

        if mask is not None:
            mask_flat = mask.flatten() > 0
            errors = errors[mask_flat]

        return float(np.mean(errors)) if len(errors) > 0 else 0.0

    def warp_thermal(
        self,
        thermal_image: np.ndarray,
        visible_image: np.ndarray,
        transform_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        将热像变换到可见光坐标系

        Args:
            thermal_image: 红外热像
            visible_image: 可见光图像 (用于获取目标尺寸)
            transform_matrix: 变换矩阵 (None时使用缓存)

        Returns:
            变换后的热像
        """
        if not CV2_AVAILABLE:
            return thermal_image

        M = transform_matrix if transform_matrix is not None else self._last_transform
        if M is None:
            return thermal_image

        h, w = visible_image.shape[:2]

        if M.shape[0] == 3:
            # 透视变换
            warped = cv2.warpPerspective(thermal_image, M, (w, h))
        else:
            # 仿射变换
            warped = cv2.warpAffine(thermal_image, M, (w, h))

        return warped

    def fuse(
        self,
        thermal_image: np.ndarray,
        visible_image: np.ndarray,
        transform_matrix: Optional[np.ndarray] = None
    ) -> FusionResult:
        """
        融合红外和可见光图像

        Args:
            thermal_image: 红外热像
            visible_image: 可见光图像
            transform_matrix: 变换矩阵 (None时先进行配准)

        Returns:
            融合结果
        """
        start_time = time.perf_counter()

        if not CV2_AVAILABLE:
            return FusionResult(
                metadata={"error": "OpenCV不可用"}
            )

        # 配准热像
        if transform_matrix is None:
            reg_result = self.register(thermal_image, visible_image)
            if reg_result.success:
                transform_matrix = reg_result.transform_matrix
            else:
                # 配准失败，使用原始热像
                thermal_aligned = cv2.resize(
                    thermal_image,
                    (visible_image.shape[1], visible_image.shape[0])
                )
                transform_matrix = None
        else:
            thermal_aligned = None

        if thermal_aligned is None and transform_matrix is not None:
            thermal_aligned = self.warp_thermal(thermal_image, visible_image, transform_matrix)
        elif thermal_aligned is None:
            thermal_aligned = cv2.resize(
                thermal_image,
                (visible_image.shape[1], visible_image.shape[0])
            )

        # 增强热像
        if self.fusion_config.enhance_thermal:
            thermal_aligned = self._enhance_thermal(thermal_aligned)

        # 融合
        fused = self._apply_fusion(visible_image, thermal_aligned)

        # 检测热点
        hotspot_mask, hotspots = self._detect_hotspots(thermal_aligned)

        return FusionResult(
            fused_image=fused,
            thermal_aligned=thermal_aligned,
            hotspot_mask=hotspot_mask,
            hotspots=hotspots,
            processing_time_ms=(time.perf_counter() - start_time) * 1000
        )

    def _enhance_thermal(self, thermal: np.ndarray) -> np.ndarray:
        """增强热像"""
        if len(thermal.shape) == 3:
            gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
        else:
            gray = thermal

        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 转为彩色图 (热力图)
        colormap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)

        return colormap

    def _apply_fusion(
        self,
        visible: np.ndarray,
        thermal: np.ndarray
    ) -> np.ndarray:
        """应用融合方法"""
        method = self.fusion_config.method

        # 确保尺寸一致
        if visible.shape[:2] != thermal.shape[:2]:
            thermal = cv2.resize(thermal, (visible.shape[1], visible.shape[0]))

        # 确保通道数一致
        if len(visible.shape) == 2:
            visible = cv2.cvtColor(visible, cv2.COLOR_GRAY2BGR)
        if len(thermal.shape) == 2:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)

        if method == FusionMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(visible, thermal)
        elif method == FusionMethod.LAPLACIAN_PYRAMID:
            return self._laplacian_pyramid_fusion(visible, thermal)
        else:
            return self._weighted_average_fusion(visible, thermal)

    def _weighted_average_fusion(
        self,
        visible: np.ndarray,
        thermal: np.ndarray
    ) -> np.ndarray:
        """加权平均融合"""
        alpha = self.fusion_config.thermal_weight
        beta = self.fusion_config.visible_weight

        fused = cv2.addWeighted(
            visible, beta,
            thermal, alpha,
            0
        )

        return fused

    def _laplacian_pyramid_fusion(
        self,
        visible: np.ndarray,
        thermal: np.ndarray
    ) -> np.ndarray:
        """拉普拉斯金字塔融合"""
        levels = self.fusion_config.pyramid_levels

        # 构建高斯金字塔
        gp_vis = [visible]
        gp_thm = [thermal]

        for _ in range(levels):
            gp_vis.append(cv2.pyrDown(gp_vis[-1]))
            gp_thm.append(cv2.pyrDown(gp_thm[-1]))

        # 构建拉普拉斯金字塔
        lp_vis = [gp_vis[-1]]
        lp_thm = [gp_thm[-1]]

        for i in range(levels - 1, -1, -1):
            size = (gp_vis[i].shape[1], gp_vis[i].shape[0])

            vis_up = cv2.pyrUp(gp_vis[i + 1], dstsize=size)
            thm_up = cv2.pyrUp(gp_thm[i + 1], dstsize=size)

            lp_vis.append(cv2.subtract(gp_vis[i], vis_up))
            lp_thm.append(cv2.subtract(gp_thm[i], thm_up))

        # 融合各层
        lp_fused = []
        alpha = self.fusion_config.thermal_weight

        for vis_lap, thm_lap in zip(lp_vis, lp_thm):
            fused_lap = cv2.addWeighted(vis_lap, 1 - alpha, thm_lap, alpha, 0)
            lp_fused.append(fused_lap)

        # 重建
        fused = lp_fused[0]
        for i in range(1, len(lp_fused)):
            size = (lp_fused[i].shape[1], lp_fused[i].shape[0])
            fused = cv2.add(cv2.pyrUp(fused, dstsize=size), lp_fused[i])

        return fused

    def _detect_hotspots(
        self,
        thermal: np.ndarray,
        threshold_percentile: float = 95
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        检测热点区域

        Args:
            thermal: 热像
            threshold_percentile: 热点阈值百分位

        Returns:
            (热点掩码, 热点列表)
        """
        if len(thermal.shape) == 3:
            gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
        else:
            gray = thermal

        # 计算阈值
        threshold = np.percentile(gray, threshold_percentile)

        # 创建掩码
        _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hotspots = []
        h, w = gray.shape

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # 过滤小区域
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            # 计算区域内最高温度
            roi = gray[y:y+bh, x:x+bw]
            max_temp = float(np.max(roi))
            mean_temp = float(np.mean(roi))

            hotspots.append({
                "bbox": {
                    "x": x / w,
                    "y": y / h,
                    "width": bw / w,
                    "height": bh / h
                },
                "area": int(area),
                "max_intensity": max_temp,
                "mean_intensity": mean_temp,
                "centroid": {
                    "x": (x + bw / 2) / w,
                    "y": (y + bh / 2) / h
                }
            })

        return mask, hotspots

    def set_calibration(self, calibration_matrix: np.ndarray):
        """设置标定矩阵"""
        self._calibration_matrix = calibration_matrix

    def get_cached_transform(self) -> Optional[np.ndarray]:
        """获取缓存的变换矩阵"""
        return self._last_transform


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "RegistrationMethod",
    "FeatureDetectorType",
    "FusionMethod",
    "RegistrationConfig",
    "FusionConfig",
    "RegistrationResult",
    "FusionResult",
    "ThermalVisibleRegistration"
]
