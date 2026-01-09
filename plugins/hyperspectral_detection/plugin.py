#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高光谱缺陷识别插件 (Hyperspectral Defect Detection Plugin)
输变电站全自动AI巡检方案 - 丰富模型与插件开发

功能:
1. 高光谱图像预处理
2. 过热检测 (Overheating)
3. 腐蚀检测 (Corrosion)
4. 绝缘老化检测 (Insulation Aging)
5. 油泄漏检测 (Oil Leakage)
6. 表面损伤检测 (Surface Damage)

作者: AI巡检系统
版本: 1.0.0
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

# 导入插件状态混入类
try:
    from platform_core.plugin_mixin import PluginStatusMixin, PluginStatus
except ImportError:
    from enum import Enum

    class PluginStatus(str, Enum):
        UNLOADED = "unloaded"
        LOADING = "loading"
        READY = "ready"
        RUNNING = "running"
        ERROR = "error"
        DISABLED = "disabled"

    class PluginStatusMixin:
        def __init_status__(self):
            self._status = PluginStatus.UNLOADED
            self._last_error = ""

        @property
        def status(self):
            return getattr(self, '_status', PluginStatus.UNLOADED)

        @status.setter
        def status(self, value):
            self._status = value

        def set_status(self, status, error: str = ""):
            if isinstance(status, str):
                try:
                    status = PluginStatus(status)
                except ValueError:
                    status = PluginStatus.ERROR
            self._status = status
            if error:
                self._last_error = error

        def get_plugin_status(self) -> dict:
            return {
                'plugin_id': getattr(self, 'PLUGIN_ID', 'unknown'),
                'name': getattr(self, 'PLUGIN_NAME', 'Unknown'),
                'version': getattr(self, 'PLUGIN_VERSION', '0.0.0'),
                'status': self._status.value if hasattr(self, '_status') else 'unknown',
                'initialized': getattr(self, '_is_initialized', False),
                'last_error': getattr(self, '_last_error', '')
            }

logger = logging.getLogger(__name__)


# =============================================================================
# 配置数据类
# =============================================================================
@dataclass
class HyperspectralConfig:
    """高光谱检测配置"""
    # 光谱配置
    num_bands: int = 224
    wavelength_min: float = 400  # nm
    wavelength_max: float = 2500  # nm
    
    # 空间分辨率
    spatial_resolution: Tuple[int, int] = (256, 256)
    
    # 预处理配置
    dark_current_subtraction: bool = True
    flat_field_correction: bool = True
    atmospheric_correction: bool = True
    
    # 特征提取配置
    feature_extraction: str = "pca"  # pca, ica, mnf
    pca_components: int = 30
    
    # 检测配置
    confidence_threshold: float = 0.6
    min_defect_area: int = 100  # 最小缺陷面积 (像素)
    
    # 缺陷类型
    defect_types: List[str] = field(default_factory=lambda: [
        "normal", "overheating", "corrosion",
        "insulation_aging", "oil_leakage", "surface_damage"
    ])
    
    # 模型配置
    model_id: str = "hyperspectral_defect"


# =============================================================================
# 缺陷类型
# =============================================================================
class DefectType:
    """缺陷类型"""
    NORMAL = "normal"
    OVERHEATING = "overheating"           # 过热
    CORROSION = "corrosion"               # 腐蚀
    INSULATION_AGING = "insulation_aging" # 绝缘老化
    OIL_LEAKAGE = "oil_leakage"          # 油泄漏
    SURFACE_DAMAGE = "surface_damage"     # 表面损伤
    PAINT_PEELING = "paint_peeling"       # 油漆剥落
    MOISTURE_INGRESS = "moisture_ingress" # 受潮
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [
            cls.NORMAL, cls.OVERHEATING, cls.CORROSION,
            cls.INSULATION_AGING, cls.OIL_LEAKAGE, cls.SURFACE_DAMAGE,
            cls.PAINT_PEELING, cls.MOISTURE_INGRESS
        ]
    
    @classmethod
    def get_severity(cls, defect_type: str) -> str:
        """获取缺陷严重程度"""
        severity_map = {
            cls.NORMAL: "info",
            cls.OVERHEATING: "critical",
            cls.CORROSION: "warning",
            cls.INSULATION_AGING: "alarm",
            cls.OIL_LEAKAGE: "alarm",
            cls.SURFACE_DAMAGE: "warning",
            cls.PAINT_PEELING: "attention",
            cls.MOISTURE_INGRESS: "warning"
        }
        return severity_map.get(defect_type, "info")


# =============================================================================
# 高光谱缺陷识别插件
# =============================================================================
class HyperspectralDetectionPlugin(PluginStatusMixin):
    """
    高光谱缺陷识别插件

    功能:
    1. 高光谱图像预处理
    2. 特征提取 (PCA/ICA/MNF)
    3. 多类型缺陷检测
    4. 缺陷分割与定位
    """

    PLUGIN_ID = "hyperspectral_detection"
    PLUGIN_NAME = "高光谱缺陷识别"
    PLUGIN_VERSION = "1.0.0"

    def __init__(self, manifest=None, plugin_dir=None):
        """
        初始化高光谱检测插件

        Args:
            manifest: 插件清单 (PluginManifest)
            plugin_dir: 插件目录 (Path)
        """
        # 初始化状态管理
        self.__init_status__()

        self.manifest = manifest
        self.plugin_dir = plugin_dir

        # 从 manifest 获取配置或使用默认值
        config = {}
        if manifest and hasattr(manifest, 'config_schema'):
            config = manifest.config_schema or {}
        self.config = HyperspectralConfig(**config)
        self._model_registry = None
        self._preprocessor = None
        self._is_initialized = False
    
    def init(self, model_registry=None) -> bool:
        """初始化插件"""
        try:
            self._model_registry = model_registry
            
            # 初始化预处理器
            self._preprocessor = HyperspectralPreprocessor(self.config)
            
            self._is_initialized = True
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}")
            return False
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理高光谱图像
        
        Args:
            inputs: {
                "hyperspectral_image": np.ndarray,  # (H, W, C) 或 (C, H, W)
                "wavelengths": List[float],         # 各波段波长 (可选)
                "device_id": str,                   # 设备ID (可选)
                "roi_id": str,                      # ROI标识 (可选)
                "timestamp": float                  # 时间戳 (可选)
            }
        
        Returns:
            {
                "success": bool,
                "defects": List[Dict],              # 检测到的缺陷列表
                "defect_map": np.ndarray,           # 缺陷分割图
                "spectral_analysis": Dict,          # 光谱分析结果
                "recommendations": List[str]
            }
        """
        if not self._is_initialized:
            return {
                "success": False,
                "error": "插件未初始化"
            }
        
        try:
            image = inputs.get("hyperspectral_image")
            if image is None:
                return {
                    "success": False,
                    "error": "缺少高光谱图像输入"
                }
            
            # 1. 预处理
            processed_image = self._preprocessor.preprocess(image)
            
            # 2. 特征提取
            features = self._extract_features(processed_image)
            
            # 3. 缺陷检测
            if self._model_registry and self._model_registry.is_model_loaded(self.config.model_id):
                detection_result = self._detect_by_deep_learning(features)
            else:
                detection_result = self._detect_by_traditional(processed_image, features)
            
            # 4. 光谱分析
            spectral_analysis = self._analyze_spectrum(
                processed_image,
                inputs.get("wavelengths")
            )
            
            # 5. 生成建议
            recommendations = self._generate_recommendations(detection_result)
            
            return {
                "success": True,
                "defects": detection_result.get("defects", []),
                "defect_map": detection_result.get("defect_map"),
                "overall_status": detection_result.get("overall_status", "normal"),
                "spectral_analysis": spectral_analysis,
                "feature_statistics": self._compute_feature_statistics(features),
                "recommendations": recommendations,
                "metadata": {
                    "device_id": inputs.get("device_id"),
                    "roi_id": inputs.get("roi_id"),
                    "timestamp": inputs.get("timestamp"),
                    "image_shape": image.shape
                }
            }
            
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """提取光谱特征"""
        # 确保图像格式为 (H, W, C)
        if image.ndim == 3 and image.shape[0] < image.shape[2]:
            image = np.transpose(image, (1, 2, 0))
        
        h, w, c = image.shape
        
        # 展平空间维度
        pixels = image.reshape(-1, c)
        
        if self.config.feature_extraction == "pca":
            features = self._pca_transform(pixels, self.config.pca_components)
        else:
            # 默认使用PCA
            features = self._pca_transform(pixels, self.config.pca_components)
        
        # 恢复空间维度
        features = features.reshape(h, w, -1)
        
        return features
    
    def _pca_transform(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """PCA降维"""
        # 中心化
        mean = np.mean(data, axis=0)
        centered = data - mean
        
        # 协方差矩阵
        cov = np.cov(centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # 选择前n个主成分
        n_components = min(n_components, eigenvectors.shape[1])
        components = eigenvectors[:, :n_components]
        
        # 投影
        transformed = np.dot(centered, components)
        
        return transformed
    
    def _detect_by_deep_learning(self, features: np.ndarray) -> Dict[str, Any]:
        """使用深度学习模型检测"""
        try:
            # 准备输入
            input_data = features.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, H, W)
            input_data = input_data.astype(np.float32)
            
            # 推理
            result = self._model_registry.infer(
                self.config.model_id,
                {"input": input_data}
            )
            
            if not result.get("success"):
                return self._detect_by_traditional(None, features)
            
            outputs = result.get("outputs", {})
            
            # 解析输出
            defect_map = outputs.get("defect_map", outputs.get("output"))
            class_probs = outputs.get("class_probs")
            
            if defect_map is not None:
                defect_map = np.array(defect_map).squeeze()
            
            # 解析缺陷
            defects = self._parse_defect_map(defect_map, class_probs)
            
            return {
                "defects": defects,
                "defect_map": defect_map,
                "overall_status": self._determine_overall_status(defects)
            }
            
        except Exception as e:
            logger.warning(f"深度学习检测失败: {e}")
            return self._detect_by_traditional(None, features)
    
    def _detect_by_traditional(self, image: np.ndarray,
                               features: np.ndarray) -> Dict[str, Any]:
        """使用传统方法检测"""
        defects = []
        h, w = features.shape[:2]
        defect_map = np.zeros((h, w), dtype=np.int32)
        
        # 基于光谱特征的简单检测
        
        # 1. 过热检测 (热红外波段能量异常高)
        # 假设最后几个通道对应近红外
        if features.shape[2] >= 10:
            nir_features = features[:, :, -10:]
            nir_energy = np.mean(nir_features, axis=2)
            
            # 使用自适应阈值
            threshold = np.mean(nir_energy) + 2 * np.std(nir_energy)
            hot_spots = nir_energy > threshold
            
            if np.any(hot_spots):
                defect_area = np.sum(hot_spots)
                if defect_area > self.config.min_defect_area:
                    defects.append({
                        "type": DefectType.OVERHEATING,
                        "severity": DefectType.get_severity(DefectType.OVERHEATING),
                        "area_pixels": int(defect_area),
                        "area_percentage": float(defect_area / (h * w) * 100),
                        "confidence": float(np.mean(nir_energy[hot_spots]) / threshold),
                        "location": self._get_defect_location(hot_spots)
                    })
                    defect_map[hot_spots] = 1
        
        # 2. 腐蚀检测 (特定波段比值)
        if features.shape[2] >= 20:
            # 铁锈在某些波段有特征吸收
            ratio = features[:, :, 10] / (features[:, :, 5] + 1e-8)
            rust_mask = ratio > 1.5
            
            if np.any(rust_mask):
                defect_area = np.sum(rust_mask)
                if defect_area > self.config.min_defect_area:
                    defects.append({
                        "type": DefectType.CORROSION,
                        "severity": DefectType.get_severity(DefectType.CORROSION),
                        "area_pixels": int(defect_area),
                        "area_percentage": float(defect_area / (h * w) * 100),
                        "confidence": float(min(np.mean(ratio[rust_mask]) / 2, 1.0)),
                        "location": self._get_defect_location(rust_mask)
                    })
                    defect_map[rust_mask] = 2
        
        # 3. 异常检测 (基于PCA重建误差)
        if features.shape[2] >= 5:
            # 使用前几个主成分重建
            reconstruction = self._pca_reconstruct(features, 5)
            error = np.mean(np.abs(features - reconstruction), axis=2)
            
            threshold = np.mean(error) + 3 * np.std(error)
            anomaly_mask = error > threshold
            
            if np.any(anomaly_mask):
                defect_area = np.sum(anomaly_mask)
                if defect_area > self.config.min_defect_area:
                    defects.append({
                        "type": DefectType.SURFACE_DAMAGE,
                        "severity": DefectType.get_severity(DefectType.SURFACE_DAMAGE),
                        "area_pixels": int(defect_area),
                        "area_percentage": float(defect_area / (h * w) * 100),
                        "confidence": float(np.mean(error[anomaly_mask]) / threshold),
                        "location": self._get_defect_location(anomaly_mask)
                    })
                    defect_map[anomaly_mask & (defect_map == 0)] = 3
        
        return {
            "defects": defects,
            "defect_map": defect_map,
            "overall_status": self._determine_overall_status(defects)
        }
    
    def _pca_reconstruct(self, features: np.ndarray, n_components: int) -> np.ndarray:
        """PCA重建 (用于异常检测)"""
        h, w, c = features.shape
        pixels = features.reshape(-1, c)
        
        # 中心化
        mean = np.mean(pixels, axis=0)
        centered = pixels - mean
        
        # 协方差矩阵
        cov = np.cov(centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # 投影到低维
        components = eigenvectors[:, :n_components]
        projected = np.dot(centered, components)
        
        # 反投影
        reconstructed = np.dot(projected, components.T) + mean
        
        return reconstructed.reshape(h, w, c)
    
    def _get_defect_location(self, mask: np.ndarray) -> Dict[str, Any]:
        """获取缺陷位置信息"""
        y_coords, x_coords = np.where(mask)
        
        if len(x_coords) == 0:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        
        return {
            "x": int(np.min(x_coords)),
            "y": int(np.min(y_coords)),
            "width": int(np.max(x_coords) - np.min(x_coords)),
            "height": int(np.max(y_coords) - np.min(y_coords)),
            "center_x": int(np.mean(x_coords)),
            "center_y": int(np.mean(y_coords))
        }
    
    def _parse_defect_map(self, defect_map: np.ndarray,
                          class_probs: np.ndarray = None) -> List[Dict]:
        """解析缺陷分割图"""
        if defect_map is None:
            return []
        
        defects = []
        defect_types = self.config.defect_types
        
        for class_idx in range(1, len(defect_types)):  # 跳过normal
            mask = defect_map == class_idx
            
            if np.any(mask):
                defect_area = np.sum(mask)
                if defect_area > self.config.min_defect_area:
                    defect_type = defect_types[class_idx]
                    
                    confidence = 1.0
                    if class_probs is not None:
                        confidence = float(np.mean(class_probs[class_idx, mask]))
                    
                    defects.append({
                        "type": defect_type,
                        "severity": DefectType.get_severity(defect_type),
                        "area_pixels": int(defect_area),
                        "area_percentage": float(defect_area / defect_map.size * 100),
                        "confidence": confidence,
                        "location": self._get_defect_location(mask)
                    })
        
        return defects
    
    def _determine_overall_status(self, defects: List[Dict]) -> str:
        """确定整体状态"""
        if not defects:
            return "normal"
        
        severities = [d.get("severity", "info") for d in defects]
        
        if "critical" in severities:
            return "critical"
        if "alarm" in severities:
            return "alarm"
        if "warning" in severities:
            return "warning"
        if "attention" in severities:
            return "attention"
        
        return "normal"
    
    def _analyze_spectrum(self, image: np.ndarray,
                          wavelengths: List[float] = None) -> Dict[str, Any]:
        """光谱分析"""
        if image.ndim == 3 and image.shape[0] < image.shape[2]:
            image = np.transpose(image, (1, 2, 0))
        
        h, w, c = image.shape
        
        # 计算平均光谱
        mean_spectrum = np.mean(image.reshape(-1, c), axis=0)
        std_spectrum = np.std(image.reshape(-1, c), axis=0)
        
        # 生成波长轴
        if wavelengths is None:
            wavelengths = np.linspace(
                self.config.wavelength_min,
                self.config.wavelength_max,
                c
            ).tolist()
        
        # 找光谱峰值
        peaks = self._find_spectral_peaks(mean_spectrum)
        
        return {
            "wavelengths": wavelengths,
            "mean_spectrum": mean_spectrum.tolist(),
            "std_spectrum": std_spectrum.tolist(),
            "spectral_peaks": [
                {"index": int(p), "wavelength": float(wavelengths[p]), "value": float(mean_spectrum[p])}
                for p in peaks
            ],
            "spectral_range": {
                "min_reflectance": float(np.min(mean_spectrum)),
                "max_reflectance": float(np.max(mean_spectrum)),
                "mean_reflectance": float(np.mean(mean_spectrum))
            }
        }
    
    def _find_spectral_peaks(self, spectrum: np.ndarray,
                             num_peaks: int = 5) -> List[int]:
        """查找光谱峰值"""
        peaks = []
        
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                peaks.append(i)
        
        # 按幅值排序
        peaks.sort(key=lambda x: spectrum[x], reverse=True)
        
        return peaks[:num_peaks]
    
    def _compute_feature_statistics(self, features: np.ndarray) -> Dict[str, Any]:
        """计算特征统计"""
        return {
            "num_components": features.shape[2] if features.ndim == 3 else features.shape[1],
            "mean_per_component": np.mean(features, axis=(0, 1)).tolist() if features.ndim == 3 else np.mean(features, axis=0).tolist(),
            "std_per_component": np.std(features, axis=(0, 1)).tolist() if features.ndim == 3 else np.std(features, axis=0).tolist()
        }
    
    def _generate_recommendations(self, detection_result: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        defects = detection_result.get("defects", [])
        
        for defect in defects:
            defect_type = defect.get("type")
            severity = defect.get("severity")
            
            if defect_type == DefectType.OVERHEATING:
                recommendations.append(
                    f"检测到过热区域 ({severity})，建议进行红外热成像复核并检查负荷情况"
                )
            elif defect_type == DefectType.CORROSION:
                recommendations.append(
                    f"检测到腐蚀区域 ({severity})，建议安排防腐处理"
                )
            elif defect_type == DefectType.INSULATION_AGING:
                recommendations.append(
                    f"检测到绝缘老化 ({severity})，建议进行绝缘电阻测试"
                )
            elif defect_type == DefectType.OIL_LEAKAGE:
                recommendations.append(
                    f"检测到油泄漏迹象 ({severity})，建议检查密封件"
                )
            elif defect_type == DefectType.SURFACE_DAMAGE:
                recommendations.append(
                    f"检测到表面损伤 ({severity})，建议人工复核"
                )
        
        if not recommendations:
            recommendations.append("高光谱检测未发现明显异常，建议保持定期监测")
        
        return recommendations
    
    def shutdown(self) -> bool:
        """关闭插件"""
        self._is_initialized = False
        logger.info(f"[{self.PLUGIN_NAME}] 已关闭")
        return True
    
    def infer(self, frame, rois, context):
        """实现BasePlugin抽象方法 - 执行推理"""
        return []

    def postprocess(self, results, rules):
        """实现BasePlugin抽象方法 - 后处理"""
        return []

    def healthcheck(self):
        """实现BasePlugin抽象方法 - 健康检查"""
        try:
            from platform_core.plugin_manager.base import HealthStatus
            return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else "未初始化")
        except ImportError:
            return {"healthy": self._is_initialized, "message": "OK" if self._is_initialized else "未初始化"}

    @property
    def plugin_info(self) -> Dict:
        return {
            "id": self.PLUGIN_ID,
            "name": self.PLUGIN_NAME,
            "version": self.PLUGIN_VERSION,
            "description": "基于高光谱成像的缺陷检测，支持过热、腐蚀、绝缘老化等多种缺陷类型",
            "capabilities": [
                "hyperspectral_imaging",
                "overheating_detection",
                "corrosion_detection",
                "insulation_aging_detection",
                "spectral_analysis"
            ],
            "defect_types": DefectType.get_all()
        }


# =============================================================================
# 高光谱预处理器
# =============================================================================
class HyperspectralPreprocessor:
    """高光谱图像预处理器"""
    
    def __init__(self, config: HyperspectralConfig):
        self.config = config
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理高光谱图像
        
        Args:
            image: 原始高光谱图像
        
        Returns:
            预处理后的图像
        """
        # 确保float类型
        image = image.astype(np.float32)
        
        # 暗电流校正
        if self.config.dark_current_subtraction:
            image = self._dark_current_correction(image)
        
        # 平场校正
        if self.config.flat_field_correction:
            image = self._flat_field_correction(image)
        
        # 归一化
        image = self._normalize(image)
        
        return image
    
    def _dark_current_correction(self, image: np.ndarray) -> np.ndarray:
        """暗电流校正"""
        # 估计暗电流 (使用最小值)
        dark = np.percentile(image, 1, axis=(0, 1) if image.ndim == 3 else 0)
        return image - dark
    
    def _flat_field_correction(self, image: np.ndarray) -> np.ndarray:
        """平场校正"""
        # 估计平场 (使用最大值)
        flat = np.percentile(image, 99, axis=(0, 1) if image.ndim == 3 else 0)
        return image / (flat + 1e-8)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化到[0, 1]"""
        min_val = np.min(image)
        max_val = np.max(image)
        
        return (image - min_val) / (max_val - min_val + 1e-8)
