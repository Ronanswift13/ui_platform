#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高光谱/短波红外缺陷检测插件
Hyperspectral/SWIR Defect Detection Plugin

功能：
1. 多光谱图像预处理与特征提取
2. 光谱特征分析（反射率、吸收特征）
3. 缺陷检测（腐蚀、涂层脱落、瓷瓶污染）
4. 深度学习与传统方法混合检测
5. 缺陷定位与严重程度评估

支持的缺陷类型：
- 油漆脱落 (Paint Peeling)
- 金属锈蚀 (Metal Corrosion)
- 瓷瓶污染 (Porcelain Pollution)
- 绝缘老化 (Insulation Aging)
- 热异常 (Thermal Anomaly)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class DefectType(Enum):
    """缺陷类型枚举"""
    PAINT_PEELING = "paint_peeling"           # 油漆脱落
    METAL_CORROSION = "metal_corrosion"       # 金属锈蚀
    PORCELAIN_POLLUTION = "porcelain_pollution"  # 瓷瓶污染
    INSULATION_AGING = "insulation_aging"     # 绝缘老化
    THERMAL_ANOMALY = "thermal_anomaly"       # 热异常
    OIL_LEAKAGE = "oil_leakage"               # 油渍泄漏
    CRACK = "crack"                           # 裂纹
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """严重程度等级"""
    NORMAL = 0       # 正常
    ATTENTION = 1    # 关注
    WARNING = 2      # 警告
    CRITICAL = 3     # 严重
    EMERGENCY = 4    # 紧急


@dataclass
class SpectralBand:
    """光谱波段配置"""
    name: str
    center_wavelength: float  # nm
    bandwidth: float          # nm
    weight: float = 1.0       # 检测权重
    
    
@dataclass
class DefectResult:
    """缺陷检测结果"""
    defect_type: DefectType
    confidence: float
    severity: SeverityLevel
    location: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    spectral_signature: Optional[np.ndarray] = None
    affected_area_ratio: float = 0.0
    description: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HyperspectralConfig:
    """高光谱检测配置"""
    # 光谱波段配置
    bands: List[SpectralBand] = field(default_factory=list)
    
    # 检测阈值
    detection_threshold: float = 0.5
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "attention": 0.3,
        "warning": 0.5,
        "critical": 0.7,
        "emergency": 0.9
    })
    
    # 预处理参数
    normalize_method: str = "minmax"  # minmax, zscore, snv
    smooth_window: int = 5
    
    # 特征提取参数
    use_pca: bool = True
    n_components: int = 10
    
    # 模型配置
    model_name: str = "hyperspectral_defect"
    use_deep_learning: bool = True
    
    def __post_init__(self):
        if not self.bands:
            # 默认波段配置（可见光+近红外+短波红外）
            self.bands = [
                SpectralBand("Blue", 450, 50, 0.8),
                SpectralBand("Green", 550, 50, 1.0),
                SpectralBand("Red", 650, 50, 1.0),
                SpectralBand("RedEdge", 720, 30, 1.2),
                SpectralBand("NIR", 850, 50, 1.0),
                SpectralBand("SWIR1", 1600, 100, 1.5),
                SpectralBand("SWIR2", 2200, 100, 1.5),
            ]


class SpectralFeatureExtractor:
    """
    光谱特征提取器
    
    提取用于缺陷检测的光谱特征：
    - 植被指数类特征（用于检测有机物污染）
    - 矿物指数（用于检测金属氧化物/锈蚀）
    - 统计特征
    - 光谱导数特征
    """
    
    def __init__(self, bands: List[SpectralBand]):
        self.bands = bands
        self.band_indices = {b.name: i for i, b in enumerate(bands)}
        
    def extract_features(self, hyperspectral_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取光谱特征
        
        Args:
            hyperspectral_image: 高光谱图像 [H, W, C] 或 [C, H, W]
            
        Returns:
            特征字典，包含各类光谱指数和统计特征
        """
        # 确保格式为 [H, W, C]
        if hyperspectral_image.ndim == 3:
            if hyperspectral_image.shape[0] < hyperspectral_image.shape[2]:
                hyperspectral_image = np.transpose(hyperspectral_image, (1, 2, 0))
        
        features = {}
        
        # 1. 光谱指数特征
        features.update(self._compute_spectral_indices(hyperspectral_image))
        
        # 2. 统计特征
        features.update(self._compute_statistical_features(hyperspectral_image))
        
        # 3. 导数特征
        features.update(self._compute_derivative_features(hyperspectral_image))
        
        # 4. 纹理特征（简化版）
        features.update(self._compute_texture_features(hyperspectral_image))
        
        return features
    
    def _get_band(self, image: np.ndarray, band_name: str) -> Optional[np.ndarray]:
        """获取指定波段的图像"""
        if band_name in self.band_indices:
            idx = self.band_indices[band_name]
            if idx < image.shape[2]:
                return image[:, :, idx].astype(np.float32)
        return None
    
    def _safe_divide(self, a: np.ndarray, b: np.ndarray, 
                     fill_value: float = 0.0) -> np.ndarray:
        """安全除法，避免除零"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b)
            result[~np.isfinite(result)] = fill_value
        return result
    
    def _compute_spectral_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """计算光谱指数"""
        indices = {}
        
        # 获取各波段
        red = self._get_band(image, "Red")
        green = self._get_band(image, "Green")
        blue = self._get_band(image, "Blue")
        nir = self._get_band(image, "NIR")
        swir1 = self._get_band(image, "SWIR1")
        swir2 = self._get_band(image, "SWIR2")
        
        # NDVI - 用于检测植被/有机污染
        if red is not None and nir is not None:
            indices["NDVI"] = self._safe_divide(nir - red, nir + red)
        
        # NDWI - 用于检测水分/油渍
        if green is not None and nir is not None:
            indices["NDWI"] = self._safe_divide(green - nir, green + nir)
        
        # 铁氧化物指数 (Iron Oxide Index) - 用于检测锈蚀
        if red is not None and blue is not None:
            indices["FeOx"] = self._safe_divide(red, blue)
        
        # 粘土矿物指数 - 用于检测瓷瓶污染
        if swir1 is not None and swir2 is not None:
            indices["Clay"] = self._safe_divide(swir1, swir2)
        
        # 热惯量指数（简化版）
        if nir is not None and swir1 is not None:
            indices["ThermalInertia"] = self._safe_divide(nir - swir1, nir + swir1)
        
        # 归一化差异指数 - 红边
        red_edge = self._get_band(image, "RedEdge")
        if red is not None and red_edge is not None:
            indices["NDRE"] = self._safe_divide(red_edge - red, red_edge + red)
        
        # 亮度指数
        if red is not None and green is not None and blue is not None:
            indices["Brightness"] = (red + green + blue) / 3.0
        
        # SWIR差异指数 - 用于检测涂层变化
        if swir1 is not None and swir2 is not None:
            indices["SWIRDiff"] = swir1 - swir2
        
        return indices
    
    def _compute_statistical_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """计算统计特征"""
        features = {}
        
        # 光谱均值
        features["SpectralMean"] = np.mean(image, axis=2)
        
        # 光谱标准差
        features["SpectralStd"] = np.std(image, axis=2)
        
        # 光谱范围
        features["SpectralRange"] = np.ptp(image, axis=2)
        
        # 光谱偏度
        mean = np.mean(image, axis=2, keepdims=True)
        std = np.std(image, axis=2, keepdims=True) + 1e-8
        features["SpectralSkewness"] = np.mean(((image - mean) / std) ** 3, axis=2)
        
        # 光谱峰度
        features["SpectralKurtosis"] = np.mean(((image - mean) / std) ** 4, axis=2) - 3
        
        return features
    
    def _compute_derivative_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """计算光谱导数特征"""
        features = {}
        
        # 一阶导数
        first_derivative = np.diff(image, axis=2)
        features["FirstDerivMax"] = np.max(np.abs(first_derivative), axis=2)
        features["FirstDerivMean"] = np.mean(first_derivative, axis=2)
        
        # 二阶导数
        if image.shape[2] > 2:
            second_derivative = np.diff(first_derivative, axis=2)
            features["SecondDerivMax"] = np.max(np.abs(second_derivative), axis=2)
        
        return features
    
    def _compute_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """计算简化的纹理特征"""
        features = {}
        
        # 使用亮度图计算纹理
        brightness = np.mean(image, axis=2)
        
        # 局部对比度（使用简单的拉普拉斯算子近似）
        kernel_size = 3
        padded = np.pad(brightness, kernel_size // 2, mode='reflect')
        
        # 简单边缘强度
        dx = np.abs(padded[1:-1, 2:] - padded[1:-1, :-2])
        dy = np.abs(padded[2:, 1:-1] - padded[:-2, 1:-1])
        features["EdgeStrength"] = np.sqrt(dx ** 2 + dy ** 2)
        
        # 局部方差
        from scipy.ndimage import uniform_filter
        try:
            mean_img = uniform_filter(brightness, size=kernel_size)
            mean_sq = uniform_filter(brightness ** 2, size=kernel_size)
            features["LocalVariance"] = mean_sq - mean_img ** 2
        except ImportError:
            # Fallback: 简单的局部方差
            features["LocalVariance"] = self._simple_local_variance(brightness, kernel_size)
        
        return features
    
    def _simple_local_variance(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """简单局部方差计算（无scipy依赖）"""
        h, w = image.shape
        result = np.zeros_like(image)
        pad = kernel_size // 2
        
        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                window = image[i-pad:i+pad+1, j-pad:j+pad+1]
                result[i, j] = np.var(window)
        
        return result


class SpectralPreprocessor:
    """
    光谱数据预处理器
    
    支持的预处理方法：
    - 归一化（Min-Max, Z-Score, SNV）
    - 平滑（Savitzky-Golay, Moving Average）
    - 基线校正
    - 大气校正（简化版）
    """
    
    def __init__(self, config: HyperspectralConfig):
        self.config = config
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        完整预处理流程
        
        Args:
            image: 原始高光谱图像 [H, W, C]
            
        Returns:
            预处理后的图像
        """
        # 1. 类型转换
        image = image.astype(np.float32)
        
        # 2. 去除无效值
        image = self._handle_invalid_values(image)
        
        # 3. 光谱平滑
        if self.config.smooth_window > 1:
            image = self._smooth_spectrum(image)
        
        # 4. 归一化
        image = self._normalize(image)
        
        return image
    
    def _handle_invalid_values(self, image: np.ndarray) -> np.ndarray:
        """处理无效值"""
        # 替换 NaN 和 Inf
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 裁剪到有效范围
        image = np.clip(image, 0, 65535)  # 假设16位数据
        
        return image
    
    def _smooth_spectrum(self, image: np.ndarray) -> np.ndarray:
        """光谱平滑"""
        window = self.config.smooth_window
        if window < 3:
            return image
            
        # 简单移动平均
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        half_window = window // 2
        for i in range(c):
            start = max(0, i - half_window)
            end = min(c, i + half_window + 1)
            result[:, :, i] = np.mean(image[:, :, start:end], axis=2)
        
        return result
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化"""
        method = self.config.normalize_method
        
        if method == "minmax":
            # Min-Max 归一化
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
                
        elif method == "zscore":
            # Z-Score 标准化
            mean = np.mean(image)
            std = np.std(image) + 1e-8
            image = (image - mean) / std
            
        elif method == "snv":
            # Standard Normal Variate (SNV) - 每个像素独立归一化
            mean = np.mean(image, axis=2, keepdims=True)
            std = np.std(image, axis=2, keepdims=True) + 1e-8
            image = (image - mean) / std
            
        return image


class RuleBasedDefectDetector:
    """
    基于规则的缺陷检测器（回退方案）
    
    使用光谱指数阈值进行缺陷判断
    """
    
    def __init__(self, config: HyperspectralConfig):
        self.config = config
        
        # 各缺陷类型的检测规则
        self.detection_rules = {
            DefectType.METAL_CORROSION: {
                "primary_index": "FeOx",
                "threshold": 1.5,
                "direction": "greater",
                "secondary_indices": [("SpectralStd", 0.1, "greater")],
            },
            DefectType.PAINT_PEELING: {
                "primary_index": "SWIRDiff",
                "threshold": 0.2,
                "direction": "greater",
                "secondary_indices": [("LocalVariance", 0.05, "greater")],
            },
            DefectType.PORCELAIN_POLLUTION: {
                "primary_index": "Clay",
                "threshold": 1.2,
                "direction": "greater",
                "secondary_indices": [("Brightness", 0.3, "less")],
            },
            DefectType.OIL_LEAKAGE: {
                "primary_index": "NDWI",
                "threshold": 0.3,
                "direction": "greater",
                "secondary_indices": [],
            },
            DefectType.THERMAL_ANOMALY: {
                "primary_index": "ThermalInertia",
                "threshold": 0.5,
                "direction": "less",
                "secondary_indices": [],
            },
        }
        
    def detect(self, features: Dict[str, np.ndarray]) -> List[DefectResult]:
        """
        基于规则检测缺陷
        
        Args:
            features: 光谱特征字典
            
        Returns:
            检测到的缺陷列表
        """
        results = []
        
        for defect_type, rules in self.detection_rules.items():
            result = self._check_defect(features, defect_type, rules)
            if result is not None:
                results.append(result)
        
        return results
    
    def _check_defect(self, features: Dict[str, np.ndarray], 
                      defect_type: DefectType,
                      rules: Dict) -> Optional[DefectResult]:
        """检查单个缺陷类型"""
        primary_index = rules["primary_index"]
        
        if primary_index not in features:
            return None
        
        index_image = features[primary_index]
        threshold = rules["threshold"]
        direction = rules["direction"]
        
        # 主指数判断
        if direction == "greater":
            mask = index_image > threshold
        else:
            mask = index_image < threshold
        
        # 次要指数判断
        for sec_index, sec_threshold, sec_direction in rules.get("secondary_indices", []):
            if sec_index in features:
                sec_image = features[sec_index]
                if sec_direction == "greater":
                    sec_mask = sec_image > sec_threshold
                else:
                    sec_mask = sec_image < sec_threshold
                mask = mask & sec_mask
        
        # 计算缺陷面积比例
        affected_ratio = np.sum(mask) / mask.size
        
        if affected_ratio < 0.01:  # 小于1%忽略
            return None
        
        # 计算置信度（基于阈值的超出程度）
        if direction == "greater":
            excess = np.mean(index_image[mask]) - threshold if np.any(mask) else 0
        else:
            excess = threshold - np.mean(index_image[mask]) if np.any(mask) else 0
        
        confidence = min(0.5 + excess * 0.5, 1.0)
        
        # 确定严重程度
        severity = self._determine_severity(affected_ratio, confidence)
        
        # 获取缺陷位置（边界框）
        location = self._get_bounding_box(mask)
        
        return DefectResult(
            defect_type=defect_type,
            confidence=confidence,
            severity=severity,
            location=location,
            affected_area_ratio=affected_ratio,
            description=f"检测到{defect_type.value}，影响面积{affected_ratio*100:.1f}%",
            recommendations=self._get_recommendations(defect_type, severity)
        )
    
    def _determine_severity(self, affected_ratio: float, 
                           confidence: float) -> SeverityLevel:
        """确定严重程度"""
        score = affected_ratio * 0.6 + confidence * 0.4
        
        thresholds = self.config.severity_thresholds
        
        if score >= thresholds["emergency"]:
            return SeverityLevel.EMERGENCY
        elif score >= thresholds["critical"]:
            return SeverityLevel.CRITICAL
        elif score >= thresholds["warning"]:
            return SeverityLevel.WARNING
        elif score >= thresholds["attention"]:
            return SeverityLevel.ATTENTION
        else:
            return SeverityLevel.NORMAL
    
    def _get_bounding_box(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """获取缺陷区域边界框"""
        if not np.any(mask):
            return None
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def _get_recommendations(self, defect_type: DefectType, 
                            severity: SeverityLevel) -> List[str]:
        """获取维护建议"""
        recommendations_map = {
            DefectType.METAL_CORROSION: [
                "检查锈蚀区域的结构完整性",
                "评估是否需要除锈和防腐处理",
                "监控锈蚀扩散速度"
            ],
            DefectType.PAINT_PEELING: [
                "评估涂层脱落范围",
                "检查基材是否受损",
                "安排重新涂装"
            ],
            DefectType.PORCELAIN_POLLUTION: [
                "进行污秽等级测量",
                "评估清洗必要性",
                "检查憎水性"
            ],
            DefectType.OIL_LEAKAGE: [
                "确认泄漏源",
                "评估泄漏量",
                "检查密封件状态"
            ],
            DefectType.THERMAL_ANOMALY: [
                "进行红外热成像复查",
                "检查电气连接",
                "评估负载情况"
            ],
        }
        
        base_recommendations = recommendations_map.get(defect_type, [])
        
        if severity >= SeverityLevel.CRITICAL:
            base_recommendations.insert(0, "建议立即安排现场检查")
        elif severity >= SeverityLevel.WARNING:
            base_recommendations.insert(0, "建议近期安排检查")
        
        return base_recommendations


class DeepLearningDefectDetector:
    """
    深度学习缺陷检测器
    
    使用预训练模型进行高光谱图像分析
    """
    
    def __init__(self, config: HyperspectralConfig):
        self.config = config
        self.model = None
        self.model_loaded = False
        
    def load_model(self):
        """加载深度学习模型"""
        try:
            from platform_core.extended_model_registry_manager import (
                get_extended_model_registry_manager
            )
            
            manager = get_extended_model_registry_manager()
            if manager:
                self.model = manager.get_model(self.config.model_name)
                self.model_loaded = self.model is not None
                
        except Exception as e:
            logger.warning(f"Failed to load hyperspectral model: {e}")
            self.model_loaded = False
    
    def detect(self, image: np.ndarray, 
               features: Dict[str, np.ndarray]) -> List[DefectResult]:
        """
        使用深度学习模型检测缺陷
        
        Args:
            image: 预处理后的高光谱图像
            features: 已提取的光谱特征
            
        Returns:
            检测到的缺陷列表
        """
        if not self.model_loaded:
            return []
        
        try:
            # 准备模型输入
            # 将光谱图像和特征拼接为模型输入
            model_input = self._prepare_input(image, features)
            
            # 模型推理
            outputs = self.model.infer(model_input)
            
            # 解析输出
            results = self._parse_outputs(outputs, image.shape)
            
            return results
            
        except Exception as e:
            logger.error(f"Deep learning detection failed: {e}")
            return []
    
    def _prepare_input(self, image: np.ndarray, 
                       features: Dict[str, np.ndarray]) -> np.ndarray:
        """准备模型输入"""
        # 方案1：仅使用高光谱图像
        # 转换为 [1, C, H, W] 格式
        if image.ndim == 3:
            input_tensor = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
        else:
            input_tensor = image[np.newaxis, ...]
        
        return input_tensor.astype(np.float32)
    
    def _parse_outputs(self, outputs: Dict[str, np.ndarray],
                       image_shape: Tuple) -> List[DefectResult]:
        """解析模型输出"""
        results = []
        
        # 假设模型输出格式：
        # - "class_scores": [N, num_classes] 每个检测的类别分数
        # - "boxes": [N, 4] 检测框
        # - "segmentation": [H, W] 语义分割图（可选）
        
        class_scores = outputs.get("class_scores", None)
        boxes = outputs.get("boxes", None)
        segmentation = outputs.get("segmentation", None)
        
        # 类别映射
        class_to_defect = {
            0: DefectType.PAINT_PEELING,
            1: DefectType.METAL_CORROSION,
            2: DefectType.PORCELAIN_POLLUTION,
            3: DefectType.INSULATION_AGING,
            4: DefectType.THERMAL_ANOMALY,
            5: DefectType.OIL_LEAKAGE,
            6: DefectType.CRACK,
        }
        
        if class_scores is not None and boxes is not None:
            for i in range(len(class_scores)):
                class_idx = np.argmax(class_scores[i])
                confidence = float(class_scores[i, class_idx])
                
                if confidence < self.config.detection_threshold:
                    continue
                
                defect_type = class_to_defect.get(class_idx, DefectType.UNKNOWN)
                box = boxes[i].astype(int)
                
                results.append(DefectResult(
                    defect_type=defect_type,
                    confidence=confidence,
                    severity=self._score_to_severity(confidence),
                    location=tuple(box),
                    description=f"DL检测: {defect_type.value}"
                ))
        
        # 处理语义分割输出
        if segmentation is not None:
            seg_results = self._parse_segmentation(segmentation, class_to_defect)
            results.extend(seg_results)
        
        return results
    
    def _parse_segmentation(self, segmentation: np.ndarray,
                           class_to_defect: Dict) -> List[DefectResult]:
        """解析语义分割结果"""
        results = []
        
        for class_idx, defect_type in class_to_defect.items():
            mask = segmentation == class_idx
            if not np.any(mask):
                continue
            
            affected_ratio = np.sum(mask) / mask.size
            if affected_ratio < 0.005:  # 忽略过小区域
                continue
            
            # 获取边界框
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
                
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            results.append(DefectResult(
                defect_type=defect_type,
                confidence=0.8,  # 分割置信度
                severity=self._score_to_severity(affected_ratio),
                location=(int(x_min), int(y_min), 
                         int(x_max - x_min), int(y_max - y_min)),
                affected_area_ratio=affected_ratio,
                description=f"分割检测: {defect_type.value}"
            ))
        
        return results
    
    def _score_to_severity(self, score: float) -> SeverityLevel:
        """分数转换为严重程度"""
        if score >= 0.9:
            return SeverityLevel.EMERGENCY
        elif score >= 0.7:
            return SeverityLevel.CRITICAL
        elif score >= 0.5:
            return SeverityLevel.WARNING
        elif score >= 0.3:
            return SeverityLevel.ATTENTION
        else:
            return SeverityLevel.NORMAL


class HyperspectralDetectionPlugin:
    """
    高光谱缺陷检测插件
    
    统一接口，整合预处理、特征提取、规则检测和深度学习检测
    """
    
    def __init__(self, config: Optional[HyperspectralConfig] = None):
        self.config = config or HyperspectralConfig()
        
        # 初始化组件
        self.preprocessor = SpectralPreprocessor(self.config)
        self.feature_extractor = SpectralFeatureExtractor(self.config.bands)
        self.rule_detector = RuleBasedDefectDetector(self.config)
        self.dl_detector = DeepLearningDefectDetector(self.config)
        
        # 加载深度学习模型
        if self.config.use_deep_learning:
            self.dl_detector.load_model()
        
        # 检测历史
        self.detection_history: List[Dict] = []
        self.max_history = 100
        
        logger.info("HyperspectralDetectionPlugin initialized")
    
    def detect(self, hyperspectral_image: np.ndarray,
               metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行高光谱缺陷检测
        
        Args:
            hyperspectral_image: 高光谱图像数据
            metadata: 可选的元数据（位置、时间等）
            
        Returns:
            检测结果字典
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # 1. 预处理
            processed = self.preprocessor.preprocess(hyperspectral_image)
            
            # 2. 特征提取
            features = self.feature_extractor.extract_features(processed)
            
            # 3. 深度学习检测
            dl_results = []
            if self.config.use_deep_learning and self.dl_detector.model_loaded:
                dl_results = self.dl_detector.detect(processed, features)
            
            # 4. 规则检测（作为补充或回退）
            rule_results = self.rule_detector.detect(features)
            
            # 5. 融合结果
            final_results = self._merge_results(dl_results, rule_results)
            
            # 6. 生成报告
            report = self._generate_report(final_results, features, metadata)
            
            # 7. 记录历史
            self._record_history(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Hyperspectral detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": timestamp
            }
    
    def _merge_results(self, dl_results: List[DefectResult],
                       rule_results: List[DefectResult]) -> List[DefectResult]:
        """
        融合深度学习和规则检测结果
        
        策略：
        - DL结果优先
        - 对于同类型缺陷，选择置信度更高的
        - 规则检测作为补充
        """
        merged = {}
        
        # 首先添加DL结果
        for result in dl_results:
            key = (result.defect_type, result.location)
            if key not in merged or result.confidence > merged[key].confidence:
                merged[key] = result
        
        # 添加规则结果（如果没有相同类型的DL结果）
        for result in rule_results:
            # 检查是否有重叠的DL结果
            has_overlap = False
            for key in merged:
                if key[0] == result.defect_type:
                    # 检查位置重叠
                    if self._boxes_overlap(key[1], result.location):
                        has_overlap = True
                        break
            
            if not has_overlap:
                key = (result.defect_type, result.location)
                merged[key] = result
        
        return list(merged.values())
    
    def _boxes_overlap(self, box1: Optional[Tuple], 
                       box2: Optional[Tuple], 
                       threshold: float = 0.3) -> bool:
        """检查两个边界框是否重叠"""
        if box1 is None or box2 is None:
            return False
        
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi
        
        if wi <= 0 or hi <= 0:
            return False
        
        intersection = wi * hi
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union > threshold
    
    def _generate_report(self, results: List[DefectResult],
                        features: Dict[str, np.ndarray],
                        metadata: Optional[Dict]) -> Dict[str, Any]:
        """生成检测报告"""
        # 统计各类缺陷
        defect_summary = {}
        for result in results:
            defect_type = result.defect_type.value
            if defect_type not in defect_summary:
                defect_summary[defect_type] = []
            defect_summary[defect_type].append({
                "confidence": result.confidence,
                "severity": result.severity.name,
                "location": result.location,
                "affected_area_ratio": result.affected_area_ratio,
                "description": result.description,
                "recommendations": result.recommendations
            })
        
        # 计算整体健康评分（0-100）
        if results:
            max_severity = max(r.severity.value for r in results)
            health_score = max(0, 100 - max_severity * 25)
        else:
            health_score = 100
        
        # 提取关键光谱统计
        spectral_stats = {}
        for name, data in features.items():
            if isinstance(data, np.ndarray):
                spectral_stats[name] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data))
                }
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "defect_count": len(results),
            "defects": defect_summary,
            "health_score": health_score,
            "spectral_statistics": spectral_stats,
            "detection_method": "hybrid" if self.dl_detector.model_loaded else "rule_based",
            "alerts": self._generate_alerts(results)
        }
    
    def _generate_alerts(self, results: List[DefectResult]) -> List[Dict]:
        """生成告警信息"""
        alerts = []
        
        for result in results:
            if result.severity.value >= SeverityLevel.WARNING.value:
                alerts.append({
                    "level": result.severity.name,
                    "type": result.defect_type.value,
                    "message": result.description,
                    "recommendations": result.recommendations,
                    "timestamp": result.timestamp
                })
        
        # 按严重程度排序
        alerts.sort(key=lambda x: SeverityLevel[x["level"]].value, reverse=True)
        
        return alerts
    
    def _record_history(self, report: Dict):
        """记录检测历史"""
        self.detection_history.append({
            "timestamp": report.get("timestamp"),
            "defect_count": report.get("defect_count", 0),
            "health_score": report.get("health_score", 100),
            "alerts_count": len(report.get("alerts", []))
        })
        
        # 保持历史长度
        if len(self.detection_history) > self.max_history:
            self.detection_history = self.detection_history[-self.max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.detection_history:
            return {"message": "No detection history"}
        
        health_scores = [h["health_score"] for h in self.detection_history]
        defect_counts = [h["defect_count"] for h in self.detection_history]
        
        return {
            "total_inspections": len(self.detection_history),
            "average_health_score": np.mean(health_scores),
            "min_health_score": np.min(health_scores),
            "total_defects_found": sum(defect_counts),
            "average_defects_per_inspection": np.mean(defect_counts),
            "model_loaded": self.dl_detector.model_loaded
        }
    
    def inspect(self, data: Any, **kwargs) -> Dict[str, Any]:
        """统一检测接口"""
        if isinstance(data, np.ndarray):
            return self.detect(data, metadata=kwargs.get("metadata"))
        elif isinstance(data, dict):
            image = data.get("image")
            if image is not None:
                return self.detect(image, metadata=data.get("metadata"))
        
        return {"success": False, "error": "Invalid input format"}


class HyperspectralDetectorEnhanced:
    """
    增强的高光谱检测器（兼容现有插件框架）
    """
    
    def __init__(self, config: Optional[Dict] = None):
        plugin_config = HyperspectralConfig()
        if config:
            for key, value in config.items():
                if hasattr(plugin_config, key):
                    setattr(plugin_config, key, value)
        
        self.plugin = HyperspectralDetectionPlugin(plugin_config)
        
    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """执行检测"""
        return self.plugin.detect(image, metadata=kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": "HyperspectralDefectDetector",
            "type": "hyperspectral",
            "model_loaded": self.plugin.dl_detector.model_loaded,
            "bands": len(self.plugin.config.bands),
            "defect_types": [d.value for d in DefectType]
        }


# 便捷函数
def create_hyperspectral_plugin(config: Optional[Dict] = None) -> HyperspectralDetectionPlugin:
    """创建高光谱检测插件实例"""
    if config:
        plugin_config = HyperspectralConfig(**config)
    else:
        plugin_config = HyperspectralConfig()
    return HyperspectralDetectionPlugin(plugin_config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    # 模拟7波段高光谱图像
    test_image = np.random.rand(256, 256, 7).astype(np.float32)
    
    # 模拟锈蚀区域（铁氧化物特征）
    test_image[100:150, 100:150, 2] *= 2.0  # 增强红色通道
    test_image[100:150, 100:150, 0] *= 0.5  # 减弱蓝色通道
    
    # 创建插件并检测
    plugin = create_hyperspectral_plugin()
    result = plugin.detect(test_image, metadata={"location": "变压器A"})
    
    print("检测结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
