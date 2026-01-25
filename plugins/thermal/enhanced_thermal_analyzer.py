#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习增强热成像分析器
========================

基于迭代方案实现:
- 多模态融合 (可见光+红外)
- 深度学习异常检测
- 温度趋势预测
- 热点自动分类

版本: 2.0.0
作者: 变电站AI巡检系统
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# =============================================================================
# 枚举和数据结构
# =============================================================================

class AnomalyType(Enum):
    """异常类型"""
    NONE = "none"
    HOTSPOT = "hotspot"              # 局部过热
    GRADIENT = "gradient"            # 温度梯度异常
    TREND = "trend"                  # 温度趋势异常
    DISTRIBUTION = "distribution"    # 温度分布异常
    CONTACT = "contact"              # 接触不良
    OVERLOAD = "overload"            # 过载发热


class SeverityLevel(Enum):
    """严重程度"""
    NORMAL = 0
    ATTENTION = 1     # 关注
    WARNING = 2       # 预警
    SERIOUS = 3       # 严重
    CRITICAL = 4      # 紧急


class EquipmentType(Enum):
    """设备类型"""
    TRANSFORMER = "transformer"
    BUSBAR = "busbar"
    SWITCH = "switch"
    CABLE_JOINT = "cable_joint"
    CAPACITOR = "capacitor"
    ARRESTER = "arrester"
    UNKNOWN = "unknown"


@dataclass
class ThermalConfig:
    """热成像分析配置"""
    # 图像配置
    input_size: Tuple[int, int] = (640, 480)
    thermal_range: Tuple[float, float] = (-20.0, 150.0)
    
    # 异常检测阈值
    temp_warning: float = 60.0
    temp_serious: float = 80.0
    temp_critical: float = 100.0
    gradient_threshold: float = 15.0
    trend_threshold: float = 5.0  # 度/小时
    
    # 模型配置
    use_deep_learning: bool = True
    model_path: str = "models/thermal_anomaly.pt"
    
    # 多模态融合
    enable_fusion: bool = True
    visible_weight: float = 0.3
    thermal_weight: float = 0.7
    
    # 趋势分析
    history_length: int = 100
    trend_window: int = 10
    
    # 分类
    num_classes: int = len(AnomalyType)


@dataclass
class ThermalRegion:
    """热区域"""
    region_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None
    equipment_type: EquipmentType = EquipmentType.UNKNOWN
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class ThermalAnomaly:
    """热异常"""
    anomaly_id: int
    anomaly_type: AnomalyType
    severity: SeverityLevel
    region: ThermalRegion
    
    # 温度数据
    max_temp: float
    min_temp: float
    avg_temp: float
    temp_gradient: float
    
    # 置信度
    confidence: float
    
    # 趋势
    temp_trend: float  # 度/小时
    time_to_critical: Optional[float] = None  # 小时
    
    # 建议
    recommendations: List[str] = field(default_factory=list)
    
    timestamp: float = 0.0


@dataclass
class ThermalAnalysisResult:
    """热分析结果"""
    frame_id: int
    timestamp: float
    
    # 整体温度
    global_max: float
    global_min: float
    global_avg: float
    
    # 检测到的异常
    anomalies: List[ThermalAnomaly]
    
    # 热图
    temperature_map: np.ndarray
    anomaly_mask: Optional[np.ndarray] = None
    
    # 统计
    num_hotspots: int = 0
    overall_health: str = "normal"


# =============================================================================
# 深度学习模型
# =============================================================================

if TORCH_AVAILABLE:
    
    class ThermalEncoder(nn.Module):
        """热图编码器"""
        
        def __init__(self, in_channels: int = 1, base_dim: int = 64):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, base_dim, 3, 2, 1),
                nn.BatchNorm2d(base_dim),
                nn.ReLU(),
                
                nn.Conv2d(base_dim, base_dim * 2, 3, 2, 1),
                nn.BatchNorm2d(base_dim * 2),
                nn.ReLU(),
                
                nn.Conv2d(base_dim * 2, base_dim * 4, 3, 2, 1),
                nn.BatchNorm2d(base_dim * 4),
                nn.ReLU(),
                
                nn.Conv2d(base_dim * 4, base_dim * 8, 3, 2, 1),
                nn.BatchNorm2d(base_dim * 8),
                nn.ReLU(),
            )
            
            self.out_channels = base_dim * 8
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)
    
    
    class VisibleEncoder(nn.Module):
        """可见光编码器"""
        
        def __init__(self, in_channels: int = 3, base_dim: int = 64):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, base_dim, 3, 2, 1),
                nn.BatchNorm2d(base_dim),
                nn.ReLU(),
                
                nn.Conv2d(base_dim, base_dim * 2, 3, 2, 1),
                nn.BatchNorm2d(base_dim * 2),
                nn.ReLU(),
                
                nn.Conv2d(base_dim * 2, base_dim * 4, 3, 2, 1),
                nn.BatchNorm2d(base_dim * 4),
                nn.ReLU(),
                
                nn.Conv2d(base_dim * 4, base_dim * 8, 3, 2, 1),
                nn.BatchNorm2d(base_dim * 8),
                nn.ReLU(),
            )
            
            self.out_channels = base_dim * 8
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)
    
    
    class MultimodalFusion(nn.Module):
        """多模态融合模块"""
        
        def __init__(self, thermal_dim: int, visible_dim: int, out_dim: int):
            super().__init__()
            
            self.thermal_proj = nn.Conv2d(thermal_dim, out_dim, 1)
            self.visible_proj = nn.Conv2d(visible_dim, out_dim, 1)
            
            # 注意力融合
            self.attention = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, 1),
                nn.Sigmoid()
            )
            
            self.fusion = nn.Conv2d(out_dim * 2, out_dim, 1)
        
        def forward(self, thermal_feat: torch.Tensor, visible_feat: torch.Tensor) -> torch.Tensor:
            t = self.thermal_proj(thermal_feat)
            v = self.visible_proj(visible_feat)
            
            # 计算注意力
            combined = torch.cat([t, v], dim=1)
            att = self.attention(combined)
            
            # 加权融合
            fused = torch.cat([t * att, v * (1 - att)], dim=1)
            return self.fusion(fused)
    
    
    class AnomalyDetectionHead(nn.Module):
        """异常检测头"""
        
        def __init__(self, in_channels: int, num_classes: int):
            super().__init__()
            
            self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(in_channels // 2)
            
            self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(in_channels // 4)
            
            # 分割头
            self.seg_head = nn.Conv2d(in_channels // 4, num_classes, 1)
            
            # 分类头
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(in_channels // 4, num_classes)
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            
            seg = self.seg_head(x)
            
            pooled = self.pool(x).flatten(1)
            cls = self.classifier(pooled)
            
            return seg, cls
    
    
    class ThermalAnomalyNet(nn.Module):
        """热成像异常检测网络"""
        
        def __init__(self, config: ThermalConfig):
            super().__init__()
            self.config = config
            
            self.thermal_encoder = ThermalEncoder(in_channels=1)
            
            if config.enable_fusion:
                self.visible_encoder = VisibleEncoder(in_channels=3)
                self.fusion = MultimodalFusion(
                    self.thermal_encoder.out_channels,
                    self.visible_encoder.out_channels,
                    512
                )
                feat_dim = 512
            else:
                self.visible_encoder = None
                self.fusion = None
                feat_dim = self.thermal_encoder.out_channels
            
            self.detection_head = AnomalyDetectionHead(feat_dim, config.num_classes)
        
        def forward(self, 
                    thermal: torch.Tensor, 
                    visible: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            thermal_feat = self.thermal_encoder(thermal)
            
            if self.config.enable_fusion and visible is not None and self.fusion is not None:
                visible_feat = self.visible_encoder(visible)
                feat = self.fusion(thermal_feat, visible_feat)
            else:
                feat = thermal_feat
            
            seg, cls = self.detection_head(feat)
            
            return seg, cls


# =============================================================================
# 温度趋势分析器
# =============================================================================

class TemperatureTrendAnalyzer:
    """温度趋势分析器"""
    
    def __init__(self, config: ThermalConfig):
        self.config = config
        self._history: Dict[int, deque] = {}
        self._timestamps: Dict[int, deque] = {}
    
    def update(self, region_id: int, temperature: float, timestamp: float = None):
        """更新温度记录"""
        timestamp = timestamp or time.time()
        
        if region_id not in self._history:
            self._history[region_id] = deque(maxlen=self.config.history_length)
            self._timestamps[region_id] = deque(maxlen=self.config.history_length)
        
        self._history[region_id].append(temperature)
        self._timestamps[region_id].append(timestamp)
    
    def get_trend(self, region_id: int) -> float:
        """获取温度趋势 (度/小时)"""
        if region_id not in self._history:
            return 0.0
        
        history = list(self._history[region_id])
        timestamps = list(self._timestamps[region_id])
        
        if len(history) < self.config.trend_window:
            return 0.0
        
        # 使用最近的数据点
        recent_temps = history[-self.config.trend_window:]
        recent_times = timestamps[-self.config.trend_window:]
        
        # 线性回归
        t = np.array(recent_times)
        y = np.array(recent_temps)
        
        t_norm = t - t[0]
        
        if t_norm[-1] == 0:
            return 0.0
        
        slope = np.polyfit(t_norm, y, 1)[0]
        
        # 转换为度/小时
        return slope * 3600
    
    def predict_time_to_threshold(self, region_id: int, threshold: float) -> Optional[float]:
        """预测达到阈值的时间 (小时)"""
        if region_id not in self._history:
            return None
        
        current_temp = self._history[region_id][-1]
        trend = self.get_trend(region_id)
        
        if trend <= 0:
            return None
        
        if current_temp >= threshold:
            return 0.0
        
        time_to_threshold = (threshold - current_temp) / trend
        return time_to_threshold
    
    def get_statistics(self, region_id: int) -> Dict[str, float]:
        """获取统计信息"""
        if region_id not in self._history:
            return {}
        
        history = list(self._history[region_id])
        
        return {
            "current": history[-1],
            "max": max(history),
            "min": min(history),
            "avg": sum(history) / len(history),
            "std": np.std(history),
            "trend": self.get_trend(region_id)
        }


# =============================================================================
# 热成像分析器
# =============================================================================

class EnhancedThermalAnalyzer:
    """增强热成像分析器"""
    
    def __init__(self, config: Optional[ThermalConfig] = None):
        self.config = config or ThermalConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self._model = None
        if self.config.use_deep_learning and TORCH_AVAILABLE:
            try:
                self._model = ThermalAnomalyNet(self.config).to(self.device)
                self._model.eval()
                logger.info("热成像分析模型加载成功")
            except Exception as e:
                logger.warning(f"模型加载失败: {e}")
        
        # 趋势分析器
        self.trend_analyzer = TemperatureTrendAnalyzer(self.config)
        
        # 帧计数
        self._frame_id = 0
        
        # 区域注册
        self._regions: Dict[int, ThermalRegion] = {}
        self._next_region_id = 0
        
        # 统计
        self._stats = {
            "frames_processed": 0,
            "anomalies_detected": 0,
            "critical_alerts": 0
        }
        
        logger.info("增强热成像分析器初始化完成")
    
    def register_region(self, 
                        bbox: Tuple[int, int, int, int],
                        equipment_type: EquipmentType = EquipmentType.UNKNOWN,
                        mask: Optional[np.ndarray] = None) -> int:
        """注册监控区域"""
        region = ThermalRegion(
            region_id=self._next_region_id,
            bbox=bbox,
            mask=mask,
            equipment_type=equipment_type
        )
        
        self._regions[self._next_region_id] = region
        self._next_region_id += 1
        
        return region.region_id
    
    def analyze(self, 
                thermal_image: np.ndarray,
                visible_image: Optional[np.ndarray] = None) -> ThermalAnalysisResult:
        """
        分析热成像
        
        Args:
            thermal_image: 热图 [H, W] 或 [H, W, 1]，温度值
            visible_image: 可见光图像 [H, W, 3] (可选)
            
        Returns:
            分析结果
        """
        self._frame_id += 1
        timestamp = time.time()
        
        # 预处理
        if thermal_image.ndim == 3:
            thermal_image = thermal_image[:, :, 0]
        
        # 全局温度统计
        global_max = float(np.max(thermal_image))
        global_min = float(np.min(thermal_image))
        global_avg = float(np.mean(thermal_image))
        
        # 检测异常
        anomalies = []
        anomaly_mask = np.zeros_like(thermal_image, dtype=np.uint8)
        
        if self._model is not None and TORCH_AVAILABLE:
            anomalies, anomaly_mask = self._analyze_with_model(thermal_image, visible_image)
        else:
            anomalies, anomaly_mask = self._analyze_with_rules(thermal_image)
        
        # 更新趋势
        for region in self._regions.values():
            x1, y1, x2, y2 = region.bbox
            region_temps = thermal_image[y1:y2, x1:x2]
            if region_temps.size > 0:
                max_temp = float(np.max(region_temps))
                self.trend_analyzer.update(region.region_id, max_temp, timestamp)
        
        # 统计
        self._stats["frames_processed"] += 1
        self._stats["anomalies_detected"] += len(anomalies)
        
        num_critical = sum(1 for a in anomalies if a.severity == SeverityLevel.CRITICAL)
        self._stats["critical_alerts"] += num_critical
        
        # 确定整体健康状态
        if num_critical > 0:
            overall_health = "critical"
        elif any(a.severity == SeverityLevel.SERIOUS for a in anomalies):
            overall_health = "serious"
        elif any(a.severity == SeverityLevel.WARNING for a in anomalies):
            overall_health = "warning"
        elif anomalies:
            overall_health = "attention"
        else:
            overall_health = "normal"
        
        return ThermalAnalysisResult(
            frame_id=self._frame_id,
            timestamp=timestamp,
            global_max=global_max,
            global_min=global_min,
            global_avg=global_avg,
            anomalies=anomalies,
            temperature_map=thermal_image,
            anomaly_mask=anomaly_mask,
            num_hotspots=len(anomalies),
            overall_health=overall_health
        )
    
    def _analyze_with_model(self, 
                            thermal_image: np.ndarray,
                            visible_image: Optional[np.ndarray]) -> Tuple[List[ThermalAnomaly], np.ndarray]:
        """使用深度学习模型分析"""
        try:
            # 准备输入
            thermal_tensor = torch.from_numpy(thermal_image).float().unsqueeze(0).unsqueeze(0)
            thermal_tensor = thermal_tensor.to(self.device)
            
            # 归一化
            t_min, t_max = self.config.thermal_range
            thermal_tensor = (thermal_tensor - t_min) / (t_max - t_min)
            
            visible_tensor = None
            if visible_image is not None and self.config.enable_fusion:
                visible_tensor = torch.from_numpy(visible_image).float().permute(2, 0, 1).unsqueeze(0)
                visible_tensor = visible_tensor.to(self.device) / 255.0
            
            # 推理
            with torch.no_grad():
                seg, cls = self._model(thermal_tensor, visible_tensor)
            
            # 处理分割结果
            seg_mask = torch.argmax(seg, dim=1).squeeze().cpu().numpy()
            cls_probs = F.softmax(cls, dim=1).squeeze().cpu().numpy()
            
            # 提取异常
            anomalies = self._extract_anomalies_from_mask(thermal_image, seg_mask, cls_probs)
            
            return anomalies, seg_mask.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            return self._analyze_with_rules(thermal_image)
    
    def _analyze_with_rules(self, thermal_image: np.ndarray) -> Tuple[List[ThermalAnomaly], np.ndarray]:
        """使用规则分析"""
        anomalies = []
        anomaly_mask = np.zeros_like(thermal_image, dtype=np.uint8)
        anomaly_id = 0
        
        # 阈值检测
        hot_mask = thermal_image > self.config.temp_warning
        
        if np.any(hot_mask):
            # 连通域分析
            try:
                from scipy import ndimage
                labeled, num_features = ndimage.label(hot_mask)
            except ImportError:
                labeled = hot_mask.astype(int)
                num_features = 1
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                
                if np.sum(component) < 10:
                    continue
                
                # 获取边界框
                ys, xs = np.where(component)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                
                # 获取温度
                temps = thermal_image[component]
                max_temp = float(np.max(temps))
                min_temp = float(np.min(temps))
                avg_temp = float(np.mean(temps))
                
                # 计算梯度
                gradient = max_temp - min_temp
                
                # 确定严重程度
                if max_temp >= self.config.temp_critical:
                    severity = SeverityLevel.CRITICAL
                elif max_temp >= self.config.temp_serious:
                    severity = SeverityLevel.SERIOUS
                elif max_temp >= self.config.temp_warning:
                    severity = SeverityLevel.WARNING
                else:
                    severity = SeverityLevel.ATTENTION
                
                # 确定异常类型
                if gradient > self.config.gradient_threshold:
                    anomaly_type = AnomalyType.GRADIENT
                else:
                    anomaly_type = AnomalyType.HOTSPOT
                
                # 生成建议
                recommendations = self._generate_recommendations(severity, anomaly_type, max_temp)
                
                region = ThermalRegion(
                    region_id=anomaly_id,
                    bbox=bbox,
                    mask=component
                )
                
                anomaly = ThermalAnomaly(
                    anomaly_id=anomaly_id,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    region=region,
                    max_temp=max_temp,
                    min_temp=min_temp,
                    avg_temp=avg_temp,
                    temp_gradient=gradient,
                    confidence=0.8,
                    temp_trend=0.0,
                    recommendations=recommendations,
                    timestamp=time.time()
                )
                
                anomalies.append(anomaly)
                anomaly_mask[component] = severity.value
                anomaly_id += 1
        
        return anomalies, anomaly_mask
    
    def _extract_anomalies_from_mask(self, 
                                      thermal_image: np.ndarray,
                                      seg_mask: np.ndarray,
                                      cls_probs: np.ndarray) -> List[ThermalAnomaly]:
        """从分割掩码提取异常"""
        anomalies = []
        anomaly_id = 0
        
        for cls_idx in range(1, len(AnomalyType)):
            component = (seg_mask == cls_idx)
            
            if not np.any(component):
                continue
            
            ys, xs = np.where(component)
            if len(xs) < 10:
                continue
            
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            
            temps = thermal_image[component]
            max_temp = float(np.max(temps))
            min_temp = float(np.min(temps))
            avg_temp = float(np.mean(temps))
            gradient = max_temp - min_temp
            
            if max_temp >= self.config.temp_critical:
                severity = SeverityLevel.CRITICAL
            elif max_temp >= self.config.temp_serious:
                severity = SeverityLevel.SERIOUS
            elif max_temp >= self.config.temp_warning:
                severity = SeverityLevel.WARNING
            else:
                severity = SeverityLevel.ATTENTION
            
            anomaly_type = list(AnomalyType)[cls_idx]
            recommendations = self._generate_recommendations(severity, anomaly_type, max_temp)
            
            region = ThermalRegion(region_id=anomaly_id, bbox=bbox, mask=component)
            
            anomaly = ThermalAnomaly(
                anomaly_id=anomaly_id,
                anomaly_type=anomaly_type,
                severity=severity,
                region=region,
                max_temp=max_temp,
                min_temp=min_temp,
                avg_temp=avg_temp,
                temp_gradient=gradient,
                confidence=float(cls_probs[cls_idx]),
                temp_trend=0.0,
                recommendations=recommendations,
                timestamp=time.time()
            )
            
            anomalies.append(anomaly)
            anomaly_id += 1
        
        return anomalies
    
    def _generate_recommendations(self, 
                                   severity: SeverityLevel,
                                   anomaly_type: AnomalyType,
                                   max_temp: float) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if severity == SeverityLevel.CRITICAL:
            recommendations.extend([
                "立即停机处理",
                "启动应急预案",
                "通知值班人员",
                "准备消防设备"
            ])
        elif severity == SeverityLevel.SERIOUS:
            recommendations.extend([
                "安排紧急检修",
                "降低负载运行",
                "持续监测温度变化"
            ])
        elif severity == SeverityLevel.WARNING:
            recommendations.extend([
                "安排计划检修",
                "增加巡检频率",
                "记录温度趋势"
            ])
        else:
            recommendations.append("持续关注")
        
        if anomaly_type == AnomalyType.CONTACT:
            recommendations.append("检查接头紧固情况")
        elif anomaly_type == AnomalyType.OVERLOAD:
            recommendations.append("检查负载分配")
        
        return recommendations
    
    def get_region_trend(self, region_id: int) -> Dict[str, Any]:
        """获取区域温度趋势"""
        stats = self.trend_analyzer.get_statistics(region_id)
        
        if not stats:
            return {}
        
        time_to_warning = self.trend_analyzer.predict_time_to_threshold(
            region_id, self.config.temp_warning
        )
        time_to_critical = self.trend_analyzer.predict_time_to_threshold(
            region_id, self.config.temp_critical
        )
        
        return {
            **stats,
            "time_to_warning": time_to_warning,
            "time_to_critical": time_to_critical
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "registered_regions": len(self._regions)
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_thermal_analyzer(
    use_deep_learning: bool = True,
    enable_fusion: bool = True,
    **kwargs
) -> EnhancedThermalAnalyzer:
    """创建热成像分析器"""
    config = ThermalConfig(
        use_deep_learning=use_deep_learning,
        enable_fusion=enable_fusion,
        **kwargs
    )
    return EnhancedThermalAnalyzer(config)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建分析器
    analyzer = create_thermal_analyzer(use_deep_learning=False)
    
    # 注册监控区域
    analyzer.register_region((100, 100, 200, 200), EquipmentType.TRANSFORMER)
    analyzer.register_region((300, 100, 400, 200), EquipmentType.BUSBAR)
    
    # 模拟分析
    for i in range(50):
        # 生成模拟热图
        thermal = np.random.normal(35, 10, (480, 640)).astype(np.float32)
        
        # 添加热点
        if np.random.random() < 0.3:
            x, y = np.random.randint(50, 590), np.random.randint(50, 430)
            thermal[y-20:y+20, x-20:x+20] += np.random.uniform(30, 60)
        
        # 分析
        result = analyzer.analyze(thermal)
        
        if result.anomalies:
            print(f"帧 {i}: 检测到 {len(result.anomalies)} 个异常")
            for anomaly in result.anomalies:
                print(f"  - {anomaly.anomaly_type.value}: {anomaly.max_temp:.1f}°C, "
                      f"严重程度: {anomaly.severity.name}")
        
        time.sleep(0.1)
    
    print(f"\n统计: {analyzer.get_statistics()}")
