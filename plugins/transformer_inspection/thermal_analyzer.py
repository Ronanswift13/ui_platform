# -*- coding: utf-8 -*-
"""
热成像分析器
支持温度分布分析、热点检测、趋势分析
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time
from collections import deque


class ThermalAnomalyType(Enum):
    """热异常类型"""
    HOTSPOT = "hotspot"                    # 热点
    COLD_SPOT = "cold_spot"                # 冷点
    UNEVEN_DISTRIBUTION = "uneven_distribution"  # 不均匀分布
    RAPID_RISE = "rapid_rise"              # 快速升温
    ABNORMAL_GRADIENT = "abnormal_gradient"  # 异常温度梯度


@dataclass
class ThermalRegion:
    """热区域"""
    x: int
    y: int
    width: int
    height: int
    mean_temp: float
    max_temp: float
    min_temp: float
    std_temp: float


@dataclass
class ThermalAnalysisResult:
    """热分析结果"""
    anomaly_type: ThermalAnomalyType
    severity: str
    confidence: float
    region: ThermalRegion
    description: str
    delta_temp: float  # 温升
    recommendations: List[str] = field(default_factory=list)


class ThermalAnalyzer:
    """
    热成像分析器
    
    功能：
    1. 温度分布分析
    2. 热点/冷点检测
    3. 温度梯度分析
    4. 趋势监测
    5. 相对温差分析
    """
    
    # 温度阈值配置（摄氏度）
    DEFAULT_THRESHOLDS = {
        'hotspot_absolute': 80.0,      # 绝对热点阈值
        'hotspot_relative': 15.0,      # 相对温升阈值
        'critical_temp': 100.0,        # 临界温度
        'warning_gradient': 20.0,      # 警告温度梯度
        'alarm_gradient': 35.0,        # 报警温度梯度
    }
    
    # 设备类型温度限值
    EQUIPMENT_LIMITS = {
        'bushing': {'normal': 60, 'warning': 75, 'alarm': 90},
        'connector': {'normal': 55, 'warning': 70, 'alarm': 85},
        'radiator': {'normal': 50, 'warning': 65, 'alarm': 80},
        'tap_changer': {'normal': 45, 'warning': 60, 'alarm': 75},
        'cable_terminal': {'normal': 50, 'warning': 65, 'alarm': 80},
        'default': {'normal': 60, 'warning': 75, 'alarm': 90}
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 阈值配置
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **self.config.get('thresholds', {})}
        
        # 历史数据（用于趋势分析）
        self.history_length = self.config.get('history_length', 100)
        self.temperature_history: Dict[str, deque] = {}
        
        # 环境温度参考
        self.ambient_temp = self.config.get('ambient_temp', 25.0)
        
    def analyze(self, thermal_image: np.ndarray,
                equipment_type: str = 'default',
                device_id: str = None,
                roi: Tuple[int, int, int, int] = None) -> List[ThermalAnalysisResult]:
        """
        分析热成像
        
        Args:
            thermal_image: 热成像数据（温度矩阵）
            equipment_type: 设备类型
            device_id: 设备ID
            roi: 感兴趣区域 (x, y, w, h)
            
        Returns:
            分析结果列表
        """
        results = []
        
        # 提取ROI
        if roi is not None:
            x, y, w, h = roi
            image = thermal_image[y:y+h, x:x+w]
            offset = (x, y)
        else:
            image = thermal_image
            offset = (0, 0)
            
        # 基本统计
        stats = self._calculate_statistics(image)
        
        # 热点检测
        hotspots = self._detect_hotspots(image, equipment_type, offset)
        results.extend(hotspots)
        
        # 冷点检测
        coldspots = self._detect_coldspots(image, offset)
        results.extend(coldspots)
        
        # 温度梯度分析
        gradient_anomalies = self._analyze_gradient(image, offset)
        results.extend(gradient_anomalies)
        
        # 分布均匀性分析
        distribution_anomalies = self._analyze_distribution(image, stats, offset)
        results.extend(distribution_anomalies)
        
        # 趋势分析（如果有历史数据）
        if device_id:
            self._update_history(device_id, stats)
            trend_anomalies = self._analyze_trend(device_id)
            results.extend(trend_anomalies)
            
        return results
    
    def analyze_relative(self, thermal_image: np.ndarray,
                         reference_regions: List[Tuple[int, int, int, int]],
                         target_region: Tuple[int, int, int, int]) -> ThermalAnalysisResult:
        """
        相对温差分析
        
        Args:
            thermal_image: 热成像数据
            reference_regions: 参考区域列表
            target_region: 目标区域
            
        Returns:
            分析结果
        """
        # 计算参考区域平均温度
        ref_temps = []
        for rx, ry, rw, rh in reference_regions:
            ref_region = thermal_image[ry:ry+rh, rx:rx+rw]
            ref_temps.append(np.mean(ref_region))
        ref_mean = np.mean(ref_temps)
        
        # 计算目标区域温度
        tx, ty, tw, th = target_region
        target = thermal_image[ty:ty+th, tx:tx+tw]
        target_mean = np.mean(target)
        target_max = np.max(target)
        
        # 计算相对温差
        delta = target_max - ref_mean
        
        # 判断严重程度
        if delta > self.thresholds['alarm_gradient']:
            severity = 'alarm'
            confidence = min(0.95, delta / 50)
        elif delta > self.thresholds['warning_gradient']:
            severity = 'warning'
            confidence = min(0.85, delta / 40)
        elif delta > self.thresholds['hotspot_relative']:
            severity = 'attention'
            confidence = min(0.75, delta / 30)
        else:
            severity = 'normal'
            confidence = 0.5
            
        region = ThermalRegion(
            x=tx, y=ty, width=tw, height=th,
            mean_temp=target_mean, max_temp=target_max,
            min_temp=np.min(target), std_temp=np.std(target)
        )
        
        return ThermalAnalysisResult(
            anomaly_type=ThermalAnomalyType.HOTSPOT if delta > 0 else ThermalAnomalyType.COLD_SPOT,
            severity=severity,
            confidence=confidence,
            region=region,
            description=f"相对温差{delta:.1f}°C（参考{ref_mean:.1f}°C，目标{target_max:.1f}°C）",
            delta_temp=delta,
            recommendations=self._get_recommendations(severity, delta)
        )
    
    def _calculate_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """计算统计信息"""
        return {
            'mean': float(np.mean(image)),
            'max': float(np.max(image)),
            'min': float(np.min(image)),
            'std': float(np.std(image)),
            'median': float(np.median(image)),
            'p95': float(np.percentile(image, 95)),
            'p5': float(np.percentile(image, 5))
        }
    
    def _detect_hotspots(self, image: np.ndarray, 
                         equipment_type: str,
                         offset: Tuple[int, int]) -> List[ThermalAnalysisResult]:
        """检测热点"""
        results = []
        
        # 获取设备温度限值
        limits = self.EQUIPMENT_LIMITS.get(equipment_type, self.EQUIPMENT_LIMITS['default'])
        
        # 计算自适应阈值
        mean_temp = np.mean(image)
        std_temp = np.std(image)
        adaptive_threshold = mean_temp + 2.5 * std_temp
        
        # 使用较低的阈值
        threshold = min(adaptive_threshold, limits['warning'])
        
        # 查找高温区域
        hot_mask = image > threshold
        
        if np.any(hot_mask):
            # 简化的连通区域分析
            hotspot_regions = self._find_connected_regions(image, hot_mask, offset)
            
            for region in hotspot_regions:
                # 确定严重程度
                if region.max_temp >= limits['alarm']:
                    severity = 'alarm'
                    confidence = 0.9
                elif region.max_temp >= limits['warning']:
                    severity = 'warning'
                    confidence = 0.8
                else:
                    severity = 'attention'
                    confidence = 0.7
                    
                delta = region.max_temp - mean_temp
                
                results.append(ThermalAnalysisResult(
                    anomaly_type=ThermalAnomalyType.HOTSPOT,
                    severity=severity,
                    confidence=confidence,
                    region=region,
                    description=f"检测到热点，最高温度{region.max_temp:.1f}°C，温升{delta:.1f}°C",
                    delta_temp=delta,
                    recommendations=self._get_recommendations(severity, delta)
                ))
                
        return results
    
    def _detect_coldspots(self, image: np.ndarray,
                          offset: Tuple[int, int]) -> List[ThermalAnalysisResult]:
        """检测冷点（可能表示堵塞或断流）"""
        results = []
        
        mean_temp = np.mean(image)
        std_temp = np.std(image)
        
        # 冷点阈值
        cold_threshold = mean_temp - 2.5 * std_temp
        
        if cold_threshold > self.ambient_temp:
            cold_mask = image < cold_threshold
            
            if np.any(cold_mask):
                cold_regions = self._find_connected_regions(image, cold_mask, offset, is_hot=False)
                
                for region in cold_regions:
                    delta = mean_temp - region.min_temp
                    
                    if delta > 15:
                        severity = 'warning'
                        confidence = 0.75
                    else:
                        severity = 'attention'
                        confidence = 0.6
                        
                    results.append(ThermalAnalysisResult(
                        anomaly_type=ThermalAnomalyType.COLD_SPOT,
                        severity=severity,
                        confidence=confidence,
                        region=region,
                        description=f"检测到冷点，温度{region.min_temp:.1f}°C，可能存在堵塞或断流",
                        delta_temp=-delta,
                        recommendations=["检查冷却系统是否堵塞", "确认油循环是否正常"]
                    ))
                    
        return results
    
    def _analyze_gradient(self, image: np.ndarray,
                          offset: Tuple[int, int]) -> List[ThermalAnalysisResult]:
        """分析温度梯度"""
        results = []
        
        # 计算梯度
        gy, gx = np.gradient(image)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # 查找高梯度区域
        threshold = self.thresholds['warning_gradient']
        high_gradient_mask = gradient_magnitude > threshold
        
        if np.any(high_gradient_mask):
            # 找到最大梯度位置
            max_idx = np.unravel_index(np.argmax(gradient_magnitude), gradient_magnitude.shape)
            max_gradient = gradient_magnitude[max_idx]
            
            y, x = max_idx
            ox, oy = offset
            
            # 创建区域
            region = ThermalRegion(
                x=x + ox - 10, y=y + oy - 10, width=20, height=20,
                mean_temp=float(image[y, x]),
                max_temp=float(np.max(image[max(0,y-5):y+5, max(0,x-5):x+5])),
                min_temp=float(np.min(image[max(0,y-5):y+5, max(0,x-5):x+5])),
                std_temp=float(np.std(image[max(0,y-5):y+5, max(0,x-5):x+5]))
            )
            
            if max_gradient > self.thresholds['alarm_gradient']:
                severity = 'alarm'
                confidence = 0.85
            else:
                severity = 'warning'
                confidence = 0.7
                
            results.append(ThermalAnalysisResult(
                anomaly_type=ThermalAnomalyType.ABNORMAL_GRADIENT,
                severity=severity,
                confidence=confidence,
                region=region,
                description=f"检测到异常温度梯度{max_gradient:.1f}°C/像素",
                delta_temp=max_gradient,
                recommendations=["检查该区域是否存在接触不良", "确认是否有局部过热"]
            ))
            
        return results
    
    def _analyze_distribution(self, image: np.ndarray,
                              stats: Dict[str, float],
                              offset: Tuple[int, int]) -> List[ThermalAnalysisResult]:
        """分析温度分布均匀性"""
        results = []
        
        # 将图像分成网格
        h, w = image.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = h // grid_h, w // grid_w
        
        cell_means = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_means.append(np.mean(cell))
                
        cell_means = np.array(cell_means)
        
        # 计算不均匀度
        cv = np.std(cell_means) / (np.mean(cell_means) + 1e-6)  # 变异系数
        
        if cv > 0.15:  # 不均匀度阈值
            ox, oy = offset
            
            region = ThermalRegion(
                x=ox, y=oy, width=w, height=h,
                mean_temp=stats['mean'], max_temp=stats['max'],
                min_temp=stats['min'], std_temp=stats['std']
            )
            
            if cv > 0.25:
                severity = 'warning'
                confidence = 0.75
            else:
                severity = 'attention'
                confidence = 0.6
                
            results.append(ThermalAnalysisResult(
                anomaly_type=ThermalAnomalyType.UNEVEN_DISTRIBUTION,
                severity=severity,
                confidence=confidence,
                region=region,
                description=f"温度分布不均匀，变异系数{cv:.2f}",
                delta_temp=stats['max'] - stats['min'],
                recommendations=["检查散热是否均匀", "确认负载分布"]
            ))
            
        return results
    
    def _analyze_trend(self, device_id: str) -> List[ThermalAnalysisResult]:
        """分析温度趋势"""
        results = []
        
        if device_id not in self.temperature_history:
            return results
            
        history = list(self.temperature_history[device_id])
        
        if len(history) < 5:
            return results
            
        # 提取最近的平均温度
        recent_means = [h['mean'] for h in history[-10:]]
        
        # 计算趋势
        if len(recent_means) >= 5:
            # 线性拟合
            x = np.arange(len(recent_means))
            coeffs = np.polyfit(x, recent_means, 1)
            slope = coeffs[0]  # 温度变化率
            
            # 快速升温检测
            if slope > 1.0:  # 每次采样升温1度以上
                severity = 'alarm' if slope > 2.0 else 'warning'
                confidence = min(0.9, slope / 3)
                
                results.append(ThermalAnalysisResult(
                    anomaly_type=ThermalAnomalyType.RAPID_RISE,
                    severity=severity,
                    confidence=confidence,
                    region=ThermalRegion(0, 0, 0, 0, recent_means[-1], max(recent_means), 
                                        min(recent_means), np.std(recent_means)),
                    description=f"检测到快速升温趋势，速率{slope:.2f}°C/采样",
                    delta_temp=recent_means[-1] - recent_means[0],
                    recommendations=["立即检查设备运行状态", "考虑降低负荷", "准备应急响应"]
                ))
                
        return results
    
    def _find_connected_regions(self, image: np.ndarray, 
                                 mask: np.ndarray,
                                 offset: Tuple[int, int],
                                 is_hot: bool = True) -> List[ThermalRegion]:
        """查找连通区域"""
        regions = []
        
        # 简化实现：找到mask中的极值点周围区域
        if is_hot:
            idx = np.unravel_index(np.argmax(image * mask), image.shape)
        else:
            masked = np.where(mask, image, np.inf)
            idx = np.unravel_index(np.argmin(masked), image.shape)
            
        y, x = idx
        ox, oy = offset
        
        # 确定区域范围
        h, w = image.shape
        x1 = max(0, x - 15)
        y1 = max(0, y - 15)
        x2 = min(w, x + 15)
        y2 = min(h, y + 15)
        
        region_data = image[y1:y2, x1:x2]
        
        regions.append(ThermalRegion(
            x=x1 + ox, y=y1 + oy, width=x2-x1, height=y2-y1,
            mean_temp=float(np.mean(region_data)),
            max_temp=float(np.max(region_data)),
            min_temp=float(np.min(region_data)),
            std_temp=float(np.std(region_data))
        ))
        
        return regions
    
    def _update_history(self, device_id: str, stats: Dict[str, float]):
        """更新历史记录"""
        if device_id not in self.temperature_history:
            self.temperature_history[device_id] = deque(maxlen=self.history_length)
            
        stats['timestamp'] = time.time()
        self.temperature_history[device_id].append(stats)
    
    def _get_recommendations(self, severity: str, delta: float) -> List[str]:
        """获取处理建议"""
        recommendations = []
        
        if severity == 'alarm':
            recommendations = [
                "立即安排现场检查",
                "准备应急处置措施",
                "考虑降低设备负荷",
                "通知调度部门"
            ]
        elif severity == 'warning':
            recommendations = [
                "安排红外测温复核",
                "加强监测频率",
                "检查接头紧固状态",
                "评估是否需要检修"
            ]
        elif severity == 'attention':
            recommendations = [
                "记录并持续监测",
                "下次巡检重点关注",
                "比对历史数据"
            ]
            
        if delta > 30:
            recommendations.insert(0, "温升过高，建议立即处理")
            
        return recommendations
    
    def get_history(self, device_id: str) -> List[Dict[str, float]]:
        """获取历史数据"""
        if device_id in self.temperature_history:
            return list(self.temperature_history[device_id])
        return []
    
    def set_ambient_temperature(self, temp: float):
        """设置环境温度"""
        self.ambient_temp = temp
        
    def clear_history(self, device_id: str = None):
        """清除历史"""
        if device_id:
            self.temperature_history.pop(device_id, None)
        else:
            self.temperature_history.clear()
