# -*- coding: utf-8 -*-
"""
热成像分析器 V2.0 - 工程级实现
输变电激光星芒监测平台

功能:
- 温度分布分析与热点检测
- 温度梯度与趋势分析
- 相对温差三相比较
- 热力图生成与可视化
- 热成像相机SDK集成
- 分析报告自动生成

版本: V2.0
更新: 2026-01-24 - 升级为工程级实现
"""

import logging
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import time
from datetime import datetime
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


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


# ==========================================
# 热成像相机接口
# ==========================================

class ThermalCameraInterface(ABC):
    """热成像相机接口基类"""

    @abstractmethod
    def connect(self) -> bool:
        """连接相机"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开相机"""
        pass

    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """捕获热成像帧"""
        pass

    @abstractmethod
    def get_temperature_range(self) -> Tuple[float, float]:
        """获取温度范围"""
        pass


class FLIRCamera(ThermalCameraInterface):
    """
    FLIR热成像相机接口

    支持型号: FLIR A系列, E系列, T系列
    通信方式: GigE Vision / USB
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ip = config.get("ip", "192.168.1.100")
        self.port = config.get("port", 80)
        self.protocol = config.get("protocol", "http")
        self._connected = False
        self._session = None
        self._temp_range = (-20.0, 150.0)

    def connect(self) -> bool:
        """连接FLIR相机"""
        try:
            if self.protocol == "http":
                # HTTP API连接
                import urllib.request
                url = f"http://{self.ip}:{self.port}/api/status"
                req = urllib.request.Request(url, method="GET")
                response = urllib.request.urlopen(req, timeout=5.0)
                if response.status == 200:
                    self._connected = True
                    logger.info(f"FLIR相机已连接: {self.ip}")
                    return True

            elif self.protocol == "gige":
                # GigE Vision协议 - 需要专门的SDK
                logger.info(f"GigE Vision连接: {self.ip}")
                self._connected = True
                return True

            return False

        except Exception as e:
            logger.warning(f"FLIR相机连接失败: {e}")
            return False

    def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
        logger.info("FLIR相机已断开")

    def capture(self) -> Optional[np.ndarray]:
        """捕获热成像帧"""
        if not self._connected:
            return None

        try:
            if self.protocol == "http":
                import urllib.request
                url = f"http://{self.ip}:{self.port}/api/thermal/frame"
                req = urllib.request.Request(url, method="GET")
                response = urllib.request.urlopen(req, timeout=5.0)

                if response.status == 200:
                    data = response.read()
                    # 解析温度数据 (假设为16位灰度图转温度)
                    frame = np.frombuffer(data, dtype=np.uint16)
                    # 转换为温度 (0-65535 -> temp_min-temp_max)
                    temp_min, temp_max = self._temp_range
                    thermal = frame.astype(np.float32) / 65535.0 * (temp_max - temp_min) + temp_min
                    return thermal.reshape(self.config.get("height", 480),
                                          self.config.get("width", 640))

            return None

        except Exception as e:
            logger.warning(f"FLIR帧捕获失败: {e}")
            return None

    def get_temperature_range(self) -> Tuple[float, float]:
        return self._temp_range


class HIKVisionThermal(ThermalCameraInterface):
    """
    海康威视热成像相机接口

    支持型号: DS-2TD系列
    通信方式: RTSP / SDK
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ip = config.get("ip", "192.168.1.100")
        self.username = config.get("username", "admin")
        self.password = config.get("password", "")
        self._connected = False
        self._temp_range = (-20.0, 150.0)

    def connect(self) -> bool:
        """连接海康相机"""
        try:
            # ISAPI认证连接
            import urllib.request
            import base64

            auth_string = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode()

            url = f"http://{self.ip}/ISAPI/System/deviceInfo"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Authorization", f"Basic {auth_string}")

            response = urllib.request.urlopen(req, timeout=5.0)
            if response.status == 200:
                self._connected = True
                logger.info(f"海康热成像相机已连接: {self.ip}")
                return True

            return False

        except Exception as e:
            logger.warning(f"海康相机连接失败: {e}")
            return False

    def disconnect(self) -> None:
        self._connected = False

    def capture(self) -> Optional[np.ndarray]:
        """捕获热成像帧"""
        if not self._connected:
            return None

        try:
            import urllib.request
            import base64

            auth_string = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode()

            url = f"http://{self.ip}/ISAPI/Thermal/channels/1/thermometry/jpegPicWithAppendData"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Authorization", f"Basic {auth_string}")

            response = urllib.request.urlopen(req, timeout=5.0)
            if response.status == 200:
                data = response.read()
                # 解析海康的温度数据格式
                # 实际实现需要根据海康SDK文档解析
                return self._parse_hikvision_thermal(data)

            return None

        except Exception as e:
            logger.warning(f"海康帧捕获失败: {e}")
            return None

    def _parse_hikvision_thermal(self, data: bytes) -> Optional[np.ndarray]:
        """解析海康热成像数据"""
        # 海康热成像数据格式解析
        # 这里需要根据实际SDK文档实现
        height = self.config.get("height", 480)
        width = self.config.get("width", 640)

        try:
            # 简化解析 - 实际需要根据海康协议
            header_size = 64
            temp_data = np.frombuffer(data[header_size:], dtype=np.float32)
            if len(temp_data) >= height * width:
                return temp_data[:height*width].reshape(height, width)
        except Exception:
            pass

        return None

    def get_temperature_range(self) -> Tuple[float, float]:
        return self._temp_range


class SimulatedThermalCamera(ThermalCameraInterface):
    """模拟热成像相机 - 用于测试"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.width = config.get("width", 640)
        self.height = config.get("height", 480)
        self._connected = False
        self._frame_count = 0

    def connect(self) -> bool:
        self._connected = True
        logger.info("模拟热成像相机已连接")
        return True

    def disconnect(self) -> None:
        self._connected = False

    def capture(self) -> Optional[np.ndarray]:
        if not self._connected:
            return None

        self._frame_count += 1

        # 生成模拟热成像数据
        base_temp = 35.0 + 5.0 * np.sin(self._frame_count * 0.1)

        # 背景温度场
        y, x = np.meshgrid(np.linspace(0, 1, self.height),
                          np.linspace(0, 1, self.width), indexing='ij')
        thermal = base_temp + 5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        # 添加模拟热点
        cx, cy = 0.3 + 0.2 * np.sin(self._frame_count * 0.05), 0.5
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        hotspot = 30 * np.exp(-dist**2 / 0.02)
        thermal += hotspot

        # 添加噪声
        thermal += np.random.normal(0, 0.5, thermal.shape)

        return thermal.astype(np.float32)

    def get_temperature_range(self) -> Tuple[float, float]:
        return (-20.0, 150.0)


def create_thermal_camera(config: Dict[str, Any]) -> ThermalCameraInterface:
    """创建热成像相机实例"""
    camera_type = config.get("type", "simulation").lower()

    camera_map = {
        "flir": FLIRCamera,
        "hikvision": HIKVisionThermal,
        "hik": HIKVisionThermal,
        "simulation": SimulatedThermalCamera,
        "simulated": SimulatedThermalCamera
    }

    camera_class = camera_map.get(camera_type, SimulatedThermalCamera)
    return camera_class(config)


# ==========================================
# 热力图生成器
# ==========================================

class HeatmapGenerator:
    """热力图生成器"""

    # 颜色映射 (温度 -> RGB)
    COLORMAPS = {
        "iron": [
            (0.0, (0, 0, 0)),       # 黑
            (0.2, (128, 0, 128)),   # 紫
            (0.4, (255, 0, 0)),     # 红
            (0.6, (255, 128, 0)),   # 橙
            (0.8, (255, 255, 0)),   # 黄
            (1.0, (255, 255, 255))  # 白
        ],
        "rainbow": [
            (0.0, (0, 0, 255)),     # 蓝
            (0.25, (0, 255, 255)),  # 青
            (0.5, (0, 255, 0)),     # 绿
            (0.75, (255, 255, 0)),  # 黄
            (1.0, (255, 0, 0))      # 红
        ],
        "grayscale": [
            (0.0, (0, 0, 0)),
            (1.0, (255, 255, 255))
        ]
    }

    def __init__(self, colormap: str = "iron"):
        self.colormap_name = colormap
        self.colormap = self.COLORMAPS.get(colormap, self.COLORMAPS["iron"])

    def generate(self, thermal_data: np.ndarray,
                 temp_min: Optional[float] = None,
                 temp_max: Optional[float] = None) -> np.ndarray:
        """
        生成热力图

        Args:
            thermal_data: 温度矩阵
            temp_min: 最低温度 (用于归一化)
            temp_max: 最高温度 (用于归一化)

        Returns:
            RGB热力图 (H, W, 3)
        """
        if temp_min is None:
            temp_min = np.min(thermal_data)
        if temp_max is None:
            temp_max = np.max(thermal_data)

        # 归一化到0-1
        if temp_max == temp_min:
            normalized = np.zeros_like(thermal_data)
        else:
            normalized = (thermal_data - temp_min) / (temp_max - temp_min)
        normalized = np.clip(normalized, 0, 1)

        # 应用颜色映射
        h, w = thermal_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(len(self.colormap) - 1):
            t1, c1 = self.colormap[i]
            t2, c2 = self.colormap[i + 1]

            mask = (normalized >= t1) & (normalized < t2)
            if i == len(self.colormap) - 2:
                mask = (normalized >= t1) & (normalized <= t2)

            if np.any(mask):
                # 线性插值
                alpha = (normalized[mask] - t1) / (t2 - t1)
                for c in range(3):
                    rgb[mask, c] = (c1[c] * (1 - alpha) + c2[c] * alpha).astype(np.uint8)

        return rgb

    def add_annotations(self, heatmap: np.ndarray,
                        results: List[ThermalAnalysisResult],
                        show_temp: bool = True) -> np.ndarray:
        """添加分析结果标注"""
        try:
            import cv2
        except ImportError:
            return heatmap

        annotated = heatmap.copy()

        for result in results:
            region = result.region
            x, y, w, h = region.x, region.y, region.width, region.height

            # 根据严重程度选择颜色
            if result.severity == 'alarm':
                color = (0, 0, 255)  # 红
            elif result.severity == 'warning':
                color = (0, 165, 255)  # 橙
            else:
                color = (0, 255, 255)  # 黄

            # 绘制边框
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # 添加温度标签
            if show_temp:
                label = f"{region.max_temp:.1f}C"
                cv2.putText(annotated, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated


# ==========================================
# 分析报告生成器
# ==========================================

class ThermalReportGenerator:
    """热分析报告生成器"""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self,
                       device_id: str,
                       results: List[ThermalAnalysisResult],
                       thermal_image: np.ndarray,
                       heatmap: Optional[np.ndarray] = None,
                       metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        生成分析报告

        Args:
            device_id: 设备ID
            results: 分析结果列表
            thermal_image: 原始热成像数据
            heatmap: 热力图 (可选)
            metadata: 元数据 (可选)

        Returns:
            报告数据字典
        """
        timestamp = datetime.now()
        report_id = f"{device_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # 统计信息
        stats = {
            "mean_temp": float(np.mean(thermal_image)),
            "max_temp": float(np.max(thermal_image)),
            "min_temp": float(np.min(thermal_image)),
            "std_temp": float(np.std(thermal_image))
        }

        # 按严重程度分类结果
        severity_counts = {"alarm": 0, "warning": 0, "attention": 0, "normal": 0}
        for r in results:
            severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1

        # 确定整体状态
        if severity_counts["alarm"] > 0:
            overall_status = "危急"
        elif severity_counts["warning"] > 0:
            overall_status = "警告"
        elif severity_counts["attention"] > 0:
            overall_status = "注意"
        else:
            overall_status = "正常"

        report = {
            "report_id": report_id,
            "device_id": device_id,
            "timestamp": timestamp.isoformat(),
            "overall_status": overall_status,
            "statistics": stats,
            "severity_summary": severity_counts,
            "anomaly_count": len(results),
            "anomalies": [
                {
                    "type": r.anomaly_type.value,
                    "severity": r.severity,
                    "confidence": r.confidence,
                    "description": r.description,
                    "delta_temp": r.delta_temp,
                    "region": {
                        "x": r.region.x,
                        "y": r.region.y,
                        "width": r.region.width,
                        "height": r.region.height,
                        "max_temp": r.region.max_temp,
                        "mean_temp": r.region.mean_temp
                    },
                    "recommendations": r.recommendations
                }
                for r in results
            ],
            "metadata": metadata or {}
        }

        # 保存报告
        report_path = self.output_dir / f"{report_id}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 保存热力图
        if heatmap is not None:
            try:
                import cv2
                heatmap_path = self.output_dir / f"{report_id}_heatmap.png"
                cv2.imwrite(str(heatmap_path), heatmap)
                report["heatmap_path"] = str(heatmap_path)
            except ImportError:
                pass

        logger.info(f"热分析报告已生成: {report_path}")
        return report

    def generate_summary_report(self,
                                device_id: str,
                                history: List[Dict],
                                time_range_hours: int = 24) -> Dict:
        """生成汇总报告"""
        if not history:
            return {"device_id": device_id, "status": "no_data"}

        # 筛选时间范围内的数据
        current_time = time.time()
        cutoff = current_time - time_range_hours * 3600

        filtered = [h for h in history if h.get("timestamp", 0) >= cutoff]

        if not filtered:
            return {"device_id": device_id, "status": "no_recent_data"}

        # 计算统计
        means = [h["mean"] for h in filtered]
        maxes = [h["max"] for h in filtered]

        return {
            "device_id": device_id,
            "time_range_hours": time_range_hours,
            "sample_count": len(filtered),
            "temperature_stats": {
                "mean_avg": float(np.mean(means)),
                "mean_max": float(np.max(means)),
                "mean_min": float(np.min(means)),
                "peak_temp": float(np.max(maxes)),
                "trend": "rising" if means[-1] > means[0] + 2 else
                        "falling" if means[-1] < means[0] - 2 else "stable"
            },
            "generated_at": datetime.now().isoformat()
        }


# ==========================================
# 增强版热成像分析器
# ==========================================

class ThermalAnalyzerEnhanced(ThermalAnalyzer):
    """
    增强版热成像分析器

    集成:
    - 热成像相机硬件接口
    - 热力图生成
    - 报告自动生成
    - 三相设备比较分析
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # 热成像相机
        camera_config = (config or {}).get("camera", {})
        self.camera: Optional[ThermalCameraInterface] = None
        if camera_config:
            self.camera = create_thermal_camera(camera_config)

        # 热力图生成器
        colormap = (config or {}).get("colormap", "iron")
        self.heatmap_generator = HeatmapGenerator(colormap)

        # 报告生成器
        report_dir = (config or {}).get("report_dir", "./thermal_reports")
        self.report_generator = ThermalReportGenerator(report_dir)

        logger.info("增强版热成像分析器 V2.0 初始化完成")

    def connect_camera(self) -> bool:
        """连接热成像相机"""
        if self.camera:
            return self.camera.connect()
        return False

    def disconnect_camera(self) -> None:
        """断开热成像相机"""
        if self.camera:
            self.camera.disconnect()

    def capture_and_analyze(self,
                            equipment_type: str = 'default',
                            device_id: str = None,
                            roi: Tuple[int, int, int, int] = None,
                            generate_report: bool = False) -> Dict[str, Any]:
        """
        捕获并分析热成像

        Returns:
            包含分析结果、热力图和报告的完整响应
        """
        if not self.camera:
            return {"error": "Camera not configured"}

        # 捕获帧
        thermal_image = self.camera.capture()
        if thermal_image is None:
            return {"error": "Failed to capture thermal frame"}

        # 分析
        results = self.analyze(thermal_image, equipment_type, device_id, roi)

        # 生成热力图
        temp_min, temp_max = self.camera.get_temperature_range()
        heatmap = self.heatmap_generator.generate(thermal_image, temp_min, temp_max)
        heatmap_annotated = self.heatmap_generator.add_annotations(heatmap, results)

        response = {
            "timestamp": time.time(),
            "device_id": device_id,
            "statistics": self._calculate_statistics(thermal_image),
            "anomaly_count": len(results),
            "results": [
                {
                    "type": r.anomaly_type.value,
                    "severity": r.severity,
                    "confidence": r.confidence,
                    "description": r.description,
                    "delta_temp": r.delta_temp
                }
                for r in results
            ],
            "max_severity": max((r.severity for r in results), default="normal")
        }

        # 生成报告
        if generate_report and device_id:
            report = self.report_generator.generate_report(
                device_id=device_id,
                results=results,
                thermal_image=thermal_image,
                heatmap=heatmap_annotated,
                metadata={"equipment_type": equipment_type}
            )
            response["report_id"] = report.get("report_id")

        return response

    def analyze_three_phase(self,
                           phase_a: np.ndarray,
                           phase_b: np.ndarray,
                           phase_c: np.ndarray,
                           equipment_type: str = 'default') -> Dict[str, Any]:
        """
        三相设备比较分析

        比较三相设备的温度分布，检测不平衡
        """
        # 计算各相统计
        stats_a = self._calculate_statistics(phase_a)
        stats_b = self._calculate_statistics(phase_b)
        stats_c = self._calculate_statistics(phase_c)

        # 计算相间差异
        means = [stats_a['mean'], stats_b['mean'], stats_c['mean']]
        maxes = [stats_a['max'], stats_b['max'], stats_c['max']]

        mean_diff = max(means) - min(means)
        max_diff = max(maxes) - min(maxes)

        # 确定不平衡程度
        if max_diff > 20:
            imbalance_level = "severe"
            severity = "alarm"
        elif max_diff > 10:
            imbalance_level = "moderate"
            severity = "warning"
        elif max_diff > 5:
            imbalance_level = "mild"
            severity = "attention"
        else:
            imbalance_level = "balanced"
            severity = "normal"

        # 找出异常相
        abnormal_phase = None
        if imbalance_level != "balanced":
            max_idx = maxes.index(max(maxes))
            abnormal_phase = ["A", "B", "C"][max_idx]

        return {
            "timestamp": time.time(),
            "phase_statistics": {
                "A": stats_a,
                "B": stats_b,
                "C": stats_c
            },
            "comparison": {
                "mean_difference": mean_diff,
                "max_difference": max_diff,
                "imbalance_level": imbalance_level,
                "severity": severity,
                "abnormal_phase": abnormal_phase
            },
            "recommendations": self._get_three_phase_recommendations(imbalance_level, abnormal_phase)
        }

    def _get_three_phase_recommendations(self, level: str, abnormal_phase: Optional[str]) -> List[str]:
        """获取三相分析建议"""
        if level == "balanced":
            return ["三相温度分布均衡，设备运行正常"]

        recommendations = []

        if level == "severe":
            recommendations.append(f"严重不平衡！{abnormal_phase}相温度明显偏高")
            recommendations.append("立即检查该相接触电阻")
            recommendations.append("考虑降低负荷或停机检修")
        elif level == "moderate":
            recommendations.append(f"中度不平衡，{abnormal_phase}相需要关注")
            recommendations.append("安排红外复测确认")
            recommendations.append("检查该相接头紧固状态")
        else:
            recommendations.append(f"轻度不平衡，{abnormal_phase}相略高")
            recommendations.append("持续监测变化趋势")

        return recommendations
