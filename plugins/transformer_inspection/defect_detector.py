# -*- coding: utf-8 -*-
"""
主变缺陷检测器
支持套管裂纹、护罩锈蚀、瓷套污秽、油位异常等检测
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time


class DefectType(Enum):
    """缺陷类型枚举"""
    BUSHING_CRACK = "bushing_crack"           # 套管裂纹
    BUSHING_CONTAMINATION = "bushing_contamination"  # 套管污秽
    SHIELD_RUST = "shield_rust"               # 护罩锈蚀
    PORCELAIN_DAMAGE = "porcelain_damage"     # 瓷套损伤
    OIL_LEAK = "oil_leak"                     # 漏油
    OIL_LEVEL_ABNORMAL = "oil_level_abnormal" # 油位异常
    CONNECTOR_OVERHEAT = "connector_overheat" # 接头过热
    RADIATOR_BLOCKAGE = "radiator_blockage"   # 散热器堵塞
    FAN_FAILURE = "fan_failure"               # 风扇故障
    PRESSURE_ABNORMAL = "pressure_abnormal"   # 压力异常
    SEAL_DAMAGE = "seal_damage"               # 密封件损伤
    PAINT_PEELING = "paint_peeling"           # 油漆剥落
    CORROSION = "corrosion"                   # 腐蚀
    BIRD_NEST = "bird_nest"                   # 鸟巢
    FOREIGN_OBJECT = "foreign_object"         # 异物


class SeverityLevel(Enum):
    """严重程度"""
    NORMAL = 0
    ATTENTION = 1
    WARNING = 2
    ALARM = 3
    CRITICAL = 4


@dataclass
class DetectionResult:
    """检测结果"""
    defect_type: DefectType
    severity: SeverityLevel
    confidence: float
    location: Tuple[int, int, int, int]  # x, y, width, height
    description: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionReport:
    """巡检报告"""
    device_id: str
    timestamp: float
    overall_status: SeverityLevel
    detections: List[DetectionResult]
    statistics: Dict[str, Any]
    recommendations: List[str]


class DefectDetector:
    """
    主变缺陷检测器
    
    支持多种检测模式：
    1. 视觉检测：套管裂纹、锈蚀、污秽等
    2. 热成像检测：过热点、温度分布异常
    3. 油位监测：油位异常、漏油检测
    4. 多模态融合检测
    """
    
    # 缺陷严重程度映射
    SEVERITY_THRESHOLDS = {
        DefectType.BUSHING_CRACK: {
            'attention': 0.3,
            'warning': 0.5,
            'alarm': 0.7,
            'critical': 0.85
        },
        DefectType.SHIELD_RUST: {
            'attention': 0.4,
            'warning': 0.6,
            'alarm': 0.75,
            'critical': 0.9
        },
        DefectType.OIL_LEAK: {
            'attention': 0.25,
            'warning': 0.45,
            'alarm': 0.65,
            'critical': 0.8
        },
        DefectType.CONNECTOR_OVERHEAT: {
            'attention': 0.35,
            'warning': 0.55,
            'alarm': 0.7,
            'critical': 0.85
        }
    }
    
    # 缺陷处理建议
    RECOMMENDATIONS = {
        DefectType.BUSHING_CRACK: [
            "立即安排专业人员现场检查",
            "准备备用套管以备更换",
            "监测套管油色谱变化",
            "考虑降低负荷运行"
        ],
        DefectType.BUSHING_CONTAMINATION: [
            "安排清洁维护",
            "检查防污等级是否满足要求",
            "考虑加装防污罩"
        ],
        DefectType.SHIELD_RUST: [
            "安排除锈和防腐处理",
            "检查防护涂层完整性",
            "评估结构强度是否受影响"
        ],
        DefectType.OIL_LEAK: [
            "立即定位泄漏点",
            "准备补油和密封材料",
            "监测油位变化速率",
            "安排检修计划"
        ],
        DefectType.OIL_LEVEL_ABNORMAL: [
            "检查油位计是否正常",
            "核实环境温度影响",
            "检查是否存在漏油点"
        ],
        DefectType.CONNECTOR_OVERHEAT: [
            "安排红外测温复核",
            "检查接头紧固状态",
            "评估是否需要停电检修",
            "准备导电膏和紧固工具"
        ],
        DefectType.RADIATOR_BLOCKAGE: [
            "清理散热器表面",
            "检查冷却油循环系统",
            "评估冷却效率"
        ],
        DefectType.FAN_FAILURE: [
            "检查风扇电机和控制回路",
            "准备备用风扇",
            "监测变压器温升"
        ],
        DefectType.BIRD_NEST: [
            "安全移除鸟巢",
            "安装驱鸟装置",
            "检查是否造成设备损伤"
        ]
    }
    
    def __init__(self, model_registry=None, config: Dict[str, Any] = None):
        """
        初始化检测器
        
        Args:
            model_registry: 模型注册中心
            config: 配置参数
        """
        self.model_registry = model_registry
        self.config = config or {}
        
        # 检测参数
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.nms_threshold = self.config.get('nms_threshold', 0.4)
        self.input_size = self.config.get('input_size', (640, 640))
        
        # 模型名称
        self.visual_model = self.config.get('visual_model', 'transformer_defect_yolo')
        self.thermal_model = self.config.get('thermal_model', 'thermal_anomaly')
        
        # 类别映射
        self.class_mapping = {
            0: DefectType.BUSHING_CRACK,
            1: DefectType.BUSHING_CONTAMINATION,
            2: DefectType.SHIELD_RUST,
            3: DefectType.PORCELAIN_DAMAGE,
            4: DefectType.OIL_LEAK,
            5: DefectType.OIL_LEVEL_ABNORMAL,
            6: DefectType.CONNECTOR_OVERHEAT,
            7: DefectType.RADIATOR_BLOCKAGE,
            8: DefectType.FAN_FAILURE,
            9: DefectType.PAINT_PEELING,
            10: DefectType.CORROSION,
            11: DefectType.BIRD_NEST,
            12: DefectType.FOREIGN_OBJECT
        }
        
        # 检测历史
        self.detection_history: Dict[str, List[DetectionResult]] = {}
        
    def detect_visual(self, image: np.ndarray, device_id: str = None) -> List[DetectionResult]:
        """
        视觉缺陷检测
        
        Args:
            image: 输入图像 (H, W, C)
            device_id: 设备ID
            
        Returns:
            检测结果列表
        """
        results = []
        
        # 预处理图像
        processed = self._preprocess_image(image)
        
        # 模型推理
        if self.model_registry:
            try:
                detections = self.model_registry.infer(
                    self.visual_model,
                    processed
                )
                results = self._parse_detections(detections, image.shape)
            except Exception as e:
                print(f"Visual detection error: {e}")
                # 回退到基于规则的检测
                results = self._rule_based_visual_detection(image)
        else:
            results = self._rule_based_visual_detection(image)
            
        # 更新历史
        if device_id:
            self._update_history(device_id, results)
            
        return results
    
    def detect_thermal(self, thermal_image: np.ndarray, 
                       temperature_range: Tuple[float, float] = None,
                       device_id: str = None) -> List[DetectionResult]:
        """
        热成像缺陷检测
        
        Args:
            thermal_image: 热成像数据
            temperature_range: 温度范围 (min, max)
            device_id: 设备ID
            
        Returns:
            检测结果列表
        """
        results = []
        
        # 温度分析
        if temperature_range:
            temp_min, temp_max = temperature_range
        else:
            temp_min, temp_max = 0, 100
            
        # 提取热点
        hotspots = self._find_hotspots(thermal_image, temp_min, temp_max)
        
        for hotspot in hotspots:
            x, y, w, h, temp, severity = hotspot
            
            if severity >= SeverityLevel.ATTENTION:
                result = DetectionResult(
                    defect_type=DefectType.CONNECTOR_OVERHEAT,
                    severity=severity,
                    confidence=min(0.95, temp / temp_max),
                    location=(x, y, w, h),
                    description=f"检测到过热点，温度约{temp:.1f}°C",
                    recommendations=self.RECOMMENDATIONS.get(DefectType.CONNECTOR_OVERHEAT, []),
                    metadata={
                        'temperature': temp,
                        'temp_rise': temp - temp_min,
                        'analysis_type': 'thermal'
                    }
                )
                results.append(result)
                
        if device_id:
            self._update_history(device_id, results)
            
        return results
    
    def detect_oil_level(self, image: np.ndarray, 
                         oil_gauge_roi: Tuple[int, int, int, int] = None,
                         device_id: str = None) -> List[DetectionResult]:
        """
        油位检测
        
        Args:
            image: 输入图像
            oil_gauge_roi: 油位计区域 (x, y, w, h)
            device_id: 设备ID
            
        Returns:
            检测结果列表
        """
        results = []
        
        # 定位油位计
        if oil_gauge_roi is None:
            oil_gauge_roi = self._locate_oil_gauge(image)
            
        if oil_gauge_roi is None:
            return results
            
        x, y, w, h = oil_gauge_roi
        gauge_image = image[y:y+h, x:x+w]
        
        # 分析油位
        oil_level, confidence = self._analyze_oil_level(gauge_image)
        
        # 判断是否异常
        normal_range = self.config.get('oil_level_range', (0.3, 0.8))
        
        if oil_level < normal_range[0]:
            severity = SeverityLevel.ALARM if oil_level < normal_range[0] * 0.5 else SeverityLevel.WARNING
            result = DetectionResult(
                defect_type=DefectType.OIL_LEVEL_ABNORMAL,
                severity=severity,
                confidence=confidence,
                location=oil_gauge_roi,
                description=f"油位偏低: {oil_level*100:.1f}%",
                recommendations=self.RECOMMENDATIONS.get(DefectType.OIL_LEVEL_ABNORMAL, []) + 
                               ["检查是否存在漏油"],
                metadata={
                    'oil_level': oil_level,
                    'normal_range': normal_range
                }
            )
            results.append(result)
        elif oil_level > normal_range[1]:
            severity = SeverityLevel.WARNING
            result = DetectionResult(
                defect_type=DefectType.OIL_LEVEL_ABNORMAL,
                severity=severity,
                confidence=confidence,
                location=oil_gauge_roi,
                description=f"油位偏高: {oil_level*100:.1f}%",
                recommendations=["检查温度是否过高", "确认油位计读数准确性"],
                metadata={
                    'oil_level': oil_level,
                    'normal_range': normal_range
                }
            )
            results.append(result)
            
        if device_id:
            self._update_history(device_id, results)
            
        return results
    
    def detect_multimodal(self, 
                         visual_image: np.ndarray = None,
                         thermal_image: np.ndarray = None,
                         acoustic_features: np.ndarray = None,
                         gas_data: Dict[str, float] = None,
                         device_id: str = None) -> InspectionReport:
        """
        多模态融合检测
        
        Args:
            visual_image: 可见光图像
            thermal_image: 热成像
            acoustic_features: 声学特征
            gas_data: 气体数据
            device_id: 设备ID
            
        Returns:
            综合巡检报告
        """
        all_detections = []
        
        # 视觉检测
        if visual_image is not None:
            visual_results = self.detect_visual(visual_image, device_id)
            all_detections.extend(visual_results)
            
        # 热成像检测
        if thermal_image is not None:
            thermal_results = self.detect_thermal(thermal_image, device_id=device_id)
            all_detections.extend(thermal_results)
            
        # 声学异常关联
        if acoustic_features is not None:
            acoustic_results = self._analyze_acoustic_correlation(acoustic_features)
            all_detections.extend(acoustic_results)
            
        # 气体异常关联
        if gas_data is not None:
            gas_results = self._analyze_gas_correlation(gas_data)
            all_detections.extend(gas_results)
            
        # 融合分析
        fused_detections = self._fuse_detections(all_detections)
        
        # 生成报告
        report = self._generate_report(device_id, fused_detections)
        
        return report
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 调整尺寸
        h, w = image.shape[:2]
        target_h, target_w = self.input_size
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 简化处理：假设已调整
        processed = np.zeros((1, 3, target_h, target_w), dtype=np.float32)
        
        # 归一化
        if len(image.shape) == 3:
            for c in range(min(3, image.shape[2])):
                processed[0, c] = image[:target_h, :target_w, c].astype(np.float32) / 255.0
                
        return processed
    
    def _parse_detections(self, model_output: np.ndarray, 
                          original_shape: Tuple[int, ...]) -> List[DetectionResult]:
        """解析模型输出"""
        results = []
        
        if model_output is None or len(model_output) == 0:
            return results
            
        # 假设输出格式: [batch, num_detections, 6] (x, y, w, h, confidence, class)
        for det in model_output[0]:
            if len(det) < 6:
                continue
                
            x, y, w, h, conf, cls = det[:6]
            
            if conf < self.confidence_threshold:
                continue
                
            cls_idx = int(cls)
            if cls_idx not in self.class_mapping:
                continue
                
            defect_type = self.class_mapping[cls_idx]
            severity = self._calculate_severity(defect_type, conf)
            
            result = DetectionResult(
                defect_type=defect_type,
                severity=severity,
                confidence=float(conf),
                location=(int(x), int(y), int(w), int(h)),
                description=f"检测到{defect_type.value}",
                recommendations=self.RECOMMENDATIONS.get(defect_type, []),
                metadata={'class_id': cls_idx}
            )
            results.append(result)
            
        return results
    
    def _rule_based_visual_detection(self, image: np.ndarray) -> List[DetectionResult]:
        """基于规则的视觉检测（备用方法）"""
        results = []
        h, w = image.shape[:2]
        
        # 颜色分析检测锈蚀
        if len(image.shape) == 3:
            # 检测棕红色区域（锈蚀）
            hsv = self._rgb_to_hsv(image)
            rust_mask = self._detect_rust_regions(hsv)
            rust_ratio = np.sum(rust_mask) / (h * w)
            
            if rust_ratio > 0.05:
                severity = SeverityLevel.WARNING if rust_ratio > 0.15 else SeverityLevel.ATTENTION
                results.append(DetectionResult(
                    defect_type=DefectType.SHIELD_RUST,
                    severity=severity,
                    confidence=min(0.8, rust_ratio * 5),
                    location=(0, 0, w, h),
                    description=f"检测到锈蚀区域，占比{rust_ratio*100:.1f}%",
                    recommendations=self.RECOMMENDATIONS.get(DefectType.SHIELD_RUST, [])
                ))
                
        return results
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """RGB转HSV"""
        rgb_normalized = rgb.astype(np.float32) / 255.0
        
        r, g, b = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
        
        v = np.maximum(np.maximum(r, g), b)
        c = v - np.minimum(np.minimum(r, g), b)
        
        s = np.where(v != 0, c / v, 0)
        
        h = np.zeros_like(v)
        mask = c != 0
        
        r_max = (v == r) & mask
        g_max = (v == g) & mask
        b_max = (v == b) & mask
        
        h[r_max] = 60 * (((g[r_max] - b[r_max]) / c[r_max]) % 6)
        h[g_max] = 60 * (((b[g_max] - r[g_max]) / c[g_max]) + 2)
        h[b_max] = 60 * (((r[b_max] - g[b_max]) / c[b_max]) + 4)
        
        return np.stack([h, s, v], axis=-1)
    
    def _detect_rust_regions(self, hsv: np.ndarray) -> np.ndarray:
        """检测锈蚀区域"""
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # 锈蚀颜色范围：橙红色到棕色
        rust_mask = ((h >= 0) & (h <= 30) | (h >= 330) & (h <= 360)) & \
                   (s >= 0.3) & (s <= 1.0) & \
                   (v >= 0.2) & (v <= 0.8)
                   
        return rust_mask
    
    def _find_hotspots(self, thermal_image: np.ndarray,
                       temp_min: float, temp_max: float) -> List[Tuple]:
        """查找热点"""
        hotspots = []
        
        # 归一化温度
        if thermal_image.max() > 1:
            normalized = (thermal_image - thermal_image.min()) / (thermal_image.max() - thermal_image.min() + 1e-6)
        else:
            normalized = thermal_image
            
        # 温度映射
        temperatures = normalized * (temp_max - temp_min) + temp_min
        
        # 计算阈值
        mean_temp = np.mean(temperatures)
        std_temp = np.std(temperatures)
        threshold = mean_temp + 2 * std_temp
        
        # 查找高温区域
        hot_mask = temperatures > threshold
        
        if np.any(hot_mask):
            # 简化：取最高温度点
            max_idx = np.unravel_index(np.argmax(temperatures), temperatures.shape)
            max_temp = temperatures[max_idx]
            
            # 确定严重程度
            if max_temp > temp_max * 0.9:
                severity = SeverityLevel.CRITICAL
            elif max_temp > temp_max * 0.7:
                severity = SeverityLevel.ALARM
            elif max_temp > temp_max * 0.5:
                severity = SeverityLevel.WARNING
            else:
                severity = SeverityLevel.ATTENTION
                
            y, x = max_idx
            hotspots.append((x-10, y-10, 20, 20, max_temp, severity))
            
        return hotspots
    
    def _locate_oil_gauge(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """定位油位计"""
        # 简化实现：返回预设区域或None
        h, w = image.shape[:2]
        # 假设油位计在图像右侧
        return (int(w * 0.8), int(h * 0.3), int(w * 0.15), int(h * 0.4))
    
    def _analyze_oil_level(self, gauge_image: np.ndarray) -> Tuple[float, float]:
        """分析油位"""
        # 简化实现：基于图像亮度估计油位
        if len(gauge_image.shape) == 3:
            gray = np.mean(gauge_image, axis=2)
        else:
            gray = gauge_image
            
        h, w = gray.shape
        
        # 分析每行的亮度变化
        row_means = np.mean(gray, axis=1)
        
        # 查找油面位置（亮度变化点）
        diff = np.diff(row_means)
        if len(diff) > 0:
            oil_line = np.argmax(np.abs(diff))
            oil_level = 1.0 - (oil_line / h)
            confidence = min(0.9, np.max(np.abs(diff)) / 50)
        else:
            oil_level = 0.5
            confidence = 0.3
            
        return oil_level, confidence
    
    def _analyze_acoustic_correlation(self, features: np.ndarray) -> List[DetectionResult]:
        """声学异常关联分析"""
        results = []
        
        # 检测局部放电特征
        if features is not None and len(features) > 0:
            # 高频能量检测
            high_freq_energy = np.mean(features[-len(features)//4:])
            
            if high_freq_energy > 0.7:
                results.append(DetectionResult(
                    defect_type=DefectType.BUSHING_CRACK,
                    severity=SeverityLevel.WARNING,
                    confidence=min(0.85, high_freq_energy),
                    location=(0, 0, 0, 0),
                    description="声学检测到可能的局部放电信号",
                    recommendations=["安排局部放电专项检测", "检查套管和绝缘状态"],
                    metadata={'source': 'acoustic', 'high_freq_energy': float(high_freq_energy)}
                ))
                
        return results
    
    def _analyze_gas_correlation(self, gas_data: Dict[str, float]) -> List[DetectionResult]:
        """气体异常关联分析"""
        results = []
        
        # 油色谱分析
        h2 = gas_data.get('H2', 0)
        ch4 = gas_data.get('CH4', 0)
        c2h2 = gas_data.get('C2H2', 0)
        c2h4 = gas_data.get('C2H4', 0)
        co = gas_data.get('CO', 0)
        
        # 判断故障类型
        total_hydrocarbon = ch4 + c2h2 + c2h4
        
        if c2h2 > 10:  # 乙炔超标
            severity = SeverityLevel.CRITICAL if c2h2 > 50 else SeverityLevel.ALARM
            results.append(DetectionResult(
                defect_type=DefectType.CONNECTOR_OVERHEAT,
                severity=severity,
                confidence=0.9,
                location=(0, 0, 0, 0),
                description=f"油色谱检测到乙炔{c2h2}ppm，可能存在电弧放电",
                recommendations=["立即安排油色谱复核", "检查内部连接和绕组", "考虑停电检修"],
                metadata={'source': 'DGA', 'gases': gas_data}
            ))
            
        if h2 > 150:  # 氢气超标
            severity = SeverityLevel.WARNING
            results.append(DetectionResult(
                defect_type=DefectType.BUSHING_CRACK,
                severity=severity,
                confidence=0.75,
                location=(0, 0, 0, 0),
                description=f"油色谱检测到氢气{h2}ppm，可能存在局部过热或放电",
                recommendations=["跟踪监测气体变化趋势", "检查绝缘状态"],
                metadata={'source': 'DGA', 'gases': gas_data}
            ))
            
        return results
    
    def _fuse_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """融合多源检测结果"""
        if not detections:
            return []
            
        # 按缺陷类型分组
        grouped = {}
        for det in detections:
            key = det.defect_type
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(det)
            
        # 融合同类检测
        fused = []
        for defect_type, dets in grouped.items():
            if len(dets) == 1:
                fused.append(dets[0])
            else:
                # 多源检测到同一问题，提升置信度和严重程度
                max_severity = max(d.severity for d in dets)
                avg_confidence = np.mean([d.confidence for d in dets])
                boosted_confidence = min(0.98, avg_confidence * (1 + 0.1 * len(dets)))
                
                # 可能需要提升严重程度
                if len(dets) >= 3 and max_severity.value < SeverityLevel.ALARM.value:
                    max_severity = SeverityLevel(max_severity.value + 1)
                    
                # 合并建议
                all_recommendations = []
                for d in dets:
                    all_recommendations.extend(d.recommendations)
                unique_recommendations = list(dict.fromkeys(all_recommendations))
                
                # 合并元数据
                merged_metadata = {'fusion_count': len(dets), 'sources': []}
                for d in dets:
                    source = d.metadata.get('source', 'visual')
                    merged_metadata['sources'].append(source)
                    merged_metadata.update(d.metadata)
                    
                fused_result = DetectionResult(
                    defect_type=defect_type,
                    severity=max_severity,
                    confidence=boosted_confidence,
                    location=dets[0].location,
                    description=f"多源融合检测确认{defect_type.value}（{len(dets)}个来源）",
                    recommendations=unique_recommendations[:5],
                    metadata=merged_metadata
                )
                fused.append(fused_result)
                
        return fused
    
    def _calculate_severity(self, defect_type: DefectType, confidence: float) -> SeverityLevel:
        """计算严重程度"""
        thresholds = self.SEVERITY_THRESHOLDS.get(defect_type, {
            'attention': 0.4,
            'warning': 0.6,
            'alarm': 0.75,
            'critical': 0.9
        })
        
        if confidence >= thresholds['critical']:
            return SeverityLevel.CRITICAL
        elif confidence >= thresholds['alarm']:
            return SeverityLevel.ALARM
        elif confidence >= thresholds['warning']:
            return SeverityLevel.WARNING
        elif confidence >= thresholds['attention']:
            return SeverityLevel.ATTENTION
        else:
            return SeverityLevel.NORMAL
    
    def _update_history(self, device_id: str, results: List[DetectionResult]):
        """更新检测历史"""
        if device_id not in self.detection_history:
            self.detection_history[device_id] = []
            
        self.detection_history[device_id].extend(results)
        
        # 保留最近100条
        if len(self.detection_history[device_id]) > 100:
            self.detection_history[device_id] = self.detection_history[device_id][-100:]
    
    def _generate_report(self, device_id: str, 
                         detections: List[DetectionResult]) -> InspectionReport:
        """生成巡检报告"""
        # 计算总体状态
        if not detections:
            overall_status = SeverityLevel.NORMAL
        else:
            overall_status = max(d.severity for d in detections)
            
        # 统计信息
        statistics = {
            'total_defects': len(detections),
            'by_severity': {},
            'by_type': {}
        }
        
        for severity in SeverityLevel:
            count = sum(1 for d in detections if d.severity == severity)
            statistics['by_severity'][severity.name] = count
            
        for det in detections:
            type_name = det.defect_type.value
            statistics['by_type'][type_name] = statistics['by_type'].get(type_name, 0) + 1
            
        # 汇总建议
        all_recommendations = []
        for det in detections:
            all_recommendations.extend(det.recommendations)
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        return InspectionReport(
            device_id=device_id or 'unknown',
            timestamp=time.time(),
            overall_status=overall_status,
            detections=detections,
            statistics=statistics,
            recommendations=unique_recommendations[:10]
        )
    
    def get_device_history(self, device_id: str) -> List[DetectionResult]:
        """获取设备检测历史"""
        return self.detection_history.get(device_id, [])
    
    def clear_history(self, device_id: str = None):
        """清除历史"""
        if device_id:
            self.detection_history.pop(device_id, None)
        else:
            self.detection_history.clear()
