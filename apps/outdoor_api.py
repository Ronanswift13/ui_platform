"""
室外监测中心API端点
====================

为室外监测中心提供统一的检测API接口

功能模块:
1. 主变巡视 - 油位、硅胶、漏油检测
2. 开关巡视 - 断路器/隔离开关状态识别
3. 母线巡视 - 绝缘子缺陷、销钉检测
4. 电容器巡视 - 倾斜/缺失/入侵检测
5. 表计读数 - 指针/数字识别
6. 鸟类监控 - 鸟类检测与驱离

版本: 3.6.0
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import base64
import numpy as np
import cv2
import hashlib
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/outdoor", tags=["室外监测"])


# =============================================================================
# 数据模型
# =============================================================================
class DetectionRequest(BaseModel):
    """检测请求"""
    module: str
    plugin_id: str
    image: str  # Base64编码的图像
    detection_types: List[str] = []
    confidence_threshold: float = 0.6


class DetectionResult(BaseModel):
    """检测结果"""
    type: str
    label: str
    confidence: float
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}


class AlarmInfo(BaseModel):
    """告警信息"""
    type: str
    level: str  # info, warning, error, critical
    message: str
    timestamp: str
    module: str
    detection_id: Optional[str] = None


class DetectionResponse(BaseModel):
    """检测响应"""
    success: bool
    module: str
    timestamp: str
    inference_time_ms: float
    detections: List[Dict[str, Any]] = []
    alarms: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


# =============================================================================
# 模块配置
# =============================================================================
MODULE_CONFIGS = {
    "transformer": {
        "plugin_id": "transformer_inspection",
        "model_id": "transformer_defect_yolov8",
        "detection_classes": {
            "oil_level": {"label": "油位", "alarm_threshold": 0.7},
            "silica_gel": {"label": "硅胶状态", "alarm_threshold": 0.8},
            "oil_leak": {"label": "漏油", "alarm_threshold": 0.6},
            "bushing_crack": {"label": "套管裂纹", "alarm_threshold": 0.5},
            "rust": {"label": "锈蚀", "alarm_threshold": 0.7},
            "thermal_anomaly": {"label": "热异常", "alarm_threshold": 0.6}
        }
    },
    "switch": {
        "plugin_id": "switch_inspection",
        "model_id": "switch_state_yolov8s",
        "detection_classes": {
            "breaker_open": {"label": "断路器分", "alarm_threshold": 0.8},
            "breaker_closed": {"label": "断路器合", "alarm_threshold": 0.8},
            "isolator_open": {"label": "隔离开关分", "alarm_threshold": 0.8},
            "isolator_closed": {"label": "隔离开关合", "alarm_threshold": 0.8},
            "indicator_red": {"label": "红色指示灯", "alarm_threshold": 0.9},
            "indicator_green": {"label": "绿色指示灯", "alarm_threshold": 0.9}
        }
    },
    "busbar": {
        "plugin_id": "busbar_inspection",
        "model_id": "busbar_defect_yolov8m",
        "detection_classes": {
            "insulator_crack": {"label": "绝缘子裂纹", "alarm_threshold": 0.5},
            "insulator_dirty": {"label": "绝缘子污损", "alarm_threshold": 0.6},
            "pin_missing": {"label": "销钉缺失", "alarm_threshold": 0.5},
            "fitting_loose": {"label": "金具松动", "alarm_threshold": 0.6},
            "fitting_rust": {"label": "金具锈蚀", "alarm_threshold": 0.7},
            "wire_damage": {"label": "导线破损", "alarm_threshold": 0.5},
            "foreign_object": {"label": "异物悬挂", "alarm_threshold": 0.6},
            "flashover": {"label": "闪络痕迹", "alarm_threshold": 0.4}
        }
    },
    "capacitor": {
        "plugin_id": "capacitor_inspection",
        "model_id": "capacitor_unit_yolov8",
        "detection_classes": {
            "capacitor_unit": {"label": "电容器单元", "alarm_threshold": 0.9},
            "capacitor_tilted": {"label": "电容器倾斜", "alarm_threshold": 0.5},
            "capacitor_fallen": {"label": "电容器倒塌", "alarm_threshold": 0.4},
            "capacitor_missing": {"label": "电容器缺失", "alarm_threshold": 0.5},
            "person_intrusion": {"label": "人员入侵", "alarm_threshold": 0.6},
            "animal_intrusion": {"label": "动物入侵", "alarm_threshold": 0.6}
        }
    },
    "meter": {
        "plugin_id": "meter_reading",
        "model_id": "meter_keypoint_hrnet",
        "detection_classes": {
            "meter_dial": {"label": "表盘", "alarm_threshold": 0.9},
            "pointer": {"label": "指针", "alarm_threshold": 0.8},
            "digital_display": {"label": "数字显示", "alarm_threshold": 0.9},
            "reading_value": {"label": "读数值", "alarm_threshold": 0.7}
        }
    },
    "bird": {
        "plugin_id": "bird_monitoring",
        "model_id": "bird_detector_yolov8",
        "detection_classes": {
            "bird": {"label": "鸟类", "alarm_threshold": 0.7},
            "bird_nest": {"label": "鸟巢", "alarm_threshold": 0.6},
            "bird_feces": {"label": "鸟粪", "alarm_threshold": 0.7},
            "high_risk_bird": {"label": "高风险鸟类", "alarm_threshold": 0.5}
        }
    }
}


# =============================================================================
# 检测器基类
# =============================================================================
class BaseDetector:
    """检测器基类"""
    
    def __init__(self, module_config: Dict[str, Any]):
        self.config = module_config
        self.model = None
        self.initialized = False
    
    def initialize(self):
        """初始化检测器"""
        # TODO: 加载实际的ONNX模型
        self.initialized = True
        return True
    
    def detect(self, image: np.ndarray, detection_types: List[str], 
               confidence_threshold: float) -> List[Dict[str, Any]]:
        """执行检测"""
        raise NotImplementedError
    
    def _simulate_detection(self, image: np.ndarray, detection_types: List[str],
                           confidence_threshold: float) -> List[Dict[str, Any]]:
        """模拟检测结果（用于开发测试）"""
        import random
        
        detections = []
        h, w = image.shape[:2]
        
        classes = self.config.get("detection_classes", {})
        
        for det_type in detection_types:
            if det_type in classes and random.random() > 0.7:
                conf = random.uniform(confidence_threshold, 1.0)
                
                # 随机生成边界框
                x = random.uniform(0.1, 0.7)
                y = random.uniform(0.1, 0.7)
                box_w = random.uniform(0.1, 0.3)
                box_h = random.uniform(0.1, 0.3)
                
                detections.append({
                    "type": det_type,
                    "label": classes[det_type]["label"],
                    "confidence": round(conf, 3),
                    "bbox": [round(x, 4), round(y, 4), round(box_w, 4), round(box_h, 4)],
                    "metadata": {
                        "class_id": list(classes.keys()).index(det_type),
                        "source": "simulation"
                    }
                })
        
        return detections


# =============================================================================
# 母线巡视检测器
# =============================================================================
class BusbarDetector(BaseDetector):
    """母线巡视检测器
    
    检测功能:
    - 绝缘子裂纹/污损检测
    - 销钉缺失检测
    - 金具松动/锈蚀检测
    - 导线破损检测
    - 异物悬挂检测
    - 闪络痕迹检测
    """
    
    def __init__(self):
        super().__init__(MODULE_CONFIGS["busbar"])
        self.tile_size = 640
        self.tile_overlap = 128
    
    def detect(self, image: np.ndarray, detection_types: List[str],
               confidence_threshold: float) -> List[Dict[str, Any]]:
        """执行母线检测"""
        
        if not self.initialized:
            self.initialize()
        
        # 图像质量评估
        quality_score = self._assess_quality(image)
        
        if quality_score < 0.3:
            return [{
                "type": "quality_warning",
                "label": "图像质量不足",
                "confidence": quality_score,
                "bbox": None,
                "metadata": {"reason": "low_quality"}
            }]
        
        # 对于高分辨率图像，使用切片检测
        h, w = image.shape[:2]
        if w > 1920 or h > 1080:
            detections = self._detect_with_tiling(image, detection_types, confidence_threshold)
        else:
            detections = self._detect_single(image, detection_types, confidence_threshold)
        
        # 后处理：NMS合并
        detections = self._nms_merge(detections)
        
        return detections
    
    def _assess_quality(self, image: np.ndarray) -> float:
        """评估图像质量"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 清晰度评估 (Laplacian方差)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity = min(laplacian.var() / 1500.0, 1.0)
        
        # 对比度评估
        contrast = (gray.max() - gray.min()) / 255.0
        
        # 曝光评估
        mean_brightness = gray.mean() / 255.0
        exposure_score = 1.0 - abs(mean_brightness - 0.5) * 2
        
        return (clarity * 0.4 + contrast * 0.3 + exposure_score * 0.3)
    
    def _detect_with_tiling(self, image: np.ndarray, detection_types: List[str],
                           confidence_threshold: float) -> List[Dict[str, Any]]:
        """切片检测（用于4K大图）"""
        h, w = image.shape[:2]
        stride = self.tile_size - self.tile_overlap
        
        all_detections = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # 提取切片
                x1 = min(x, w - self.tile_size)
                y1 = min(y, h - self.tile_size)
                x2 = x1 + self.tile_size
                y2 = y1 + self.tile_size
                
                tile = image[y1:y2, x1:x2]
                
                # 检测切片
                tile_detections = self._detect_single(tile, detection_types, confidence_threshold)
                
                # 转换坐标到原图
                for det in tile_detections:
                    if det.get("bbox"):
                        bx, by, bw, bh = det["bbox"]
                        det["bbox"] = [
                            (x1 + bx * self.tile_size) / w,
                            (y1 + by * self.tile_size) / h,
                            bw * self.tile_size / w,
                            bh * self.tile_size / h
                        ]
                    all_detections.append(det)
        
        return all_detections
    
    def _detect_single(self, image: np.ndarray, detection_types: List[str],
                      confidence_threshold: float) -> List[Dict[str, Any]]:
        """单图检测"""
        # 如果有实际模型，在这里调用
        # return self._model_inference(image, detection_types, confidence_threshold)
        
        # 模拟检测
        return self._simulate_detection(image, detection_types, confidence_threshold)
    
    def _nms_merge(self, detections: List[Dict[str, Any]], 
                   iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """NMS合并重叠检测"""
        if len(detections) <= 1:
            return detections
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)
        
        kept = []
        for det in detections:
            if not det.get("bbox"):
                kept.append(det)
                continue
            
            should_keep = True
            for kept_det in kept:
                if not kept_det.get("bbox"):
                    continue
                if det["type"] != kept_det["type"]:
                    continue
                
                iou = self._calculate_iou(det["bbox"], kept_det["bbox"])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(det)
        
        return kept
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


# =============================================================================
# 电容器巡视检测器
# =============================================================================
class CapacitorDetector(BaseDetector):
    """电容器巡视检测器
    
    检测功能:
    - 电容器单元识别
    - 倾斜/倒塌检测
    - 缺失检测
    - 人员/动物入侵检测
    """
    
    def __init__(self):
        super().__init__(MODULE_CONFIGS["capacitor"])
        self.reference_positions = []  # 电容器参考位置
    
    def detect(self, image: np.ndarray, detection_types: List[str],
               confidence_threshold: float) -> List[Dict[str, Any]]:
        """执行电容器检测"""
        
        if not self.initialized:
            self.initialize()
        
        detections = []
        
        # 电容器结构检测
        if any(t in detection_types for t in ["capacitor_unit", "capacitor_tilted", 
                                               "capacitor_fallen", "capacitor_missing"]):
            structure_detections = self._detect_structure(image, confidence_threshold)
            detections.extend(structure_detections)
        
        # 入侵检测
        if any(t in detection_types for t in ["person_intrusion", "animal_intrusion"]):
            intrusion_detections = self._detect_intrusion(image, confidence_threshold)
            detections.extend(intrusion_detections)
        
        return detections
    
    def _detect_structure(self, image: np.ndarray, 
                         confidence_threshold: float) -> List[Dict[str, Any]]:
        """检测电容器结构"""
        # 模拟检测
        return self._simulate_detection(
            image, 
            ["capacitor_unit", "capacitor_tilted", "capacitor_fallen", "capacitor_missing"],
            confidence_threshold
        )
    
    def _detect_intrusion(self, image: np.ndarray,
                         confidence_threshold: float) -> List[Dict[str, Any]]:
        """检测入侵"""
        return self._simulate_detection(
            image,
            ["person_intrusion", "animal_intrusion"],
            confidence_threshold
        )


# =============================================================================
# 检测器工厂
# =============================================================================
DETECTORS = {
    "busbar": BusbarDetector(),
    "capacitor": CapacitorDetector(),
}


def get_detector(module: str) -> BaseDetector:
    """获取检测器实例"""
    if module in DETECTORS:
        return DETECTORS[module]
    
    # 返回通用检测器
    if module in MODULE_CONFIGS:
        detector = BaseDetector(MODULE_CONFIGS[module])
        detector.detect = lambda img, types, conf: detector._simulate_detection(img, types, conf)
        return detector
    
    raise ValueError(f"未知模块: {module}")


# =============================================================================
# API端点
# =============================================================================
@router.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """
    执行室外监测检测
    
    支持的模块:
    - transformer: 主变巡视
    - switch: 开关巡视
    - busbar: 母线巡视
    - capacitor: 电容器巡视
    - meter: 表计读数
    - bird: 鸟类监控
    """
    start_time = time.time()
    
    try:
        # 解码图像
        image_data = request.image
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像")
        
        # 获取检测器
        detector = get_detector(request.module)
        
        # 执行检测
        detection_types = request.detection_types
        if not detection_types and request.module in MODULE_CONFIGS:
            detection_types = list(MODULE_CONFIGS[request.module]["detection_classes"].keys())
        
        detections = detector.detect(image, detection_types, request.confidence_threshold)
        
        # 生成告警
        alarms = []
        config = MODULE_CONFIGS.get(request.module, {})
        classes = config.get("detection_classes", {})
        
        for det in detections:
            det_type = det.get("type")
            conf = det.get("confidence", 0)
            
            if det_type in classes:
                threshold = classes[det_type].get("alarm_threshold", 0.7)
                if conf >= threshold and det_type not in ["capacitor_unit", "meter_dial"]:
                    alarms.append({
                        "type": det_type,
                        "level": "warning" if conf < 0.8 else "error",
                        "message": f"检测到{det.get('label', det_type)}，置信度{conf:.1%}",
                        "timestamp": datetime.now().isoformat(),
                        "module": request.module,
                        "detection_id": hashlib.md5(f"{det_type}{conf}{time.time()}".encode()).hexdigest()[:8]
                    })
        
        inference_time = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            success=True,
            module=request.module,
            timestamp=datetime.now().isoformat(),
            inference_time_ms=round(inference_time, 2),
            detections=detections,
            alarms=alarms,
            metadata={
                "image_size": [image.shape[1], image.shape[0]],
                "detection_count": len(detections),
                "alarm_count": len(alarms)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modules")
async def get_modules():
    """获取所有模块配置"""
    return {
        "modules": {
            module_id: {
                "plugin_id": config["plugin_id"],
                "detection_classes": list(config["detection_classes"].keys())
            }
            for module_id, config in MODULE_CONFIGS.items()
        }
    }


@router.get("/module/{module_id}/config")
async def get_module_config(module_id: str):
    """获取模块配置"""
    if module_id not in MODULE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"模块不存在: {module_id}")
    
    return MODULE_CONFIGS[module_id]


@router.post("/module/{module_id}/config")
async def update_module_config(module_id: str, config: Dict[str, Any] = Body(...)):
    """更新模块配置"""
    if module_id not in MODULE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"模块不存在: {module_id}")
    
    # 更新配置
    for key, value in config.items():
        if key in MODULE_CONFIGS[module_id]:
            MODULE_CONFIGS[module_id][key] = value
    
    return {"success": True, "module": module_id}


# =============================================================================
# 注册路由
# =============================================================================
def register_outdoor_routes(app):
    """注册室外监测路由"""
    app.include_router(router)
    logger.info("[API] 室外监测API已注册")
