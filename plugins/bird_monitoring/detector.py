"""
鸟类检测器 - 基础版本
输变电激光星芒破夜绘明监测平台 (F组)

基于YOLOv8的鸟类检测和跟踪
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class BirdDetector:
    """
    基础鸟类检测器
    
    使用YOLOv8进行鸟类目标检测
    """
    
    # 检测类别
    CLASSES = [
        "sparrow",      # 麻雀
        "magpie",       # 喜鹊
        "crow",         # 乌鸦
        "eagle",        # 老鹰
        "egret",        # 白鹭
        "swallow",      # 燕子
        "dove",         # 斑鸠
        "heron",        # 苍鹭
        "nest",         # 鸟巢
        "bird_generic", # 通用鸟类
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.session = None
        
        # 推理参数
        model_config = config.get("model", {})
        self.input_size = tuple(model_config.get("input_size", [640, 640]))
        self.device = model_config.get("device", "cpu")
        
        inference_config = config.get("inference", {})
        self.confidence_threshold = inference_config.get("confidence_threshold", 0.5)
        self.nms_threshold = inference_config.get("nms_threshold", 0.45)
        self.tracking_enabled = inference_config.get("tracking_enabled", True)
        self.max_track_age = inference_config.get("max_track_age", 30)
        
        # 跟踪状态
        self._tracks: Dict[int, Dict] = {}
        self._next_track_id = 1
        self._frame_count = 0
        
        # 加载模型
        self._load_model(model_config.get("path"))
    
    def _load_model(self, model_path: Optional[str] = None):
        """加载ONNX模型"""
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "bird_yolov8n.onnx"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"[BirdDetector] 模型文件不存在: {model_path}")
            print(f"[BirdDetector] 使用模拟检测模式")
            return
        
        if ort is None:
            print(f"[BirdDetector] onnxruntime未安装，使用模拟模式")
            return
        
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            print(f"[BirdDetector] 模型加载成功: {model_path}")
        except Exception as e:
            print(f"[BirdDetector] 模型加载失败: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测图像中的鸟类
        
        Args:
            image: BGR格式图像
            
        Returns:
            检测结果列表
        """
        self._frame_count += 1
        
        if self.session is None:
            # 模拟模式
            return self._simulate_detection(image)
        
        # 预处理
        input_tensor = self._preprocess(image)
        
        # 推理
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # 后处理
        detections = self._postprocess(outputs, image.shape[:2])
        
        # 跟踪
        if self.tracking_enabled:
            detections = self._update_tracks(detections)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        if cv2 is None:
            return np.zeros((1, 3, *self.input_size), dtype=np.float32)
        
        # 缩放
        resized = cv2.resize(image, self.input_size)
        
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        
        # NHWC -> NCHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 添加batch维度
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _postprocess(self, outputs: List[np.ndarray], original_size: Tuple[int, int]) -> List[Dict]:
        """后处理检测结果"""
        detections = []
        
        # YOLOv8输出格式: [batch, num_detections, 5+num_classes]
        # 5 = x_center, y_center, width, height, confidence
        output = outputs[0]
        
        if len(output.shape) == 3:
            output = output[0]  # 移除batch维度
        
        # 转置如果需要
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        h, w = original_size
        
        for detection in output:
            confidence = detection[4]
            if confidence < self.confidence_threshold:
                continue
            
            # 获取类别
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            if class_confidence < self.confidence_threshold:
                continue
            
            # 坐标转换
            x_center, y_center, box_w, box_h = detection[:4]
            x1 = (x_center - box_w / 2) / self.input_size[0]
            y1 = (y_center - box_h / 2) / self.input_size[1]
            x2 = (x_center + box_w / 2) / self.input_size[0]
            y2 = (y_center + box_h / 2) / self.input_size[1]
            
            detections.append({
                "class_id": int(class_id),
                "class_name": self.CLASSES[class_id] if class_id < len(self.CLASSES) else "unknown",
                "confidence": float(confidence * class_confidence),
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                },
                "track_id": 0,
                "status": "flying",
                "distance": 20.0,  # 默认距离，需要3D估计
                "speed": 0.0,
                "heading": 0.0,
            })
        
        # NMS
        detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """非极大值抑制"""
        if len(detections) == 0:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                det for det in detections
                if self._iou(best["bbox"], det["bbox"]) < self.nms_threshold
            ]
        
        return keep
    
    def _iou(self, box1: Dict, box2: Dict) -> float:
        """计算IoU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """更新跟踪"""
        # 简单的IoU跟踪
        for det in detections:
            best_track_id = None
            best_iou = 0.3  # IoU阈值
            
            for track_id, track in self._tracks.items():
                if track["age"] > self.max_track_age:
                    continue
                
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # 更新现有跟踪
                det["track_id"] = best_track_id
                self._tracks[best_track_id] = {
                    "bbox": det["bbox"],
                    "age": 0,
                    "class_name": det["class_name"],
                }
            else:
                # 创建新跟踪
                det["track_id"] = self._next_track_id
                self._tracks[self._next_track_id] = {
                    "bbox": det["bbox"],
                    "age": 0,
                    "class_name": det["class_name"],
                }
                self._next_track_id += 1
        
        # 增加未匹配跟踪的年龄
        matched_ids = {det["track_id"] for det in detections}
        for track_id in list(self._tracks.keys()):
            if track_id not in matched_ids:
                self._tracks[track_id]["age"] += 1
                if self._tracks[track_id]["age"] > self.max_track_age:
                    del self._tracks[track_id]
        
        return detections
    
    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """模拟检测（用于测试）"""
        # 不产生模拟数据，返回空列表
        return []
    
    def cleanup(self):
        """清理资源"""
        self._tracks.clear()
        self.session = None


class BirdDetectorEnhanced(BirdDetector):
    """
    增强版鸟类检测器
    
    额外功能:
    - 多尺度检测
    - 3D距离估计
    - 飞行轨迹预测
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 3D估计参数
        self.camera_height_m = config.get("camera", {}).get("height_m", 15.0)
        self.camera_fov_deg = config.get("camera", {}).get("fov_deg", 60.0)
        self.reference_bird_size_m = 0.3  # 参考鸟类尺寸
        
        # 轨迹历史
        self._trajectory_history: Dict[int, List[Dict]] = {}
        self._max_trajectory_length = 30
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """增强检测"""
        detections = super().detect(image)
        
        # 估计3D距离
        for det in detections:
            det["distance"] = self._estimate_distance(det, image.shape[:2])
            det["status"] = self._determine_status(det)
        
        # 更新轨迹并预测
        for det in detections:
            self._update_trajectory(det)
            det["speed"], det["heading"] = self._estimate_velocity(det["track_id"])
        
        return detections
    
    def _estimate_distance(self, detection: Dict, image_size: Tuple[int, int]) -> float:
        """估计鸟类与输电线的距离"""
        bbox = detection["bbox"]
        
        # 基于边界框大小估计距离
        box_height = bbox["height"] * image_size[0]
        
        if box_height < 1:
            return 50.0  # 太小，认为很远
        
        # 简单的透视距离估计
        # 假设鸟类平均高度约0.3米
        focal_length = image_size[1] / (2 * np.tan(np.radians(self.camera_fov_deg / 2)))
        distance = (self.reference_bird_size_m * focal_length) / box_height
        
        return float(np.clip(distance, 1.0, 100.0))
    
    def _determine_status(self, detection: Dict) -> str:
        """判断鸟类状态"""
        track_id = detection["track_id"]
        
        if track_id in self._trajectory_history:
            history = self._trajectory_history[track_id]
            if len(history) >= 5:
                # 检查位置变化
                recent = history[-5:]
                x_var = np.var([h["x"] for h in recent])
                y_var = np.var([h["y"] for h in recent])
                
                if x_var < 0.001 and y_var < 0.001:
                    return "perched"  # 停留
        
        # 检查是否接近
        if detection.get("distance", 50) < 10:
            return "approaching"
        
        return "flying"
    
    def _update_trajectory(self, detection: Dict):
        """更新轨迹历史"""
        track_id = detection["track_id"]
        bbox = detection["bbox"]
        
        point = {
            "x": bbox["x"] + bbox["width"] / 2,
            "y": bbox["y"] + bbox["height"] / 2,
            "frame": self._frame_count,
        }
        
        if track_id not in self._trajectory_history:
            self._trajectory_history[track_id] = []
        
        self._trajectory_history[track_id].append(point)
        
        # 限制长度
        if len(self._trajectory_history[track_id]) > self._max_trajectory_length:
            self._trajectory_history[track_id].pop(0)
    
    def _estimate_velocity(self, track_id: int) -> Tuple[float, float]:
        """估计速度和航向"""
        if track_id not in self._trajectory_history:
            return 0.0, 0.0
        
        history = self._trajectory_history[track_id]
        if len(history) < 2:
            return 0.0, 0.0
        
        # 使用最近两个点计算
        p1, p2 = history[-2], history[-1]
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        dt = p2["frame"] - p1["frame"]
        
        if dt == 0:
            return 0.0, 0.0
        
        # 速度（像素/帧 -> 米/秒，假设30fps）
        speed_px = np.sqrt(dx**2 + dy**2) / dt
        speed_ms = speed_px * 30 * 0.1  # 简化转换
        
        # 航向
        heading = np.degrees(np.arctan2(dy, dx))
        
        return float(speed_ms), float(heading)
    
    def cleanup(self):
        """清理资源"""
        super().cleanup()
        self._trajectory_history.clear()
