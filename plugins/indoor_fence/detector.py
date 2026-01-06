"""
室内检测器 - 基础版本
输变电激光星芒破夜绘明监测平台 (G组)

基于YOLOv8的人员检测和视觉-雷达融合
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


class IndoorDetector:
    """
    室内人员检测器
    
    使用YOLOv8进行人员目标检测
    """
    
    CLASSES = ["person"]
    
    def __init__(self, config: Dict[str, Any]):
        """初始化检测器"""
        self.config = config
        self.session = None
        
        model_config = config.get("model", {})
        self.input_size = tuple(model_config.get("input_size", [640, 640]))
        self.device = model_config.get("device", "cpu")
        
        inference_config = config.get("inference", {})
        self.confidence_threshold = inference_config.get("confidence_threshold", 0.5)
        self.nms_threshold = inference_config.get("nms_threshold", 0.45)
        
        # 跟踪
        self._tracks: Dict[int, Dict] = {}
        self._next_track_id = 1
        
        # 加载模型
        self._load_model(model_config.get("path"))
    
    def _load_model(self, model_path: Optional[str] = None):
        """加载ONNX模型"""
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "person_yolov8n.onnx"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"[IndoorDetector] 模型文件不存在: {model_path}")
            return
        
        if ort is None:
            print(f"[IndoorDetector] onnxruntime未安装")
            return
        
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            print(f"[IndoorDetector] 模型加载成功")
        except Exception as e:
            print(f"[IndoorDetector] 模型加载失败: {e}")
    
    def detect_persons(self, image: np.ndarray) -> List[Dict]:
        """检测图像中的人员"""
        if self.session is None:
            return self._simulate_detection(image)
        
        # 预处理
        input_tensor = self._preprocess(image)
        
        # 推理
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # 后处理
        detections = self._postprocess(outputs, image.shape[:2])
        
        # 跟踪
        detections = self._update_tracks(detections)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理"""
        if cv2 is None:
            return np.zeros((1, 3, *self.input_size), dtype=np.float32)
        
        resized = cv2.resize(image, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)
    
    def _postprocess(self, outputs: List[np.ndarray], original_size: Tuple[int, int]) -> List[Dict]:
        """后处理"""
        detections = []
        output = outputs[0]
        
        if len(output.shape) == 3:
            output = output[0]
        
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        h, w = original_size
        
        for det in output:
            confidence = det[4]
            if confidence < self.confidence_threshold:
                continue
            
            # 只检测人员 (class 0)
            class_id = 0
            if len(det) > 5:
                class_scores = det[5:]
                if class_scores[0] < self.confidence_threshold:
                    continue
            
            x_center, y_center, box_w, box_h = det[:4]
            x1 = (x_center - box_w / 2) / self.input_size[0]
            y1 = (y_center - box_h / 2) / self.input_size[1]
            x2 = (x_center + box_w / 2) / self.input_size[0]
            y2 = (y_center + box_h / 2) / self.input_size[1]
            
            detections.append({
                "class_id": 0,
                "class_name": "person",
                "confidence": float(confidence),
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                },
                "track_id": 0,
                "foot_position": {
                    "x": float((x1 + x2) / 2),
                    "y": float(y2),
                },
            })
        
        return self._nms(detections)
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """非极大值抑制"""
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if self._iou(best["bbox"], d["bbox"]) < self.nms_threshold]
        
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
        for det in detections:
            best_id = None
            best_iou = 0.3
            
            for tid, track in self._tracks.items():
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            
            if best_id:
                det["track_id"] = best_id
                self._tracks[best_id] = {"bbox": det["bbox"], "age": 0}
            else:
                det["track_id"] = self._next_track_id
                self._tracks[self._next_track_id] = {"bbox": det["bbox"], "age": 0}
                self._next_track_id += 1
        
        # 清理旧跟踪
        matched = {d["track_id"] for d in detections}
        for tid in list(self._tracks.keys()):
            if tid not in matched:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > 30:
                    del self._tracks[tid]
        
        return detections
    
    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """模拟检测"""
        return []
    
    def cleanup(self):
        """清理"""
        self._tracks.clear()
        self.session = None


class IndoorDetectorEnhanced(IndoorDetector):
    """
    增强版室内检测器
    
    额外功能:
    - 姿态估计
    - 行为分析
    - 透视变换坐标映射
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 透视变换矩阵 (需要标定)
        self._homography: Optional[np.ndarray] = None
        self._load_homography()
    
    def _load_homography(self):
        """加载透视变换矩阵"""
        h_path = Path(__file__).parent / "configs" / "homography.npy"
        if h_path.exists():
            try:
                self._homography = np.load(str(h_path))
                print("[IndoorDetectorEnhanced] 透视变换矩阵已加载")
            except Exception as e:
                print(f"[IndoorDetectorEnhanced] 加载透视变换矩阵失败: {e}")
    
    def detect_persons(self, image: np.ndarray) -> List[Dict]:
        """增强检测"""
        detections = super().detect_persons(image)
        
        # 映射到地面坐标
        for det in detections:
            det["ground_position"] = self._map_to_ground(det["foot_position"], image.shape[:2])
        
        return detections
    
    def _map_to_ground(self, foot_pos: Dict, image_size: Tuple[int, int]) -> Dict:
        """将脚部位置映射到地面坐标"""
        if self._homography is None:
            # 无标定时使用简单映射
            return {
                "x": foot_pos["x"],
                "y": foot_pos["y"],
                "z": 0.0,
            }
        
        # 像素坐标
        px = foot_pos["x"] * image_size[1]
        py = foot_pos["y"] * image_size[0]
        
        # 透视变换
        pt = np.array([px, py, 1.0])
        ground_pt = self._homography @ pt
        ground_pt = ground_pt / ground_pt[2]
        
        return {
            "x": float(ground_pt[0]),
            "y": float(ground_pt[1]),
            "z": 0.0,
        }
