#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机适配器 V2.1 - 工程级实现
==========================================

封装视觉检测和跟踪:
- YOLO/ONNX人体检测 (真实推理)
- OpenCV视频流采集
- ByteTrack轻量跟踪
- 边界框和脚点输出
- 统一数据格式

支持的输入源:
- USB摄像头 (设备索引)
- RTSP流 (网络摄像机)
- HTTP流 (MJPEG)
- 本地视频文件

作者: G组 | 版本: 2.1.0
"""

from __future__ import annotations
import logging
import time
import random
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from queue import Queue, Empty
import numpy as np

from .base_adapter import BaseAdapter, AdapterConfig, AdapterStatus

logger = logging.getLogger(__name__)

# 尝试导入OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    logger.warning("OpenCV未安装，相机功能将使用模拟模式")

# 尝试导入ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None
    logger.warning("ONNXRuntime未安装，检测功能将使用模拟模式")


@dataclass
class CameraConfig(AdapterConfig):
    """相机配置"""
    # 连接参数
    source: str = "0"               # 视频源: 设备索引/RTSP URL/文件路径
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30

    # 检测模型参数
    model_path: str = "models/indoor/person_yolov8n.onnx"
    device: str = "cpu"             # cpu/cuda
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45

    # 跟踪参数
    tracking_enabled: bool = True
    max_track_age: int = 30
    min_hits: int = 3

    # 相机安装参数 (用于坐标变换)
    height_m: float = 3.0           # 安装高度
    tilt_deg: float = 40.0          # 俯仰角度

    # 内参 (需要标定)
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0

    # 缓冲配置
    buffer_size: int = 5
    use_threading: bool = True


@dataclass
class PersonDetection:
    """人员检测结果"""
    detection_id: str               # 临时检测ID
    bbox: Dict[str, float]          # {x, y, width, height} 归一化坐标
    foot_pixel: Tuple[float, float] # 脚部像素坐标 (原始分辨率)
    confidence: float               # 置信度
    timestamp: float                # 时间戳

    # 可选跟踪ID
    track_id: Optional[str] = None

    # 额外信息
    class_name: str = "person"
    metadata: Dict[str, Any] = field(default_factory=dict)


class YOLOv8Detector:
    """
    YOLOv8 ONNX检测器

    支持YOLOv8n/s/m模型的ONNX推理
    """

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.5, nms_threshold: float = 0.45):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = (640, 640)  # YOLOv8默认输入尺寸

        # 类别 (COCO person = 0)
        self.target_classes = [0]  # 只检测人

    def load(self) -> bool:
        """加载模型"""
        if not ONNX_AVAILABLE:
            logger.error("ONNXRuntime未安装")
            return False

        try:
            # 配置执行提供者
            providers = ['CPUExecutionProvider']
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            # 创建会话
            self.session = ort.InferenceSession(self.model_path, providers=providers)

            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            # 获取输入尺寸
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) == 4:
                self.input_shape = (input_shape[2], input_shape[3])

            logger.info(f"YOLOv8模型加载成功: {self.model_path}")
            logger.info(f"输入尺寸: {self.input_shape}, 设备: {self.device}")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测图像中的人员

        Args:
            frame: BGR图像 (H, W, 3)

        Returns:
            检测结果列表 [{bbox, confidence, class_id}, ...]
        """
        if self.session is None:
            return []

        try:
            # 预处理
            input_tensor, scale, pad = self._preprocess(frame)

            # 推理
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # 后处理
            detections = self._postprocess(outputs[0], frame.shape, scale, pad)

            return detections

        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """预处理图像"""
        h, w = frame.shape[:2]
        target_h, target_w = self.input_shape

        # 计算缩放比例 (保持宽高比)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 填充到目标尺寸
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # 转换格式: BGR -> RGB, HWC -> CHW, uint8 -> float32
        input_tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

        # 添加batch维度
        input_tensor = input_tensor[np.newaxis, ...]

        return input_tensor, scale, (pad_w, pad_h)

    def _postprocess(self, output: np.ndarray, orig_shape: Tuple[int, int, int],
                     scale: float, pad: Tuple[int, int]) -> List[Dict]:
        """后处理检测结果"""
        h, w = orig_shape[:2]
        pad_w, pad_h = pad

        # YOLOv8输出格式: (1, 84, 8400) 或 (1, 5+nc, num_boxes)
        # 84 = 4 (bbox) + 80 (classes)
        if len(output.shape) == 3:
            output = output[0]  # 移除batch维度

        # 转置: (84, 8400) -> (8400, 84)
        if output.shape[0] < output.shape[1]:
            output = output.T

        detections = []

        for det in output:
            # 获取类别置信度
            class_scores = det[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # 只保留目标类别 (人)
            if class_id not in self.target_classes:
                continue

            if confidence < self.conf_threshold:
                continue

            # 获取边界框 (center_x, center_y, width, height)
            cx, cy, bw, bh = det[:4]

            # 转换为角点坐标
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # 还原到原始图像坐标
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale

            # 裁剪到图像范围
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': int(class_id)
            })

        # NMS
        if detections:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """非极大值抑制"""
        if not detections:
            return []

        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        # 计算面积
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # 按置信度排序
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IoU小于阈值的
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]


class SimpleTracker:
    """
    简单的多目标跟踪器

    基于IoU匹配的轻量级跟踪
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._tracks: Dict[int, Dict] = {}
        self._next_id = 1

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新跟踪器

        Args:
            detections: 检测结果列表 [{bbox, confidence}, ...]

        Returns:
            跟踪结果列表 [{bbox, track_id, confidence}, ...]
        """
        if not detections:
            # 更新所有轨迹年龄
            self._age_tracks()
            return []

        # 匹配检测和轨迹
        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        # 更新匹配的轨迹
        for det_idx, track_id in matched:
            det = detections[det_idx]
            self._tracks[track_id]['bbox'] = det['bbox']
            self._tracks[track_id]['confidence'] = det['confidence']
            self._tracks[track_id]['hits'] += 1
            self._tracks[track_id]['age'] = 0

        # 创建新轨迹
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'hits': 1,
                'age': 0,
                'track_id': track_id
            }

        # 更新未匹配轨迹年龄
        for track_id in unmatched_tracks:
            self._tracks[track_id]['age'] += 1

        # 删除过期轨迹
        self._remove_old_tracks()

        # 返回有效轨迹
        results = []
        for track_id, track in self._tracks.items():
            if track['hits'] >= self.min_hits:
                results.append({
                    'bbox': track['bbox'],
                    'track_id': track_id,
                    'confidence': track['confidence']
                })

        return results

    def _match(self, detections: List[Dict]) -> Tuple[List, List, List]:
        """匹配检测和轨迹"""
        if not self._tracks:
            return [], list(range(len(detections))), []

        # 计算IoU矩阵
        det_boxes = np.array([d['bbox'] for d in detections])
        track_ids = list(self._tracks.keys())
        track_boxes = np.array([self._tracks[tid]['bbox'] for tid in track_ids])

        iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)

        # 贪婪匹配
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(track_ids)

        while True:
            if not unmatched_dets or not unmatched_tracks:
                break

            # 找最大IoU
            max_iou = 0
            max_det_idx = -1
            max_track_idx = -1

            for det_idx in unmatched_dets:
                for i, track_id in enumerate(unmatched_tracks):
                    track_idx = track_ids.index(track_id)
                    iou = iou_matrix[det_idx, track_idx]
                    if iou > max_iou and iou >= self.iou_threshold:
                        max_iou = iou
                        max_det_idx = det_idx
                        max_track_idx = track_id

            if max_det_idx >= 0:
                matched.append((max_det_idx, max_track_idx))
                unmatched_dets.remove(max_det_idx)
                unmatched_tracks.remove(max_track_idx)
            else:
                break

        return matched, unmatched_dets, unmatched_tracks

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算IoU矩阵"""
        n1 = len(boxes1)
        n2 = len(boxes2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])

        return iou_matrix

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter / (area1 + area2 - inter + 1e-8)
        return iou

    def _age_tracks(self):
        """增加所有轨迹年龄"""
        for track_id in list(self._tracks.keys()):
            self._tracks[track_id]['age'] += 1

    def _remove_old_tracks(self):
        """删除过期轨迹"""
        to_remove = [tid for tid, t in self._tracks.items() if t['age'] > self.max_age]
        for tid in to_remove:
            del self._tracks[tid]

    def reset(self):
        """重置跟踪器"""
        self._tracks.clear()
        self._next_id = 1


class CameraAdapter(BaseAdapter):
    """
    相机适配器 V2.1

    提供统一的人员检测接口:
    - get_person_detections() -> List[PersonDetection]

    内部封装YOLO检测和跟踪
    """

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self.config: CameraConfig = config

        # 视频捕获
        self._capture = None
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_queue: Queue = Queue(maxsize=config.buffer_size)
        self._capture_running = False

        # 检测器 (YOLO)
        self._detector: Optional[YOLOv8Detector] = None

        # 跟踪器
        self._tracker: Optional[SimpleTracker] = None

        # 帧计数
        self._frame_count = 0
        self._detection_id_counter = 0

        # 最新帧缓存
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # 仿真注入 (由 plugin.py 设置, 与生产路径隔离)
        self._simulator = None           # Simulator 实例
        self._sim_renderer = None        # SimulationRenderer 实例

        logger.info(f"相机适配器初始化: source={config.source}")

    def connect(self) -> bool:
        """连接相机"""
        self._stats['connect_attempts'] += 1
        self._status = AdapterStatus.CONNECTING

        try:
            # 1. 尝试打开相机
            if CV2_AVAILABLE:
                success = self._connect_camera()
                if success:
                    # 2. 加载检测模型
                    model_loaded = self._load_detector()

                    # 3. 初始化跟踪器
                    if self.config.tracking_enabled:
                        self._tracker = SimpleTracker(
                            max_age=self.config.max_track_age,
                            min_hits=self.config.min_hits
                        )

                    # 4. 启动采集线程
                    if self.config.use_threading:
                        self._start_capture_thread()

                    self._status = AdapterStatus.CONNECTED
                    self._connected = True
                    logger.info(f"相机连接成功: {self.config.source}")
                    logger.info(f"检测模型: {'已加载' if model_loaded else '使用模拟'}")
                    return True

            # 硬件不可用，尝试模拟模式
            if self.config.simulate_if_unavailable:
                self._set_simulated()
                self._tracker = SimpleTracker(
                    max_age=self.config.max_track_age,
                    min_hits=self.config.min_hits
                )
                logger.info("相机适配器: 使用模拟模式 (硬件不可用)")
                return True
            else:
                self._set_error("相机连接失败,且未启用模拟模式")
                return False

        except Exception as e:
            if self.config.simulate_if_unavailable:
                self._set_simulated()
                self._tracker = SimpleTracker(
                    max_age=self.config.max_track_age,
                    min_hits=self.config.min_hits
                )
                logger.warning(f"相机连接异常，使用模拟模式: {e}")
                return True
            else:
                self._set_error(f"相机连接异常: {e}")
                return False

    def _connect_camera(self) -> bool:
        """连接相机硬件"""
        try:
            # 解析视频源
            source = self.config.source
            if source.isdigit():
                source = int(source)

            # 打开视频流
            self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                logger.error(
                    f"FALLBACK: 无法打开视频源: {self.config.source}",
                    extra={
                        "component": "camera",
                        "reason": "camera_open_failed",
                        "fallback_to": "simulation",
                        "source": str(self.config.source),
                    }
                )
                return False

            # 设置分辨率
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)

            # 读取测试帧
            ret, frame = self._capture.read()
            if not ret:
                logger.error(
                    "FALLBACK: 无法读取视频帧",
                    extra={
                        "component": "camera",
                        "reason": "frame_read_failed",
                        "fallback_to": "simulation",
                    }
                )
                return False

            logger.info(f"相机已打开: {frame.shape[1]}x{frame.shape[0]} @ {self.config.fps}fps")
            return True

        except Exception as e:
            logger.error(
                f"FALLBACK: 相机连接失败: {e}",
                extra={
                    "component": "camera",
                    "reason": "camera_connection_error",
                    "fallback_to": "simulation",
                    "error": str(e),
                }
            )
            return False

    def _load_detector(self) -> bool:
        """加载检测模型"""
        model_path = Path(self.config.model_path)

        if not model_path.exists():
            logger.error(
                f"FALLBACK: 模型文件不存在: {model_path}, 使用模拟检测",
                extra={
                    "component": "camera",
                    "reason": "model_file_not_found",
                    "fallback_to": "simulation_detection",
                    "model_path": str(model_path),
                }
            )
            return False

        self._detector = YOLOv8Detector(
            model_path=str(model_path),
            device=self.config.device,
            conf_threshold=self.config.confidence_threshold,
            nms_threshold=self.config.nms_threshold
        )

        load_success = self._detector.load()
        if not load_success:
            logger.error(
                f"FALLBACK: 模型加载失败: {model_path}, 使用模拟检测",
                extra={
                    "component": "camera",
                    "reason": "model_load_failed",
                    "fallback_to": "simulation_detection",
                }
            )
        return load_success

    def _start_capture_thread(self):
        """启动采集线程"""
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="camera_capture"
        )
        self._capture_thread.start()

    def _capture_loop(self):
        """采集循环"""
        while self._capture_running and self._capture is not None:
            try:
                ret, frame = self._capture.read()
                if ret:
                    with self._frame_lock:
                        self._latest_frame = frame

                    # 也放入队列
                    try:
                        self._frame_queue.put_nowait(frame)
                    except:
                        # 队列满，丢弃旧帧
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait(frame)
                        except:
                            pass

            except Exception as e:
                logger.error(f"采集异常: {e}")
                time.sleep(0.1)

            # 控制帧率
            time.sleep(1.0 / self.config.fps)

    def disconnect(self):
        """断开相机"""
        # 停止采集线程
        self._capture_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        # 释放相机
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        # 清理
        self._detector = None
        self._tracker = None
        self._latest_frame = None
        self._connected = False
        self._simulated = False
        self._status = AdapterStatus.DISCONNECTED

        logger.info("相机适配器已断开")

    def set_simulator(self, simulator, renderer=None) -> None:
        """
        注入仿真器和渲染器 (由 plugin.py 在仿真模式下调用).

        隔离设计: CameraAdapter 不 import Simulator/SimulationRenderer,
        只通过鸭子类型使用 .step()/.render() 接口.
        """
        self._simulator = simulator
        self._sim_renderer = renderer
        if simulator is not None:
            logger.info(
                f"仿真器已注入: {simulator.num_persons} persons, "
                f"renderer={'attached' if renderer else 'none'}"
            )

    def healthcheck(self) -> bool:
        """健康检查"""
        if self._simulated:
            return True

        if self._capture is None:
            return False

        return self._capture.isOpened()

    def get_person_detections(self, frame: Optional[np.ndarray] = None) -> List[PersonDetection]:
        """
        获取人员检测结果

        Args:
            frame: 输入帧 (可选,如果为None则从相机读取)

        Returns:
            List[PersonDetection]: 检测结果列表
        """
        if not self.is_connected and not self.is_simulated:
            logger.warning("相机未连接")
            return []

        self._frame_count += 1
        current_time = time.time()

        # 模拟模式
        if self._simulated:
            return self._simulate_detections(current_time)

        try:
            # 1. 获取帧
            if frame is None:
                frame = self._get_frame()
                if frame is None:
                    return []

            # 2. 检测
            if self._detector is not None:
                raw_detections = self._detector.detect(frame)
            else:
                raw_detections = self._simulate_raw_detections(frame.shape)

            # 3. 跟踪
            if self._tracker is not None and self.config.tracking_enabled:
                tracked = self._tracker.update(raw_detections)
            else:
                tracked = raw_detections

            # 4. 转换为PersonDetection格式
            detections = self._convert_detections(tracked, frame.shape, current_time)

            self._stats['successful_reads'] += 1
            self._stats['last_read_time'] = current_time

            return detections

        except Exception as e:
            self._stats['failed_reads'] += 1
            logger.error(f"检测失败: {e}")
            return []

    def _get_frame(self) -> Optional[np.ndarray]:
        """获取帧"""
        if self.config.use_threading:
            with self._frame_lock:
                return self._latest_frame.copy() if self._latest_frame is not None else None
        else:
            if self._capture is not None:
                ret, frame = self._capture.read()
                return frame if ret else None
        return None

    def _convert_detections(self, detections: List[Dict], frame_shape: Tuple,
                            timestamp: float) -> List[PersonDetection]:
        """转换检测结果格式"""
        h, w = frame_shape[:2]
        results = []

        for det in detections:
            self._detection_id_counter += 1
            det_id = f"D{self._detection_id_counter:05d}"

            bbox = det['bbox']  # [x1, y1, x2, y2]

            # 归一化坐标
            x = bbox[0] / w
            y = bbox[1] / h
            width = (bbox[2] - bbox[0]) / w
            height = (bbox[3] - bbox[1]) / h

            # 脚部位置 (边界框底部中心)
            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]

            track_id = None
            if 'track_id' in det:
                track_id = f"T{det['track_id']:03d}"

            results.append(PersonDetection(
                detection_id=det_id,
                bbox={'x': x, 'y': y, 'width': width, 'height': height},
                foot_pixel=(foot_x, foot_y),
                confidence=det.get('confidence', 0.8),
                timestamp=timestamp,
                track_id=track_id,
            ))

        return results

    def _simulate_detections(self, timestamp: float) -> List[PersonDetection]:
        """模拟检测结果"""
        # 生成随机检测
        raw_dets = self._simulate_raw_detections((480, 640, 3))

        # 通过跟踪器
        if self._tracker is not None:
            tracked = self._tracker.update(raw_dets)
        else:
            tracked = raw_dets

        # 转换格式
        return self._convert_detections(tracked, (480, 640, 3), timestamp)

    def _simulate_raw_detections(self, frame_shape: Tuple) -> List[Dict]:
        """调度器: 有 Simulator 时走轨迹仿真, 否则 fallback 到随机检测"""
        if self._simulator is not None:
            return self._simulate_from_simulator(frame_shape)
        return self._simulate_random_detections(frame_shape)

    def _simulate_from_simulator(self, frame_shape: Tuple) -> List[Dict]:
        """从 Simulator 获取轨迹驱动的检测结果"""
        from ..protocols import SensorType

        h, w = frame_shape[:2]
        step_data = self._simulator.step()

        camera_data = step_data.get(SensorType.CAMERA)
        if camera_data is None:
            return self._simulate_random_detections(frame_shape)

        detections = []
        for det in camera_data.data.get("detections", []):
            pos = det.get("position", [0, 0])
            world_x, world_y = pos[0], pos[1]

            bounds = self._simulator._config.scene_bounds
            x_min, y_min, x_max, y_max = bounds
            norm_x = (world_x - x_min) / (x_max - x_min + 1e-8)
            norm_y = (world_y - y_min) / (y_max - y_min + 1e-8)

            px = norm_x * w
            py = norm_y * h

            bw = random.uniform(0.06 * w, 0.12 * w)
            bh = random.uniform(0.25 * h, 0.35 * h)

            x1 = max(0, px - bw / 2)
            y1 = max(0, py - bh)
            x2 = min(w, px + bw / 2)
            y2 = min(h, py)

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': det.get("confidence", 0.85),
                'class_id': 0,
            })

        return detections

    def _simulate_random_detections(self, frame_shape: Tuple) -> List[Dict]:
        """纯随机检测 fallback (无 Simulator 时使用)"""
        h, w = frame_shape[:2]
        detections = []
        num_persons = random.randint(1, 3)

        for i in range(num_persons):
            cx = random.uniform(0.2 * w, 0.8 * w)
            cy = random.uniform(0.3 * h, 0.7 * h)
            bw = random.uniform(0.05 * w, 0.15 * w)
            bh = random.uniform(0.2 * h, 0.4 * h)

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': random.uniform(0.7, 0.98),
                'class_id': 0,
            })

        return detections

    def get_frame(self) -> Optional[np.ndarray]:
        """获取当前帧 (调度器: 有渲染器时走仿真俯视图)"""
        if self._simulated:
            if self._sim_renderer is not None and self._simulator is not None:
                return self._generate_renderer_frame()
            return self._generate_legacy_simulation_frame()

        return self._get_frame()

    def _generate_renderer_frame(self) -> np.ndarray:
        """使用 SimulationRenderer 渲染工程级俯视图帧"""
        persons = []
        for p in self._simulator._persons:
            persons.append({
                "id": p.person_id,
                "x": p.x,
                "y": p.y,
                "z": getattr(p, 'z', 0.0),
            })
        return self._sim_renderer.render(persons, self._simulator._step_count)

    def _generate_legacy_simulation_frame(self) -> np.ndarray:
        """旧版模拟帧 fallback (网格 + 随机检测框)"""
        w, h = self.config.resolution[0], self.config.resolution[1]

        frame = np.full((h, w, 3), 26, dtype=np.uint8)

        if not CV2_AVAILABLE:
            return frame

        grid_spacing = 40
        grid_color = (50, 45, 45)
        for gx in range(0, w, grid_spacing):
            cv2.line(frame, (gx, 0), (gx, h), grid_color, 1)
        for gy in range(0, h, grid_spacing):
            cv2.line(frame, (0, gy), (w, gy), grid_color, 1)

        raw_dets = self._simulate_raw_detections((h, w, 3))
        for det in raw_dets:
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            conf = det.get('confidence', 0.8)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)

            label = f"Person {conf:.0%}"
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1,
            )

        cv2.putText(
            frame, "SIM", (w - 55, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )

        self._frame_count += 1
        info = f"Frame: {self._frame_count} | Persons: {len(raw_dets)}"
        cv2.putText(
            frame, info, (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
        )

        return frame


__all__ = ['CameraConfig', 'PersonDetection', 'CameraAdapter', 'YOLOv8Detector', 'SimpleTracker']
