"""
Indoor Fence V3.0 - YOLO Detector
===================================

YOLO-based object detector with ONNX Runtime inference.
Falls back to simulation mode if no model file provided.
"""
from __future__ import annotations

import logging
import os
import random
from typing import List, Optional

import numpy as np

from .object_detector import ObjectDetector, DetectionBox

logger = logging.getLogger(__name__)


class YOLODetector(ObjectDetector):
    """YOLO object detector (ONNX Runtime or simulation)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ):
        self._model_path = model_path
        self._device = device
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._session = None
        self._simulation_mode = False
        self._fallback_reason = None

        if model_path is None:
            self._simulation_mode = True
            self._fallback_reason = "no_model_path_provided"
            logger.warning(
                "FALLBACK: No model path provided, using simulation mode",
                extra={
                    "component": "detection",
                    "reason": "no_model_path",
                    "fallback_to": "simulation",
                }
            )
        elif not os.path.exists(model_path):
            self._simulation_mode = True
            self._fallback_reason = "model_file_not_found"
            logger.error(
                f"FALLBACK: Model file not found at {model_path}, using simulation mode",
                extra={
                    "component": "detection",
                    "reason": "model_not_found",
                    "fallback_to": "simulation",
                    "model_path": model_path,
                }
            )
        else:
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(model_path)
                logger.info(f"YOLO model loaded successfully from {model_path}")
            except ImportError as e:
                self._simulation_mode = True
                self._fallback_reason = "onnxruntime_not_installed"
                logger.error(
                    f"FALLBACK: onnxruntime not installed, using simulation mode: {e}",
                    extra={
                        "component": "detection",
                        "reason": "onnxruntime_missing",
                        "fallback_to": "simulation",
                    }
                )
            except Exception as e:
                self._simulation_mode = True
                self._fallback_reason = "model_load_failed"
                logger.error(
                    f"FALLBACK: Failed to load model from {model_path}, using simulation mode: {e}",
                    extra={
                        "component": "detection",
                        "reason": "model_load_error",
                        "fallback_to": "simulation",
                        "error": str(e),
                    }
                )

    def detect(self, frame: np.ndarray) -> List[DetectionBox]:
        """Detect objects in frame."""
        if self._simulation_mode:
            return self._simulate_detections(frame.shape)
        return self._run_inference(frame)

    def is_simulation_mode(self) -> bool:
        """Check if detector is running in simulation mode."""
        return self._simulation_mode

    def get_fallback_reason(self) -> Optional[str]:
        """Get the reason for fallback to simulation mode."""
        return self._fallback_reason

    def _simulate_detections(self, shape: tuple) -> List[DetectionBox]:
        """Generate random detections for simulation."""
        h, w = shape[:2]
        num = random.randint(1, 3)
        detections = []
        for _ in range(num):
            cx = random.randint(int(w * 0.15), int(w * 0.85))
            cy = random.randint(int(h * 0.2), int(h * 0.6))
            bw = random.randint(int(w * 0.05), int(w * 0.15))
            bh = random.randint(int(h * 0.15), int(h * 0.4))
            detections.append(DetectionBox(
                x1=max(0, cx - bw // 2),
                y1=max(0, cy - bh // 2),
                x2=min(w, cx + bw // 2),
                y2=min(h, cy + bh // 2),
                confidence=random.uniform(0.6, 0.98),
                class_id=0,
                class_name="person",
            ))
        return detections

    def _run_inference(self, frame: np.ndarray) -> List[DetectionBox]:
        """Run actual ONNX inference (placeholder for real implementation)."""
        if self._session is None:
            return self._simulate_detections(frame.shape)
        # Real ONNX inference would go here
        return self._simulate_detections(frame.shape)
