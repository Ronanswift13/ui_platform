"""
Local Inference Engine - Lightweight standalone inference.

Supports ONNX Runtime and PyTorch with graceful fallback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class BackendType(Enum):
    ONNX_CPU = "onnx_cpu"
    ONNX_CUDA = "onnx_cuda"
    PYTORCH_CPU = "pytorch_cpu"
    PYTORCH_CUDA = "pytorch_cuda"
    MOCK = "mock"


@dataclass
class ModelInfo:
    """Loaded model metadata."""
    model_id: str
    model_path: str
    backend: BackendType
    input_size: Tuple[int, int] = (640, 640)
    class_names: List[str] = field(default_factory=list)
    loaded: bool = False


class LocalInferenceEngine:
    """
    Lightweight inference engine for standalone plugin operation.

    Supports ONNX Runtime and PyTorch. Falls back to mock mode
    when no ML backend is available.
    """

    def __init__(self, device: str = "auto"):
        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._device = self._resolve_device(device)
        self._backend = self._detect_backend()
        logger.info(f"LocalInferenceEngine initialized: backend={self._backend.value}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            if ONNX_AVAILABLE:
                providers = ort.get_available_providers() if ort else []
                if "CUDAExecutionProvider" in providers:
                    return "cuda"
            return "cpu"
        return device

    def _detect_backend(self) -> BackendType:
        if ONNX_AVAILABLE:
            if self._device == "cuda":
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    return BackendType.ONNX_CUDA
            return BackendType.ONNX_CPU
        if TORCH_AVAILABLE:
            return BackendType.PYTORCH_CUDA if self._device == "cuda" else BackendType.PYTORCH_CPU
        logger.warning("No ML backend available, using mock mode")
        return BackendType.MOCK

    def load_model(
        self,
        model_id: str,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        class_names: Optional[List[str]] = None,
    ) -> bool:
        """Load a model from file."""
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        info = ModelInfo(
            model_id=model_id,
            model_path=str(path),
            backend=self._backend,
            input_size=input_size,
            class_names=class_names or [],
        )

        try:
            suffix = path.suffix.lower()
            if suffix == ".onnx" and ONNX_AVAILABLE:
                providers = ["CPUExecutionProvider"]
                if self._backend == BackendType.ONNX_CUDA:
                    providers.insert(0, "CUDAExecutionProvider")
                session = ort.InferenceSession(str(path), providers=providers)
                self._models[model_id] = session
                info.loaded = True
            elif suffix in (".pt", ".pth") and TORCH_AVAILABLE:
                model = torch.load(str(path), map_location=self._device, weights_only=False)
                if hasattr(model, "eval"):
                    model.eval()
                self._models[model_id] = model
                info.loaded = True
            else:
                logger.warning(f"Unsupported model format: {suffix}, using mock")
                self._models[model_id] = None
                info.backend = BackendType.MOCK
                info.loaded = True

            self._model_info[model_id] = info
            logger.info(f"Model loaded: {model_id} ({info.backend.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False

    def infer(
        self,
        model_id: str,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on an image.

        Returns list of detections:
        [{"bbox": [x1,y1,x2,y2], "score": float, "class_id": int, "class_name": str}]
        """
        if model_id not in self._models:
            logger.warning(f"Model {model_id} not loaded")
            return []

        info = self._model_info[model_id]
        model = self._models[model_id]

        if info.backend == BackendType.MOCK or model is None:
            return self._mock_infer(image, info)

        try:
            preprocessed = self._preprocess(image, info.input_size)

            if info.backend in (BackendType.ONNX_CPU, BackendType.ONNX_CUDA):
                return self._onnx_infer(model, preprocessed, info, conf_threshold, nms_threshold, image.shape)
            elif info.backend in (BackendType.PYTORCH_CPU, BackendType.PYTORCH_CUDA):
                return self._torch_infer(model, preprocessed, info, conf_threshold, image.shape)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []

    @staticmethod
    def _preprocess(image: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess image for inference."""
        try:
            import cv2
            resized = cv2.resize(image, input_size)
        except ImportError:
            h, w = image.shape[:2]
            th, tw = input_size
            resized = image[:min(h, th), :min(w, tw)]
            if resized.shape[:2] != (th, tw):
                resized = np.zeros((th, tw, 3), dtype=np.uint8)

        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, 0)   # Add batch dim
        return blob

    def _onnx_infer(
        self, session, blob, info, conf_threshold, nms_threshold, orig_shape
    ) -> List[Dict[str, Any]]:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: blob})

        detections = []
        if len(outputs) > 0:
            output = outputs[0]
            if output.ndim == 3 and output.shape[0] == 1:
                output = output[0]
                if output.shape[0] < output.shape[1]:
                    output = output.T

                h_orig, w_orig = orig_shape[:2]
                h_input, w_input = info.input_size

                for row in output:
                    if len(row) >= 5:
                        scores = row[4:]
                        class_id = int(np.argmax(scores))
                        score = float(scores[class_id])
                        if score >= conf_threshold:
                            cx, cy, w, h = row[:4]
                            x1 = (cx - w / 2) * w_orig / w_input
                            y1 = (cy - h / 2) * h_orig / h_input
                            x2 = (cx + w / 2) * w_orig / w_input
                            y2 = (cy + h / 2) * h_orig / h_input
                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "score": score,
                                "class_id": class_id,
                                "class_name": info.class_names[class_id] if class_id < len(info.class_names) else f"class_{class_id}",
                            })

        detections = self._nms(detections, nms_threshold)
        return detections

    def _torch_infer(self, model, blob, info, conf_threshold, orig_shape) -> List[Dict[str, Any]]:
        tensor = torch.from_numpy(blob).to(self._device)
        with torch.no_grad():
            output = model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "cpu"):
            output = output.cpu().numpy()
        return []

    @staticmethod
    def _mock_infer(image: np.ndarray, info: ModelInfo) -> List[Dict[str, Any]]:
        """Return empty results in mock mode."""
        logger.debug(f"Mock inference for {info.model_id}")
        return []

    @staticmethod
    def _nms(detections: List[Dict], threshold: float) -> List[Dict]:
        if not detections:
            return []
        detections.sort(key=lambda d: d["score"], reverse=True)
        keep = []
        for det in detections:
            overlap = False
            for kept in keep:
                iou = _compute_iou(det["bbox"], kept["bbox"])
                if iou > threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
        return keep

    def unload_model(self, model_id: str) -> None:
        self._models.pop(model_id, None)
        self._model_info.pop(model_id, None)

    def list_models(self) -> List[ModelInfo]:
        return list(self._model_info.values())

    @property
    def backend(self) -> BackendType:
        return self._backend

    @property
    def device(self) -> str:
        return self._device


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)
