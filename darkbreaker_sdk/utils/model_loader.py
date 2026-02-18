"""
ONNX model loader utility.

Provides a helper function to load ONNX models using onnxruntime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_onnx_model(
    path: str | Path,
    providers: list[str] | None = None,
) -> Optional[Any]:
    """
    Load an ONNX model using onnxruntime.

    Args:
        path: Path to the .onnx model file.
        providers: List of execution providers
            (e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]).
            Defaults to ["CPUExecutionProvider"] if not specified.

    Returns:
        An onnxruntime.InferenceSession instance, or None if loading fails
        (e.g. file not found, onnxruntime not installed).
    """
    path = Path(path)

    if not path.exists():
        logger.warning("ONNX model file not found: %s", path)
        return None

    if providers is None:
        providers = ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime is not installed. "
            "Install it with: pip install onnxruntime"
        )
        return None

    try:
        session = ort.InferenceSession(str(path), providers=providers)
        logger.info("Loaded ONNX model: %s (providers=%s)", path.name, providers)
        return session
    except Exception as e:
        logger.error("Failed to load ONNX model %s: %s", path, e)
        return None
