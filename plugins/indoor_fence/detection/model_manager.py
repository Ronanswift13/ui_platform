"""
Indoor Fence V3.0 - Model Lifecycle Manager
=============================================

Manages registration, loading, and status tracking of ML models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ModelStatus(str, Enum):
    REGISTERED = "registered"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADED = "unloaded"


@dataclass
class ModelInfo:
    """Information about a registered model."""
    model_id: str
    model_path: str
    model_type: str = "detector"
    status: ModelStatus = ModelStatus.REGISTERED
    version: str = ""
    metadata: Dict = field(default_factory=dict)


class ModelManager:
    """Manages ML model lifecycle."""

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}

    def register(
        self,
        model_id: str,
        model_path: str,
        model_type: str = "detector",
        version: str = "",
    ) -> ModelInfo:
        """Register a new model."""
        info = ModelInfo(
            model_id=model_id,
            model_path=model_path,
            model_type=model_type,
            version=version,
        )
        self._models[model_id] = info
        return info

    def get(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        return self._models.get(model_id)

    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self._models.values())

    def set_status(self, model_id: str, status: ModelStatus) -> None:
        """Update model status."""
        if model_id in self._models:
            self._models[model_id].status = status

    def unregister(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False
