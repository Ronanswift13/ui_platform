"""
Indoor Fence V3.0 - Training API Routes
=========================================

FastAPI routes for training pipeline control.
"""
from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter

from .training_pipeline import TrainingPipeline, TrainingConfig


def create_training_router(
    pipeline: Optional[TrainingPipeline] = None,
) -> APIRouter:
    """Create FastAPI router for training pipeline control."""
    router = APIRouter(tags=["training"])
    _pipeline = pipeline or TrainingPipeline()

    @router.get("/status")
    def get_status() -> Dict:
        return {
            "status": _pipeline.status.value,
            "datasets": {k: {"num_images": v.num_images, "num_labels": v.num_labels}
                        for k, v in _pipeline.datasets.items()},
        }

    @router.post("/dataset/register")
    def register_dataset(dataset_id: str, path: str) -> Dict:
        info = _pipeline.register_dataset(dataset_id, path)
        return {
            "dataset_id": info.dataset_id,
            "num_images": info.num_images,
            "num_labels": info.num_labels,
        }

    @router.post("/start")
    def start_training(
        dataset_id: str,
        model_type: str = "yolov8n",
        epochs: int = 100,
        batch_size: int = 16,
    ) -> Dict:
        config = TrainingConfig(
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
        )
        _pipeline.configure(config)
        result = _pipeline.start_training(dataset_id)
        return {
            "status": result.status.value,
            "epochs_completed": result.epochs_completed,
        }

    return router
