"""Shared fixtures for transformer_inspection tests."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def loaded_config() -> dict:
    """Load the plugin's default config once per test session."""
    with open(PLUGIN_DIR / "configs" / "default.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def plugin(loaded_config):
    """Create a standalone plugin instance with default config."""
    from plugins.transformer_inspection.plugin import TransformerInspectionPlugin

    return TransformerInspectionPlugin.create_standalone(config=copy.deepcopy(loaded_config))


@pytest.fixture
def detector(loaded_config):
    """Create an initialized detector instance."""
    from plugins.transformer_inspection.detector_enhanced import TransformerDetectorEnhanced

    instance = TransformerDetectorEnhanced(copy.deepcopy(loaded_config))
    instance.initialize()
    return instance


@pytest.fixture
def sample_context():
    """Create a sample plugin context."""
    from darkbreaker_sdk.interfaces import PluginContext

    return PluginContext(
        task_id="test-task-001",
        site_id="test-site",
        device_id="test-camera",
        component_id="transformer-01",
    )


@pytest.fixture
def make_roi():
    """Factory for ROI objects with current SDK-required fields."""
    from darkbreaker_sdk.schemas import BoundingBox, ROI, ROIType

    def _factory(
        *,
        roi_id: str,
        name: str,
        roi_type: ROIType,
        bbox: BoundingBox | None = None,
    ) -> ROI:
        return ROI(
            id=roi_id,
            name=name,
            component_id="transformer-01",
            roi_type=roi_type,
            bbox=bbox or BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
        )

    return _factory


@pytest.fixture
def defect_frame():
    """Create a deterministic frame with dark and rust-colored regions."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[80:220, 60:220] = [20, 20, 20]
    frame[260:360, 300:430] = [0, 100, 180]
    return frame


@pytest.fixture
def blue_silica_frame():
    """Create a frame that strongly matches the blue silica-gel range."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [255, 0, 0]
    return frame


@pytest.fixture
def oil_level_frame():
    """Create a frame with a visible yellow oil region in the lower half."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[240:470, 80:560] = [0, 255, 255]
    return frame


@pytest.fixture
def thermal_frame():
    """Create a grayscale thermal frame with one hot block."""
    frame = np.zeros((480, 640), dtype=np.uint8)
    frame[140:260, 180:320] = 255
    return frame
