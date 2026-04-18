"""Shared fixtures for hyperspectral_detection tests."""
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
    with open(PLUGIN_DIR / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def plugin(loaded_config):
    from plugins.hyperspectral_detection.plugin import HyperspectralDetectionPlugin
    return HyperspectralDetectionPlugin.create_standalone(config=copy.deepcopy(loaded_config))


@pytest.fixture
def sample_context():
    from darkbreaker_sdk.interfaces import PluginContext
    return PluginContext(
        task_id="test-task-001",
        site_id="test-site",
        device_id="test-camera",
        component_id="hs-region-01",
    )


@pytest.fixture
def make_roi():
    from darkbreaker_sdk.schemas import BoundingBox, ROI, ROIType
    def _factory(*, roi_id="roi-hs-01", name="hyperspectral_region", roi_type=ROIType.DEFECT, bbox=None):
        return ROI(
            id=roi_id,
            name=name,
            component_id="hs-region-01",
            roi_type=roi_type,
            bbox=bbox or BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
        )
    return _factory


@pytest.fixture
def spectral_frame():
    """Simulated hyperspectral cube: bands x height x width."""
    return np.random.rand(224, 256, 256).astype(np.float32)


@pytest.fixture
def visual_frame():
    """Standard BGR frame for visual fallback."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
