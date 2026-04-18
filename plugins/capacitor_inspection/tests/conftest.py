"""Shared fixtures for capacitor_inspection tests."""

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
    from plugins.capacitor_inspection.plugin import CapacitorInspectionPlugin

    return CapacitorInspectionPlugin.create_standalone(config=copy.deepcopy(loaded_config))


@pytest.fixture
def detector(loaded_config):
    """Create an initialized detector instance."""
    from plugins.capacitor_inspection.detector_enhanced import CapacitorDetectorEnhanced

    instance = CapacitorDetectorEnhanced(copy.deepcopy(loaded_config))
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
        component_id="capacitor-bank-01",
    )


@pytest.fixture
def make_roi():
    """Factory for SDK ROI objects."""
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
            component_id="capacitor-bank-01",
            roi_type=roi_type,
            bbox=bbox or BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
        )

    return _factory


@pytest.fixture
def structural_frame():
    """Create a simple frame with several capacitor-like vertical regions."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[120:420, 100:150] = [20, 20, 20]
    frame[110:420, 250:300] = [20, 20, 20]
    frame[130:420, 400:450] = [20, 20, 20]
    return frame


@pytest.fixture
def intrusion_frame():
    """Create a generic frame for intrusion-route tests."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
