"""Shared fixtures for thermal plugin tests."""

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
    from plugins.thermal.plugin import ThermalPlugin

    return ThermalPlugin.create_standalone(config=copy.deepcopy(loaded_config))


@pytest.fixture
def sample_context():
    """Create a sample plugin context."""
    from darkbreaker_sdk.interfaces import PluginContext

    return PluginContext(
        task_id="test-task-001",
        site_id="test-site",
        device_id="test-camera",
        component_id="thermal-region-01",
    )


@pytest.fixture
def make_roi():
    """Factory for SDK ROI objects."""
    from darkbreaker_sdk.schemas import BoundingBox, ROI, ROIType

    def _factory(
        *,
        roi_id: str,
        name: str,
        roi_type: ROIType = ROIType.THERMAL,
        bbox: BoundingBox | None = None,
    ) -> ROI:
        return ROI(
            id=roi_id,
            name=name,
            component_id="thermal-region-01",
            roi_type=roi_type,
            bbox=bbox or BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
        )

    return _factory


@pytest.fixture
def thermal_frame():
    """Create a synthetic thermal frame with normal temperatures."""
    return np.random.normal(35, 10, (480, 640)).astype(np.float32)


@pytest.fixture
def thermal_frame_with_hotspot():
    """Create a thermal frame with a prominent hotspot region."""
    frame = np.random.normal(35, 10, (480, 640)).astype(np.float32)
    # 在中心区域注入高温热点 (> 80 度)
    frame[200:260, 280:360] = np.random.uniform(85, 110, (60, 80)).astype(np.float32)
    return frame
