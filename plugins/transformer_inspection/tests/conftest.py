"""Pytest configuration for transformer_inspection tests."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def plugin():
    """Create a standalone transformer inspection plugin instance."""
    from plugins.transformer_inspection.plugin import TransformerInspectionPlugin
    return TransformerInspectionPlugin.create_standalone()


@pytest.fixture
def dummy_frame():
    """Create a dummy BGR image frame."""
    import numpy as np
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


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
def sample_rois():
    """Create sample ROIs."""
    from darkbreaker_sdk.schemas import ROI, BoundingBox
    return [
        ROI(
            id="roi-test",
            name="transformer_body",
            component_id="transformer-01",
            bbox=BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8),
        )
    ]
