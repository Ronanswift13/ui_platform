#!/usr/bin/env python3
"""Switch Inspection - Test Configuration"""
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import yaml

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_plugin_root = Path(__file__).resolve().parent.parent

from darkbreaker_sdk.interfaces.base_plugin import PluginContext
from darkbreaker_sdk.schemas.common import ROIType
from darkbreaker_sdk.schemas.detection import BoundingBox


@pytest.fixture
def default_config():
    return yaml.safe_load((_plugin_root / "configs" / "default.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def plugin_context():
    return PluginContext(
        task_id="task-001",
        site_id="site-001",
        device_id="device-001",
        component_id="component-001",
    )


@pytest.fixture
def roi_factory():
    def _make_roi(name: str, bbox: Optional[BoundingBox] = None):
        return SimpleNamespace(
            id=f"roi-{name}",
            name=name,
            roi_type=name,
            bbox=bbox or BoundingBox(x=0.1, y=0.1, width=0.4, height=0.4),
            recognition_types=[ROIType.STATE.value],
        )

    return _make_roi
