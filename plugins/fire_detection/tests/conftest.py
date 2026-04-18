#!/usr/bin/env python3
"""Fire Detection - Test Configuration"""
import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@pytest.fixture
def plugin_dir():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def default_config(plugin_dir):
    from darkbreaker_sdk.utils import load_plugin_config

    return load_plugin_config(plugin_dir / "configs" / "default.yaml")


@pytest.fixture
def plugin(default_config):
    from plugins.fire_detection.plugin import FireDetectionPlugin

    return FireDetectionPlugin.create_standalone(default_config)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fire_frame(blank_frame):
    frame = blank_frame.copy()
    frame[100:220, 180:300] = (0, 140, 255)
    return frame


@pytest.fixture
def smoke_frame(blank_frame):
    frame = blank_frame.copy()
    frame[120:280, 200:360] = (160, 160, 160)
    return frame


@pytest.fixture
def thermal_hotspot_frame():
    frame = np.full((120, 160), 40, dtype=np.uint8)
    frame[30:90, 50:110] = 90
    return frame


@pytest.fixture
def sensor_alarm_data():
    return {
        "smoke_concentration": 35.0,
        "temperature": 85.0,
        "co_concentration": 30.0,
        "humidity": 45.0,
    }
