"""
Pytest配置和共享fixtures
"""
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def plugin_dir():
    """插件目录路径"""
    return PLUGIN_DIR


@pytest.fixture
def default_config(plugin_dir):
    """默认配置"""
    import yaml
    config_path = plugin_dir / "configs" / "default.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_frame():
    """生成测试用BGR帧 (480x640x3 uint8)"""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def valid_roi():
    """有效的归一化ROI"""
    return {
        "x": 0.2,
        "y": 0.2,
        "w": 0.4,
        "h": 0.4
    }


@pytest.fixture
def meter_ranges():
    """表计量程映射"""
    return {
        "pressure_gauge": {"min": 0, "max": 1.6, "unit": "MPa"},
        "temperature_gauge": {"min": -20, "max": 100, "unit": "\u00b0C"},
        "oil_level_gauge": {"min": 0, "max": 100, "unit": "%"},
    }


@pytest.fixture
def detector(default_config):
    """创建已初始化的检测器实例"""
    from plugins.meter_reading.detector_enhanced import MeterReadingDetectorEnhanced
    det = MeterReadingDetectorEnhanced(default_config)
    det.initialize()
    return det


@pytest.fixture
def red_led_image():
    """纯红色LED图像 (64x64)"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # BGR red
    return img


@pytest.fixture
def green_led_image():
    """纯绿色LED图像 (64x64)"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 255, 0)  # BGR green
    return img


@pytest.fixture
def yellow_led_image():
    """纯黄色LED图像 (64x64)"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 255, 255)  # BGR yellow
    return img


@pytest.fixture
def off_led_image():
    """灭灯LED图像 (64x64)"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (10, 10, 10)  # very dark
    return img
