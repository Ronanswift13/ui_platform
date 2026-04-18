#!/usr/bin/env python3
"""Animal Detection - Test Configuration"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 插件目录也需要在 sys.path 中，以支持 `from core.xxx import ...`
_plugin_root = Path(__file__).resolve().parent.parent
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))


# === Pytest Fixtures (集中注册) ===

import numpy as np
import pytest


# --- 帧数据 fixtures ---

@pytest.fixture
def blank_frame():
    """全黑 BGR 测试帧 (640x480)"""
    from tests.fixtures.frame_factories import blank_frame
    return blank_frame()


@pytest.fixture
def noise_frame():
    """可复现噪声帧 (seed=42)"""
    from tests.fixtures.frame_factories import noise_frame
    return noise_frame()


@pytest.fixture
def sample_frame():
    """标准测试帧 (640x480 随机 BGR)"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_small():
    """小尺寸测试帧 (320x240)"""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def uniform_thermal():
    """均匀温度热成像帧 (25°C)"""
    from tests.fixtures.frame_factories import uniform_thermal
    return uniform_thermal()


@pytest.fixture
def thermal_with_hotspot():
    """带热点的热成像帧 (背景25°C, 热点30°C)"""
    from tests.fixtures.frame_factories import thermal_with_hotspot
    return thermal_with_hotspot()


# --- 检测结果 fixtures ---

@pytest.fixture
def mouse_detection():
    """标准鼠检测结果 (confidence=0.85)"""
    from tests.fixtures.detection_factories import make_mouse
    return make_mouse()


@pytest.fixture
def snake_detection():
    """标准蛇检测结果 (confidence=0.75, CRITICAL)"""
    from tests.fixtures.detection_factories import make_snake
    return make_snake()


@pytest.fixture
def cat_detection():
    """标准猫检测结果 (confidence=0.80)"""
    from tests.fixtures.detection_factories import make_cat
    return make_cat()


# --- 配置 fixtures ---

@pytest.fixture
def simulation_config():
    """仿真模式配置"""
    from tests.fixtures.config_factories import simulation_config
    return simulation_config()


@pytest.fixture
def production_config(tmp_path):
    """生产模式配置（模型路径指向 tmp）"""
    from tests.fixtures.config_factories import production_config
    return production_config(model_path=str(tmp_path / "test_model.onnx"))


# --- ONNX 模型 fixtures ---

@pytest.fixture
def fake_onnx_model(tmp_path):
    """不可加载的假模型路径 (用于测试降级)"""
    model_path = tmp_path / "fake_model.onnx"
    model_path.write_bytes(b"not a real onnx model")
    return str(model_path)


@pytest.fixture
def nonexistent_model():
    """不存在的模型路径"""
    return "/tmp/does_not_exist_animal_model.onnx"
