"""bird_monitoring 测试夹具"""
import sys
from pathlib import Path
import pytest
import numpy as np

# 确保插件和项目根在 sys.path
_plugin_dir = Path(__file__).resolve().parent.parent
_project_root = _plugin_dir.parent.parent
for p in [str(_plugin_dir), str(_project_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def plugin_dir():
    return _plugin_dir


@pytest.fixture
def default_config():
    """最小默认配置"""
    return {
        "model": {"path": "nonexistent.onnx", "device": "cpu", "input_size": [640, 640]},
        "inference": {"confidence_threshold": 0.5, "nms_threshold": 0.45},
        "risk_assessment": {
            "safe_distance_m": 15.0,
            "warning_distance_m": 10.0,
            "danger_distance_m": 5.0,
            "critical_distance_m": 2.0,
            "distance_weight": 0.4,
            "species_weight": 0.3,
            "wingspan_weight": 0.2,
            "behavior_weight": 0.1,
        },
        "quality": {
            "min_dimension": 64,
            "clarity_threshold": 50.0,
            "review_confidence_threshold": 0.5,
            "species_confidence_threshold": 0.6,
        },
        "deterrent": {"enabled": False},
    }


@pytest.fixture
def plugin_instance(default_config):
    """创建已初始化的 plugin 实例"""
    from plugins.bird_monitoring.plugin import BirdMonitoringPlugin
    instance = BirdMonitoringPlugin.create_standalone(config=default_config)
    return instance


@pytest.fixture
def sample_frame():
    """合成测试帧 (640x480 BGR)"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def tiny_frame():
    """极小帧，用于测试质量失败"""
    return np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)


@pytest.fixture
def dark_frame():
    """过暗帧"""
    return np.zeros((480, 640, 3), dtype=np.uint8) + 5


@pytest.fixture
def make_context():
    """工厂函数: 创建 PluginContext"""
    from darkbreaker_sdk.interfaces import PluginContext

    def _make(task_id="test_task", **kwargs):
        return PluginContext(
            task_id=task_id,
            site_id=kwargs.get("site_id", "site_001"),
            device_id=kwargs.get("device_id", "cam_001"),
            component_id=kwargs.get("component_id", "line_001"),
            timestamp=kwargs.get("timestamp", 0),
            metadata=kwargs.get("metadata", {}),
        )
    return _make


@pytest.fixture
def make_roi():
    """工厂函数: 创建 ROI"""
    from darkbreaker_sdk.schemas import ROI, BoundingBox
    from darkbreaker_sdk.schemas.common import ROIType

    def _make(roi_id="roi_001", x=0.0, y=0.0, w=1.0, h=1.0):
        return ROI(
            id=roi_id,
            name=f"test_{roi_id}",
            component_id="line_001",
            roi_type=ROIType.DEFECT,
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
        )
    return _make
