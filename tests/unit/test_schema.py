"""
Schema校验单元测试
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from platform_core.schema.models import (
    BoundingBox,
    ROI,
    ROIType,
    RecognitionResult,
    PluginOutput,
    Site,
    Device,
    DeviceType,
)
from platform_core.schema.validator import SchemaValidator, validate_plugin_output
from platform_core.exceptions import SchemaValidationError


class TestBoundingBox:
    """测试边界框模型"""

    def test_valid_bbox(self):
        """测试有效边界框"""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        assert bbox.x == 0.1
        assert bbox.y == 0.2

    def test_invalid_bbox(self):
        """测试无效边界框"""
        with pytest.raises(ValueError):
            BoundingBox(x=1.5, y=0.2, width=0.3, height=0.4)


class TestRecognitionResult:
    """测试识别结果模型"""

    def test_create_result(self):
        """测试创建识别结果"""
        result = RecognitionResult(
            task_id="task_001",
            site_id="site_001",
            device_id="dev_001",
            component_id="comp_001",
            roi_id="roi_001",
            bbox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
            label="正常",
            confidence=0.95,
        )
        assert result.label == "正常"
        assert result.confidence == 0.95


class TestPluginOutput:
    """测试插件输出模型"""

    def test_valid_output(self):
        """测试有效输出"""
        output = PluginOutput(
            task_id="task_001",
            plugin_id="test_plugin",
            plugin_version="1.0.0",
            code_hash="abc123",
            success=True,
            results=[],
            alarms=[],
        )
        assert output.success is True

    def test_validate_output(self):
        """测试输出验证"""
        data = {
            "task_id": "task_001",
            "plugin_id": "test_plugin",
            "plugin_version": "1.0.0",
            "code_hash": "abc123",
            "success": True,
            "results": [],
            "alarms": [],
        }
        output = validate_plugin_output(data, "test_plugin")
        assert output.task_id == "task_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
