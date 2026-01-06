"""
插件管理器单元测试
"""

import pytest
from pathlib import Path
import sys

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from platform_core.plugin_manager.base import PluginManifest, PluginCapability


class TestPluginManifest:
    """测试插件清单"""

    def test_from_dict(self):
        """测试从字典创建清单"""
        data = {
            "id": "test_plugin",
            "name": "测试插件",
            "version": "1.0.0",
            "description": "测试用插件",
            "capabilities": ["defect_detection", "state_recognition"],
            "device_types": ["transformer"],
        }

        manifest = PluginManifest.from_dict(data)

        assert manifest.id == "test_plugin"
        assert manifest.name == "测试插件"
        assert manifest.version == "1.0.0"
        assert PluginCapability.DEFECT_DETECTION in manifest.capabilities
        assert "transformer" in manifest.device_types

    def test_default_values(self):
        """测试默认值"""
        data = {
            "id": "minimal_plugin",
            "name": "最小插件",
            "version": "0.1.0",
        }

        manifest = PluginManifest.from_dict(data)

        assert manifest.entrypoint == "plugin.py"
        assert manifest.plugin_class == "Plugin"
        assert manifest.capabilities == []


class TestPluginCapability:
    """测试插件能力枚举"""

    def test_all_capabilities(self):
        """测试所有能力枚举"""
        capabilities = [
            PluginCapability.DEFECT_DETECTION,
            PluginCapability.STATE_RECOGNITION,
            PluginCapability.METER_READING,
            PluginCapability.THERMAL_ANALYSIS,
            PluginCapability.INTRUSION_DETECTION,
        ]

        for cap in capabilities:
            assert isinstance(cap.value, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
