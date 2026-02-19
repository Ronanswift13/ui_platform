"""Tests for PluginInstaller."""
import json
import tempfile
import unittest
from pathlib import Path

import sys
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class TestPluginInstaller(unittest.TestCase):
    """Test PluginInstaller functionality."""

    def setUp(self):
        from platform_core.plugin_manager.installer import PluginInstaller
        self.installer = PluginInstaller(plugins_dir="plugins/")

    def test_list_available_plugins(self):
        available = self.installer.list_available()
        self.assertGreaterEqual(len(available), 15)
        ids = [p.id for p in available]
        self.assertIn("indoor_fence", ids)
        self.assertIn("transformer_inspection", ids)

    def test_plugin_info_has_fields(self):
        available = self.installer.list_available()
        plugin = available[0]
        self.assertTrue(hasattr(plugin, "id"))
        self.assertTrue(hasattr(plugin, "name"))
        self.assertTrue(hasattr(plugin, "version"))
        self.assertTrue(hasattr(plugin, "category"))
        self.assertTrue(hasattr(plugin, "enabled"))
        self.assertTrue(hasattr(plugin, "has_standalone"))

    def test_plugin_info_to_dict(self):
        available = self.installer.list_available()
        plugin = available[0]
        d = plugin.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("id", d)
        self.assertIn("enabled", d)

    def test_enable_disable_plugin(self):
        # Disable first
        self.installer.disable("indoor_fence")
        self.assertNotIn("indoor_fence", self.installer.get_enabled())
        # Re-enable
        self.installer.enable("indoor_fence")
        self.assertIn("indoor_fence", self.installer.get_enabled())

    def test_is_enabled(self):
        self.installer.enable("indoor_fence")
        self.assertTrue(self.installer.is_enabled("indoor_fence"))

    def test_check_dependencies(self):
        missing = self.installer.check_dependencies("indoor_fence")
        self.assertIsInstance(missing, list)

    def test_get_plugin_info(self):
        info = self.installer.get_plugin_info("indoor_fence")
        self.assertIsNotNone(info)
        self.assertEqual(info.id, "indoor_fence")
        self.assertTrue(hasattr(info, "enabled"))

    def test_list_by_category(self):
        categories = self.installer.list_by_category()
        self.assertIn("indoor", categories)
        self.assertIn("outdoor", categories)
        self.assertIn("monitoring", categories)

    def test_enable_all(self):
        self.installer.enable_all()
        enabled = self.installer.get_enabled()
        self.assertGreaterEqual(len(enabled), 15)

    def test_get_disabled(self):
        self.installer.disable("indoor_fence")
        disabled = self.installer.get_disabled()
        self.assertIn("indoor_fence", disabled)
        # Re-enable
        self.installer.enable("indoor_fence")


if __name__ == "__main__":
    unittest.main()
