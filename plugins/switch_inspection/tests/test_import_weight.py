"""
回归测试: 模块级 import 不应触发重型初始化 (torch / OpenMP)

根因: plugins._model_resolution → training → torch 链在模块级被触发，
导致 clean-start 场景 (multiprocessing spawn / gunicorn preload) 失败。
修复: _model_resolution import 延迟到 init() 调用时。

对应 03_test_strategy L0 targeted 级别。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestImportWeight:
    """验证 switch_inspection 模块导入不触发 torch/training 初始化。"""

    def test_plugin_module_import_no_torch(self):
        """import plugin.py 不应加载 torch（延迟到 init）。"""
        mods_to_remove = [
            k for k in sys.modules
            if k.startswith("plugins.switch_inspection.plugin")
        ]
        for m in mods_to_remove:
            del sys.modules[m]

        if "torch" in sys.modules:
            pytest.skip("torch already loaded by prior test; run in isolation")

        import importlib
        mod = importlib.import_module("plugins.switch_inspection.plugin")

        assert hasattr(mod, "Plugin")
        assert hasattr(mod, "SwitchInspectionPlugin")
        assert hasattr(mod, "_get_model_resolver")
        assert callable(mod._get_model_resolver)

    def test_package_init_import_no_torch(self):
        """import plugins.switch_inspection 包级别不应加载 plugin.py。"""
        if "torch" in sys.modules:
            pytest.skip("torch already loaded by prior test; run in isolation")

        mods_to_remove = [
            k for k in sys.modules
            if k.startswith("plugins.switch_inspection")
        ]
        for m in mods_to_remove:
            del sys.modules[m]

        import importlib
        importlib.import_module("plugins.switch_inspection")

        assert "plugins.switch_inspection.plugin" not in sys.modules

    def test_create_standalone_succeeds(self):
        """create_standalone() 完整链路仍可用。"""
        from plugins.switch_inspection.plugin import Plugin

        p = Plugin.create_standalone()
        assert p.status.value == "ready" or str(p.status) == "PluginStatus.READY"

    def test_lazy_getattr_works(self):
        """PEP 562 __getattr__ 延迟加载导出名称可用。"""
        import plugins.switch_inspection as pkg

        SwitchPlugin = pkg.SwitchInspectionPlugin
        assert SwitchPlugin is not None
        assert hasattr(SwitchPlugin, "create_standalone")
