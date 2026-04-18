import importlib
import json
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_plugin_alias_and_local_entrypoint_files_exist():
    module = importlib.import_module("plugins.action_event_monitoring.plugin")

    assert module.Plugin is module.ActionEventMonitoringPlugin
    assert (PLUGIN_DIR / "requirements.txt").is_file()
    assert (PLUGIN_DIR / "demo" / "run_demo.py").is_file()
    assert (PLUGIN_DIR / "__main__.py").is_file()
    assert (PLUGIN_DIR / "run_standalone.py").is_file()


def test_manifest_declares_local_standalone_entrypoint_and_port():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    standalone = manifest["standalone"]

    assert manifest["category"] == "outdoor"
    assert manifest["entrypoint"] == "plugin.py"
    assert manifest["plugin_class"] == "ActionEventMonitoringPlugin"
    assert manifest["runtime_requirements"] == "requirements.txt"
    assert standalone["enabled"] is True
    assert standalone["entrypoint"] == "run_standalone.py"
    assert standalone["module"] == "plugins.action_event_monitoring"
    assert standalone["port"] == 8097
    assert standalone["smoke_path"] == "/api/action-event/smoke"


def test_demo_module_exposes_main_without_starting_server():
    demo = importlib.import_module("plugins.action_event_monitoring.demo.run_demo")
    package_main = importlib.import_module("plugins.action_event_monitoring.__main__")
    run_standalone = importlib.import_module("plugins.action_event_monitoring.run_standalone")

    assert callable(demo.main)
    assert callable(package_main.main)
    assert callable(run_standalone.main)
