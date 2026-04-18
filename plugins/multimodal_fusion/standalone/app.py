#!/usr/bin/env python3
"""Multimodal Fusion - Standalone"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from darkbreaker_sdk.standalone import StandalonePluginRunner
from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin as Plugin


def create_runner(plugin=None) -> StandalonePluginRunner:
    plugin = plugin or Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent / "templates",
        plugin_static_dir=Path(__file__).parent / "static",
    )
    runner.app.get("/health")(runner._get_health)
    runner.app.get("/status")(runner._get_status)
    return runner


def create_app(plugin=None):
    runner = create_runner(plugin=plugin)
    return runner.app


def main():
    runner = create_runner()
    runner.run(host="0.0.0.0", port=8096)


if __name__ == "__main__":
    main()
