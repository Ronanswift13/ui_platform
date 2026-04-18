#!/usr/bin/env python3
"""SLAM Mapping - Standalone Application."""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from darkbreaker_sdk.standalone import StandalonePluginRunner
from plugins.slam_mapping.plugin import Plugin
from plugins.slam_mapping.standalone.simulator import register_simulation_routes


def create_runner():
    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent / "templates",
        plugin_static_dir=Path(__file__).parent / "static",
    )
    register_simulation_routes(runner.app)
    return runner


def main():
    runner = create_runner()
    runner.run(host="0.0.0.0", port=8084)


if __name__ == "__main__":
    main()
