#!/usr/bin/env python3
"""Capacitor Inspection - Standalone"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from darkbreaker_sdk.standalone import StandalonePluginRunner
from plugins.capacitor_inspection.plugin import CapacitorInspectionPlugin as Plugin


def main():
    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parent / "templates",
        plugin_static_dir=Path(__file__).parent / "static",
    )
    runner.run(host="0.0.0.0", port=8090)


if __name__ == "__main__":
    main()
