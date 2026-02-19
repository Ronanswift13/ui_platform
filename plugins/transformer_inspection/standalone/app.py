#!/usr/bin/env python3
"""
Standalone server for the Transformer Inspection plugin.

Usage:
    python -m plugins.transformer_inspection.standalone.app
    python plugins/transformer_inspection/standalone/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plugins.transformer_inspection.plugin import TransformerInspectionPlugin
from darkbreaker_sdk.standalone import StandalonePluginRunner


def main():
    plugin = TransformerInspectionPlugin.create_standalone()

    runner = StandalonePluginRunner(
        plugin=plugin,
        title="Transformer Inspection - Standalone",
        port=8087,
        plugin_templates_dir=Path(__file__).resolve().parent / "templates",
    )
    runner.run()


if __name__ == "__main__":
    main()
