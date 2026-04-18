#!/usr/bin/env python3
"""Run action_event_monitoring as a standalone plugin."""

from __future__ import annotations

import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent
REPO_DIR = PLUGIN_DIR.parent.parent

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from plugins.action_event_monitoring.standalone.app import main


if __name__ == "__main__":
    main()
