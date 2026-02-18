"""
Plugin configuration loader.

Supports loading configuration from YAML or JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_plugin_config(path: str | Path) -> dict[str, Any]:
    """
    Load plugin configuration from a YAML or JSON file.

    If the file does not exist, returns an empty dictionary.
    File format is determined by extension (.yaml/.yml for YAML, .json for JSON).

    Args:
        path: Path to the configuration file.

    Returns:
        Configuration dictionary (empty dict if file missing).
    """
    path = Path(path)

    if not path.exists():
        return {}

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    else:
        # Try JSON first, then YAML
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, ValueError):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
            except ImportError:
                return {}
