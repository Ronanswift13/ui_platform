"""
Local Model Registry - Discover and manage models from local directories.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = {".onnx", ".pt", ".pth", ".trt", ".engine", ".xml", ".bin"}


@dataclass
class ModelEntry:
    """Registry entry for a discovered model."""
    model_id: str
    path: str
    format: str
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalModelRegistry:
    """
    Discover and track models from local directories.

    Scans plugin's models/ directory and optional global models/ directory.
    """

    def __init__(self, search_dirs: Optional[List[str]] = None):
        self._entries: Dict[str, ModelEntry] = {}
        self._search_dirs = [Path(d) for d in (search_dirs or [])]

    def scan(self, directory: Optional[str] = None) -> int:
        """Scan a directory for model files. Returns count of models found."""
        dirs = [Path(directory)] if directory else self._search_dirs
        count = 0
        for d in dirs:
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.suffix.lower() in MODEL_EXTENSIONS and f.is_file():
                    model_id = f.stem
                    if model_id in self._entries:
                        model_id = f"{f.parent.name}_{model_id}"
                    entry = ModelEntry(
                        model_id=model_id,
                        path=str(f),
                        format=f.suffix.lower().lstrip("."),
                        size_bytes=f.stat().st_size,
                    )
                    meta_path = f.with_suffix(".json")
                    if meta_path.exists():
                        try:
                            entry.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    self._entries[model_id] = entry
                    count += 1
                    logger.debug(f"Found model: {model_id} ({entry.format})")
        logger.info(f"Model registry: scanned {len(dirs)} dirs, found {count} models")
        return count

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._entries.get(model_id)

    def list_models(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def list_by_format(self, fmt: str) -> List[ModelEntry]:
        return [e for e in self._entries.values() if e.format == fmt]

    def register(self, model_id: str, path: str, metadata: Optional[Dict] = None) -> ModelEntry:
        """Manually register a model."""
        p = Path(path)
        entry = ModelEntry(
            model_id=model_id,
            path=str(p),
            format=p.suffix.lower().lstrip("."),
            size_bytes=p.stat().st_size if p.exists() else 0,
            metadata=metadata or {},
        )
        self._entries[model_id] = entry
        return entry

    def remove(self, model_id: str) -> bool:
        return self._entries.pop(model_id, None) is not None
