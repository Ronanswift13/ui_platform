#!/usr/bin/env python3
"""Compatibility shim for the experimental advanced bird detector.

Production inference is `plugin.py -> detector.py::BirdDetector`.
The advanced detector contains concept/demo logic and must not be wired into
the production chain without an explicit opt-in contract and tests.
"""

if __package__:
    from .experimental.advanced_bird_detector import *  # noqa: F401,F403
else:  # pragma: no cover - direct legacy script compatibility
    from experimental.advanced_bird_detector import *  # noqa: F401,F403
