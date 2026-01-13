#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shim package to support `python -m plugins.indoor_fence.demo` from this folder."""

from __future__ import annotations

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Expose the real top-level plugins package path.
__path__ = [os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
