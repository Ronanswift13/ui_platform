#!/usr/bin/env python3
"""Convenience entry: python run_standalone.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plugins.thermal.standalone.app import main

main()
