"""Tests for adaptive threshold module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.rules.adaptive_threshold import AdaptiveThreshold


def test_adaptive_threshold_creation():
    at = AdaptiveThreshold(
        name="warning_distance",
        initial_value=0.3,
        min_value=0.1,
        max_value=1.0,
    )
    assert at.current_value == 0.3


def test_adaptive_threshold_feedback():
    at = AdaptiveThreshold(
        name="warning_distance",
        initial_value=0.3,
        min_value=0.1,
        max_value=1.0,
        learning_rate=0.1,
    )
    # Report false positives -> threshold should increase
    for _ in range(20):
        at.report_event(false_positive=True)
    assert at.current_value > 0.3


def test_adaptive_threshold_true_positive():
    at = AdaptiveThreshold(
        name="warning_distance",
        initial_value=0.5,
        min_value=0.1,
        max_value=1.0,
        learning_rate=0.1,
    )
    # Report true positives (missed alarms) -> threshold should decrease
    for _ in range(20):
        at.report_event(missed_alarm=True)
    assert at.current_value < 0.5


def test_adaptive_threshold_bounds():
    at = AdaptiveThreshold(
        name="test",
        initial_value=0.5,
        min_value=0.2,
        max_value=0.8,
        learning_rate=0.5,
    )
    # Push to boundary
    for _ in range(100):
        at.report_event(false_positive=True)
    assert at.current_value <= 0.8

    for _ in range(100):
        at.report_event(missed_alarm=True)
    assert at.current_value >= 0.2
