"""
Indoor Fence V3.0 - Adaptive Threshold
========================================

Self-tuning thresholds that adjust based on feedback:
- False positives → increase threshold (less sensitive)
- Missed alarms → decrease threshold (more sensitive)
Bounded by min/max values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class AdaptiveThreshold:
    """A threshold that adapts based on feedback events."""

    def __init__(
        self,
        name: str,
        initial_value: float,
        min_value: float = 0.0,
        max_value: float = 1.0,
        learning_rate: float = 0.01,
    ):
        self._name = name
        self._value = initial_value
        self._min = min_value
        self._max = max_value
        self._lr = learning_rate
        self._history: List[float] = [initial_value]

    @property
    def name(self) -> str:
        return self._name

    @property
    def current_value(self) -> float:
        return self._value

    @property
    def history(self) -> List[float]:
        return list(self._history)

    def report_event(
        self,
        false_positive: bool = False,
        missed_alarm: bool = False,
    ) -> None:
        """Report an event to adjust the threshold.

        Args:
            false_positive: Alarm triggered when it shouldn't have.
                           → Increase threshold (less sensitive).
            missed_alarm: Real event missed.
                         → Decrease threshold (more sensitive).
        """
        if false_positive:
            # Too many false alarms → raise threshold
            self._value += self._lr
        elif missed_alarm:
            # Missed real events → lower threshold
            self._value -= self._lr

        # Clamp to bounds
        self._value = max(self._min, min(self._max, self._value))
        self._history.append(self._value)

    def reset(self, value: Optional[float] = None) -> None:
        """Reset to a specific value or initial."""
        if value is not None:
            self._value = max(self._min, min(self._max, value))
        else:
            self._value = self._history[0] if self._history else self._min
        self._history = [self._value]
