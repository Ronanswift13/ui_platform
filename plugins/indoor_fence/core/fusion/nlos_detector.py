"""
Indoor Fence V3.0 - NLOS Detector
===================================

Classifies UWB signals as line-of-sight (LOS) or non-line-of-sight (NLOS)
using first-path power, total power, and delay spread features.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NLOSFeatures:
    """Features extracted from a UWB channel impulse response."""
    first_path_power: float  # dBm
    total_power: float  # dBm
    delay_spread_ns: float  # nanoseconds


class NLOSDetector:
    """Classifies UWB measurements as LOS or NLOS.

    Uses a simple heuristic model based on:
    - Power difference (total - first_path): large = NLOS
    - Delay spread: high = multipath = NLOS
    """

    def __init__(
        self,
        power_diff_threshold: float = 15.0,
        delay_spread_threshold_ns: float = 20.0,
    ):
        self._power_diff_threshold = power_diff_threshold
        self._delay_spread_threshold = delay_spread_threshold_ns

    def classify(
        self,
        first_path_power: float,
        total_power: float,
        delay_spread_ns: float,
    ) -> float:
        """Return NLOS probability [0.0 (LOS) .. 1.0 (NLOS)].

        Uses a soft classification based on power difference and delay spread.
        """
        # Power difference feature (higher = more NLOS-like)
        power_diff = total_power - first_path_power
        power_score = min(1.0, max(0.0, power_diff / self._power_diff_threshold))

        # Delay spread feature (higher = more multipath = NLOS)
        delay_score = min(1.0, max(0.0, delay_spread_ns / self._delay_spread_threshold))

        # Weighted combination
        probability = 0.5 * power_score + 0.5 * delay_score
        return float(min(1.0, max(0.0, probability)))
