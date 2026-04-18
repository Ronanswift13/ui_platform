"""
Indoor Fence V3.0 - Multi-Dimensional Risk Scorer
====================================================

Computes a composite risk score [0..1] based on:
- Person state severity
- Distance to danger zone
- Authorization status
- Behavior classification
"""
from __future__ import annotations

from typing import Optional

from ...protocols import PersonStateV3


class RiskScorer:
    """Computes a composite risk score from multiple factors."""

    # Base risk per state
    _STATE_RISK = {
        PersonStateV3.NORMAL: 0.0,
        PersonStateV3.ON_LINE: 0.3,
        PersonStateV3.MISPLACED: 0.4,
        PersonStateV3.PROLONGED_STAY: 0.35,
        PersonStateV3.CROSS_LINE: 0.7,
        PersonStateV3.MULTI_PERSON: 0.5,
        PersonStateV3.CLIMBING: 0.8,
        PersonStateV3.HIGH_RISK: 0.9,
        PersonStateV3.FALLEN: 0.95,
    }

    # Behavior risk modifier
    _BEHAVIOR_RISK = {
        "normal": 0.0,
        "normal_walk": 0.0,
        "prolonged_stay": 0.15,
        "climbing": 0.3,
        "crossing": 0.25,
        "fallen": 0.4,
    }

    def __init__(
        self,
        distance_weight: float = 0.2,
        auth_weight: float = 0.15,
        behavior_weight: float = 0.15,
        state_weight: float = 0.5,
    ):
        self._distance_w = distance_weight
        self._auth_w = auth_weight
        self._behavior_w = behavior_weight
        self._state_w = state_weight

    def score(
        self,
        person_state: PersonStateV3,
        distance_to_danger: float = 10.0,
        is_authorized: bool = True,
        behavior: str = "normal",
    ) -> float:
        """Compute composite risk score [0.0 .. 1.0].

        Args:
            person_state: Current person state from state machine
            distance_to_danger: Distance to nearest danger zone (meters)
            is_authorized: Whether person is authorized in current zone
            behavior: Behavior classification string

        Returns:
            Risk score between 0.0 (safe) and 1.0 (critical)
        """
        # State component
        state_risk = self._STATE_RISK.get(person_state, 0.0)

        # Distance component (closer = higher risk)
        # Sigmoid-like: 1.0 at dist=0, ~0.0 at dist>2m
        distance_risk = max(0.0, 1.0 - distance_to_danger / 2.0)

        # Authorization component
        auth_risk = 0.0 if is_authorized else 0.5

        # Behavior component
        behavior_risk = self._BEHAVIOR_RISK.get(behavior, 0.0)

        # Weighted combination
        total = (
            self._state_w * state_risk
            + self._distance_w * distance_risk
            + self._auth_w * auth_risk
            + self._behavior_w * behavior_risk
        )

        return float(min(1.0, max(0.0, total)))
