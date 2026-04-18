"""Tests for V3 state machine with expanded states."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.state_machine import StateMachine, PersonState
from plugins.indoor_fence.core.geometry import Point2D
from plugins.indoor_fence.core.zone_config import ZoneConfigLoader


def test_person_state_has_new_states():
    assert hasattr(PersonState, 'CLIMBING')
    assert hasattr(PersonState, 'PROLONGED_STAY')
    assert hasattr(PersonState, 'FALLEN')
    assert hasattr(PersonState, 'MULTI_PERSON')


def test_climbing_state():
    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 3.0),
        metadata={"behavior": "climbing"}
    )
    assert result.state == PersonState.CLIMBING


def test_fallen_state():
    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 1.0),
        metadata={"behavior": "fallen"}
    )
    assert result.state == PersonState.FALLEN


def test_prolonged_stay_state():
    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 3.0),
        metadata={"behavior": "prolonged_stay"}
    )
    assert result.state == PersonState.PROLONGED_STAY


def test_behavior_overrides_position():
    """Behavior-based states should take priority when applicable."""
    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)
    # Person in normal zone but fallen
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 1.0),
        metadata={"behavior": "fallen"}
    )
    assert result.state == PersonState.FALLEN  # Behavior overrides normal position
