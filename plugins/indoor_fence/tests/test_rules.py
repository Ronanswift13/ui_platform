"""Tests for rule engine and risk scorer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.core.rules.rule_engine import (
    RuleEngine, Rule, RuleAction,
)
from plugins.indoor_fence.core.rules.risk_scorer import RiskScorer
from plugins.indoor_fence.protocols import PersonStateV3, RiskAssessment


def test_rule_creation():
    rule = Rule(
        rule_id="cross_line_alarm",
        condition="person_state == 'cross_line'",
        action=RuleAction.ALARM_RED,
        priority=10,
        cooldown_seconds=5.0,
    )
    assert rule.rule_id == "cross_line_alarm"


def test_rule_engine_evaluate():
    engine = RuleEngine()
    engine.add_rule(Rule(
        rule_id="cross_line",
        condition="person_state == 'cross_line'",
        action=RuleAction.ALARM_RED,
        priority=10,
        cooldown_seconds=0,
    ))
    actions = engine.evaluate({
        "person_state": "cross_line",
        "zone_id": "cabinet_1",
    })
    assert RuleAction.ALARM_RED in actions


def test_rule_engine_cooldown():
    engine = RuleEngine()
    engine.add_rule(Rule(
        rule_id="test",
        condition="person_state == 'on_line'",
        action=RuleAction.ALARM_YELLOW,
        priority=5,
        cooldown_seconds=10.0,
    ))
    # First evaluation triggers
    actions1 = engine.evaluate({"person_state": "on_line"})
    assert RuleAction.ALARM_YELLOW in actions1
    # Immediate re-evaluation should be cooled down
    actions2 = engine.evaluate({"person_state": "on_line"})
    assert RuleAction.ALARM_YELLOW not in actions2


def test_rule_engine_from_yaml(tmp_path):
    yaml_content = """
rules:
  - rule_id: cross_line
    condition: "person_state == 'cross_line'"
    action: alarm_red
    priority: 10
    cooldown_seconds: 5
  - rule_id: climbing
    condition: "person_state == 'climbing'"
    action: alarm_red
    priority: 9
    cooldown_seconds: 3
"""
    yaml_file = tmp_path / "rules.yaml"
    yaml_file.write_text(yaml_content)
    engine = RuleEngine.from_yaml(str(yaml_file))
    assert len(engine.rules) == 2


def test_risk_scorer():
    scorer = RiskScorer()
    score = scorer.score(
        person_state=PersonStateV3.CROSS_LINE,
        distance_to_danger=0.05,
        is_authorized=False,
        behavior="normal",
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Cross line + unauthorized + near danger should be elevated


def test_risk_scorer_low_risk():
    scorer = RiskScorer()
    score = scorer.score(
        person_state=PersonStateV3.NORMAL,
        distance_to_danger=2.0,
        is_authorized=True,
        behavior="normal",
    )
    assert score < 0.3  # Normal + authorized + far from danger
