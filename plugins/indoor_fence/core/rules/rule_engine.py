"""
Indoor Fence V3.0 - Configurable Rule Engine
==============================================

Evaluates rules against context data (person state, zone, behavior)
and produces alarm actions with cooldown management.
Rules can be loaded from YAML configuration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class RuleAction(str, Enum):
    """Actions that can be triggered by rules."""
    NONE = "none"
    ALARM_GREEN = "alarm_green"
    ALARM_YELLOW = "alarm_yellow"
    ALARM_RED = "alarm_red"
    ALARM_CRITICAL = "alarm_critical"
    NOTIFY = "notify"
    LOG_ONLY = "log_only"


@dataclass
class Rule:
    """A single evaluation rule."""
    rule_id: str
    condition: str  # Simple expression like "person_state == 'cross_line'"
    action: RuleAction
    priority: int = 0
    cooldown_seconds: float = 5.0
    enabled: bool = True
    description: str = ""


class RuleEngine:
    """Evaluates rules against context and manages cooldowns."""

    def __init__(self):
        self._rules: List[Rule] = []
        self._last_fired: Dict[str, float] = {}  # rule_id -> last fire time

    @property
    def rules(self) -> List[Rule]:
        """Get all registered rules."""
        return list(self._rules)

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self._rules.append(rule)
        # Keep sorted by priority (higher first)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule by ID."""
        self._rules = [r for r in self._rules if r.rule_id != rule_id]

    def evaluate(self, context: Dict[str, Any]) -> List[RuleAction]:
        """Evaluate all rules against the given context.

        Returns list of triggered actions (after cooldown filtering).
        """
        now = time.time()
        actions: List[RuleAction] = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            # Check cooldown
            last = self._last_fired.get(rule.rule_id, 0.0)
            if (now - last) < rule.cooldown_seconds:
                continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, context):
                actions.append(rule.action)
                self._last_fired[rule.rule_id] = now

        return actions

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple condition string against context.

        Supports: ==, !=, >, <, >=, <=
        Format: "key operator value"
        """
        try:
            # Parse simple conditions: "key == 'value'" or "key > number"
            for op in ("==", "!=", ">=", "<=", ">", "<"):
                if op in condition:
                    parts = condition.split(op, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        expected = parts[1].strip().strip("'\"")
                        actual = str(context.get(key, ""))

                        if op == "==":
                            return actual == expected
                        elif op == "!=":
                            return actual != expected
                        elif op == ">":
                            return float(actual) > float(expected)
                        elif op == "<":
                            return float(actual) < float(expected)
                        elif op == ">=":
                            return float(actual) >= float(expected)
                        elif op == "<=":
                            return float(actual) <= float(expected)
        except (ValueError, TypeError) as e:
            logger.warning(f"Rule condition evaluation failed: {condition}: {e}")
            return False

        return False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RuleEngine":
        """Load rules from a YAML file."""
        try:
            import yaml
        except ImportError:
            # Fallback: try to parse manually if PyYAML not installed
            return cls._from_yaml_fallback(yaml_path)

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        engine = cls()
        for rule_data in data.get("rules", []):
            action_str = rule_data.get("action", "none")
            action = RuleAction(action_str)
            rule = Rule(
                rule_id=rule_data["rule_id"],
                condition=rule_data["condition"],
                action=action,
                priority=rule_data.get("priority", 0),
                cooldown_seconds=rule_data.get("cooldown_seconds", 5.0),
                enabled=rule_data.get("enabled", True),
                description=rule_data.get("description", ""),
            )
            engine.add_rule(rule)

        return engine

    @classmethod
    def _from_yaml_fallback(cls, yaml_path: str) -> "RuleEngine":
        """Minimal YAML parser fallback when PyYAML is not installed."""
        import re

        engine = cls()
        with open(yaml_path, "r") as f:
            content = f.read()

        # Simple regex-based extraction
        rule_blocks = re.findall(
            r'rule_id:\s*(\S+).*?condition:\s*"([^"]*)".*?action:\s*(\S+).*?priority:\s*(\d+).*?cooldown_seconds:\s*(\d+(?:\.\d+)?)',
            content,
            re.DOTALL,
        )

        for rule_id, condition, action_str, priority, cooldown in rule_blocks:
            try:
                action = RuleAction(action_str)
            except ValueError:
                action = RuleAction.NONE
            rule = Rule(
                rule_id=rule_id,
                condition=condition,
                action=action,
                priority=int(priority),
                cooldown_seconds=float(cooldown),
            )
            engine.add_rule(rule)

        return engine
