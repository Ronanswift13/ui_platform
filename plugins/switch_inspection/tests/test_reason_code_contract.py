#!/usr/bin/env python3
"""Reason-code contract tests for switch_inspection."""

from plugins.switch_inspection.plugin import Plugin, SwitchInspectionPlugin


def test_reason_codes_align_with_yaml(default_config):
    yaml_codes = {int(key): value for key, value in default_config["reason_codes"].items()}
    assert set(SwitchInspectionPlugin.REASON_CODES) == set(yaml_codes)
    for code, message in SwitchInspectionPlugin.REASON_CODES.items():
        assert yaml_codes[code]
        assert message


def test_plugin_reason_codes_cover_interlock_and_clarity(default_config):
    plugin = Plugin.create_standalone(default_config)
    assert plugin.REASON_CODES[1001] == "清晰度过低"
    assert plugin.REASON_CODES[2001] == "互锁逻辑异常"
