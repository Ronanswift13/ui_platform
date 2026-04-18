"""标签契约冻结一致性测试。"""

from __future__ import annotations

from pathlib import Path
import re

import yaml

from plugins.busbar_inspection.detector_enhanced import BusbarDetectorEnhanced
from plugins.busbar_inspection.label_contract import (
    canonicalize_label,
    get_label_contract_snapshot,
    is_runtime_supported_defect_label,
)
from plugins.busbar_inspection.plugin import BusbarInspectionPlugin


PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_readme_contract() -> dict:
    readme_text = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"<!-- LABEL_CONTRACT_START -->\s*```yaml\s*(.*?)\s*```\s*<!-- LABEL_CONTRACT_END -->",
        readme_text,
        re.DOTALL,
    )
    assert match is not None, "README.md 必须包含 LABEL_CONTRACT 标记块"
    return yaml.safe_load(match.group(1))


def test_readme_default_yaml_detector_and_plugin_share_one_label_contract():
    """README / default.yaml / detector / plugin 必须共享同一标签契约。"""
    contract = get_label_contract_snapshot()
    readme_contract = _load_readme_contract()
    yaml_contract = yaml.safe_load(
        (PLUGIN_DIR / "configs" / "default.yaml").read_text(encoding="utf-8")
    )["labels"]

    assert readme_contract == yaml_contract
    assert readme_contract["runtime_supported"] == contract["runtime_supported"]
    assert readme_contract["aliases"] == contract["aliases"]
    assert readme_contract["planned"] == contract["planned"]
    assert tuple(contract["runtime_supported"]) == BusbarInspectionPlugin.SUPPORTED_RUNTIME_LABELS
    assert tuple(contract["runtime_supported"]) == BusbarDetectorEnhanced.RUNTIME_SUPPORTED_LABELS
    assert tuple(contract["runtime_supported_defects"]) == (
        BusbarInspectionPlugin.SUPPORTED_RUNTIME_DEFECT_LABELS
    )
    assert tuple(contract["runtime_supported_defects"]) == (
        BusbarDetectorEnhanced.RUNTIME_SUPPORTED_DEFECT_LABELS
    )
    assert set(BusbarInspectionPlugin.LABEL_NAMES) == set(contract["runtime_supported"])


def test_runtime_supported_labels_are_frozen_to_baseline_four():
    """当前 runtime supported labels 必须固定为四类基线。"""
    contract = get_label_contract_snapshot()

    assert contract["runtime_supported"] == [
        "pin_missing",
        "crack",
        "foreign_object",
        "quality_failed",
    ]
    assert contract["runtime_supported_defects"] == [
        "pin_missing",
        "crack",
        "foreign_object",
    ]
    assert canonicalize_label("loose_fitting") == "fitting_loose"
    assert is_runtime_supported_defect_label("loose_fitting") is False
    assert contract["planned"]["broken_part"] == "blocked"
    assert contract["planned"]["fitting_loose"] == "blocked"
