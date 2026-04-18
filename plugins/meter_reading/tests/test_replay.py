"""
Minimal replay baseline tests.

These tests validate the replay contract and execute the single mock LED pilot
fixture. Placeholder analog/digital/perspective slots are intentionally not
treated as verified real-image samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plugins.meter_reading.detector_enhanced import MeterReadingDetectorEnhanced, MeterType


PLUGIN_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLUGIN_DIR.parents[1]
FIXTURES_DIR = PLUGIN_DIR / "tests" / "fixtures"
EXPECTED_RESULTS_PATH = PLUGIN_DIR / "tests" / "replay" / "expected_results.json"

REQUIRED_FIXTURE_GROUPS = {
    "analog_normal",
    "analog_boundary",
    "analog_quality_fail",
    "digital_display",
    "led_indicator",
    "glare_or_tilt_review_required",
}


def _load_expected_results() -> dict:
    return json.loads(EXPECTED_RESULTS_PATH.read_text(encoding="utf-8"))


def _resolve_repo_path(relpath: str) -> Path:
    return PROJECT_ROOT / relpath


def _build_mock_image(fixture: dict) -> np.ndarray:
    image_spec = fixture["image"]
    if image_spec["kind"] != "solid_bgr":
        raise ValueError(f"Unsupported mock image kind: {image_spec['kind']}")
    shape = tuple(image_spec["shape"])
    bgr = np.array(image_spec["bgr"], dtype=np.uint8)
    image = np.zeros(shape, dtype=np.uint8)
    image[:, :] = bgr
    return image


def test_replay_fixture_directory_structure():
    """最小试点 fixture 分类目录必须存在。"""
    existing_groups = {path.name for path in FIXTURES_DIR.iterdir() if path.is_dir()}
    missing = REQUIRED_FIXTURE_GROUPS - existing_groups
    assert not missing


def test_expected_results_schema_fields():
    """expected_results.json 必须声明顶层、sample 和输出 metadata 字段。"""
    expected = _load_expected_results()
    schema = expected["expected_results_schema"]

    for field in schema["required_top_level_fields"]:
        assert field in expected, f"missing top-level field: {field}"

    assert expected["plugin_id"] == "meter_reading"
    assert expected["samples"], "replay baseline must contain sample slots"

    sample_required = set(schema["required_sample_fields"])
    for sample in expected["samples"]:
        missing = sample_required - set(sample)
        assert not missing, f"{sample['sample_id']} missing fields: {sorted(missing)}"
        metadata_required = set(schema["required_output_metadata_fields"])
        declared_required = set(sample["expected_output_contract"]["metadata_required"])
        assert metadata_required <= declared_required


def test_minimum_pilot_sample_slots_exist():
    """analog / digital / led / glare-or-tilt 六类槽位必须固定。"""
    expected = _load_expected_results()
    groups = {sample["fixture_group"] for sample in expected["samples"]}
    assert REQUIRED_FIXTURE_GROUPS <= groups


def test_placeholder_samples_do_not_claim_real_assets():
    """planned placeholder 不能被误标成 present_labeled。"""
    expected = _load_expected_results()
    placeholders = [s for s in expected["samples"] if s["source_type"] == "placeholder"]
    assert placeholders
    for sample in placeholders:
        assert sample["collection_status"] == "planned"
        assert sample.get("acceptance_blockers")


def test_real_labeled_image_replay_assets_are_explicitly_absent():
    """真实标注图像还未进入 replay，必须显式 skip 而不是假装已验证。"""
    expected = _load_expected_results()
    ready_real_images = [
        sample
        for sample in expected["samples"]
        if sample["source_type"] == "real"
        and sample["asset_kind"] == "image"
        and sample["collection_status"] == "present_labeled"
    ]
    if not ready_real_images:
        pytest.skip("No present_labeled real-image replay samples yet.")

    for sample in ready_real_images:
        assert _resolve_repo_path(sample["asset_relpath"]).exists()


def test_led_mock_replay_baseline(default_config):
    """执行最小 LED mock replay，验证结果与 output metadata 契约。"""
    expected = _load_expected_results()
    led_sample = next(
        sample
        for sample in expected["samples"]
        if sample["sample_id"] == "meter_led_indicator_green_001"
    )

    fixture_path = _resolve_repo_path(led_sample["asset_relpath"])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    image = _build_mock_image(fixture)

    detector = MeterReadingDetectorEnhanced(default_config)
    detector.initialize()
    result = detector.read_meter(image, MeterType.LED_INDICATOR, roi_id=led_sample["sample_id"])

    expected_contract = led_sample["expected_output_contract"]
    expected_targets = led_sample["expected_numeric_targets"]
    assert result.status.value == expected_contract["reading_status"]
    assert result.value == expected_targets["value"]
    assert result.unit == expected_targets["unit"]

    metadata = result.metadata
    for field in expected_contract["metadata_required"]:
        assert field in metadata, f"missing metadata field: {field}"
    assert metadata["runtime_mode"] == led_sample["expected_runtime_mode"]
    assert metadata["review_status"] == led_sample["expected_review_status"]
    assert metadata["failure_reason"] == fixture["expected"]["failure_reason"]
    assert metadata["color_class"] == expected_targets["color_class"]
