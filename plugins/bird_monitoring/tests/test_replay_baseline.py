"""bird_monitoring replay baseline — simulation 主链可复现的最小回归。

本测试只覆盖 `simulation` 主链能稳定复现的两类样本：
- `no_bird_clear` → expected label `no_bird`
- `quality_dark`  → expected label `quality_failed`

其它 label（bird_safe / unknown_bird / review_required / bird_danger / …）必须
依赖真实图像 + real_dl 模型，在当前阶段 `collection_status=planned_blocked_by_model`，
不得由合成数据伪造。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[1]
REGRESSION_DIR = PLUGIN_DIR / "tests" / "regression"
FIXTURE_DIR = REGRESSION_DIR / "fixtures"


def _ensure_fixtures() -> None:
    """按需构建 synthetic fixture；不覆盖已存在文件。"""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    # 懒加载 builder；避免顶层 sys.path 污染
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_synthetic_fixtures",
        REGRESSION_DIR / "build_synthetic_fixtures.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    no_bird_path = FIXTURE_DIR / "no_bird_clear.npy"
    quality_dark_path = FIXTURE_DIR / "quality_dark.npy"
    if not no_bird_path.exists():
        np.save(no_bird_path, mod.build_no_bird_clear())
    if not quality_dark_path.exists():
        np.save(quality_dark_path, mod.build_quality_dark())


@pytest.fixture(scope="module", autouse=True)
def _fixtures_ready():
    _ensure_fixtures()


def _load(name: str) -> np.ndarray:
    return np.load(FIXTURE_DIR / f"{name}.npy")


class TestReplaySimulationSafeLabels:
    """simulation 主链可复现 label 的 replay 回归。"""

    def test_no_bird_clear_replay(self, plugin_instance, make_context, make_roi):
        frame = _load("no_bird_clear")
        results = plugin_instance.infer(frame, [make_roi()], make_context())
        assert len(results) == 1
        assert results[0].label == "no_bird"
        assert results[0].metadata["runtime_mode"] == "simulation"
        assert results[0].metadata["input_quality"]["status"] in ("pass", "soft_fail")

    def test_quality_dark_replay(self, plugin_instance, make_context, make_roi):
        frame = _load("quality_dark")
        results = plugin_instance.infer(frame, [make_roi()], make_context())
        assert len(results) == 1
        assert results[0].label == "quality_failed"
        quality = results[0].metadata["input_quality"]
        assert quality["status"] == "hard_fail"
        assert quality["is_valid"] is False


class TestReplayBlockedLabels:
    """blocked label 仍保留槽位，不得用合成数据伪造闭环。"""

    def test_expected_results_declares_blocked_slots(self):
        expected_path = PLUGIN_DIR / "tests" / "replay" / "expected_results.json"
        data = json.loads(expected_path.read_text(encoding="utf-8"))
        blocked_labels = {
            "bird_safe", "bird_warning", "bird_danger", "bird_critical",
            "unknown_bird", "review_required",
        }
        for sample in data["samples"]:
            labels = set(sample.get("expected_primary_labels", []))
            if labels & blocked_labels:
                status = sample.get("collection_status", "")
                assert "blocked_by_model" in status, (
                    f"{sample['sample_id']} 声称产出 blocked label 但未标注 "
                    f"planned_blocked_by_model"
                )
