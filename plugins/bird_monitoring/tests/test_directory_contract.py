"""Directory and entrypoint contract tests for bird_monitoring."""

from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_entrypoints_keep_clear_roles():
    expected_paths = [
        "plugin.py",
        "detector.py",
        "standalone/app.py",
        "run_standalone.py",
        "__main__.py",
        "main.py",
        "demo/run_demo.py",
    ]

    for rel_path in expected_paths:
        assert (PLUGIN_DIR / rel_path).exists(), rel_path


def test_advanced_detector_is_experimental_with_root_compatibility_shim():
    wrapper = PLUGIN_DIR / "advanced_bird_detector.py"
    experimental = PLUGIN_DIR / "experimental" / "advanced_bird_detector.py"

    assert wrapper.exists()
    assert experimental.exists()

    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert ".experimental.advanced_bird_detector" in wrapper_text
    assert "Production inference is `plugin.py -> detector.py::BirdDetector`" in wrapper_text
    assert "np.random" not in wrapper_text


def test_standalone_simulation_is_isolated_from_production_detector():
    app_text = (PLUGIN_DIR / "standalone" / "app.py").read_text(encoding="utf-8")
    simulator_text = (
        PLUGIN_DIR / "standalone" / "bird_simulator.py"
    ).read_text(encoding="utf-8")

    assert "/api/simulator/" in app_text
    assert 'runtime_mode="standalone_simulation"' in simulator_text
    assert "from plugins.bird_monitoring.plugin" not in simulator_text
    assert "from plugins.bird_monitoring.detector" not in simulator_text


def test_supporting_directories_are_explicit():
    expected_dirs = [
        "experimental",
        "standalone/static",
        "docs",
        "prompts",
        "tests/regression",
    ]
    expected_docs = [
        "experimental/README.md",
        "standalone/static/README.md",
        "docs/README.md",
        "prompts/README.md",
        "tests/regression/README.md",
    ]

    for rel_path in expected_dirs:
        assert (PLUGIN_DIR / rel_path).is_dir(), rel_path
    for rel_path in expected_docs:
        assert (PLUGIN_DIR / rel_path).exists(), rel_path


def test_enhanced_detector_requires_explicit_opt_in(monkeypatch):
    """`BirdDetectorEnhanced` 必须是 opt-in shim，默认不得可实例化。"""
    import os
    import pytest

    from plugins.bird_monitoring.detector import BirdDetectorEnhanced

    # 确保默认环境变量未设置
    monkeypatch.delenv("BIRD_ENABLE_ENHANCED_DETECTOR", raising=False)
    with pytest.raises(RuntimeError, match="experimental"):
        BirdDetectorEnhanced({})


def test_enhanced_detector_body_lives_in_experimental():
    experimental = PLUGIN_DIR / "experimental" / "enhanced_detector.py"
    assert experimental.exists()
    text = experimental.read_text(encoding="utf-8")
    assert "class EnhancedBirdDetector(BirdDetector)" in text
    detector_text = (PLUGIN_DIR / "detector.py").read_text(encoding="utf-8")
    # detector.py 仅保留 opt-in shim；不得再内嵌真实实现
    assert "_OPT_IN_ENV" in detector_text
    assert "BIRD_ENABLE_ENHANCED_DETECTOR" in detector_text
