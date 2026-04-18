"""bird_monitoring 冒烟测试 — 验证插件可初始化、推理、健康检查、standalone 仿真隔离"""
from pathlib import Path

import pytest
import numpy as np


class TestPluginStandalone:
    """插件基础生命周期"""

    def test_create_standalone(self, plugin_instance):
        assert plugin_instance is not None
        assert plugin_instance._initialized is True

    def test_production_chain_uses_base_detector(self, plugin_instance):
        assert plugin_instance._detector.__class__.__name__ == "BirdDetector"

    def test_healthcheck_after_init(self, plugin_instance):
        health = plugin_instance.healthcheck()
        assert health.healthy is True
        assert "ready" in health.details.get("status", "")
        assert health.details.get("runtime_mode") in ("simulation", "traditional_fallback", "real_dl")
        assert health.details.get("real_model_loaded") is False
        assert health.details.get("onnx_session_ready") is False
        assert health.details.get("simulation_enabled") is True

    def test_infer_returns_results(self, plugin_instance, sample_frame, make_context, make_roi):
        ctx = make_context()
        rois = [make_roi()]
        results = plugin_instance.infer(sample_frame, rois, ctx)
        assert isinstance(results, list)
        assert len(results) >= 1
        # 模拟模式下应返回 no_bird
        for r in results:
            assert r.label in ("no_bird", "bird_safe", "bird_warning", "bird_danger",
                               "bird_critical", "review_required", "unknown_bird",
                               "quality_failed", "error")

    def test_simulation_chain_does_not_emit_bird_detection(
        self, plugin_instance, sample_frame, make_context, make_roi
    ):
        results = plugin_instance.infer(sample_frame, [make_roi()], make_context())
        assert [r.label for r in results] == ["no_bird"]
        assert results[0].metadata["runtime_mode"] == "simulation"

    def test_infer_no_bird_has_metadata(self, plugin_instance, sample_frame, make_context, make_roi):
        ctx = make_context()
        results = plugin_instance.infer(sample_frame, [make_roi()], ctx)
        for r in results:
            assert "runtime_mode" in r.metadata
            assert "training_placeholders" in r.metadata

    def test_cleanup(self, plugin_instance):
        plugin_instance.cleanup()
        assert plugin_instance._initialized is False

    def test_code_hash_stable(self, plugin_instance):
        h1 = plugin_instance.code_hash
        h2 = plugin_instance.code_hash
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_get_bird_database(self, plugin_instance):
        db = plugin_instance.get_bird_database()
        assert isinstance(db, dict)
        assert len(db) >= 8  # 内置 8 种

    def test_get_ui_config(self, plugin_instance):
        ui = plugin_instance.get_ui_config()
        assert "detection_types" in ui
        assert "runtime_mode" in ui

    def test_postprocess_no_crash_on_empty(self, plugin_instance):
        alarms = plugin_instance.postprocess([], [])
        assert alarms == []


class TestPluginNotInitialized:
    """未初始化时的行为"""

    def test_infer_before_init(self, plugin_dir, make_context, make_roi, sample_frame):
        from plugins.bird_monitoring.plugin import BirdMonitoringPlugin
        from darkbreaker_sdk.interfaces import PluginManifest
        manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
        instance = BirdMonitoringPlugin(manifest, plugin_dir)
        # 未调用 init
        results = instance.infer(sample_frame, [make_roi()], make_context())
        assert len(results) == 1
        assert results[0].label == "error"
        assert results[0].failure_reason == "9000"


class TestStandaloneAssetsExist:
    """standalone UI 资源闭环 — 模板/静态目录/入口必须可用"""

    def test_templates_dir_exists(self, plugin_dir):
        tpl = plugin_dir / "standalone" / "templates" / "bird_monitoring.html"
        assert tpl.exists(), "standalone 模板缺失: bird_monitoring.html"

    def test_static_dir_exists(self, plugin_dir):
        # 目录可以为空（SDK /static 兜底），但不能不存在
        sd = plugin_dir / "standalone" / "static"
        assert sd.exists() and sd.is_dir(), "standalone/static 目录缺失"

    def test_template_marks_simulation_runtime_mode(self, plugin_dir):
        tpl = plugin_dir / "standalone" / "templates" / "bird_monitoring.html"
        text = tpl.read_text(encoding="utf-8")
        # 边界门禁: 仿真页必须显式标记 standalone_simulation
        assert "standalone_simulation" in text
        # 必须呈现 runtime_mode 字段（避免把仿真伪装成真实）
        assert "runtime_mode" in text

    def test_template_exposes_quality_tristate_and_placeholder_download(self, plugin_dir):
        tpl = plugin_dir / "standalone" / "templates" / "bird_monitoring.html"
        text = tpl.read_text(encoding="utf-8")
        # 真实上传侧质量门三态徽章
        assert 'id="q-status"' in text
        assert "soft_fail" in text and "hard_fail" in text
        # 训练占位下载入口
        assert "downloadTrainingPlaceholders" in text
        assert "bird_monitoring_training_placeholders.json" in text

    def test_app_registers_simulator_routes(self):
        """standalone/app.py 必须挂载 /api/simulator/* 路由。"""
        from plugins.bird_monitoring.standalone import app as standalone_app
        from plugins.bird_monitoring.plugin import BirdMonitoringPlugin
        from darkbreaker_sdk.standalone import StandalonePluginRunner

        plugin = BirdMonitoringPlugin.create_standalone()
        runner = StandalonePluginRunner(
            plugin,
            plugin_templates_dir=Path(standalone_app.__file__).parent / "templates",
            plugin_static_dir=Path(standalone_app.__file__).parent / "static",
        )
        standalone_app._register_simulation_routes(runner.app)

        routes = {r.path for r in runner.app.routes if hasattr(r, "path")}
        assert "/api/simulator/scenarios" in routes
        assert "/api/simulator/load" in routes
        assert "/api/simulator/step" in routes
        assert "/api/simulator/status" in routes
        assert "/api/simulator/frame" in routes
        # 生产 detect 必须仍然存在且独立
        assert "/api/detect" in routes


class TestSimulatorIsolation:
    """standalone 仿真器边界契约 — 与生产 plugin/detector 完全隔离"""

    def test_simulator_default_is_no_bird(self):
        from plugins.bird_monitoring.standalone.bird_simulator import (
            BirdMonitoringSimulator,
            RUNTIME_MODE,
        )
        sim = BirdMonitoringSimulator(seed=1)
        payload = sim.step()
        assert payload["runtime_mode"] == RUNTIME_MODE == "standalone_simulation"
        assert payload["detections"]
        assert payload["detections"][0]["label"] == "no_bird"
        assert payload["detections"][0]["runtime_mode"] == "standalone_simulation"

    def test_simulator_marks_every_detection_standalone(self):
        from plugins.bird_monitoring.standalone.bird_simulator import BirdMonitoringSimulator
        sim = BirdMonitoringSimulator(seed=2)
        for sc in ("eagle_critical", "low_confidence_review", "unknown_species",
                   "quality_dark", "magpie_warning"):
            assert sim.load_scenario(sc), f"内置场景缺失: {sc}"
            payload = sim.step()
            assert payload["runtime_mode"] == "standalone_simulation"
            for det in payload["detections"]:
                assert det["runtime_mode"] == "standalone_simulation", det
                assert det["metadata"]["runtime_mode"] == "standalone_simulation"
            for alarm in payload["alarms"]:
                assert alarm["runtime_mode"] == "standalone_simulation"

    def test_simulator_quality_failed_emits_no_bird_bbox(self):
        """质量门场景下不能输出任何 bird detection。"""
        from plugins.bird_monitoring.standalone.bird_simulator import BirdMonitoringSimulator
        sim = BirdMonitoringSimulator()
        assert sim.load_scenario("quality_dark")
        payload = sim.step()
        labels = [d["label"] for d in payload["detections"]]
        assert labels == ["quality_failed"]
        assert payload["quality"]["is_valid"] is False

    def test_simulator_scenarios_cover_label_contract(self):
        """仿真场景必须覆盖 9 个 label 中的关键路径。"""
        from plugins.bird_monitoring.standalone.bird_simulator import BirdMonitoringSimulator
        sim = BirdMonitoringSimulator()
        labels_seen = set()
        for sc in sim.list_scenarios():
            sim.load_scenario(sc["scenario_id"])
            payload = sim.step()
            for det in payload["detections"]:
                labels_seen.add(det["label"])
        # 必须覆盖的核心契约
        required = {"no_bird", "bird_safe", "bird_warning", "bird_danger",
                    "bird_critical", "review_required", "unknown_bird",
                    "quality_failed"}
        missing = required - labels_seen
        assert not missing, f"仿真场景缺失 label: {missing}"

    def test_simulator_does_not_call_plugin_infer(self):
        """边界: 仿真器与 plugin.infer 必须解耦。"""
        from plugins.bird_monitoring.standalone import bird_simulator
        src = Path(bird_simulator.__file__).read_text(encoding="utf-8")
        assert "from plugins.bird_monitoring.plugin" not in src
        assert "from plugins.bird_monitoring.detector" not in src

    def test_production_chain_runtime_mode_is_not_standalone_simulation(
        self, plugin_instance, sample_frame, make_context, make_roi
    ):
        """回归: 生产 plugin.infer 不得使用 standalone_simulation 模式。"""
        results = plugin_instance.infer(sample_frame, [make_roi()], make_context())
        for r in results:
            assert r.metadata["runtime_mode"] != "standalone_simulation"
