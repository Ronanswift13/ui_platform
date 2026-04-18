"""
L1 集成测试 - 插件全链路测试
(合约 7.8, 7.9)
"""
import pytest
import numpy as np
from pathlib import Path
from darkbreaker_sdk.interfaces import PluginContext, PluginManifest
from darkbreaker_sdk.schemas import ROI, BoundingBox
from plugins.meter_reading.plugin import MeterReadingPlugin


class TestPluginIntegration:
    """插件集成测试"""

    @pytest.fixture
    def plugin(self, default_config):
        plugin_dir = Path(__file__).parent.parent
        manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
        plugin = MeterReadingPlugin(manifest, plugin_dir)
        plugin.init(default_config)
        return plugin

    @pytest.fixture
    def context(self):
        return PluginContext(
            task_id="test_task",
            site_id="test_site",
            device_id="test_device",
            component_id="test_component"
        )

    @pytest.fixture
    def test_roi(self):
        from darkbreaker_sdk.schemas import ROIType
        return ROI(
            id="roi_1",
            name="pressure_gauge",
            component_id="comp1",
            roi_type=ROIType.METER,
            bbox=BoundingBox(x=0.2, y=0.2, width=0.4, height=0.4)
        )

    def test_plugin_init_success(self, plugin):
        """plugin.init() -> True"""
        assert plugin._initialized is True
        assert plugin._detector is not None

    def test_plugin_infer_returns_list(self, plugin, sample_frame, test_roi, context):
        """plugin.infer() -> list[RecognitionResult]"""
        results = plugin.infer(sample_frame, [test_roi], context)
        assert isinstance(results, list)

    def test_plugin_infer_result_has_metadata(self, plugin, sample_frame, test_roi, context):
        """推理结果metadata包含必填字段"""
        results = plugin.infer(sample_frame, [test_roi], context)
        if results:
            meta = results[0].metadata
            for key in ["meter_type", "reading_status", "pipeline_stage", "fallback_level", "timestamp_ms"]:
                assert key in meta, f"缺少metadata字段: {key}"

    def test_plugin_postprocess(self, plugin, sample_frame, test_roi, context):
        """plugin.postprocess() -> list[Alarm]"""
        results = plugin.infer(sample_frame, [test_roi], context)
        alarms = plugin.postprocess(results, [])
        assert isinstance(alarms, list)

    def test_plugin_healthcheck_ready(self, plugin):
        """plugin.healthcheck() -> healthy=True"""
        health = plugin.healthcheck()
        assert health.healthy is True
        assert "status" in health.details

    def test_plugin_multiple_rois(self, plugin, sample_frame, context):
        """多个ROI处理"""
        from darkbreaker_sdk.schemas import ROIType
        rois = [
            ROI(id="roi_1", name="roi1", component_id="comp1",
                roi_type=ROIType.METER,
                bbox=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.3)),
            ROI(id="roi_2", name="roi2", component_id="comp2",
                roi_type=ROIType.METER,
                bbox=BoundingBox(x=0.5, y=0.5, width=0.3, height=0.3)),
        ]
        results = plugin.infer(sample_frame, rois, context)
        assert isinstance(results, list)

    def test_plugin_uninit_returns_empty(self, default_config):
        """未初始化时infer返回空列表"""
        plugin_dir = Path(__file__).parent.parent
        manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
        plugin = MeterReadingPlugin(manifest, plugin_dir)
        context = PluginContext(task_id="t", site_id="s", device_id="d", component_id="c")
        results = plugin.infer(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [],
            context,
        )
        assert results == []

    def test_plugin_healthcheck_uninit(self):
        """未初始化时healthcheck返回unhealthy"""
        plugin_dir = Path(__file__).parent.parent
        manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
        plugin = MeterReadingPlugin(manifest, plugin_dir)
        health = plugin.healthcheck()
        assert health.healthy is False
