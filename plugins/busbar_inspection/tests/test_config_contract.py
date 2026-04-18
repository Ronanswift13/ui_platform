"""配置映射契约测试

测试配置适配器是否正确映射 YAML 配置到统一视图。
遵循 02_algorithm_contract.md 第 2.6 节和 03_test_strategy.md 第 5 节。
"""

import pytest
import yaml
import tempfile
import os

from plugins.busbar_inspection.config_adapter import ConfigAdapter


class TestConfigContract:
    """配置映射契约测试"""

    @pytest.fixture
    def sample_config(self):
        """创建示例配置"""
        return {
            'thresholds': {
                'conf_thr': 0.30,
                'nms_iou': 0.45,
            },
            'tiling': {
                'tile_size': 1024,
                'overlap': 256,
            },
            'quality': {
                'blur_thr': 0.40,
            },
            'zoom': {
                'min_obj_px': 20,
                'target_px': 100,
                'zmin': 1.5,
                'zmax': 10.0,
            },
        }

    @pytest.fixture
    def config_adapter(self, sample_config):
        """创建配置适配器实例"""
        return ConfigAdapter(sample_config)

    def test_confidence_threshold_mapping(self, config_adapter):
        """测试置信度阈值映射"""
        assert config_adapter.get('confidence_threshold') == 0.30

    def test_nms_threshold_mapping(self, config_adapter):
        """测试 NMS 阈值映射"""
        assert config_adapter.get('nms_threshold') == 0.45

    def test_tile_size_mapping(self, config_adapter):
        """测试切片大小映射"""
        assert config_adapter.get('tile_size') == 1024

    def test_tile_overlap_mapping(self, config_adapter):
        """测试切片重叠映射"""
        assert config_adapter.get('tile_overlap') == 256

    def test_clarity_threshold_mapping(self, config_adapter):
        """测试清晰度阈值映射"""
        assert config_adapter.get('clarity_threshold') == 0.40

    def test_min_object_size_px_mapping(self, config_adapter):
        """测试最小目标尺寸映射"""
        assert config_adapter.get('min_object_size_px') == 20

    def test_target_size_px_mapping(self, config_adapter):
        """测试目标尺寸映射"""
        assert config_adapter.get('target_size_px') == 100

    def test_zoom_min_mapping(self, config_adapter):
        """测试最小变焦倍率映射"""
        assert config_adapter.get('zoom_min') == 1.5

    def test_zoom_max_mapping(self, config_adapter):
        """测试最大变焦倍率映射"""
        assert config_adapter.get('zoom_max') == 10.0

    def test_missing_key_uses_default(self):
        """测试缺失键使用默认值"""
        config = {}
        adapter = ConfigAdapter(config)
        assert adapter.get('confidence_threshold') == 0.25
        assert adapter.get('nms_threshold') == 0.50
        assert adapter.get('tile_size') == 1280

    def test_partial_config_uses_defaults(self):
        """测试部分配置使用默认值"""
        config = {
            'thresholds': {
                'conf_thr': 0.35,
            }
        }
        adapter = ConfigAdapter(config)
        assert adapter.get('confidence_threshold') == 0.35
        assert adapter.get('nms_threshold') == 0.50

    def test_unknown_key_raises_error(self, config_adapter):
        """测试未知键抛出异常"""
        with pytest.raises(KeyError, match="Unknown config key"):
            config_adapter.get('unknown_key')

    def test_yaml_value_change_reflected(self):
        """测试 YAML 值变更能被算法层读取（T-009）"""
        config1 = {'thresholds': {'conf_thr': 0.25}}
        adapter1 = ConfigAdapter(config1)
        assert adapter1.get('confidence_threshold') == 0.25

        config2 = {'thresholds': {'conf_thr': 0.50}}
        adapter2 = ConfigAdapter(config2)
        assert adapter2.get('confidence_threshold') == 0.50

    def test_from_yaml_file(self):
        """测试从 YAML 文件加载配置"""
        config_data = {
            'thresholds': {'conf_thr': 0.28},
            'tiling': {'tile_size': 1536},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            adapter = ConfigAdapter.from_yaml(temp_path)
            assert adapter.get('confidence_threshold') == 0.28
            assert adapter.get('tile_size') == 1536
        finally:
            os.unlink(temp_path)

    def test_cache_consistency(self, config_adapter):
        """测试缓存一致性"""
        val1 = config_adapter.get('confidence_threshold')
        val2 = config_adapter.get('confidence_threshold')
        assert val1 == val2
        assert val1 is val2

    def test_custom_default_override(self, config_adapter):
        """测试自定义默认值覆盖"""
        config = {}
        adapter = ConfigAdapter(config)
        assert adapter.get('confidence_threshold', 0.99) == 0.99
