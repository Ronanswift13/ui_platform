"""原因码映射契约测试

测试内部原因码到外部原因码的映射是否正确。
遵循 02_algorithm_contract.md 第 2.4 节和 03_test_strategy.md 第 5 节。
"""

import pytest
from reason_code_mapper import ReasonCodeMapper


class TestReasonCodeContract:
    """原因码映射契约测试"""

    def test_internal_1001_maps_to_103(self):
        """测试内部码 1001 映射到外部码 103（模糊/失焦）"""
        assert ReasonCodeMapper.map_to_external(1001) == 103

    def test_internal_1002_maps_to_101(self):
        """测试内部码 1002 映射到外部码 101（过曝）"""
        assert ReasonCodeMapper.map_to_external(1002) == 101

    def test_internal_1003_maps_to_101(self):
        """测试内部码 1003 映射到外部码 101（低对比）"""
        assert ReasonCodeMapper.map_to_external(1003) == 101

    def test_internal_1004_maps_to_102(self):
        """测试内部码 1004 映射到外部码 102（遮挡）"""
        assert ReasonCodeMapper.map_to_external(1004) == 102

    def test_internal_1005_maps_to_101(self):
        """测试内部码 1005 映射到外部码 101（欠曝）"""
        assert ReasonCodeMapper.map_to_external(1005) == 101

    def test_internal_2001_maps_to_201(self):
        """测试内部码 2001 映射到外部码 201（目标过小）"""
        assert ReasonCodeMapper.map_to_external(2001) == 201

    def test_internal_2002_maps_to_202(self):
        """测试内部码 2002 映射到外部码 202（检测不稳定）"""
        assert ReasonCodeMapper.map_to_external(2002) == 202

    def test_internal_3001_maps_to_301(self):
        """测试内部码 3001 映射到外部码 301（环境干扰-鸟类）"""
        assert ReasonCodeMapper.map_to_external(3001) == 301

    def test_internal_3002_maps_to_301(self):
        """测试内部码 3002 映射到外部码 301（环境干扰-飞虫）"""
        assert ReasonCodeMapper.map_to_external(3002) == 301

    def test_internal_3003_maps_to_301(self):
        """测试内部码 3003 映射到外部码 301（环境干扰-水滴）"""
        assert ReasonCodeMapper.map_to_external(3003) == 301

    def test_all_internal_codes_mapped(self):
        """测试所有内部码都有映射（T-008）"""
        internal_codes = [1001, 1002, 1003, 1004, 1005, 2001, 2002, 3001, 3002, 3003]
        for code in internal_codes:
            external = ReasonCodeMapper.map_to_external(code)
            assert external is not None, f"Internal code {code} has no mapping"

    def test_none_input_returns_none(self):
        """测试 None 输入返回 None"""
        assert ReasonCodeMapper.map_to_external(None) is None

    def test_unknown_code_returns_none(self):
        """测试未知码返回 None"""
        assert ReasonCodeMapper.map_to_external(9999) is None

    def test_external_code_101_description(self):
        """测试外部码 101 的描述"""
        desc = ReasonCodeMapper.get_description(101)
        assert "光照异常" in desc

    def test_external_code_102_description(self):
        """测试外部码 102 的描述"""
        desc = ReasonCodeMapper.get_description(102)
        assert "遮挡" in desc or "不可见" in desc

    def test_external_code_103_description(self):
        """测试外部码 103 的描述"""
        desc = ReasonCodeMapper.get_description(103)
        assert "模糊" in desc or "失焦" in desc

    def test_external_code_201_description(self):
        """测试外部码 201 的描述"""
        desc = ReasonCodeMapper.get_description(201)
        assert "目标过小" in desc or "变焦" in desc

    def test_external_code_301_description(self):
        """测试外部码 301 的描述"""
        desc = ReasonCodeMapper.get_description(301)
        assert "环境干扰" in desc

    def test_none_code_description(self):
        """测试 None 码的描述"""
        desc = ReasonCodeMapper.get_description(None)
        assert desc == "未知原因"

    def test_unknown_code_description(self):
        """测试未知码的描述"""
        desc = ReasonCodeMapper.get_description(9999)
        assert desc == "未知原因"

    def test_map_and_describe_1001(self):
        """测试映射并描述内部码 1001"""
        external, desc = ReasonCodeMapper.map_and_describe(1001)
        assert external == 103
        assert "模糊" in desc or "失焦" in desc

    def test_map_and_describe_1002(self):
        """测试映射并描述内部码 1002"""
        external, desc = ReasonCodeMapper.map_and_describe(1002)
        assert external == 101
        assert "光照异常" in desc

    def test_map_and_describe_none(self):
        """测试映射并描述 None"""
        external, desc = ReasonCodeMapper.map_and_describe(None)
        assert external is None
        assert desc == "未知原因"

    def test_map_and_describe_unknown(self):
        """测试映射并描述未知码"""
        external, desc = ReasonCodeMapper.map_and_describe(9999)
        assert external is None
        assert desc == "未知原因"
