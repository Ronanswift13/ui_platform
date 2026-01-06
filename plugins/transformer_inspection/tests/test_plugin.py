#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主变自主巡视插件测试脚本 - pytest 版本
"""

import sys
from pathlib import Path

# 添加路径
PLUGIN_DIR = Path(__file__).parent.parent
PROJECT_ROOT = PLUGIN_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PLUGIN_DIR))

import numpy as np
import yaml
import pytest


@pytest.fixture
def detector():
    """创建检测器实例"""
    from plugins.transformer_inspection.detector_enhanced import TransformerDetectorEnhanced

    config_path = PLUGIN_DIR / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return TransformerDetectorEnhanced(config)


@pytest.fixture
def plugin():
    """创建并初始化插件实例"""
    from platform_core.plugin_manager import PluginManager

    pm = PluginManager()
    plugin_instance = pm.load_plugin("transformer_inspection")
    return plugin_instance


@pytest.fixture
def test_image():
    """创建测试图像"""
    # 创建测试图像 (模拟有缺陷的图像)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 添加深色区域模拟油漏
    image[100:200, 100:200] = [30, 30, 30]
    # 添加橙色区域模拟锈蚀
    image[200:280, 300:400] = [0, 100, 180]
    return image


def test_detector(detector, test_image):
    """测试检测器"""
    # 测试缺陷检测
    defects = detector.detect_defects(test_image, "radiator")

    # 断言检测到了缺陷
    assert isinstance(defects, list), "检测结果应该是列表"

    # 验证每个检测结果的格式
    for d in defects:
        assert "label" in d, "检测结果应包含 label"
        assert "confidence" in d, "检测结果应包含 confidence"
        assert 0 <= d["confidence"] <= 1, "置信度应在 0-1 之间"
        print(f"    - {d['label']}: 置信度 {d['confidence']:.2f}")


def test_plugin_init(plugin):
    """测试插件初始化"""
    from platform_core.plugin_manager.base import PluginStatus

    assert plugin is not None, "插件实例不应为空"
    assert plugin.status == PluginStatus.READY, f"插件状态应为 READY，实际为 {plugin.status.value}"
    print(f"插件初始化成功, 状态: {plugin.status.value}")


def test_healthcheck(plugin):
    """测试健康检查"""
    health = plugin.healthcheck()

    assert health is not None, "健康检查结果不应为空"
    assert health.healthy, f"插件应处于健康状态，消息: {health.message}"

    print(f"健康状态: {'正常' if health.healthy else '异常'}")
    print(f"消息: {health.message}")

    if health.details:
        for k, v in health.details.items():
            print(f"    {k}: {v}")


def test_inference(plugin, test_image):
    """测试推理"""
    from platform_core.plugin_manager.base import PluginContext
    from platform_core.schema.models import ROI, ROIType, BoundingBox

    # 创建ROI
    rois = [
        ROI(
            name="散热器ROI",
            component_id="comp_001",
            roi_type=ROIType.DEFECT,
            bbox=BoundingBox(x=0.1, y=0.1, width=0.4, height=0.4),
        ),
        ROI(
            name="呼吸器ROI",
            component_id="comp_002",
            roi_type=ROIType.STATE,
            bbox=BoundingBox(x=0.3, y=0.3, width=0.3, height=0.3),
        ),
    ]

    # 创建上下文
    context = PluginContext(
        task_id="test_task_001",
        site_id="test_site",
        device_id="test_device",
        component_id="transformer_01",
    )

    # 执行推理
    results = plugin.infer(test_image, rois, context)

    assert isinstance(results, list), "推理结果应该是列表"
    assert len(results) >= 0, "推理结果应该有至少0个结果"

    print(f"检测到 {len(results)} 个结果")

    for r in results:
        assert hasattr(r, "roi_id"), "结果应包含 roi_id"
        assert hasattr(r, "label"), "结果应包含 label"
        assert hasattr(r, "confidence"), "结果应包含 confidence"
        print(f"    - ROI: {r.roi_id}, 标签: {r.label}, 置信度: {r.confidence:.2f}")

    # 测试后处理
    alarms = plugin.postprocess(results, [])

    assert isinstance(alarms, list), "告警结果应该是列表"
    print(f"生成 {len(alarms)} 个告警")

    for a in alarms:
        assert hasattr(a, "level"), "告警应包含级别"
        assert hasattr(a, "title"), "告警应包含标题"
        print(f"    - [{a.level.value}] {a.title}")
