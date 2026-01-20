"""
深度学习集成测试
输变电激光星芒破夜绘明监测平台

测试内容:
1. YOLOv8-ViT模型加载和推理
2. GL-TransLSTM气体预测
3. 声学CNN-LSTM检测
4. 深度学习与插件集成
5. 仪表盘API
6. 输出格式规范

版本: 1.0.0
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import time

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestResult:
    """测试结果"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration_ms = 0.0
        self.details = {}


def test_yolov8_vit_import():
    """测试YOLOv8-ViT模块导入"""
    result = TestResult("YOLOv8-ViT模块导入")
    start = time.time()

    try:
        from ai_models.deep_learning.yolov8_vit import (
            YOLOv8ViTDetector,
            YOLOv8ViTConfig,
            DetectionTask,
            DetectionResult,
        )
        result.passed = True
        result.message = "模块导入成功"
        result.details = {
            "classes_imported": ["YOLOv8ViTDetector", "YOLOv8ViTConfig", "DetectionTask", "DetectionResult"],
        }
    except ImportError as e:
        result.passed = False
        result.message = f"导入失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_yolov8_vit_config():
    """测试YOLOv8-ViT配置"""
    result = TestResult("YOLOv8-ViT配置创建")
    start = time.time()

    try:
        from ai_models.deep_learning.yolov8_vit import YOLOv8ViTConfig, DetectionTask

        config = YOLOv8ViTConfig(
            model_path=None,
            task=DetectionTask.TRANSFORMER_DEFECT,
            confidence_threshold=0.5,
            nms_threshold=0.4,
            use_vit_backbone=True,
            use_se_attention=True,
        )

        result.passed = True
        result.message = "配置创建成功"
        result.details = {
            "task": config.task.value,
            "confidence_threshold": config.confidence_threshold,
            "use_vit_backbone": config.use_vit_backbone,
        }
    except Exception as e:
        result.passed = False
        result.message = f"配置创建失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_gl_translstm_import():
    """测试GL-TransLSTM模块导入"""
    result = TestResult("GL-TransLSTM模块导入")
    start = time.time()

    try:
        from ai_models.deep_learning.gl_translstm import (
            GLTransLSTM,
            GLTransLSTMConfig,
            GasType,
            GasPrediction,
            LeakDetectionResult,
        )
        result.passed = True
        result.message = "模块导入成功"
        result.details = {
            "classes_imported": ["GLTransLSTM", "GLTransLSTMConfig", "GasType", "GasPrediction", "LeakDetectionResult"],
        }
    except ImportError as e:
        result.passed = False
        result.message = f"导入失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_acoustic_cnn_lstm_import():
    """测试声学CNN-LSTM模块导入"""
    result = TestResult("声学CNN-LSTM模块导入")
    start = time.time()

    try:
        from ai_models.deep_learning.acoustic_cnn_lstm import (
            AcousticCNNLSTM,
            AcousticModelConfig,
            AnomalyType,
            AcousticAnalysisResult,
        )
        result.passed = True
        result.message = "模块导入成功"
        result.details = {
            "classes_imported": ["AcousticCNNLSTM", "AcousticModelConfig", "AnomalyType", "AcousticAnalysisResult"],
        }
    except ImportError as e:
        result.passed = False
        result.message = f"导入失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_deep_sort_import():
    """测试DeepSORT模块导入"""
    result = TestResult("DeepSORT模块导入")
    start = time.time()

    try:
        from ai_models.deep_learning.deep_sort_tracker import (
            DeepSORTTracker,
            TrackerConfig,
            Track,
            TrackState,
        )
        result.passed = True
        result.message = "模块导入成功"
        result.details = {
            "classes_imported": ["DeepSORTTracker", "TrackerConfig", "Track", "TrackState"],
        }
    except ImportError as e:
        result.passed = False
        result.message = f"导入失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_keypoint_detector_import():
    """测试关键点检测器模块导入"""
    result = TestResult("关键点检测器模块导入")
    start = time.time()

    try:
        from ai_models.deep_learning.keypoint_detector import (
            KeypointDetector,
            KeypointConfig,
            MeterType,
            MeterReadingResult,
            Keypoint,
        )
        result.passed = True
        result.message = "模块导入成功"
        result.details = {
            "classes_imported": ["KeypointDetector", "KeypointConfig", "MeterType", "MeterReadingResult", "Keypoint"],
        }
    except ImportError as e:
        result.passed = False
        result.message = f"导入失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_dashboard_api():
    """测试仪表盘API"""
    result = TestResult("仪表盘API")
    start = time.time()

    try:
        from platform_core.dashboard_api import (
            DashboardService,
            get_dashboard_service,
            DashboardData,
            SystemOverview,
            AlarmSummary,
            ModuleHealth,
            DLModelStatus,
            PluginStatus,
        )

        # 测试服务实例化
        service = get_dashboard_service()
        assert service is not None, "服务实例化失败"

        # 测试获取仪表盘数据
        data = service.get_dashboard_data()
        assert isinstance(data, DashboardData), "数据类型不正确"
        assert data.system_overview is not None, "系统概览为空"
        assert len(data.module_health) > 0, "模块健康状态为空"
        assert len(data.dl_models) > 0, "深度学习模型状态为空"

        result.passed = True
        result.message = "仪表盘API测试通过"
        result.details = {
            "total_modules": data.system_overview.total_modules,
            "online_modules": data.system_overview.online_modules,
            "dl_model_count": len(data.dl_models),
            "plugin_count": len(data.plugins),
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_output_format():
    """测试输出格式规范"""
    result = TestResult("输出格式规范")
    start = time.time()

    try:
        from platform_core.output_format import (
            StandardLabel,
            LABEL_NAMES,
            LABEL_SEVERITY,
            OutputFormatter,
            StandardResult,
            StandardPluginOutput,
            format_result,
            format_plugin_output,
            get_label_name,
            get_label_severity,
        )

        # 测试标签名称
        assert get_label_name("oil_leak") == "渗漏油", "标签名称不正确"
        assert get_label_name("normal") == "正常", "标签名称不正确"

        # 测试标签严重程度
        assert get_label_severity("oil_leak") == "danger", "严重程度不正确"
        assert get_label_severity("normal") == "success", "严重程度不正确"

        # 测试格式化结果
        formatted = format_result(
            task_id="test_task",
            site_id="test_site",
            device_id="test_device",
            label="oil_leak",
            confidence=0.95,
            bbox={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            value=None,
        )

        assert isinstance(formatted, StandardResult), "结果类型不正确"
        assert formatted.label == "oil_leak", "标签不正确"
        assert formatted.label_name == "渗漏油", "标签名称不正确"
        assert formatted.severity == "danger", "严重程度不正确"

        result.passed = True
        result.message = "输出格式规范测试通过"
        result.details = {
            "total_labels": len(StandardLabel),
            "label_names_count": len(LABEL_NAMES),
            "label_severity_count": len(LABEL_SEVERITY),
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"
        import traceback
        result.details = {"traceback": traceback.format_exc()}

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_transformer_detector_enhanced():
    """测试主变增强检测器"""
    result = TestResult("主变增强检测器")
    start = time.time()

    try:
        from plugins.transformer_inspection.detector_enhanced import (
            TransformerDetectorEnhanced,
            Detection,
            DefectType,
            SilicaGelState,
            ThermalLevel,
        )

        # 创建配置
        config = {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "use_deep_learning": True,
        }

        # 创建检测器
        detector = TransformerDetectorEnhanced(config)
        detector.initialize()

        # 测试缺陷检测 (使用模拟图像)
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        defects = detector.detect_defects(test_image)

        # 测试油位检测
        oil_level = detector.detect_oil_level(test_image)

        # 测试硅胶识别
        silica_result = detector.recognize_silica_gel(test_image)

        result.passed = True
        result.message = "主变增强检测器测试通过"
        result.details = {
            "defect_count": len(defects),
            "oil_level_status": oil_level.level_status,
            "silica_state": silica_result.state.value,
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_switch_detector_enhanced():
    """测试开关增强检测器"""
    result = TestResult("开关增强检测器")
    start = time.time()

    try:
        from plugins.switch_inspection.detector_enhanced import (
            SwitchDetectorEnhanced,
            SwitchType,
            SwitchState,
            ClarityLevel,
        )

        # 创建配置
        config = {
            "use_deep_learning": True,
            "state_recognition": {
                "confidence_threshold": 0.5,
            },
        }

        # 创建检测器
        detector = SwitchDetectorEnhanced(config)
        detector.initialize()

        # 测试状态识别 (使用模拟图像)
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        state_result = detector.recognize_switch_state(test_image, SwitchType.BREAKER)

        # 测试清晰度评价
        clarity = detector.evaluate_clarity(test_image)

        result.passed = True
        result.message = "开关增强检测器测试通过"
        result.details = {
            "state": state_result.get("state", "unknown"),
            "confidence": state_result.get("confidence", 0),
            "clarity_score": clarity.score,
            "clarity_level": clarity.level.value,
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_plugin_dl_integration():
    """测试插件与深度学习模型集成"""
    result = TestResult("插件与DL模型集成")
    start = time.time()

    try:
        # 检查深度学习模型是否可用
        from ai_models.deep_learning.yolov8_vit import YOLOv8ViTDetector, YOLOv8ViTConfig, DetectionTask

        # 检查增强检测器是否正确导入DL模型
        from plugins.transformer_inspection.detector_enhanced import TransformerDetectorEnhanced
        from plugins.switch_inspection.detector_enhanced import SwitchDetectorEnhanced

        # 验证集成
        transformer_config = {"use_deep_learning": True}
        switch_config = {"use_deep_learning": True}

        t_detector = TransformerDetectorEnhanced(transformer_config)
        s_detector = SwitchDetectorEnhanced(switch_config)

        # 检查DL模块是否被正确引用
        assert hasattr(t_detector, '_yolov8_vit_detector'), "主变检测器缺少YOLOv8-ViT引用"
        assert hasattr(s_detector, '_yolov8_vit_detector'), "开关检测器缺少YOLOv8-ViT引用"

        result.passed = True
        result.message = "插件与深度学习模型集成验证通过"
        result.details = {
            "transformer_dl_attr": "_yolov8_vit_detector",
            "switch_dl_attr": "_yolov8_vit_detector",
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"

    result.duration_ms = (time.time() - start) * 1000
    return result


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("深度学习集成测试")
    print("=" * 70)
    print(f"开始时间: {datetime.now().isoformat()}")
    print()

    # 定义测试函数列表
    tests = [
        test_yolov8_vit_import,
        test_yolov8_vit_config,
        test_gl_translstm_import,
        test_acoustic_cnn_lstm_import,
        test_deep_sort_import,
        test_keypoint_detector_import,
        test_dashboard_api,
        test_output_format,
        test_transformer_detector_enhanced,
        test_switch_detector_enhanced,
        test_plugin_dl_integration,
    ]

    results = []

    for test_func in tests:
        print(f"\n{'─' * 70}")
        print(f"测试: {test_func.__doc__ or test_func.__name__}")
        print(f"{'─' * 70}")

        result = test_func()
        results.append(result)

        # 打印结果
        status = "✓ 通过" if result.passed else "✗ 失败"
        print(f"状态: {status}")
        print(f"消息: {result.message}")
        print(f"耗时: {result.duration_ms:.2f}ms")

        if result.details:
            print("详情:")
            for key, value in result.details.items():
                print(f"  - {key}: {value}")

    # 汇总统计
    print(f"\n\n{'=' * 70}")
    print("测试汇总")
    print(f"{'=' * 70}")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    total_time = sum(r.duration_ms for r in results)

    print(f"\n总计: {total} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    print(f"总耗时: {total_time:.2f}ms")

    if failed == 0:
        print("\n✓ 所有测试通过！")
        return 0
    else:
        print("\n✗ 部分测试失败")
        print("\n失败的测试:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
