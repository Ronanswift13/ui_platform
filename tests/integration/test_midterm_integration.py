"""
中期迭代集成测试
输变电激光星芒破夜绘明监测平台

测试内容:
1. 点云处理模块
2. 时序分析模块
3. 鸟类风险预测模块
4. MLOps流水线模块
5. 微服务架构模块
6. 系统监控模块

版本: 1.0.0
"""

import sys
from pathlib import Path
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


def test_point_cloud_processor():
    """测试点云处理模块"""
    result = TestResult("点云处理模块")
    start = time.time()

    try:
        from ai_models.deep_learning.point_cloud_processor import (
            PointCloud,
            SensorType,
            CameraIntrinsics,
            DepthToPointCloud,
            PointCloudProcessorConfig,
        )

        # 创建相机内参
        intrinsics = CameraIntrinsics(
            fx=525.0,
            fy=525.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )

        # 测试深度图转点云
        converter = DepthToPointCloud(intrinsics)

        # 创建模拟深度图
        depth_image = np.ones((480, 640), dtype=np.float32) * 2.0  # 2米深度
        point_cloud = converter.convert(depth_image)

        assert point_cloud is not None, "点云转换失败"
        assert point_cloud.size > 0, "点云为空"

        # 测试点云下采样
        downsampled = point_cloud.downsample(voxel_size=0.1)
        assert downsampled is not None, "下采样失败"

        # 测试点云过滤
        filtered = point_cloud.filter_by_range(0.5, 5.0)
        assert filtered is not None, "过滤失败"

        # 测试配置
        config = PointCloudProcessorConfig(
            voxel_size=0.05,
            depth_min=0.1,
            depth_max=10.0,
        )
        assert config is not None, "配置创建失败"

        result.passed = True
        result.message = "点云处理模块测试通过"
        result.details = {
            "original_points": point_cloud.size,
            "downsampled_points": downsampled.size,
            "filtered_points": filtered.size,
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"
        import traceback
        result.details = {"traceback": traceback.format_exc()}

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_temporal_analyzer():
    """测试时序分析模块"""
    result = TestResult("时序分析模块")
    start = time.time()

    try:
        from ai_models.deep_learning.temporal_analyzer import (
            SwitchState,
            TransitionType,
            AnomalyType,
            TemporalAnalyzerConfig,
            FrameObservation,
            TemporalDecision,
            TemporalAnalyzer,
        )

        # 创建配置
        config = TemporalAnalyzerConfig(
            window_size=10,
            confidence_threshold=0.7,
        )

        # 创建分析器
        analyzer = TemporalAnalyzer(config)

        # 添加模拟观测
        for i in range(10):
            obs = FrameObservation(
                timestamp=float(i * 0.04),  # 25 FPS
                state=SwitchState.OPEN,
                confidence=0.85,
                angle=-55.0,
                features=np.random.rand(4).astype(np.float32),
            )
            analyzer.add_observation(obs)

        # 执行分析
        decision = analyzer.analyze()

        assert decision is not None, "分析结果为空"
        assert isinstance(decision, TemporalDecision), "返回类型不正确"
        assert decision.final_state is not None, "状态为空"

        result.passed = True
        result.message = "时序分析模块测试通过"
        result.details = {
            "final_state": decision.final_state.value,
            "confidence": decision.confidence,
            "transition_type": decision.transition_type.value,
            "anomaly_type": decision.anomaly_type.value,
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"
        import traceback
        result.details = {"traceback": traceback.format_exc()}

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_bird_risk_predictor():
    """测试鸟类风险预测模块"""
    result = TestResult("鸟类风险预测模块")
    start = time.time()

    try:
        from ai_models.deep_learning.bird_risk_predictor import (
            BirdSpecies,
            BirdBehavior,
            RiskLevel,
            DeterrenceAction,
            BirdRiskPredictorConfig,
            BirdDetection,
            BirdTrack,
            RiskAssessment,
            BirdClassifier,
            TrajectoryPredictor,
            RiskAssessor,
            BirdRiskPredictor,
        )

        # 创建配置
        config = BirdRiskPredictorConfig(
            max_tracks=100,
            track_max_age=5.0,
        )

        # 创建预测器
        predictor = BirdRiskPredictor(config)

        # 测试分类器配置
        from ai_models.deep_learning.bird_risk_predictor import BirdClassifierConfig
        classifier_config = BirdClassifierConfig()
        classifier = BirdClassifier(classifier_config)
        assert classifier is not None, "分类器创建失败"

        # 测试轨迹预测器
        trajectory_predictor = TrajectoryPredictor()
        assert trajectory_predictor is not None, "轨迹预测器创建失败"

        # 测试风险评估器
        assessor = RiskAssessor()
        assert assessor is not None, "风险评估器创建失败"

        # 创建模拟检测
        detection = BirdDetection(
            track_id=1,
            bbox=(0.1, 0.1, 0.1, 0.1),
            confidence=0.9,
            timestamp=time.time(),
        )

        # 创建轨迹
        track = BirdTrack(track_id=1)
        track.add_detection(detection)

        # 模拟图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # 处理检测 (以字典形式)
        det_dict = {
            "track_id": 1,
            "bbox": (0.1, 0.1, 0.1, 0.1),
            "confidence": 0.9,
        }

        tracks, risk = predictor.process_detections(
            [det_dict],
            test_image,
            time.time()
        )

        assert risk is not None, "风险评估为空"
        assert risk.risk_level is not None, "风险级别为空"

        result.passed = True
        result.message = "鸟类风险预测模块测试通过"
        result.details = {
            "track_count": len(tracks),
            "risk_level": risk.risk_level.value,
            "risk_score": risk.risk_score,
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"
        import traceback
        result.details = {"traceback": traceback.format_exc()}

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_mlops_pipeline():
    """测试MLOps流水线模块"""
    result = TestResult("MLOps流水线模块")
    start = time.time()

    try:
        from platform_core.mlops_pipeline import (
            ModelStatus,
            SamplingStrategy,
            ModelVersion,
            TrainingJob,
            DatasetVersion,
            ModelRegistry,
            TrainingScheduler,
            ActiveLearningEngine,
            ActiveLearningConfig,
            DeploymentManager,
            MLOpsPipeline,
        )

        # 创建MLOps流水线
        pipeline = MLOpsPipeline()

        # 测试模型注册 - 使用register方法
        version = pipeline.registry.register(
            model_id="yolov8_vit",
            version="v1.0.0",
            name="YOLOv8-ViT检测器",
            model_path="/tmp/test_model.pt",  # 测试路径
            metrics={"mAP": 0.85, "precision": 0.9},
        )

        assert version is not None, "模型版本注册失败"
        assert version.model_id == "yolov8_vit", "模型ID不正确"

        # 测试训练调度 - 使用submit方法
        job = pipeline.scheduler.submit(
            model_id="yolov8_vit",
            config={"epochs": 10, "batch_size": 32},
            dataset_path="/tmp/test_dataset",
        )

        assert job is not None, "训练任务创建失败"
        assert job.job_id is not None, "任务ID为空"

        # 测试主动学习
        al_config = ActiveLearningConfig(
            strategy=SamplingStrategy.UNCERTAINTY,
            sample_size=100,
        )
        al_engine = ActiveLearningEngine(al_config)

        # 添加样本
        al_engine.add_sample(
            sample_id="sample_1",
            features=np.random.rand(512),
            predictions={"class_0": 0.6, "class_1": 0.4},
        )

        stats = al_engine.get_statistics()
        assert stats["pending_samples"] > 0, "样本添加失败"

        # 测试部署管理器 - 需要传入registry参数
        deployment_manager = DeploymentManager(pipeline.registry)
        assert deployment_manager is not None, "部署管理器创建失败"

        result.passed = True
        result.message = "MLOps流水线模块测试通过"
        result.details = {
            "version_id": version.version,
            "job_id": job.job_id,
            "al_pending_samples": stats["pending_samples"],
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"
        import traceback
        result.details = {"traceback": traceback.format_exc()}

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_microservice():
    """测试微服务架构模块"""
    result = TestResult("微服务架构模块")
    start = time.time()

    try:
        from platform_core.microservice import (
            ServiceStatus,
            TaskStatus,
            TaskPriority,
            MessageType,
            LoadBalanceStrategy,
            ServiceInfo,
            RouteConfig,
            AsyncTask,
            WebSocketMessage,
            ServiceRegistry,
            APIGateway,
            TaskQueue,
            WebSocketManager,
            MicroserviceOrchestrator,
        )

        # 创建协调器
        orchestrator = MicroserviceOrchestrator({
            "max_workers": 2,
            "heartbeat_interval": 5.0,
        })

        # 启动服务
        orchestrator.start()

        # 注册服务
        service = ServiceInfo(
            service_id="test_service_1",
            name="test_plugin",
            version="1.0.0",
            host="localhost",
            port=8001,
        )
        orchestrator.registry.register(service)

        # 验证服务注册
        services = orchestrator.registry.get_services("test_plugin")
        assert len(services) > 0, "服务注册失败"

        # 添加路由
        route = RouteConfig(
            path_pattern="/api/test/*",
            service_name="test_plugin",
            methods=["GET", "POST"],
        )
        orchestrator.gateway.add_route(route)

        # 注册任务处理器
        def test_handler(data):
            return {"processed": True, "data": data}

        orchestrator.register_task_handler("test_task", test_handler)

        # 提交任务
        task_id = orchestrator.submit_task(
            name="测试任务",
            func_name="test_task",
            kwargs={"input": "test"},
            priority=TaskPriority.NORMAL,
        )

        # 等待任务完成
        time.sleep(2)

        # 获取任务状态
        task = orchestrator.task_queue.get_task(task_id)

        # 获取系统状态
        status = orchestrator.get_system_status()

        # 停止服务
        orchestrator.stop()

        result.passed = True
        result.message = "微服务架构模块测试通过"
        result.details = {
            "task_id": task_id,
            "task_status": task.status.value if task else "unknown",
            "registered_services": len(services),
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"
        import traceback
        result.details = {"traceback": traceback.format_exc()}

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_monitoring():
    """测试系统监控模块"""
    result = TestResult("系统监控模块")
    start = time.time()

    try:
        from platform_core.monitoring import (
            MetricType,
            LogLevel,
            AlertSeverity,
            HealthStatus,
            MetricsCollector,
            StructuredLogger,
            Tracer,
            AlertManager,
            HealthCheckManager,
            MonitoringSystem,
            AlertRule,
        )

        # 初始化监控系统
        monitoring = MonitoringSystem("test_service", {
            "log_level": "info",
            "log_json": False,
        })

        # 启动监控
        monitoring.start()

        # 测试指标收集
        monitoring.metrics.counter_inc("test_counter", 1)
        monitoring.metrics.gauge_set("test_gauge", 42.0)
        monitoring.metrics.histogram_observe("test_histogram", 0.5)

        # 验证指标
        counter_value = monitoring.metrics.counter_get("test_counter")
        gauge_value = monitoring.metrics.gauge_get("test_gauge")

        assert counter_value == 1.0, f"计数器值不正确: {counter_value}"
        assert gauge_value == 42.0, f"仪表值不正确: {gauge_value}"

        # 测试请求记录
        monitoring.record_request("/api/test", "GET", 200, 0.1)
        monitoring.record_inference("yolov8", 0.05, True)

        # 测试追踪
        with monitoring.tracer.trace("test_operation"):
            time.sleep(0.01)

        # 测试健康检查
        health_status = monitoring.get_health_status()
        assert "status" in health_status, "健康状态格式不正确"

        # 获取Prometheus指标
        prometheus_metrics = monitoring.get_prometheus_metrics()
        assert len(prometheus_metrics) > 0, "Prometheus指标为空"

        # 停止监控
        monitoring.stop()

        result.passed = True
        result.message = "系统监控模块测试通过"
        result.details = {
            "health_status": health_status.get("status"),
            "prometheus_lines": len(prometheus_metrics.split("\n")),
            "counter_value": counter_value,
            "gauge_value": gauge_value,
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"
        import traceback
        result.details = {"traceback": traceback.format_exc()}

    result.duration_ms = (time.time() - start) * 1000
    return result


def test_module_integration():
    """测试模块间集成"""
    result = TestResult("模块间集成")
    start = time.time()

    try:
        # 验证所有模块可以同时导入
        from ai_models.deep_learning.point_cloud_processor import (
            PointCloud,
            PointCloudProcessorConfig,
        )
        from ai_models.deep_learning.temporal_analyzer import (
            TemporalAnalyzer,
            TemporalAnalyzerConfig,
        )
        from ai_models.deep_learning.bird_risk_predictor import (
            BirdRiskPredictor,
            BirdRiskPredictorConfig,
        )
        from platform_core.mlops_pipeline import MLOpsPipeline
        from platform_core.microservice import MicroserviceOrchestrator
        from platform_core.monitoring import MonitoringSystem

        # 验证可以创建实例
        pc_config = PointCloudProcessorConfig()
        ta_config = TemporalAnalyzerConfig()
        br_config = BirdRiskPredictorConfig()

        temporal_analyzer = TemporalAnalyzer(ta_config)
        bird_predictor = BirdRiskPredictor(br_config)
        mlops = MLOpsPipeline()
        monitoring = MonitoringSystem("integration_test")

        # 验证模块间通信
        # 记录MLOps指标到监控系统
        monitoring.metrics.gauge_set(
            "mlops_models_registered",
            float(len(mlops.registry.list_models())),
        )

        result.passed = True
        result.message = "模块间集成测试通过"
        result.details = {
            "modules_imported": 6,
            "instances_created": 5,
        }
    except Exception as e:
        result.passed = False
        result.message = f"测试失败: {e}"
        import traceback
        result.details = {"traceback": traceback.format_exc()}

    result.duration_ms = (time.time() - start) * 1000
    return result


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("中期迭代集成测试")
    print("=" * 70)
    print(f"开始时间: {datetime.now().isoformat()}")
    print()

    # 定义测试函数列表
    tests = [
        test_point_cloud_processor,
        test_temporal_analyzer,
        test_bird_risk_predictor,
        test_mlops_pipeline,
        test_microservice,
        test_monitoring,
        test_module_integration,
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
                if key != "traceback":
                    print(f"  - {key}: {value}")
                else:
                    print(f"  - {key}:")
                    for line in str(value).split("\n")[:10]:
                        print(f"      {line}")

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
        print("\n✓ 所有中期迭代测试通过！")
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
