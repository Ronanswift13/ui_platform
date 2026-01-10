# -*- coding: utf-8 -*-
"""
多模态融合系统单元测试
测试融合引擎、数据采集管理器、训练流水线和全景监测API
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import asyncio

# =============================================================================
# 测试融合引擎
# =============================================================================

class TestFusionEngine:
    """融合引擎测试"""
    
    @pytest.fixture
    def setup_fusion_engine(self):
        """设置融合引擎测试环境"""
        # 这里模拟导入，实际测试时需要确保模块可用
        from plugins.multimodal_fusion.fusion_engine_enhanced import (
            EnhancedFusionEngine,
            FusionConfig,
            ModalityData,
            FusionStrategy
        )
        
        config = FusionConfig(
            default_strategy="hybrid",
            enable_bayesian=True,
            enable_attention=True
        )
        engine = EnhancedFusionEngine(config)
        return engine, ModalityData, FusionStrategy
    
    def test_fusion_engine_initialization(self, setup_fusion_engine):
        """测试融合引擎初始化"""
        engine, _, _ = setup_fusion_engine
        assert engine is not None
        assert engine.strategy_manager is not None
    
    def test_early_fusion(self, setup_fusion_engine):
        """测试早期融合"""
        engine, ModalityData, _ = setup_fusion_engine
        
        # 准备测试数据
        modality_data = {
            "visual": ModalityData(
                modality_type="visual",
                features=np.random.randn(128),
                confidence=0.9,
                detections=[{"type": "overheating", "confidence": 0.85}]
            ),
            "thermal": ModalityData(
                modality_type="thermal",
                features=np.random.randn(64),
                confidence=0.85,
                detections=[{"type": "hotspot", "confidence": 0.8}]
            )
        }
        
        # 切换到早期融合策略
        engine.strategy_manager.switch_strategy("EARLY")
        
        # 执行融合
        result = engine.fuse(modality_data)
        
        assert result is not None
        assert result.success
        assert result.overall_status in ["normal", "attention", "warning", "alarm", "critical"]
    
    def test_attention_fusion(self, setup_fusion_engine):
        """测试注意力融合"""
        engine, ModalityData, _ = setup_fusion_engine
        
        modality_data = {
            "acoustic": ModalityData(
                modality_type="acoustic",
                features=np.random.randn(256),
                confidence=0.8,
                detections=[]
            ),
            "gas": ModalityData(
                modality_type="gas",
                features=np.random.randn(32),
                confidence=0.95,
                detections=[{"type": "sf6_leak", "confidence": 0.7}]
            )
        }
        
        engine.strategy_manager.switch_strategy("ATTENTION")
        result = engine.fuse(modality_data)
        
        assert result is not None
        assert "modality_contributions" in dir(result) or hasattr(result, "modality_contributions")
    
    def test_bayesian_inference(self, setup_fusion_engine):
        """测试贝叶斯推断"""
        engine, ModalityData, _ = setup_fusion_engine
        
        # 模拟多种检测结果
        modality_data = {
            "thermal": ModalityData(
                modality_type="thermal",
                features=np.random.randn(64),
                confidence=0.9,
                detections=[
                    {"type": "thermal.hotspot", "confidence": 0.9},
                    {"type": "thermal.overheating", "confidence": 0.85}
                ]
            ),
            "gas": ModalityData(
                modality_type="gas",
                features=np.random.randn(32),
                confidence=0.85,
                detections=[{"type": "gas.H2_high", "confidence": 0.8}]
            )
        }
        
        engine.strategy_manager.switch_strategy("BAYESIAN")
        result = engine.fuse(modality_data)
        
        assert result is not None
        if hasattr(result, 'diagnostic_report'):
            assert result.diagnostic_report is not None
    
    def test_strategy_auto_selection(self, setup_fusion_engine):
        """测试策略自动选择"""
        engine, _, _ = setup_fusion_engine
        
        # 测试不同设备类型的策略选择
        strategy = engine.strategy_manager.auto_select_strategy(
            device_type="transformer",
            available_modalities=["visual", "thermal", "gas", "acoustic"]
        )
        assert strategy is not None
        
        # 测试传感器不完整时的策略选择
        strategy = engine.strategy_manager.auto_select_strategy(
            device_type="transformer",
            available_modalities=["visual"]  # 只有一种模态
        )
        assert strategy is not None


class TestBayesianDecisionNetwork:
    """贝叶斯决策网络测试"""
    
    @pytest.fixture
    def setup_bayesian(self):
        from plugins.multimodal_fusion.fusion_engine_enhanced import BayesianDecisionNetwork
        return BayesianDecisionNetwork()
    
    def test_posterior_computation(self, setup_bayesian):
        """测试后验概率计算"""
        network = setup_bayesian
        
        evidence = ["thermal.hotspot", "gas.H2_high"]
        posterior = network.compute_posterior(evidence)
        
        assert posterior is not None
        assert isinstance(posterior, dict)
        assert sum(posterior.values()) <= 1.01  # 允许小误差
    
    def test_fault_chain_inference(self, setup_bayesian):
        """测试故障链推断"""
        network = setup_bayesian
        
        evidence = ["thermal.overheating", "acoustic.partial_discharge"]
        fault_chain = network.infer_fault_chain(evidence)
        
        assert fault_chain is not None
        assert "root_fault" in fault_chain
        assert "related_faults" in fault_chain


class TestCrossAttention:
    """跨模态注意力测试"""
    
    @pytest.fixture
    def setup_attention(self):
        from plugins.multimodal_fusion.fusion_engine_enhanced import CrossAttentionLayer
        return CrossAttentionLayer(d_model=256, n_heads=8)
    
    def test_attention_computation(self, setup_attention):
        """测试注意力计算"""
        layer = setup_attention
        
        query = np.random.randn(1, 10, 256)  # (batch, seq, dim)
        key = np.random.randn(1, 15, 256)
        value = np.random.randn(1, 15, 256)
        
        output, attention_weights = layer.forward(query, key, value)
        
        assert output.shape == (1, 10, 256)
        assert attention_weights.shape[0] == 1
        assert attention_weights.shape[1] == 8  # n_heads


# =============================================================================
# 测试数据采集管理器
# =============================================================================

class TestDataCollectionManager:
    """数据采集管理器测试"""
    
    @pytest.fixture
    def setup_data_manager(self, tmp_path):
        """设置数据采集管理器测试环境"""
        from platform_core.data_collection_manager import (
            DataCollectionManager,
            ModalityType,
            AcousticCollector,
            GasCollector
        )
        
        manager = DataCollectionManager(data_dir=str(tmp_path / "test_data"))
        manager.register_collector(ModalityType.ACOUSTIC, AcousticCollector())
        manager.register_collector(ModalityType.GAS, GasCollector())
        
        return manager, ModalityType
    
    def test_create_collection_task(self, setup_data_manager):
        """测试创建采集任务"""
        manager, ModalityType = setup_data_manager
        
        task = manager.create_collection_task(
            name="测试任务",
            modalities=[ModalityType.ACOUSTIC, ModalityType.GAS],
            device_ids=["T001", "T002"],
            duration_hours=24,
            interval_seconds=3600
        )
        
        assert task is not None
        assert task.name == "测试任务"
        assert len(task.modalities) == 2
        assert len(task.device_ids) == 2
    
    def test_collect_sample(self, setup_data_manager):
        """测试采集样本"""
        manager, ModalityType = setup_data_manager
        
        # 创建任务
        task = manager.create_collection_task(
            name="采集测试",
            modalities=[ModalityType.ACOUSTIC],
            device_ids=["T001"],
            duration_hours=1,
            interval_seconds=60
        )
        
        # 采集样本
        raw_data = {"waveform": np.random.randn(16000).tolist()}
        sample = manager.collect_sample(
            task_id=task.task_id,
            modality=ModalityType.ACOUSTIC,
            device_id="T001",
            raw_data=raw_data
        )
        
        assert sample is not None
        assert sample.device_id == "T001"
        assert sample.modality == ModalityType.ACOUSTIC
    
    def test_data_alignment(self, setup_data_manager):
        """测试数据对齐"""
        manager, ModalityType = setup_data_manager
        
        # 创建多模态采集任务
        task = manager.create_collection_task(
            name="对齐测试",
            modalities=[ModalityType.ACOUSTIC, ModalityType.GAS],
            device_ids=["T001"],
            duration_hours=1,
            interval_seconds=60
        )
        
        # 采集不同模态的样本（时间接近）
        import time
        base_time = time.time()
        
        sample1 = manager.collect_sample(
            task_id=task.task_id,
            modality=ModalityType.ACOUSTIC,
            device_id="T001",
            raw_data={"waveform": [0.1, 0.2, 0.3]}
        )
        
        sample2 = manager.collect_sample(
            task_id=task.task_id,
            modality=ModalityType.GAS,
            device_id="T001",
            raw_data={"SF6": 99.5, "H2": 0.01}
        )
        
        # 执行对齐
        aligned_count = manager.align_samples(time_window_seconds=60)
        
        assert aligned_count >= 0
    
    def test_label_management(self, setup_data_manager):
        """测试标签管理"""
        manager, ModalityType = setup_data_manager
        
        # 创建任务并采集样本
        task = manager.create_collection_task(
            name="标签测试",
            modalities=[ModalityType.ACOUSTIC],
            device_ids=["T001"],
            duration_hours=1,
            interval_seconds=60
        )
        
        sample = manager.collect_sample(
            task_id=task.task_id,
            modality=ModalityType.ACOUSTIC,
            device_id="T001",
            raw_data={"waveform": [0.1, 0.2, 0.3]}
        )
        
        # 添加标签
        success = manager.add_label(
            sample_id=sample.sample_id,
            fault_type="partial_discharge",
            fault_level="medium",
            annotator="test_user"
        )
        
        assert success
        
        # 验证标签
        updated_sample = manager.samples.get(sample.sample_id)
        assert updated_sample.labels.get("fault_type") == "partial_discharge"
    
    def test_export_dataset(self, setup_data_manager):
        """测试数据集导出"""
        manager, ModalityType = setup_data_manager
        
        # 创建任务并采集样本
        task = manager.create_collection_task(
            name="导出测试",
            modalities=[ModalityType.ACOUSTIC],
            device_ids=["T001"],
            duration_hours=1,
            interval_seconds=60
        )
        
        for i in range(5):
            sample = manager.collect_sample(
                task_id=task.task_id,
                modality=ModalityType.ACOUSTIC,
                device_id="T001",
                raw_data={"waveform": [0.1 * i, 0.2 * i, 0.3 * i]}
            )
            manager.add_label(
                sample_id=sample.sample_id,
                fault_type="normal" if i < 3 else "abnormal",
                fault_level="low",
                annotator="test"
            )
        
        # 导出数据集
        export_info = manager.export_dataset(
            output_name="test_export",
            modalities=[ModalityType.ACOUSTIC]
        )
        
        assert export_info is not None
        assert "export_path" in export_info


# =============================================================================
# 测试训练流水线
# =============================================================================

class TestTrainingPipeline:
    """训练流水线测试"""
    
    @pytest.fixture
    def setup_pipeline(self, tmp_path):
        """设置训练流水线测试环境"""
        from platform_core.training_pipeline import (
            TrainingPipeline,
            TrainingConfig,
            ModelType
        )
        
        pipeline = TrainingPipeline(models_dir=str(tmp_path / "models"))
        return pipeline, TrainingConfig, ModelType
    
    def test_pipeline_initialization(self, setup_pipeline):
        """测试流水线初始化"""
        pipeline, _, _ = setup_pipeline
        assert pipeline is not None
    
    def test_acoustic_trainer(self, setup_pipeline):
        """测试声学异常检测训练器"""
        from platform_core.training_pipeline import AcousticAnomalyTrainer, TrainingConfig, ModelType
        
        config = TrainingConfig(
            model_type=ModelType.ACOUSTIC_ANOMALY,
            model_name="test_acoustic",
            input_dim=128,
            output_dim=5,
            hidden_dim=64,
            num_layers=2,
            batch_size=4,
            max_epochs=2
        )
        
        trainer = AcousticAnomalyTrainer(config)
        trainer.build_model()
        
        assert trainer.model is not None
    
    def test_gas_forecast_trainer(self, setup_pipeline):
        """测试气体预测训练器"""
        from platform_core.training_pipeline import GasForecastTrainer, TrainingConfig, ModelType
        
        config = TrainingConfig(
            model_type=ModelType.GAS_FORECAST,
            model_name="test_gas",
            input_dim=7,
            output_dim=7,
            hidden_dim=32,
            num_layers=1,
            batch_size=4,
            max_epochs=2
        )
        
        trainer = GasForecastTrainer(config)
        trainer.build_model()
        
        assert trainer.model is not None
    
    def test_version_management(self, setup_pipeline):
        """测试版本管理"""
        pipeline, TrainingConfig, ModelType = setup_pipeline
        
        # 模拟训练结果
        config = TrainingConfig(
            model_type=ModelType.ACOUSTIC_ANOMALY,
            model_name="version_test",
            input_dim=128,
            output_dim=5
        )
        
        # 创建多个版本
        for i in range(3):
            pipeline._update_version(
                config.model_name,
                model_path=f"model_v{i}.pt",
                onnx_path=f"model_v{i}.onnx",
                metrics={"accuracy": 0.8 + i * 0.05}
            )
        
        # 检查版本历史
        stats = pipeline.get_training_statistics()
        assert stats is not None


# =============================================================================
# 测试全景监测API
# =============================================================================

class TestPanoramicMonitoringAPI:
    """全景监测API测试"""
    
    @pytest.fixture
    def setup_monitoring(self):
        """设置全景监测测试环境"""
        from apps.panoramic_monitoring_api import (
            CentralStateCache,
            PanoramicMonitoringService,
            DeviceInfo,
            DeviceStatus,
            AlarmInfo,
            AlarmLevel,
            MonitoringData
        )
        
        cache = CentralStateCache()
        service = PanoramicMonitoringService(cache)
        
        return cache, service, DeviceInfo, DeviceStatus, AlarmInfo, AlarmLevel, MonitoringData
    
    def test_device_registration(self, setup_monitoring):
        """测试设备注册"""
        cache, service, DeviceInfo, DeviceStatus, _, _, _ = setup_monitoring
        
        device = DeviceInfo(
            device_id="T001",
            device_name="变压器1号",
            device_type="transformer",
            location={"x": 100, "y": 200, "z": 0},
            modalities=["visual", "thermal", "gas"]
        )
        
        cache.register_device(device)
        
        assert "T001" in cache.devices
        assert cache.devices["T001"].device_name == "变压器1号"
    
    def test_device_status_update(self, setup_monitoring):
        """测试设备状态更新"""
        cache, service, DeviceInfo, DeviceStatus, _, _, _ = setup_monitoring
        
        # 注册设备
        device = DeviceInfo(
            device_id="T002",
            device_name="变压器2号",
            device_type="transformer",
            location={"x": 0, "y": 0, "z": 0},
            modalities=["visual"]
        )
        cache.register_device(device)
        
        # 更新状态
        cache.update_device_status("T002", DeviceStatus.WARNING)
        
        assert cache.devices["T002"].status == DeviceStatus.WARNING
    
    def test_alarm_management(self, setup_monitoring):
        """测试告警管理"""
        cache, service, DeviceInfo, DeviceStatus, AlarmInfo, AlarmLevel, _ = setup_monitoring
        
        # 注册设备
        device = DeviceInfo(
            device_id="T003",
            device_name="变压器3号",
            device_type="transformer",
            location={"x": 0, "y": 0, "z": 0},
            modalities=["visual"]
        )
        cache.register_device(device)
        
        # 添加告警
        alarm = AlarmInfo(
            alarm_id="ALM001",
            device_id="T003",
            level=AlarmLevel.WARNING,
            message="温度异常",
            source="thermal_monitoring",
            timestamp=datetime.now().timestamp(),
            details={"temperature": 85.5}
        )
        cache.add_alarm(alarm)
        
        assert "ALM001" in cache.alarms
        
        # 确认告警
        cache.acknowledge_alarm("ALM001", "test_operator")
        assert cache.alarms["ALM001"].acknowledged
        
        # 解决告警
        cache.resolve_alarm("ALM001", "test_operator")
        assert cache.alarms["ALM001"].resolved
    
    def test_monitoring_data_addition(self, setup_monitoring):
        """测试监测数据添加"""
        cache, service, DeviceInfo, DeviceStatus, _, _, MonitoringData = setup_monitoring
        
        # 注册设备
        device = DeviceInfo(
            device_id="T004",
            device_name="变压器4号",
            device_type="transformer",
            location={"x": 0, "y": 0, "z": 0},
            modalities=["thermal"]
        )
        cache.register_device(device)
        
        # 添加监测数据
        data = MonitoringData(
            device_id="T004",
            modality="thermal",
            timestamp=datetime.now().timestamp(),
            values={"max_temp": 75.5, "avg_temp": 65.0},
            status="normal",
            confidence=0.95,
            detections=[],
            metadata={}
        )
        cache.add_monitoring_data("T004", "thermal", data)
        
        assert "T004" in cache.monitoring_data
        assert "thermal" in cache.monitoring_data["T004"]
        assert len(cache.monitoring_data["T004"]["thermal"]) > 0
    
    def test_device_overview(self, setup_monitoring):
        """测试设备概览"""
        cache, service, DeviceInfo, DeviceStatus, _, _, _ = setup_monitoring
        
        # 注册多个设备
        for i in range(5):
            device = DeviceInfo(
                device_id=f"D00{i}",
                device_name=f"设备{i}",
                device_type="transformer",
                location={"x": i * 10, "y": 0, "z": 0},
                modalities=["visual"]
            )
            cache.register_device(device)
        
        # 更新部分设备状态
        cache.update_device_status("D001", DeviceStatus.WARNING)
        cache.update_device_status("D002", DeviceStatus.ALARM)
        
        # 获取概览
        overview = cache.get_device_overview()
        
        assert overview["total_devices"] == 5
        assert overview["status_counts"][DeviceStatus.WARNING.value] == 1
        assert overview["status_counts"][DeviceStatus.ALARM.value] == 1


class TestAlarmLinkageService:
    """告警联动服务测试"""
    
    @pytest.fixture
    def setup_alarm_service(self):
        """设置告警联动服务测试环境"""
        from apps.panoramic_monitoring_api import AlarmLinkageService, AlarmInfo, AlarmLevel
        
        service = AlarmLinkageService()
        service.configure({
            'http_endpoint': 'http://localhost:8080/alarm',
            'email_recipients': ['admin@test.com'],
            'escalation_rules': [
                {'threshold': 3, 'escalate_to': 'critical'}
            ]
        })
        
        return service, AlarmInfo, AlarmLevel
    
    def test_alarm_handling(self, setup_alarm_service):
        """测试告警处理"""
        service, AlarmInfo, AlarmLevel = setup_alarm_service
        
        alarm = AlarmInfo(
            alarm_id="ALM_TEST_001",
            device_id="T001",
            level=AlarmLevel.WARNING,
            message="测试告警",
            source="test",
            timestamp=datetime.now().timestamp(),
            details={}
        )
        
        # 处理告警（不实际发送通知）
        with patch.object(service, '_send_notifications'):
            service.handle_alarm(alarm)
        
        # 检查告警计数
        assert service.alarm_counts.get("T001", 0) >= 0


# =============================================================================
# 测试集成模块
# =============================================================================

class TestMultimodalIntegration:
    """多模态系统集成测试"""
    
    @pytest.fixture
    def setup_integration(self, tmp_path):
        """设置集成测试环境"""
        from platform_core.multimodal_integration import (
            MultimodalSystemIntegration,
            IntegrationConfig
        )
        
        config = IntegrationConfig(
            data_dir=str(tmp_path / "data"),
            models_dir=str(tmp_path / "models"),
            enable_auto_training=False,  # 测试时禁用
            enable_panoramic_monitoring=True,
            enable_alarm_linkage=False  # 测试时禁用
        )
        
        integration = MultimodalSystemIntegration(config)
        return integration
    
    def test_integration_initialization(self, setup_integration):
        """测试集成初始化"""
        integration = setup_integration
        
        # 尝试初始化（可能因为缺少依赖而失败，但不应抛出异常）
        try:
            result = integration.initialize()
            # 如果成功初始化
            if result:
                assert integration._initialized
                status = integration.get_system_status()
                assert status["status"] == "initialized"
        except ImportError:
            # 如果缺少依赖，跳过测试
            pytest.skip("缺少必要的依赖模块")
    
    def test_plugin_registration(self, setup_integration):
        """测试插件注册"""
        integration = setup_integration
        
        # 创建模拟插件
        mock_plugin = Mock()
        mock_plugin.healthcheck.return_value = True
        
        integration.register_plugin("test_plugin", mock_plugin)
        
        assert "test_plugin" in integration.plugins
    
    def test_event_handling(self, setup_integration):
        """测试事件处理"""
        integration = setup_integration
        
        events_received = []
        
        def test_handler(data):
            events_received.append(data)
        
        integration.register_event_handler('detection', test_handler)
        integration._emit_event('detection', {"test": "data"})
        
        assert len(events_received) == 1
        assert events_received[0]["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_health_check(self, setup_integration):
        """测试健康检查"""
        integration = setup_integration
        
        health = await integration.health_check()
        
        assert health is not None
        assert "timestamp" in health
        assert "overall" in health
        assert "components" in health


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
