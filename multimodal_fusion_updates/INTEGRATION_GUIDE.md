# 多模态融合系统集成指南

## 📋 更新内容概述

本次更新根据"下一步方案.docx"文档实现了以下功能模块：

### 1. 多模态融合算法完善
**文件**: `plugins/multimodal_fusion/fusion_engine_enhanced.py`

- ✅ 基于Transformer的cross-attention模块
- ✅ 贝叶斯决策网络（故障链推断）
- ✅ 动态策略切换接口（early/late/attention/hybrid/bayesian）
- ✅ 自适应模态权重调整

### 2. 数据采集管理
**文件**: `platform_core/data_collection_manager.py`

- ✅ 多模态数据采集任务管理
- ✅ 数据对齐（时间窗口匹配）
- ✅ 故障日志对接与自动标注
- ✅ 训练数据集导出（train/val/test划分）

### 3. 模型训练流水线
**文件**: `platform_core/training_pipeline.py`

- ✅ Transformer声学异常检测模型
- ✅ LSTM气体浓度预测模型
- ✅ ONNX格式导出
- ✅ 版本控制与回滚
- ✅ 自动化重训练调度

### 4. 全景监测仪表盘API
**文件**: `apps/panoramic_monitoring_api.py`

- ✅ 设备状态中央缓存
- ✅ 实时WebSocket推送
- ✅ 告警联动服务
- ✅ 趋势数据API
- ✅ RESTful接口

### 5. 插件兼容性修复
**文件**: `platform_core/plugin_compatibility_fix.py`

- ✅ 统一插件加载机制
- ✅ 错误处理与恢复
- ✅ 多种实例化方式支持

---

## 🔧 集成步骤

### 步骤1：复制新文件到项目目录

```bash
# 复制增强版融合引擎
cp project_updates/plugins/multimodal_fusion/fusion_engine_enhanced.py \
   your_project/plugins/multimodal_fusion/

# 复制平台核心模块
cp project_updates/platform_core/data_collection_manager.py \
   your_project/platform_core/
cp project_updates/platform_core/training_pipeline.py \
   your_project/platform_core/
cp project_updates/platform_core/plugin_compatibility_fix.py \
   your_project/platform_core/

# 复制全景监测API
cp project_updates/apps/panoramic_monitoring_api.py \
   your_project/apps/

# 复制配置文件
cp project_updates/configs/extended_models_config.yaml \
   your_project/configs/
```

### 步骤2：更新多模态融合插件

在 `plugins/multimodal_fusion/plugin.py` 中添加导入：

```python
# 在文件顶部添加
from .fusion_engine_enhanced import (
    EnhancedFusionEngine,
    FusionStrategyManager,
    ModalityData,
    FusionResult,
    BayesianDecisionNetwork
)

# 在 MultimodalFusionPlugin.__init__() 中初始化
class MultimodalFusionPlugin:
    def __init__(self, manifest=None, plugin_dir=None, config=None):
        # ... 现有代码 ...
        
        # 添加增强融合引擎
        self.enhanced_engine = None
        self.strategy_manager = None
    
    def init(self, config: Dict = None) -> bool:
        # ... 现有代码 ...
        
        # 初始化增强融合引擎
        try:
            self.enhanced_engine = EnhancedFusionEngine(self._config)
            self.strategy_manager = self.enhanced_engine.strategy_manager
            logger.info("增强融合引擎初始化成功")
        except Exception as e:
            logger.warning(f"增强融合引擎初始化失败: {e}")
        
        return True
```

### 步骤3：集成数据采集管理器

在主应用或单独的数据服务中：

```python
from platform_core.data_collection_manager import (
    DataCollectionManager,
    ModalityType,
    AcousticCollector,
    GasCollector,
    AcousticPreprocessor,
    assess_acoustic_quality
)

# 初始化数据管理器
data_manager = DataCollectionManager(data_dir="./collected_data")

# 注册采集器
data_manager.register_collector(ModalityType.ACOUSTIC, AcousticCollector())
data_manager.register_collector(ModalityType.GAS, GasCollector())

# 注册预处理器
data_manager.register_preprocessor(ModalityType.ACOUSTIC, AcousticPreprocessor())

# 注册质量评估器
data_manager.register_quality_assessor(ModalityType.ACOUSTIC, assess_acoustic_quality)

# 创建采集任务示例
task = data_manager.create_collection_task(
    name="变压器T001多模态监测",
    modalities=[ModalityType.ACOUSTIC, ModalityType.GAS, ModalityType.THERMAL],
    device_ids=["T001"],
    duration_hours=168,  # 一周
    interval_seconds=3600  # 每小时
)
```

### 步骤4：集成训练流水线

```python
from platform_core.training_pipeline import (
    TrainingPipeline,
    AutoTrainingScheduler,
    TrainingConfig,
    ModelType
)

# 初始化训练流水线
pipeline = TrainingPipeline(
    models_dir="./models",
    config_path="./configs/extended_models_config.yaml"
)

# 初始化自动训练调度器
scheduler = AutoTrainingScheduler(
    pipeline=pipeline,
    data_manager=data_manager,
    min_new_samples=100,  # 达到100个新样本时触发重训练
    check_interval_hours=24  # 每24小时检查一次
)

# 手动训练示例
config = TrainingConfig(
    model_type=ModelType.ACOUSTIC_ANOMALY,
    model_name="acoustic_transformer_v2",
    input_dim=128,
    output_dim=5,
    hidden_dim=256,
    num_layers=4,
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)

# 导出数据集
dataset_info = data_manager.export_dataset(
    output_name="acoustic_dataset_v2",
    modalities=[ModalityType.ACOUSTIC],
    label_filter=["fault", "normal"],
    quality_filter=[DataQuality.HIGH, DataQuality.MEDIUM],
    split_ratios=(0.7, 0.15, 0.15)
)

# 加载数据并训练（需实现数据加载逻辑）
# train_data = load_training_data(dataset_info)
# result = pipeline.train_model(config, train_data, export_onnx=True)
```

### 步骤5：集成全景监测API

在 `apps/api_server.py` 或主应用中：

```python
from fastapi import FastAPI
from apps.panoramic_monitoring_api import (
    integrate_panoramic_routes,
    CentralStateCache,
    PanoramicMonitoringService,
    AlarmLinkageService,
    DeviceInfo,
    DeviceStatus
)

app = FastAPI()

# 初始化组件
cache = CentralStateCache()
service = PanoramicMonitoringService(cache)
alarm_service = AlarmLinkageService()

# 配置告警联动
alarm_service.configure({
    'http_endpoint': 'http://alarm-service:8080/api/alarm',
    'email_recipients': ['admin@example.com', 'operator@example.com'],
    'sms_recipients': ['+86138xxxx1234'],
    'escalation_rules': [
        {'threshold': 3, 'escalate_to': 'critical', 'notify_manager': True}
    ],
    'silence_rules': [
        {'device_pattern': 'TEST_*', 'duration_minutes': 60}
    ]
})

# 注册告警处理器
service.register_alarm_handler(alarm_service.handle_alarm)

# 集成路由
integrate_panoramic_routes(app, cache, service)

# 注册设备
devices = [
    DeviceInfo(
        device_id="T001",
        device_name="1号主变压器",
        device_type="transformer",
        location={"x": 100, "y": 200, "z": 0, "zone": "A区"},
        modalities=["visual", "thermal", "acoustic", "gas"]
    ),
    DeviceInfo(
        device_id="S001", 
        device_name="1号开关",
        device_type="switch",
        location={"x": 150, "y": 200, "z": 0, "zone": "A区"},
        modalities=["visual", "thermal", "acoustic"]
    ),
]

for device in devices:
    cache.register_device(device)

# 启动应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 步骤6：应用插件兼容性修复

在 `apps/advanced_plugins_api.py` 中更新插件加载逻辑：

```python
from platform_core.plugin_compatibility_fix import (
    PluginCompatibilityLayer,
    safe_plugin_load,
    validate_plugin_interface
)

# 在 AdvancedPluginManager.initialize_plugins() 中使用
def initialize_plugins(self, force_reload: bool = False):
    compat_layer = PluginCompatibilityLayer()
    
    for plugin_id, module_path, class_name in plugins_to_load:
        try:
            plugin_class = safe_plugin_load(module_path, class_name)
            
            if plugin_class:
                # 验证插件接口
                if validate_plugin_interface(plugin_class):
                    # 使用兼容层包装
                    plugin = compat_layer.wrap_plugin(plugin_class)
                    self._plugins[plugin_id] = plugin
                    logger.info(f"插件 {plugin_id} 加载成功")
                else:
                    logger.warning(f"插件 {plugin_id} 接口不完整")
        except Exception as e:
            logger.error(f"加载插件 {plugin_id} 失败: {e}")
```

---

## 📡 API 接口说明

### 全景监测 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/panoramic/overview` | GET | 获取设备概览统计 |
| `/api/panoramic/devices` | GET | 获取设备列表（支持过滤） |
| `/api/panoramic/devices/{device_id}` | GET | 获取设备详情 |
| `/api/panoramic/devices/{device_id}/trend` | GET | 获取趋势数据 |
| `/api/panoramic/alarms` | GET | 获取告警列表 |
| `/api/panoramic/alarms/{alarm_id}/acknowledge` | POST | 确认告警 |
| `/api/panoramic/alarms/{alarm_id}/resolve` | POST | 解决告警 |
| `/api/panoramic/fusion/{device_id}` | GET | 获取融合结果 |
| `/api/panoramic/statistics` | GET | 获取统计信息 |
| `/api/panoramic/ws` | WebSocket | 实时数据推送 |

### WebSocket 消息格式

```json
// 订阅设备
{"type": "subscribe", "device_ids": ["T001", "S001"]}

// 告警推送
{
  "type": "alarm",
  "data": {
    "alarm_id": "ALM_001",
    "device_id": "T001",
    "level": "warning",
    "message": "温度异常升高",
    "timestamp": 1234567890.123
  }
}

// 状态更新
{
  "type": "status_update",
  "data": {
    "device_id": "T001",
    "status": "attention",
    "health_index": 0.85
  }
}
```

---

## 🔄 融合策略说明

### 可用策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `early` | 特征级融合，拼接所有模态特征 | 传感器数据质量高且同步 |
| `late` | 决策级融合，加权投票 | 各模态独立性强 |
| `attention` | Transformer注意力融合 | 需要动态权重分配 |
| `hybrid` | 混合融合（注意力+贝叶斯） | 综合诊断场景 |
| `bayesian` | 贝叶斯决策网络 | 故障链推断 |

### 策略切换示例

```python
from plugins.multimodal_fusion.fusion_engine_enhanced import (
    EnhancedFusionEngine,
    FusionStrategy
)

engine = EnhancedFusionEngine(config)

# 手动切换策略
engine.strategy_manager.switch_strategy(FusionStrategy.ATTENTION)

# 自动选择策略
strategy = engine.strategy_manager.auto_select_strategy(
    device_type="transformer",
    available_sensors=["thermal", "acoustic", "gas"]
)

# 获取历史最佳策略
best = engine.strategy_manager.get_best_strategy()
```

---

## 📊 贝叶斯故障诊断

### 支持的故障类型

- `transformer_fault` - 变压器故障
- `insulation_fault` - 绝缘故障
- `gas_leak` - 气体泄漏
- `partial_discharge` - 局部放电
- `mechanical_fault` - 机械故障
- `overheating` - 过热
- `corrosion` - 腐蚀
- `aging` - 老化
- `contamination` - 污染
- `normal` - 正常状态

### 故障链推断示例

```python
from plugins.multimodal_fusion.fusion_engine_enhanced import BayesianDecisionNetwork

bayesian = BayesianDecisionNetwork()

# 输入各模态检测证据
evidence = {
    'thermal_anomaly': True,
    'gas_H2_elevated': True,
    'acoustic_abnormal': False
}

# 计算后验概率
posterior = bayesian.compute_posterior(evidence)
# 输出: {'transformer_fault': 0.72, 'overheating': 0.65, ...}

# 推断故障链
fault_chain = bayesian.infer_fault_chain(evidence)
# 输出: {
#   'root_fault': 'overheating',
#   'related_faults': ['transformer_fault', 'insulation_fault'],
#   'confidence': 0.78
# }
```

---

## 📁 文件清单

```
project_updates/
├── apps/
│   └── panoramic_monitoring_api.py      # 全景监测API (30KB)
├── configs/
│   └── extended_models_config.yaml      # 扩展模型配置 (14KB)
├── platform_core/
│   ├── data_collection_manager.py       # 数据采集管理器 (32KB)
│   ├── training_pipeline.py             # 训练流水线 (34KB)
│   ├── plugin_compatibility_fix.py      # 插件兼容性修复 (33KB)
│   └── multimodal_integration.py        # 多模态集成 (32KB)
└── plugins/
    └── multimodal_fusion/
        └── fusion_engine_enhanced.py    # 增强融合引擎 (44KB)
```

---

## ⚠️ 注意事项

1. **依赖安装**
   ```bash
   pip install numpy fastapi uvicorn pydantic pyyaml
   ```

2. **模型训练**
   - 当前实现为纯NumPy版本，适合理解和调试
   - 生产环境建议使用PyTorch/TensorFlow实现并导出ONNX

3. **ONNX导出**
   - 当前`export_onnx()`方法导出JSON格式权重
   - 实际部署需替换为torch.onnx.export()或tf2onnx

4. **WebSocket连接**
   - 需要在前端实现心跳检测（ping/pong）
   - 建议实现断线重连机制

5. **告警升级**
   - 默认配置基于告警计数触发升级
   - 可根据实际需求调整阈值和通知方式

---

## 🔜 后续开发建议

1. **前端仪表盘开发**
   - 基于Vue/React开发全景监测页面
   - 集成3D SLAM地图可视化
   - 实现实时趋势图表

2. **真实模型训练**
   - 收集实际现场数据
   - 使用PyTorch训练Transformer/LSTM模型
   - 验证ONNX推理性能

3. **性能优化**
   - 融合算法GPU加速
   - 数据库索引优化
   - WebSocket消息批处理

4. **测试完善**
   - 编写单元测试
   - 添加模拟数据生成器
   - 集成测试覆盖

---

*文档版本: 1.0*
*更新日期: 2026-01-09*
