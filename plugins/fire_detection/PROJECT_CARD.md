# PROJECT_CARD: fire_detection

## 1. 项目名称
消防监测插件（fire_detection）

## 2. 项目类型
plugin（独立插件，含 YOLOv8 火焰/烟雾检测 + 多模态传感器融合 + 灭火联动 + 应急指导）

## 3. 输入源
- 摄像头帧：`BGR np.ndarray`，单帧推理。
- 热成像帧：灰度 ndarray，可选。
- 传感器数据：烟雾浓度、温度、CO 浓度、湿度（可选）。
- 监测区域：polygon + priority 配置。

## 4. 输出目标
- 检测结果：火焰/烟雾/火花/热点，含 bbox、置信度、跟踪 ID。
- 火灾等级：none → smoldering → small → medium → large → critical。
- 融合置信度：视觉+传感器加权。
- 灭火动作：喷淋/断电/声光告警。
- 疏散路线：应急照明引导。

## 5. 关键约束
- 依赖 `darkbreaker_sdk` 插件契约。
- ONNX 模型为必需依赖，缺失时回退到模拟模式。
- 灭火联动需外部硬件支持（喷淋/断电）。
- 传感器融合需实际传感器数据。

## 6. 验收标准
- 当前仅 test_standalone.py 1 个测试可通过。

## 7. 已知参考物
- `plugin.py`：SDK 适配层，含 drill 模拟。
- `detector.py`：YOLOv8 ONNX 检测器 + 传感器融合 + DeepSORT。
- `manifest.json`：插件元数据。

## 8. Phase 2 能力三分表 (2026-04-16 审计)

| 能力 | 分类 | 证据 |
|------|------|------|
| fire_detection | experimental | detector.py 含完整 YOLOv8 推理+后处理代码，但模型不存在、仅 1 smoke test |
| smoke_detection | experimental | 与 fire_detection 共用检测器，classes 含 smoke |
| drill_simulation | experimental | start_drill()/stop_drill() 代码存在，配置可控 |
| thermal_anomaly_detection | **blocked** | thermal_anomaly_cnn.onnx 不存在 |
| multi_sensor_fusion | **blocked** | 融合代码存在但无传感器硬件集成测试 |
| active_suppression_control | **blocked** | 喷淋/断电控制代码路径存在但依赖硬件接口 |
| evacuation_guidance | **blocked** | 应急照明代码路径存在但无验证 |

**runtime_mode_support**: real_dl=false traditional_fallback=false simulation=true
**测试状态**: 仅 test_standalone.py 通过；无 fixtures、无 regression
**关键差距**: 7 个宣称能力中 4 个 blocked；manifest 描述远超实际可用能力
