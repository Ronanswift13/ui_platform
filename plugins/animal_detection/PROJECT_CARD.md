# PROJECT_CARD: animal_detection

## 1. 项目名称
动物入侵检测插件（animal_detection）

## 2. 项目类型
plugin（独立插件，含 YOLO ONNX 检测 + 多目标跟踪 + 热成像验证 + 驱离控制）

## 3. 输入源
- 摄像头帧：`BGR np.ndarray`，单帧推理，最大 4096×4096。
- 热成像帧：灰度 `uint8 ndarray`，可选（`thermal.enabled` 控制）。
- 配置文件：`configs/default.yaml`。
- 模型文件：`models/` 下 ONNX 格式（≤ 50MB）。

## 4. 输出目标
- 检测结果：`AnimalDetectionResult`（动物类别、置信度、bbox、跟踪 ID、热验证状态）。
- 统一事件：`AnimalEvent`（10 种事件类型，含证据链 + 处置建议 + trace_id）。
- 驱离动作：`DeterrentAction`（音频/灯光/组合，含结果评估）。
- 统计报告：日/周汇总、趋势数据、CSV/JSON 导出。
- 健康状态：`healthcheck()` 输出推理计数与模型加载状态。

## 5. 关键约束
### 工程约束
- 必须遵循 `darkbreaker_sdk` 插件契约。
- `plugin.py` 仅做 SDK 适配，不承载核心算法逻辑。
- `core/` 不得依赖 `standalone/`、`demo/`、`tests/`。
- 所有阈值必须来自 `configs/default.yaml`，不允许推理路径硬编码 (QR-1)。

### 业务约束
- 支持 8 类动物检测：鼠、猫、蛇、鸟、狗、家禽、昆虫、其他。
- 蛇类检测风险等级固定为 CRITICAL。
- 单帧检测延迟 P95 ≤ 100ms（CPU, 640×480）。
- GPU 为可选，CPU 必须可运行。
- 召回率 ≥ 85%，精确率 ≥ 80%，误报率 < 5%。

### 安全约束
- 不访问外部网络。
- 不持久化原始图像到未授权目录（证据链路除外）。
- standalone 默认仅监听 localhost。
- 运行入口必须含 venv guard (QR-6)。

## 6. 验收标准
- `./scripts/run_targeted_tests.sh all` 通过。
- `./scripts/run_regression_tests.sh` 全部通过。
- `./scripts/run_quality_gate.sh` 全部通过。
- `.agent_skills/00~09` 完整，且规则可执行可验证。
- 测试覆盖率 ≥ 70%（core/ ≥ 80%）。

## 7. 禁止事项
- 禁止修改 SDK 接口签名。
- 禁止新增硬编码业务阈值到推理主路径。
- 禁止使用 `except: pass`。
- 禁止在生产路径新增 `print()`。
- 禁止删除既有降级路径（模型缺失→空检测、热成像缺失→跳过验证）。

## 8. 已知参考物
- `plugin.py`：SDK 适配层。
- `core/detector.py`：YOLO 检测核心实现。
- `core/onnx_inference.py`：ONNX 推理引擎。
- `core/event_schema.py`：统一事件契约定义。
- `configs/default.yaml`：阈值与运行参数。
- `tests/`：当前可运行测试基线。

## 9. 当前任务
- agent skills 结构治理（08 task routing + 09 runtime repair）。
- 补齐 PROJECT_CARD / CLAUDE.md / README.md。
- 脚本体系与 busbar 金模板对齐。

## 10. Phase 2 能力三分表 (2026-04-16 审计)

| 能力 | 分类 | 证据 |
|------|------|------|
| animal_detection | **verified** | test_detector.py + test_plugin.py 通过；factory fixtures 生成合成输入 |
| intrusion_statistics | **verified** | test_event_schema_contract.py 通过；统计模块有独立测试 |
| species_classification | experimental | 代码路径存在(8类映射)，但 species_classifier 模型不存在 |
| behavior_tracking | experimental | tracker.py 代码存在，但测试仅覆盖 smoke 级 |
| deterrent_control | experimental | deterrent.py 代码完整，但依赖外部硬件接口 |
| thermal_fusion_detection | **blocked** | thermal_validator.py 存在但 thermal_animal_cnn.onnx 不存在 |
| real_dl_onnx_inference | **blocked** | animal_yolov8n.onnx 不存在，推理降级为空检测 |

**runtime_mode_support**: real_dl=❌ traditional_fallback=✅ simulation=✅
**测试状态**: 全量通过 (含 8 skipped regression)；fixtures 使用 factory 模式
**关键差距**: 测试基于合成数据而非真实图像；ONNX 模型全部缺失
