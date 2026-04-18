# PROJECT_CARD: busbar_inspection

## 1. 项目名称
母线自主巡视插件（busbar_inspection）

## 2. 项目类型
plugin_update（已有插件规则化治理 + 工程骨架补齐）

## 3. 输入源
- 巡检图像帧：`BGR np.ndarray`，单帧推理，支持 1080p/4K。
- ROI 列表：平台传入 `ROI`（归一化 `x/y/width/height`）。
- 插件上下文：`task_id/site_id/device_id/component_id/timestamp`。
- 配置文件：`configs/default.yaml`。

## 4. 输出目标
- 缺陷识别结果：`RecognitionResult`（`pin_missing/crack/foreign_object/quality_failed`）。
- 告警结果：`Alarm`（按标签映射到 `ERROR/WARNING`）。
- 质量解释：`failure_reason` + `metadata.quality` + `metadata.suggested_action`。
- 变焦建议：`metadata.suggested_zoom`、`metadata.suggested_action`。
- 健康状态：`healthcheck()` 输出计数与最近推理时间。

## 5. 关键约束
### 工程约束
- 必须遵循 `darkbreaker_sdk.interfaces.BasePlugin` 契约。
- `plugin.py` 仅做 SDK 适配，不承载核心检测算法。
- `detector_enhanced.py` 不得依赖 `darkbreaker_sdk`。
- 所有阈值必须来自 `configs/default.yaml` 映射。

### 业务约束
- 单帧多 ROI 独立处理，单 ROI 异常不得拖垮整帧。
- 质量门禁失败必须可解释（原因码 + 建议动作）。
- 对目标过小场景必须给出变焦建议，不允许仅返回失败。

### 安全约束
- 不访问外部网络。
- 不持久化原始图像到未授权目录。
- 日志中不得输出敏感标识符原文拼接。

## 6. 验收标准（本轮治理）
- `./scripts/run_targeted_tests.sh standalone` 通过。
- `./scripts/run_regression_tests.sh` 可执行且阶段化输出明确。
- `.agent_skills/00~04` 完整，且规则可执行可验证。
- 新增 `.agent_skills/05~07`，形成闭环治理。
- `PROJECT_CARD.md`、`CLAUDE.md`、`.claude/commands` 均为本项目定制内容。

## 7. 禁止事项
- 禁止修改 SDK 接口签名。
- 禁止新增硬编码业务阈值到推理主路径。
- 禁止使用 `except: pass`。
- 禁止在生产路径新增 `print()`。
- 禁止删除既有降级路径（深度学习 -> 传统方法）。

## 8. 已知参考物
- `README.md`：业务目标与输出字段说明。
- `plugin.py`：SDK 适配层。
- `detector_enhanced.py`：检测与质量门禁核心实现。
- `configs/default.yaml`：阈值与运行参数。
- `tests/test_standalone.py`：当前可运行测试基线。

## 9. 当前任务
- 解压并应用 agentic 模板到本插件。
- 按母线场景补齐规则文档、命令脚本与质量门禁。
- 给出第一轮最适合实现模块并形成执行入口。
