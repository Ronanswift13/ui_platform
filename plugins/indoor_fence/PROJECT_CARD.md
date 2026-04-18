# PROJECT_CARD: Indoor Fence Plugin

## 1. 项目名称
`indoor_fence` - 室内电子围栏监控插件（DarkBreaker）

## 2. 项目类型
`plugin_update`（已有插件迭代，当前同时维护主插件运行时与 V3 多传感器演练链路）

## 3. 输入源
- **主插件运行时**
  - Camera 帧与人体检测
  - 2D LiDAR 扫描与聚类
  - `configs/default.yaml`
  - `standalone/configs/zone.yaml`
- **V3 演练链路**
  - UWB 定位（可 simulation）
  - IMU 惯性（可 simulation）
  - `configs/scenarios/*.json`
  - `protocols.py` 定义的多传感器数据模型

## 4. 输出目标
- `RecognitionResult` 标签：
  - `safe`
  - `on_line`
  - `line_cross`
  - `unauthorized`
  - `high_risk`
  - `multi_person`
  - `no_person`
  - `error`
- 告警输出：黄线越界、未授权操作、同柜多人、高风险区域
- standalone HTTP / MJPEG surface：
  - `/api/indoor-fence/config`
  - `/api/indoor-fence/zones`
  - `/api/indoor-fence/events`
  - `/api/indoor-fence/stream`
  - `/api/indoor-fence/snapshot`
  - `/api/indoor-fence/simulator/*`
  - `/api/indoor-fence/tracking`
- 可选集成面：`api_integration.py` 中的 FastAPI / WebSocket 集成（非默认 standalone surface）

## 5. 关键约束
- `core/` 禁止依赖 `adapters/` 或 `standalone/`
- SDK 入口保持 `manifest.json -> plugin.py -> IndoorFencePlugin`
- `run_standalone.py` 必须保留 venv 自动切换
- `update_config()` 失败必须回滚
- `update_zone_config(..., persist=False)` 不能强制落盘
- 无硬件 / 无模型时必须仍能走 simulation / placeholder 路径
- 当前质量门禁覆盖率阈值是 `70%`（由脚本与 `.coveragerc` 约束），工程目标仍是 active 模块 `80%+`
- 禁止新增未经批准的网络依赖或扩大对外接口面

## 6. 验收标准
- `./scripts/run_targeted_tests.sh <module>` 通过
- `./scripts/run_regression_tests.sh` 通过
- `./scripts/run_quality_gate.sh` 通过
- `python run_standalone.py` 可在 mock / simulation 路径启动
- fallback、config rollback、video stream、scenario / zone 相关测试不回退

## 7. 禁止事项
- 修改 `manifest.json` 核心入口字段却不声明影响
- 在 `core/` 中引入硬件 I/O 或 web I/O
- 在 touched production files 中新增 `except: pass`、裸 `print()`、未解释 `TODO/FIXME`
- 把新功能默认继续堆到 legacy `detector.py`
- 改 route / config / zone / scenario 但不补测试和文档

## 8. 已知参考物
- `.agent_skills/08_task_routing.md`
- `.agent_skills/04_quality_audit.md`
- `.agent_skills/07_learning_log.md`
- `docs/plans/2026-03-02-indoor-fence-v3-design.md`
- `docs/plans/2026-03-02-indoor-fence-v3-implementation.md`
- `tests/test_matrix_template.md`

## 9. 当前任务
<!-- 每轮任务开始时填写本字段 -->
[由使用者在每轮任务开始时填写]
