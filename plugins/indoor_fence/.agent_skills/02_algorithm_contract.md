# 02 算法与降级契约

## 1. When to use
- 修改 `plugin.py::infer()`、`postprocess()`、`healthcheck()`
- 修改 Camera / LiDAR / UWB / IMU 适配器
- 修改 `detection/*`、`core/fusion/*`、`core/tracking/*`
- 修改 `standalone/video_stream.py`、`standalone/realtime_pipeline.py`
- 修改配置更新、zone 更新、scenario 加载逻辑

## 2. Inputs
- 图像帧 `frame`、ROI、`PluginContext`
- `configs/default.yaml`
- `standalone/configs/zone.yaml`
- `configs/scenarios/*.json`
- Camera / LiDAR 主链路数据，及 UWB / IMU / replay / simulation 数据

## 3. Outputs
- 主插件运行时输出：`RecognitionResult` 列表与 `Alarm` 列表
- V3 演练链路输出：`SensorData` / `FusionOutput` / 跟踪结果 / MJPEG 帧
- 运行时状态输出：config revision、zone config、recent events、standalone route 响应

## 4. Hard Constraints
- `plugin.py::infer()` 失败时必须返回统一错误结果，`label="error"`，失败原因沿用 `9000` / `9001`
- 无硬件、无模型、无 `cv2`、无回放数据等情况必须走降级或占位路径，不能直接 crash
- `update_config()` 失败必须回滚；`update_zone_config(..., persist=False)` 不能强制落盘
- 场景文件仍为 `configs/scenarios/*.json`；区域配置仍为 `standalone/configs/zone.yaml`
- 对外识别标签保持稳定：`safe` / `on_line` / `line_cross` / `unauthorized` / `high_risk` / `multi_person` / `no_person` / `error`

## 5. Algorithm / Logic Contract

### A. 主插件运行时契约
- `plugin.py` 当前稳定链路主要使用：
  - `CameraAdapter.get_person_detections()`
  - `LidarAdapter.get_scan()` / `get_clusters()`
  - 状态机 + 黄线 / 机柜授权判定
- 输出关注点：
  - 黄线距离
  - 授权机柜 `allow_list`
  - 同柜多人 `multi_person`
  - `healthcheck()` 的适配器状态与延迟摘要

### B. V3 演练链路契约
- `protocols.py` 定义 `SensorData` / `FusionInput` / `FusionOutput` / `RiskAssessment`
- `core/fusion/*` 与 `core/tracking/*` 承接多传感器融合、Hungarian 关联、3D 跟踪
- `standalone/realtime_pipeline.py` 与 `standalone/video_stream.py` 用于仿真、可视化、轨迹和快照

### C. 降级矩阵

| 触发场景 | 当前行为 | 验证点 |
|---------|---------|--------|
| 相机硬件不可用 | CameraAdapter 切到 simulation | `tests/test_camera_adapter.py` |
| 雷达硬件不可用 | LidarAdapter 切到 simulation | `tests/test_lidar_adapter.py` |
| YOLO 模型缺失 / 加载失败 | `detection/yolo_detector.py` 切到 simulation mode，并可查询 fallback reason | `tests/test_detection.py` |
| UWB / IMU 使用 `protocol="simulate"` | 生成仿真 `SensorData` | `tests/test_uwb_adapter.py` / `tests/test_imu_adapter.py` |
| 视频流无相机或无 `cv2` | 返回可视化占位帧 / 快照 | `tests/test_video_stream.py` |
| 配置非法 | `update_config()` 抛错并回滚 | `tests/test_config_updates.py` |
| zone 仅内存更新 | `persist=False` 时不写回文件 | `tests/test_config_updates.py` |
| scenario 切换 | 从 `configs/scenarios/*.json` 重载仿真器和 renderer | `tests/test_simulator_integration.py` |

### D. 日志与可追溯性
- 审计日志目录是 `logs/indoor_fence/`
- 降级事件应使用 `FALLBACK` 前缀 + 结构化 `extra`
- 不要再假设存在 `logs/fallback.log`

## 6. Validation Rules

```bash
# 主插件与配置
./scripts/run_targeted_tests.sh plugin

# 检测与降级
./scripts/run_targeted_tests.sh detection
./scripts/run_targeted_tests.sh adapters

# 融合 / 跟踪 / 规则
./scripts/run_targeted_tests.sh fusion
./scripts/run_targeted_tests.sh logic

# standalone
./scripts/run_targeted_tests.sh standalone
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 失败时返回空列表而非 `error` 结果 | 上游无法区分“无人”和“异常” | 回到统一错误输出 |
| 只改主插件链路，忘记 V3 演练链路 | standalone / 协议测试漂移 | 明确链路归属并补对应测试 |
| 场景 JSON / zone YAML 混改 | 仿真与黄线行为不一致 | 分别维护场景与区域配置 |
| 降级无日志或不可查询 | 故障不可追踪 | 保留 `FALLBACK` 日志和查询方法 |
| 配置更新无回滚 | runtime 配置损坏 | 保持 validate -> apply -> rollback 流程 |

## 8. Required Tests
- `tests/test_detection.py`
- `tests/test_camera_adapter.py`
- `tests/test_lidar_adapter.py`
- `tests/test_uwb_adapter.py`
- `tests/test_imu_adapter.py`
- `tests/test_config_updates.py`
- `tests/test_video_stream.py`
- `tests/test_simulator_integration.py`

