# 00 项目上下文

## 1. When to use
- 首次接手 `plugins/indoor_fence/`
- 需要判断任务属于主插件运行时还是 V3 演练链路
- 修改配置、standalone、测试或脚本前确认真实入口
- 校对 skill / command / script 是否仍与当前代码一致

## 2. Inputs
- `plugin.py`, `manifest.json`, `core/config_manager.py`
- `configs/default.yaml`, `standalone/configs/zone.yaml`, `configs/scenarios/*.json`
- `run_standalone.py`, `standalone/app.py`
- `tests/`, `.coveragerc`, `PROJECT_CARD.md`, `CLAUDE.md`

## 3. Outputs
- 明确当前任务落点与依赖边界
- 理解配置来源、fallback 机制、standalone API 面
- 确认真实测试入口、覆盖率门槛和 legacy 模块范围

## 4. Hard Constraints
- 真实导入路径是 `plugins.indoor_fence`，不是顶层 `indoor_fence`
- SDK 入口保持 `manifest.json -> plugin.py -> IndoorFencePlugin`，并保留 `Plugin = IndoorFencePlugin`
- 当前仓内并存两条链路，修改前必须先判定：
  1. 主插件运行时：`plugin.py` + `adapters/camera_adapter.py` + `adapters/lidar_adapter.py` + `core/*`
  2. V3 演练链路：`protocols.py` + `detection/*` + `core/fusion/*` + `core/tracking/*` + `standalone/realtime_pipeline.py`
- `configs/scenarios/*.json` 是仿真场景文件，`standalone/configs/zone.yaml` 是区域/黄线配置，不能混用
- `detector.py` 属于 legacy / 兼容层；新增检测能力默认落到 `detection/*`

## 5. Algorithm / Logic Contract

### 目录定位

```text
indoor_fence/
├── plugin.py                    # 主插件运行时：init / infer / postprocess / standalone routes
├── manifest.json               # SDK 入口与对外契约
├── core/
│   ├── config_manager.py       # default.yaml 归一化、校验、zone 路径解析
│   ├── fusion/                 # V3 融合链路（EKF、NLOS、权重）
│   ├── tracking/               # V3 跟踪链路（Hungarian、多目标）
│   ├── rules/                  # 规则引擎、风险评分、自适应阈值
│   └── state_machine.py        # 主插件运行时的人员/机柜/黄线状态机
├── adapters/                   # Camera / LiDAR / Light + UWB / IMU / Simulator / BaseAdapter
├── detection/                  # YOLO / Pose / Behavior / Auto Fence / ModelManager
├── standalone/                 # app / video_stream / realtime_pipeline / recorder / replayer / training
├── configs/default.yaml        # 运行时配置基线
├── configs/scenarios/*.json    # 仿真场景
├── standalone/configs/zone.yaml# 区域与黄线布局
└── tests/                      # 插件、适配器、检测、融合、standalone、集成测试
```

### 当前真实运行面
- 主插件运行时由 `plugin.py` 驱动，当前稳定路径仍以 Camera + LiDAR + 状态机为主。
- UWB / IMU / `protocols.py` / `standalone/realtime_pipeline.py` 属于 V3 多传感器演练链路，已有测试与 standalone 支撑，但不是所有能力都直接接入 `plugin.py::infer()`。
- standalone API 由 `plugin.py::get_standalone_routes()` 暴露，当前对外路径是：
  - `/api/indoor-fence/config`
  - `/api/indoor-fence/zones`
  - `/api/indoor-fence/events`
  - `/api/indoor-fence/stream`
  - `/api/indoor-fence/snapshot`
  - `/api/indoor-fence/simulator/*`
  - `/api/indoor-fence/tracking`

### 配置与资源加载
- `plugin.py::create_standalone()` 读取 `configs/default.yaml`
- `core/config_manager.py` 负责 normalize + validate + zone path resolve
- `update_config()` 失败时必须回滚到上一版配置
- `update_zone_config(..., persist=False)` 支持只更新内存，不落盘
- `run_standalone.py` 先做 venv 切换，再导入项目模块

### 降级与模拟
- Camera / LiDAR 可在 `simulate_if_unavailable=True` 时降级到模拟模式
- `detection/yolo_detector.py`、`detection/pose_estimator.py` 缺模型或缺依赖时降级到 simulation mode
- `adapters/base_adapter.py` 统一支持 `LIVE / SIMULATED / REPLAY`
- `standalone/video_stream.py` 在无相机或无 `cv2` 时仍要提供可视化占位帧/快照

## 6. Validation Rules

```bash
# 推荐启动
python run_standalone.py

# 备用启动
python -m plugins.indoor_fence.standalone.app

# 插件与配置烟测
./scripts/run_targeted_tests.sh plugin

# 全量回归
./scripts/run_regression_tests.sh
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 误把任务落到错误链路 | 改了测试不生效或改错入口 | 先判定主插件运行时 / V3 演练链路 |
| 把场景 JSON 当 zone YAML 修改 | standalone 仿真异常 | 场景改 `configs/scenarios/*.json`，区域改 `standalone/configs/zone.yaml` |
| 使用 `indoor_fence` 顶层导入 | 本地运行或测试导入失败 | 统一使用 `plugins.indoor_fence` |
| 改了 standalone 路由但未补测试/模板 | Web UI 断裂 | 同步更新 `tests/test_standalone.py` / `tests/test_api_routes.py` / `tests/test_video_stream.py` |
| 忽略 `.coveragerc` 的 legacy 排除 | 误判覆盖率下降 | 先看 `.coveragerc` 再解释覆盖率 |

## 8. Required Tests
- `tests/test_standalone.py`
- `tests/test_config_updates.py`
- `tests/test_integration.py`
- `tests/test_api_routes.py`

