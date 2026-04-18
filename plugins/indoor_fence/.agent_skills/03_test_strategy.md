# 03 测试策略

## 1. When to use
- 新增功能、规则、standalone 路由或配置项
- 修复 bug 并补防回归测试
- 判断应该跑哪个 targeted 模块
- 解释覆盖率门槛与 `.coveragerc` 排除项

## 2. Inputs
- 功能需求或缺陷描述
- 受影响源码文件
- `tests/test_matrix_template.md`
- `.coveragerc`

## 3. Outputs
- 最近测试文件的新增或更新
- 模块级 targeted 验证结果
- 回归与覆盖率结果

## 4. Hard Constraints
- 每次改动都要更新最近的测试文件，不能只跑已有用例
- 测试必须可在无硬件环境下运行；优先使用 simulation / replay / `tmp_path`
- 当前质量门禁以 `.coveragerc` + `run_regression_tests.sh` 为准，默认覆盖率门槛为 `70%`
- 工程目标仍是：对本次触达的 active 模块尽量维持 `80%+`，且不能让现有覆盖率倒退
- `tests/fixtures/` 当前不是既定公共入口；共享 fixture 优先放 `tests/conftest.py`

## 5. Algorithm / Logic Contract

### 模块到测试映射

| 变更面 | 优先测试 |
|------|---------|
| 插件入口 / config / zone | `tests/test_standalone.py`, `tests/test_config_updates.py`, `tests/test_integration.py` |
| adapters | `tests/test_adapters_base.py`, `tests/test_camera_adapter.py`, `tests/test_lidar_adapter.py`, `tests/test_uwb_adapter.py`, `tests/test_imu_adapter.py`, `tests/test_simulator.py` |
| detection | `tests/test_detection.py`, `tests/test_model_manager.py`, `tests/test_pose.py`, `tests/test_behavior.py`, `tests/test_auto_fence.py` |
| fusion / protocol | `tests/test_ekf.py`, `tests/test_fusion_v3.py`, `tests/test_nlos_weights.py`, `tests/test_protocols.py` |
| tracking / rules / state | `tests/test_realtime_tracking.py`, `tests/test_tracker_v3_new.py`, `tests/test_state_machine_v3.py`, `tests/test_rules.py`, `tests/test_adaptive.py` |
| standalone | `tests/test_api_routes.py`, `tests/test_video_stream.py`, `tests/test_data_recorder.py`, `tests/test_data_replayer.py`, `tests/test_training.py`, `tests/test_simulation_renderer.py` |
| 仿真/集成 | `tests/test_integration.py`, `tests/test_simulator_integration.py` |

### targeted 模块名约定

| 命令 | 说明 |
|------|------|
| `./scripts/run_targeted_tests.sh plugin` | 插件入口、config/zone 更新、基础 standalone |
| `./scripts/run_targeted_tests.sh adapters` | Camera / LiDAR / UWB / IMU / BaseAdapter / Simulator |
| `./scripts/run_targeted_tests.sh detection` | YOLO / Pose / Behavior / Auto Fence / ModelManager |
| `./scripts/run_targeted_tests.sh fusion` | EKF、fusion v3、NLOS、protocols |
| `./scripts/run_targeted_tests.sh logic` | 规则、状态机、实时跟踪、多目标关联 |
| `./scripts/run_targeted_tests.sh standalone` | API routes、video stream、record/replay、training、renderer |
| `./scripts/run_targeted_tests.sh integration` | 插件全链路与 scenario 集成 |
| `./scripts/run_targeted_tests.sh all` | 非 regression 全量快速门禁 |

### 覆盖率边界
- `.coveragerc` 当前排除了 legacy 或外部环境强依赖模块，例如：
  - `detector.py`
  - `plugin.py`
  - `core/tracker_v3.py`
  - `core/enhanced_tracking.py`
  - `api_integration.py`
- 因此覆盖率解释必须结合 `.coveragerc`，不能只看总数字

## 6. Validation Rules

```bash
# 最近模块先跑 targeted
./scripts/run_targeted_tests.sh <module>

# 回归 + 覆盖率
./scripts/run_regression_tests.sh

# 交付前质量门禁
./scripts/run_quality_gate.sh
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 只跑全量不补最近测试 | 根因未被固定 | 先补最近模块测试，再跑 regression |
| 把 simulation / replay 改坏 | 无硬件环境测试失真 | 优先检查 adapter / video stream / simulator 测试 |
| 忽略 `.coveragerc` 排除 | 误判质量回退 | 先解释覆盖率范围，再下结论 |
| standalone 路由改了没补 UI 相关测试 | 页面或 API 断裂 | 同步跑 `standalone` targeted |
| 只改场景文件不测 | 仿真行为漂移 | 跑 `tests/test_simulator_integration.py` |

## 8. Required Tests
- `tests/test_config_updates.py`
- `tests/test_detection.py`
- `tests/test_camera_adapter.py`
- `tests/test_lidar_adapter.py`
- `tests/test_fusion_v3.py`
- `tests/test_realtime_tracking.py`
- `tests/test_video_stream.py`
- `tests/test_api_routes.py`

