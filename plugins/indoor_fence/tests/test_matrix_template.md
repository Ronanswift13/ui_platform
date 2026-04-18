# 测试矩阵模板

## 测试维度

### 1. 功能维度
- 传感器融合
- 行为检测
- 围栏判定
- 状态管理
- 数据录制/回放

### 2. 质量维度
- 正常场景
- 边界条件
- 异常处理
- 性能压力
- 并发安全

### 3. 环境维度
- 硬件可用
- 硬件缺失
- 模型可用
- 模型缺失
- 配置变化

## 测试矩阵

| 功能模块 | 测试场景 | 输入条件 | 期望输出 | 测试文件 | 优先级 |
|---------|---------|---------|---------|---------|--------|
| **传感器融合** |
| EKF 初始化 | 正常启动 | UWB + IMU 数据 | 状态收敛 | test_ekf.py::test_init | P0 |
| EKF 更新 | 连续数据流 | 10Hz UWB, 100Hz IMU | 位置平滑 | test_ekf.py::test_update | P0 |
| EKF 发散检测 | 高噪声输入 | std=5.0m | 触发重置 | test_ekf.py::test_divergence | P0 |
| EKF 发散恢复 | 重置后 | 新测量值 | 重新收敛 | test_ekf.py::test_recovery | P1 |
| 纯 UWB 模式 | IMU 缺失 | 仅 UWB | 定位可用 | test_fusion_v3.py::test_uwb_only | P1 |
| 纯 IMU 模式 | UWB 缺失 | 仅 IMU | 航位推算 | test_fusion_v3.py::test_imu_only | P1 |
| **硬件适配器** |
| UWB 正常读取 | 串口可用 | 有效帧 | 解析成功 | test_uwb_adapter.py::test_read | P0 |
| UWB 串口断开 | 串口不可用 | 无设备 | 降级到模拟器 | test_uwb_adapter.py::test_fallback | P0 |
| UWB 数据损坏 | CRC 错误 | 损坏帧 | 丢弃该帧 | test_uwb_adapter.py::test_corrupted | P1 |
| IMU 正常读取 | 串口可用 | 有效数据 | 解析成功 | test_imu_adapter.py::test_read | P0 |
| IMU 串口断开 | 串口不可用 | 无设备 | 返回零值 | test_imu_adapter.py::test_fallback | P0 |
| 模拟器轨迹 | 无硬件 | 配置文件 | 生成轨迹 | test_simulator.py::test_trajectory | P1 |
| **行为检测** |
| ML 模型推理 | 模型可用 | 速度+加速度 | 行为分类 | test_detection.py::test_ml | P0 |
| 模型缺失降级 | 模型不存在 | 速度+加速度 | 规则引擎 | test_detection.py::test_fallback | P0 |
| 规则引擎 | 无模型 | v=0.3 m/s | WALKING | test_detection.py::test_rules | P1 |
| 静止检测 | 低速 | v<0.1 m/s | STATIONARY | test_detection.py::test_stationary | P1 |
| 跑步检测 | 高速 | v>2.0 m/s | RUNNING | test_detection.py::test_running | P1 |
| **围栏判定** |
| 内部位置 | 正常 | dist=3.0m | INSIDE | test_rules.py::test_inside | P0 |
| 边界位置 | 临界 | dist=5.0m | ON_BOUNDARY | test_rules.py::test_boundary | P0 |
| 外部位置 | 越界 | dist=6.0m | OUTSIDE | test_rules.py::test_outside | P0 |
| 自适应围栏 | 行为变化 | RUNNING | 扩大半径 | test_auto_fence.py::test_adaptive | P1 |
| 围栏配置更新 | 运行时 | 新半径 | 立即生效 | test_rules.py::test_update_config | P2 |
| **状态机** |
| 状态初始化 | 启动 | 无输入 | INIT | test_state_machine_v3.py::test_init | P0 |
| 正常运行 | 数据流 | 连续帧 | TRACKING | test_state_machine_v3.py::test_tracking | P0 |
| 信号丢失 | 5 帧缺失 | gap=0.5s | SIGNAL_LOST | test_state_machine_v3.py::test_signal_loss | P0 |
| 信号恢复 | 新帧到达 | 恢复数据 | TRACKING | test_state_machine_v3.py::test_recovery | P1 |
| 越界告警 | 位置变化 | dist>5.0m | ALERT | test_state_machine_v3.py::test_alert | P1 |
| **数据管理** |
| 数据录制 | 运行中 | 60s | JSON 文件 | test_data_recorder.py::test_record | P1 |
| 数据回放 | 离线 | JSON 文件 | 重现轨迹 | test_data_replayer.py::test_replay | P1 |
| 训练数据生成 | 标注 | 轨迹+标签 | 训练集 | test_training.py::test_generate | P2 |
| **集成测试** |
| 端到端流程 | 完整系统 | UWB+IMU | 正确定位 | test_integration.py::test_e2e | P0 |
| 多传感器切换 | 动态降级 | 串口断开 | 自动切换 | test_integration.py::test_switch | P1 |
| 长时间运行 | 稳定性 | 1 小时 | 无崩溃 | test_integration.py::test_stability | P2 |

## 优先级定义

- **P0**: 核心功能，必须通过
- **P1**: 重要功能，应该通过
- **P2**: 增强功能，可选通过

## 测试覆盖率目标

| 模块 | 目标覆盖率 | 当前覆盖率 |
|------|-----------|-----------|
| core/fusion/ | 90% | - |
| core/rules/ | 85% | - |
| adapters/ | 80% | - |
| detection/ | 85% | - |
| standalone/ | 70% | - |
| **总体** | **80%** | - |

## 测试数据集

### 标准数据集
- `tests/data/trajectories/straight_line.json` - 直线运动
- `tests/data/trajectories/circle.json` - 圆周运动
- `tests/data/edge_cases/signal_loss.json` - 信号丢失
- `tests/data/edge_cases/high_noise.json` - 高噪声

### 真实数据集
- `tests/data/real_world/warehouse_patrol.json` - 仓库巡逻
- `tests/data/real_world/office_walk.json` - 办公室行走

## 执行命令

```bash
# 运行所有 P0 测试
pytest tests/ -m "priority_p0"

# 运行特定模块
./scripts/run_targeted_tests.sh fusion

# 运行回归测试
./scripts/run_regression_tests.sh

# 生成覆盖率报告
pytest --cov=indoor_fence --cov-report=html tests/
```

## 测试报告模板

```markdown
## 测试执行报告

**日期**: 2026-03-06
**版本**: v2.1
**执行人**: [姓名]

### 测试结果

| 优先级 | 总数 | 通过 | 失败 | 跳过 |
|--------|------|------|------|------|
| P0 | 20 | 20 | 0 | 0 |
| P1 | 15 | 14 | 1 | 0 |
| P2 | 5 | 4 | 0 | 1 |

### 失败用例

1. **test_auto_fence.py::test_adaptive**
   - 原因: 自适应半径计算错误
   - 影响: 中等
   - 修复: 已创建 issue #123

### 覆盖率

- 总体覆盖率: 87%
- 未覆盖模块: standalone/training_pipeline.py (45%)

### 建议

- 增加 training_pipeline 的单元测试
- 修复 test_adaptive 失败用例
```
