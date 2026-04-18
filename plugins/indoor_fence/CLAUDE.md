# Claude 开发指南 - Indoor Fence Plugin

## 项目概述

`indoor_fence` 是 DarkBreaker 的室内电子围栏插件。
当前稳定交付面是 `plugin.py` 主插件运行时；仓内同时保留 V3 多传感器演练链路（`protocols.py`、`detection/*`、`core/fusion/*`、`core/tracking/*`、`standalone/realtime_pipeline.py`）。

## 工作流

| 阶段 | 命令 | 说明 |
|------|------|------|
| 起盘 | `/bootstrap` | 初始化环境、读取上下文 |
| 实现 | `/implement` | 先读路由表，再 TDD、实现、验证 |
| 修复 | `/repair` | 根因分析、失败测试、最小修复、知识回灌 |
| 审计 | `/audit` | 质量闸门 + 人工分级审计 |
| 扩散 | `/propagate` | 大范围受控传播 |

**强制规则：** `/implement`、`/repair`、`/audit` 必须先读取 `.agent_skills/08_task_routing.md`。  
若任务沉淀出通用规则或复发模式，必须回写 `.agent_skills/04_quality_audit.md` 与 `.agent_skills/07_learning_log.md`。

## 启动命令

```bash
python run_standalone.py
python -m plugins.indoor_fence.standalone.app
./scripts/run_targeted_tests.sh plugin
./scripts/run_regression_tests.sh
./scripts/run_quality_gate.sh
```

## 技能文件索引

| 文件 | 用途 |
|------|------|
| `.agent_skills/00_project_context.md` | 项目真实入口、链路划分、配置来源 |
| `.agent_skills/01_architecture_rules.md` | 架构边界与路由收口规则 |
| `.agent_skills/02_algorithm_contract.md` | 输出、fallback、rollback 契约 |
| `.agent_skills/03_test_strategy.md` | 最近测试与 targeted 模块映射 |
| `.agent_skills/04_quality_audit.md` | blocker / high-risk / debt 审计清单 |
| `.agent_skills/05_security_boundary.md` | 路径、日志、网络、启动 guard 边界 |
| `.agent_skills/06_refactor_policy.md` | 受控扩散顺序与重构边界 |
| `.agent_skills/07_learning_log.md` | 根因经验回灌 |
| `.agent_skills/08_task_routing.md` | implement / repair / audit 统一路由表 |

## targeted 模块

| 命令 | 说明 |
|------|------|
| `plugin` | 插件入口、config / zone 更新 |
| `adapters` | Camera / LiDAR / UWB / IMU / BaseAdapter / Simulator |
| `detection` | YOLO / Pose / Behavior / Auto Fence |
| `fusion` | EKF / fusion v3 / NLOS / protocols |
| `logic` | 规则、状态机、实时跟踪 |
| `standalone` | API routes / stream / recorder / replayer / training / renderer |
| `integration` | 插件全链路与 scenario 集成 |
| `all` | 非 regression 全量快速门禁 |

## 关键原则

1. 先判定改动属于主插件运行时还是 V3 演练链路。
2. 新检测能力默认落在 `detection/*`，不要继续堆到 `detector.py`。
3. standalone 新接口统一经 `plugin.py::get_standalone_routes()` 暴露。
4. 配置更新必须保持 validate -> apply -> rollback。
5. 任何变更都先补最近测试，再跑 targeted / regression / quality gate。

