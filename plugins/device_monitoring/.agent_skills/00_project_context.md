# 00_project_context

## 0. 治理等级

**标准治理基线** — 当前已补齐 detector 层边界测试、process 合同测试、现场式设备数据 replay fixture，以及 sanity/targeted 两类脚本入口。健康指数仍是确定性公式，第二阶段可再接入真实遥测和模型预测。

## 1. 固定母版规则（跨插件统一）

1. **接口契约冻结**：插件必须实现 `init/detect/healthcheck`，签名与 SDK 一致。
2. **配置优先**：阈值与权重必须来自 `configs/default.yaml`，不允许在主路径硬编码。
3. **降级可观测**：降级路径必须输出 `failure_reason` + `metadata`。
4. **输出可校验**：`health_index` ∈ `[0, 100]`，`anomaly_score` ∈ `[0, 1]`。
5. **测试分层**：至少包含 L0（单测）、L1（集成）分层执行入口。

## 2. 本项目差异规则（device_monitoring 专属）

1. **非图像插件**：输入为设备遥测数据 `device_readings`（CPU 温度、内存、网络质量等），不处理图像帧。`infer()` 返回空列表，核心入口为 `detect()`。
2. **健康指数为确定性公式**：`DeviceHealthCalculator` 基于加权扣分模型，非 ML 推理。
3. **故障预测可选**：`prediction.enabled` 控制是否启用统计/模型预测，默认 statistical 方法。
4. **维护工单自动生成**：健康指数低于 `health_alarm` 时自动创建 `MaintenanceTicket`，优先级由配置映射。
5. **设备列表配置化**：`managed_devices` 在 YAML 中定义，`scan_devices()` 为模拟扫描（生成随机指标）。

## 3. 当前目录与职责边界

```
device_monitoring/
├── plugin.py                   # SDK 适配层（init、detect 编排、告警生成、scan_devices 模拟）
├── detector.py                 # 算法层（DeviceHealthCalculator 健康计算、
│                               #   DeviceMonitorDetector 异常检测 + 故障预测 + 工单生成）
├── configs/default.yaml        # 运行参数唯一来源（权重、阈值、设备列表、协议配置）
├── manifest.json               # 插件注册信息
├── tests/
│   ├── test_config_contract.py # manifest/YAML/配置契约
│   ├── test_process_contract.py# 统一时序输入输出壳
│   ├── test_detector.py        # L0 健康指数/工单阈值边界
│   ├── test_device_replay.py   # 现场式遥测 fixture 回放
│   └── test_standalone.py      # L1 standalone smoke
├── scripts/
│   ├── run_sanity_checks.sh    # contract + standalone 最小门禁
│   └── run_targeted_tests.sh   # detector + replay 快速回归
├── standalone/                 # 独立运行 Web 仪表盘
│   ├── app.py
│   └── templates/
├── demo/run_demo.py            # 演示脚本
└── .agent_skills/              # AI 代理规则（本目录）
```

## 4. 模块职责边界

| 模块 | 职责 | 不应包含 |
|------|------|----------|
| `plugin.py` | SDK 适配、配置加载、detect() 编排、告警生成、scan_devices 模拟、UI 配置 | 健康计算公式、异常评分算法 |
| `detector.py` | 健康指数加权计算、异常评分、状态判定、故障预测、工单生成、历史记录 | SDK schema、告警级别映射 |
| `standalone/app.py` | Web 仪表盘运行 | 算法逻辑 |

### 关键数据流

```
device_readings (List[Dict])
    → plugin.detect()
        → detector.detect()
            → DeviceHealthCalculator.calculate() → health_index, issues, recs
            → _calc_anomaly() → anomaly_score
            → _predict_failure() → predicted_failure (可选)
            → _create_ticket() → MaintenanceTicket (条件触发)
        → plugin._gen_alarms() → alarms
    → 输出 Dict
```

## 5. AI 自动闭环 vs 人工确认

### 可自动闭环
- `.agent_skills/` 规则维护
- `tests/` 测试补齐与执行
- 配置键一致性检查
- 反模式扫描

### 需人工确认
- 健康指数权重调整（`health_weights`）
- 告警阈值变更（影响工单触发策略）
- 工单优先级映射（`ticket_priority_map`）
- 协议启用（SNMP/Modbus，涉及网络访问）
- `managed_devices` 列表变更

## 6. 可执行校验命令

```bash
# 配置可解析
python -c "import yaml; yaml.safe_load(open('plugins/device_monitoring/configs/default.yaml'))"

# 插件可导入
python -c "from plugins.device_monitoring.plugin import DeviceMonitoringPlugin; print(DeviceMonitoringPlugin.__name__)"

# 最小质量门
cd plugins/device_monitoring && ./scripts/run_sanity_checks.sh

# detector/replay targeted
cd plugins/device_monitoring && ./scripts/run_targeted_tests.sh

# 全量插件测试
python -m pytest plugins/device_monitoring/tests/ -q

# 性能基准
python -m plugins.device_monitoring.scripts.benchmark
```
