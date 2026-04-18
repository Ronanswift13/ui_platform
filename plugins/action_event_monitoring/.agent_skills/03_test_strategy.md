# 03_test_strategy

## 1. 当前真实现状

1. 当前目录已有 `tests/`：config/process/entrypoints/standalone 合同测试。
2. 当前目录已有 `scripts/run_sanity_checks.sh`，作为正式最小质量门。
3. 当前目录已有 `standalone/app.py`、`run_standalone.py`、`__main__.py`、`demo/run_demo.py`。
4. 当前没有 targeted / regression / coverage 质量门；不得伪造覆盖率或完整回归结论。

## 2. 当前可执行验证方式

### 2.1 最小 sanity 校验

`scripts/run_sanity_checks.sh` 是当前正式最小校验入口。

该校验只覆盖：

1. `manifest.json` 可解析
2. `configs/default.yaml` 可解析
3. `plugin.py` 可导入
4. `init(default_config)` 成功
5. `process(sample_event)` 成功
6. `get_status()` 统计字段存在
7. `Plugin` 别名、manifest standalone 端口 `8097`、本地 entrypoints 存在
8. pytest 合同测试全部通过

### 2.2 不应误报的能力

以下能力当前仍不存在：

1. 覆盖率统计
2. regression/quality gate 脚本
3. UI/dashboard/cockpit 前端入口
4. 真实协议服务依赖下的 smoke
5. `.claude/commands/` 本地命令入口

## 3. 后续补齐路线（建议顺序）

### 当前已覆盖

- `tests/test_config_contract.py`
- `tests/test_process_contract.py`
- `tests/test_entrypoints.py`
- `tests/test_standalone.py`
- `scripts/run_sanity_checks.sh`

### P2：协议与拓扑边界

建议新增：

- `tests/test_topology_loading.py`
- `tests/test_protocol_init_fallback.py`
- `tests/test_iec104_callback_normalization.py`
- `tests/test_iec61850_callback_normalization.py`

### P3：质量门禁脚本

有了基础测试后，再考虑新增：

- `scripts/run_targeted_tests.sh`
- `scripts/run_quality_gate.sh`

在此之前不要伪造这些文件。

## 4. 当前阻断条件

1. 修改了 `plugin.py` 却没有跑 sanity 脚本。
2. 把当前合同测试宣称为“覆盖率达标”或“完整回归通过”。
3. 把历史文档中的测试矩阵当成当前本地已存在测试。
4. 把 standalone smoke 宣称为真实 OPC UA / IEC104 / IEC61850 协议联调通过。
