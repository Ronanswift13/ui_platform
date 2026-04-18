# 03_test_strategy

## 1. 固定母版规则

1. 每个硬约束至少对应 1 个自动化测试或 1 个明确人工回放入口。
2. 每个 bug 修复必须新增防回归测试。
3. 已有 smoke 测试不等于功能正确性测试。
4. 测试结论必须区分“已本地执行”和“建议补齐”。

## 2. 当前测试现状

| 文件 | 层级 | 内容 |
|---|---|---|
| `tests/test_standalone.py` | L1 smoke | `create_standalone`、`healthcheck`、`infer`、`process_point_cloud`、`runner` 创建 |
| `tests/conftest.py` | — | 路径配置 |

本地已验证：

```bash
python3 -m pytest plugins/slam_mapping/tests/test_standalone.py -q
```

当前可通过，但带 1 个 pytest config warning。

## 3. 当前测试缺口

1. 未验证 `shutdown()` / `healthcheck()` 语义一致性。
2. 未验证未初始化时是否应允许处理点云。
3. 未验证 `init(dict_config)` 误判为 model registry 的问题。
4. 未验证 `process_point_cloud()` 的语义正确性，只验证“返回 dict”。
5. 未验证路径规划、沉降监测、地图导出、设备查询。
6. 未验证 `semantic_slam_plugin.py` 与主实现边界。

## 4. 分层建议

### L0 Targeted

优先覆盖：

1. `PointCloudProcessor.preprocess()`
2. `segment_ground()`
3. `ICPMatcher.align()`
4. `PathPlanner.plan_path()`
5. `SubsidenceMonitor` 告警与趋势
6. `init(dict)` 配置误判
7. `healthcheck()` / `shutdown()` 一致性

### L1 Integration

优先覆盖：

1. `create_standalone()` -> `process_point_cloud()` -> `get_status()`
2. `demo/run_demo.py`
3. `tests/test_standalone.py`
4. `get_map_data('2d')`
5. `plan_inspection_path()` 基础正/反例

### L2 Manual Service

仅在需要验证 standalone 页面时执行：

1. `python3 -m plugins.slam_mapping`
2. 打开 `http://localhost:8084`
3. 检查服务启动与模板加载

## 5. 最小验证命令

```bash
# 当前 smoke 测试
python3 -m pytest plugins/slam_mapping/tests/test_standalone.py -q

# demo 回放
python3 -m plugins.slam_mapping.demo.run_demo

# 初始化边界
python3 - <<'PY'
import sys
from pathlib import Path
import numpy as np
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.slam_mapping.plugin import SLAMMappingPlugin
p = SLAMMappingPlugin()
print(p.process_point_cloud(np.random.randn(50, 3)))
print(p.healthcheck())
PY
```

## 6. 待补齐测试文件（优先级）

1. `tests/test_plugin_contract.py`
   - `init()` / `shutdown()` / `healthcheck()`
   - 未初始化时的行为
   - `init(dict)` 配置误判
2. `tests/test_point_cloud_pipeline.py`
   - preprocess / ground segmentation / empty point cloud
3. `tests/test_services.py`
   - register / locate / nearest / plan path / map export

## 7. 后续最值得补的第一个脚本

建议先补：

`scripts/run_sanity_checks.sh`

理由：

1. 当前已有最小测试，但没有统一脚本入口。
2. 它可以串起 `pytest smoke + demo 回放 + 初始化边界检查`。
3. 它还能把当前最重要的契约问题固定成显式检查点：`init(dict)` 误判、`shutdown()` 后健康检查仍为 OK。
