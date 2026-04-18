# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`
2. 禁止在生产主链路新增 `print()`
3. 禁止把并行实现写成当前 manifest 现状
4. 禁止修改合同却不补回归验证
5. 禁止把 smoke 通过当成算法正确性证明

## 2. 本项目高优先级审计项

1. **`init(dict)` 配置误判**
   - 当前签名是 `init(model_registry=None)`
   - 传入非空 dict 会被当成 registry
   - `dl_enabled=True`，但真实配置仍未应用
2. **健康检查失真**
   - `healthcheck()` 始终返回 `healthy=True`
   - `shutdown()` 不会清理 `_is_initialized` / 状态
3. **未初始化仍可处理**
   - `process_point_cloud()` 当前没有初始化保护
4. **业务入口非标准**
   - 当前没有统一 `process()`，只有 `process_point_cloud()`
5. **依赖声明漂移**
   - manifest 声明 `scipy`
   - `requirements.txt` 未列出
6. **版本/实现漂移**
   - `manifest.json` / `plugin.py` 是 `1.0.0`
   - `__init__.py` 导出语义 SLAM 并标 `2.0.0`
7. **demo 与真实返回字段漂移**
   - demo 打印 `obstacles_detected`
   - 当前 `process_point_cloud()` 不返回该字段
8. **数据库假象**
   - `data/results.db` 存在
   - 当前未见主链路写入

## 3. 反模式清单

| 反模式 | 检测方法 | 严重度 |
|---|---|---|
| init 把配置当 registry | 复现 `init({'voxel_size': 0.2})` | 阻断 |
| healthcheck 恒 OK | 复现未初始化 / shutdown 后状态 | 高 |
| 未初始化仍处理 | 直接实例化后调用 `process_point_cloud()` | 高 |
| 依赖漂移 | 对比 `manifest.json` 与 `requirements.txt` | 中 |
| 版本漂移 | 对比 `manifest.json`、`plugin.py`、`__init__.py` | 中 |
| 并行实现误用 | `rg 'SemanticSLAMPlugin|semantic_slam_plugin'` | 中 |
| 假持久化 | `rg 'results.db|sqlite|detection_results'` | 中 |

## 4. 审计命令

```bash
# 当前 smoke
python3 -m pytest plugins/slam_mapping/tests/test_standalone.py -q

# demo 回放
python3 -m plugins.slam_mapping.demo.run_demo

# 契约问题复现
python3 - <<'PY'
import sys
from pathlib import Path
import numpy as np
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.slam_mapping.plugin import SLAMMappingPlugin
p = SLAMMappingPlugin()
print('process_without_init', p.process_point_cloud(np.random.randn(20, 3)))
print('health_no_init', p.healthcheck())
p.init({'voxel_size': 0.2})
print('dl_enabled', p.dl_enabled, 'voxel_size', p.point_processor.voxel_size)
p.shutdown()
print('health_after_shutdown', p.healthcheck())
PY
```

## 5. 当前阻断/高风险问题

1. `init(dict)` 配置语义错误。
2. `healthcheck()` 与真实状态不一致。
3. `process_point_cloud()` 未受初始化状态保护。

## 6. 当前建议级问题

1. demo 与真实返回字段不一致。
2. `semantic_slam_plugin.py` 与当前主实现版本漂移，容易误导阅读者。
3. `requirements.txt` 与 manifest 依赖列表应同步。
