# 00_project_context

## 1. 输入完备性审计（当前真实状态）

| 输入项 | 状态 | 来源 | 处置规则 |
|---|---|---|---|
| `plugin.py` | 可用 | 当前插件目录 | 当前 manifest 指向的唯一主实现，平台契约和核心算法都以它为准 |
| `semantic_slam_plugin.py` | 可用但未接入当前 manifest 主链路 | 当前插件目录 | 视为并行/候选实现，不得默认当成当前插件现状 |
| `manifest.json` | 可用 | 当前插件目录 | 插件标识、声明依赖、默认配置来源 |
| `requirements.txt` | 可用但与 manifest 依赖有漂移 | 当前插件目录 | 真实 Python 安装入口 |
| `run_standalone.py` / `standalone/app.py` | 可用 | 当前插件目录 | 独立服务入口，默认端口 `8084` |
| `demo/run_demo.py` | 可用 | 当前插件目录 | 当前最佳人工回放入口，已本地验证可执行 |
| `tests/test_standalone.py` | 可用 | 当前插件目录 | 当前唯一自动化测试入口，偏 L1 smoke |
| `tests/conftest.py` | 可用 | 当前插件目录 | 路径配置 |
| `configs/default.yaml` | 不存在 | 当前插件目录 | 不得假设已有 YAML 配置；`create_standalone()` 依赖 SDK 缺省兜底 |
| `scripts/` | 不存在 | 当前插件目录 | 当前没有任何脚本化门禁入口 |
| `data/results.db` | 可用但未见主链路写入 | 当前插件目录 | 不得宣称插件当前会持久化处理结果 |

## 2. 治理等级

**基础治理**。

理由：

1. 有 `standalone/`、`run_standalone.py`、`demo/`、`tests/test_standalone.py`。
2. 但缺少 `configs/default.yaml` 和任何 `scripts/`。
3. 当前测试偏 smoke，不覆盖核心语义正确性。
4. 初始化/健康检查/配置注入存在明显契约问题。

## 3. 固定母版规则（跨插件统一）

1. 事实源优先级固定：`plugin.py` > `tests/` > `demo/` > `manifest.json` > `requirements.txt`。
2. 不因目录里有 `semantic_slam_plugin.py` 就把它写成当前 manifest 主链路。
3. 不因插件名叫 SLAM 就默认存在统一 `process()` 契约；当前主业务入口是 `process_point_cloud()`。
4. 没有 `configs/`、没有 `scripts/` 必须如实说明。
5. 平台兼容接口与业务接口要分开描述。

## 4. 当前插件真实结构

```text
slam_mapping/
├── plugin.py
├── semantic_slam_plugin.py
├── manifest.json
├── requirements.txt
├── __main__.py
├── run_standalone.py
├── demo/
│   └── run_demo.py
├── standalone/
│   ├── app.py
│   └── templates/
├── tests/
│   ├── conftest.py
│   └── test_standalone.py
├── data/
│   └── results.db
└── .agent_skills/             # 本轮补齐 00~08
```

## 5. 当前已核验事实

以下事实已本地轻量验证：

1. `Plugin.create_standalone()` 可执行。
2. `python3 -m plugins.slam_mapping.demo.run_demo` 可执行。
3. `python3 -m pytest plugins/slam_mapping/tests/test_standalone.py -q` 本地通过。
4. `process_point_cloud()` 会返回字典结果。
5. `process_point_cloud()` 在未 `init()` 时也能运行。
6. `healthcheck()` 在未初始化和 `shutdown()` 后仍返回 `healthy=True`。
7. 将配置 dict 传给 `init()` 时，会被误当成 `model_registry`，并把 `dl_enabled` 置为 `True`，但真实算法配置不生效。

## 6. AI 自动闭环 vs 人工确认

### 可由 AI 自动闭环

- 维护 `.agent_skills/00~08`
- 新增最小测试和 sanity 脚本
- 校验 import / standalone / demo / pytest smoke
- 识别配置、健康检查、契约漂移问题

### 必须人工确认

- 是否切换到 `semantic_slam_plugin.py` 作为主实现
- 是否引入真实语义分割模型或外部地图服务
- 是否修改平台主入口契约
- 是否启用地图导出到外部路径
- 是否接入真实数据库写入

## 7. 最小可执行校验命令

```bash
# demo 回放
python3 -m plugins.slam_mapping.demo.run_demo

# 当前 smoke 测试
python3 -m pytest plugins/slam_mapping/tests/test_standalone.py -q

# 基础 import + point cloud 处理
python3 - <<'PY'
import sys
from pathlib import Path
import numpy as np
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.slam_mapping.plugin import SLAMMappingPlugin
p = SLAMMappingPlugin()
print(p.process_point_cloud(np.random.randn(100, 3)))
PY
```
