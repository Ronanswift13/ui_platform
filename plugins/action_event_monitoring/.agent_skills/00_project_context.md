# 00_project_context

## 1. 输入完备性审计（当前真实状态）

| 输入项 | 状态 | 来源 | 处置规则 |
|---|---|---|---|
| `plugin.py` | 可用 | 当前插件目录 | 当前插件唯一的核心实现文件，生命周期与处理契约以此为准 |
| `manifest.json` | 可用 | 当前插件目录 | 插件标识、能力、依赖、配置文件路径的权威来源 |
| `configs/default.yaml` | 可用 | 当前插件目录 | 协议、订阅、分析、告警的运行参数来源 |
| `configs/topology/` | 可用 | 当前插件目录 | 厂站、间隔、设备、信号点拓扑来源 |
| `requirements.txt` | 可用 | 当前插件目录 | 本地依赖声明 |
| `demo/run_demo.py` | 可用 | 当前插件目录 | 最小本地事件回放入口 |
| `__main__.py` / `run_standalone.py` | 可用 | 当前插件目录 | 模块入口与 standalone 入口 |
| `tests/` | 可用 | 当前插件目录 | config/process/entrypoints/standalone 合同测试 |
| `scripts/` | 可用 | 当前插件目录 | `scripts/run_sanity_checks.sh` 为正式最小质量门 |
| `standalone/` | 可用 | 当前插件目录 | 本地 standalone app 与 `/api/action-event/smoke` |
| `.claude/commands/` | 不存在 | 当前插件目录 | 不得引用 `/implement`、`/repair` 等本地命令文件 |

## 2. 固定母版规则（跨插件通用）

1. 真实代码优先于历史笔记：`plugin.py` 和 `manifest.json` 是单一事实源。
2. 缺什么就如实写什么：当前已有 tests / standalone / demo / entrypoints；仍没有 `.claude/commands/`、UI/cockpit 接线和真实协议联调环境。
3. 最小治理先于伪完整治理：先保证 import/init/process 可验证，再谈复杂门禁。
4. 协议接入类插件默认包含外部系统依赖，校验必须区分“本地可运行部分”和“外部适配器依赖部分”。

## 3. 当前插件的真实结构

```text
action_event_monitoring/
├── plugin.py                      # 单文件插件实现
├── manifest.json                  # 插件元数据
├── requirements.txt               # 本地依赖声明
├── __main__.py                    # python -m plugins.action_event_monitoring
├── run_standalone.py              # 顶层 standalone 启动入口
├── demo/run_demo.py               # 最小回放入口
├── configs/default.yaml           # 协议/订阅/分析/告警配置
├── configs/topology/*.yaml        # 厂站拓扑与信号映射
├── standalone/app.py              # StandalonePluginRunner 入口，端口 8097
├── tests/
│   ├── test_config_contract.py
│   ├── test_entrypoints.py
│   ├── test_process_contract.py
│   └── test_standalone.py
├── scripts/run_sanity_checks.sh
└── .agent_skills/                 # 本轮补齐 00~08
```

## 4. 当前实现事实（经本地核验）

1. 插件主类是 `ActionEventMonitoringPlugin`，并已提供 `Plugin = ActionEventMonitoringPlugin` 标准别名。
2. 当前生命周期是：`init()` -> `start()` -> `process()` -> `stop()` -> `shutdown()`。
3. `create_standalone()` 会加载 `configs/default.yaml` 并初始化插件。
4. `process()` 支持统一事件壳：`events`、`signal_changes`、`state_change_events`、`protocol_ingested_data`，并兼容单条完整事件 dict。
5. 本地实测：
   - `plugin.py` 可导入。
   - 默认配置可解析。
   - `init(default_config)` 返回 `True`。
   - `process(sample_event)` 返回统一 `success/status/label/value/confidence/metadata/results` 壳，并保留 `stored_event_ids`、`analysis_triggered`、`analysis_result`。
   - standalone smoke 可通过 `/api/action-event/smoke` 提交模拟样本。
6. 全局统一层已接入：
   - `platform_core/plugin_manager/installer.py`：`category=monitoring`，`port=8097`。
   - `tests/integration/test_plugin_standalone.py`：已包含 `action_event_monitoring`。
   - `tests/integration/test_all_standalone_boot.py`：已包含 `("action_event_monitoring", 8097)`。
7. `start()` 在默认协议配置下，即使本地缺少真实外部环境，也不得作为“协议采集正常”证明；本地 smoke 只证明模拟输入闭环。

## 5. AI 自动闭环 vs 人工确认

### 可由 AI 自动闭环

- 维护 `.agent_skills/00~08`
- 维护最小 sanity 脚本、pytest 合同测试与 standalone smoke
- 校验 manifest / config / import / init / process / entrypoints / standalone 这条最小链路
- 识别并标注当前实现风险

### 必须人工确认

- 是否要把插件升级为 SDK 标准 `BasePlugin` 形态
- 是否要引入真实协议联调环境（OPC UA / IEC104 / IEC61850）
- 是否要新增本地 API 层、人工复核接口、候选事件接口
- 是否要将当前单文件插件拆分为 `services/` 等更完整结构
- 是否要把 UI / dashboard / cockpit 统一入口接线到前端目录

## 6. 当前治理级别判断依据

以下事实说明它当前已进入“标准治理基线”，但尚未达到完整平台 UI 闭环：

1. 已有 `tests/`、`standalone/`、`demo/`、`requirements.txt`、`__main__.py`、`run_standalone.py`。
2. 已有全局 installer 分类/端口映射和 integration standalone 清单。
3. 无 `.claude/commands/`。
4. 核心实现仍集中在单个 `plugin.py`。
5. 协议接入依赖外部驱动与运行环境；本地 smoke 不依赖真实协议服务。

升级到“高频开发级”至少需要：真实协议适配 smoke、UI/cockpit 接线、CandidateEvent/人工复核 API 是否落地的明确决策。
