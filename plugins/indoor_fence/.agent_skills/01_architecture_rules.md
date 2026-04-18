# 01 架构规则

## 1. When to use
- 修改 `import`、目录结构、公开入口或 standalone 路由
- 在 `plugin.py`、`core/`、`adapters/`、`detection/`、`standalone/` 之间移动逻辑
- 需要判断某个改动应该落在主插件运行时还是 V3 演练链路

## 2. Inputs
- 代码 diff
- 受影响文件清单
- `manifest.json`、`plugin.py`、`core/config_manager.py`

## 3. Outputs
- 架构合规判定
- 依赖方向与接口面说明
- 需要补的测试/文档清单

## 4. Hard Constraints
- `core/` 禁止依赖 `adapters/` 或 `standalone/`
- `protocols.py` 只能承载跨层数据模型；不得反向依赖 `plugin.py`、`adapters/` 或 web 框架
- `plugin.py` 负责 orchestration、配置热更新、standalone route 注册；复杂检测/融合逻辑不要继续堆进 `plugin.py`
- 新的检测/姿态/行为能力优先放在 `detection/*`，不要默认继续扩写 legacy `detector.py`
- 对外可见的配置字段若变化，必须同步检查 `configs/default.yaml`、`manifest.json`、`tests/test_config_updates.py`
- standalone HTTP 面只能通过 `plugin.py::get_standalone_routes()` 扩展；不要把路由定义散落到随机模块

## 5. Algorithm / Logic Contract

### 依赖方向

```text
configs/default.yaml + zone.yaml + scenario.json
        ↓
core/config_manager.py / ZoneConfigLoader
        ↓
adapters/* + detection/* + core/*
        ↓
plugin.py
        ↓
standalone routes / RecognitionResult / Alarm

protocols.py + core/fusion/* + core/tracking/* + standalone/realtime_pipeline.py
        └── 属于 V3 演练链路，和主插件运行时并存
```

### 目录职责
- `plugin.py`
  - 保持 SDK 生命周期、`create_standalone()`、`Plugin = IndoorFencePlugin`
  - 注册 `/api/indoor-fence/*` 路由
  - 处理配置更新回滚、zone 更新落盘策略
- `core/`
  - 纯算法、状态、规则、配置解析
  - 不做硬件 I/O 和 web I/O
- `adapters/`
  - 硬件/模拟/回放适配
  - 所有 live/simulated/replay 模式切换都经 `BaseAdapter`
- `detection/`
  - 新版 YOLO / Pose / Behavior / Auto Fence 逻辑
- `standalone/`
  - Web UI、视频流、仿真、训练 scaffold、录制回放
  - 可以依赖 `plugin.py` 或 `adapters/`，但不反向侵入 `core/`

### 特殊边界
- `detector.py`、`core/tracker_v3.py`、`core/enhanced_tracking.py` 当前更像 legacy / 历史兼容模块；除非任务明确要求，否则不要把新功能落回这些文件
- `run_standalone.py` 的 venv guard 属于启动契约，不要移到导入之后
- route 变化必须同步检查模板和前端脚本是否仍引用同名路径/元素

## 6. Validation Rules

```bash
# core 不得依赖 adapters / standalone
rg -n "from .*adapters|import .*adapters|from .*standalone|import .*standalone" core/

# 主入口契约
rg -n "def create_standalone|Plugin = IndoorFencePlugin" plugin.py

# standalone 核心路由是否还在
rg -n "/api/indoor-fence/(config|zones|events|stream|snapshot|tracking)" plugin.py

# 启动 guard 是否还在
rg -n "PROJECT_VENV_PYTHON|os\\.execv" run_standalone.py
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| `core/` 反向 import `adapters/` | 算法层不可测 | 回退到数据注入/协议模型 |
| 新路由绕过 `get_standalone_routes()` | standalone surface 漂移 | 路由统一收口到 `plugin.py` |
| 新检测逻辑写进 `detector.py` | 主链路与 V3 链路继续分叉 | 优先迁入 `detection/*` |
| 配置字段变更未同步 schema/test | runtime update 崩溃 | 同步更新 `manifest.json` 与配置测试 |
| 启动 guard 被删 | 系统 Python 启动缺依赖 | 立即恢复 venv 切换逻辑 |

## 8. Required Tests
- `tests/test_standalone.py`
- `tests/test_config_updates.py`
- `tests/test_api_routes.py`
- `tests/test_integration.py`

