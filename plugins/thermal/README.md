# thermal - 占位态

当前状态：**未落地占位态**，不具备 skill 化条件。

## 现有文件

| 文件 | 说明 |
|---|---|
| `__init__.py` | 模块导出（EnhancedThermalAnalyzer 等） |
| `enhanced_thermal_analyzer.py` | 热成像分析单体模块 |

## 缺失的最小治理文件

启动 `.agent_skills/` 建设前，需补齐以下文件：

1. **`plugin.py`** - 标准插件入口，实现 `register() / start() / stop()` 生命周期
2. **`manifest.json`** - 插件元数据（名称、版本、依赖声明）
3. **`configs/default.yaml`** - 默认配置
4. **`tests/`** - 至少一个冒烟测试（`test_plugin.py`）

## 升级条件

满足以下全部条件时，可从占位态升级为最小治理态：

- [ ] `plugin.py` 已实现并可被平台 `PluginManager` 加载
- [ ] `manifest.json` 存在且通过 schema 校验
- [ ] `configs/default.yaml` 存在且 plugin.py 能正确读取
- [ ] `tests/test_plugin.py` 存在且 `pytest` 通过
- [ ] 在 `.enabled_plugins.json` 中注册
