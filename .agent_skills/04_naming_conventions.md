# 04 — 命名与编码规范

## Python 代码

| 元素 | 规范 | 示例 |
|---|---|---|
| 模块 | snake_case | `gas_detection.py` |
| 类 | PascalCase | `EnhancedThermalAnalyzer` |
| 函数/方法 | snake_case | `detect_anomaly()` |
| 常量 | UPPER_SNAKE | `MAX_RETRY_COUNT` |
| 私有 | 前缀 `_` | `_internal_state` |
| 异步方法 | 后缀 `_async` 或前缀 `async_` | `process_async()` |

## 目录命名

| 类型 | 规范 | 示例 |
|---|---|---|
| 插件目录 | snake_case，业务名词 | `transformer_inspection/` |
| 配置目录 | `configs/` (复数) | — |
| 测试目录 | `tests/` (复数) | — |

## 插件四件套命名

```
plugins/<plugin_name>/
├── __init__.py
├── plugin.py          # 固定名 — 入口
├── manifest.json      # 固定名 — 元数据
├── configs/
│   └── default.yaml   # 默认配置
├── tests/
│   └── test_plugin.py # 冒烟测试
└── *.py               # 业务模块 (自由命名)
```

## 格式化

- **black**: line-length=100, target py310/py311
- **ruff**: E/F/W/I/N/UP/B/C4 规则集
- 提交前必须通过 `black --check .` 和 `ruff check .`

## 注释与文档

- 模块顶部: 中文 docstring，标注版本
- 函数/类: 中文说明 + 英文参数类型注解
- TODO 格式: `# TODO(姓名): 描述`
- 技术术语保留英文 (FastAPI, YOLO, SLAM 等)

## Git Commit

- 简洁英文，一句话 why
- 格式: `<type>: <description>`
- type: feat / fix / refactor / docs / test / chore
