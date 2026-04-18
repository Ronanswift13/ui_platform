# 06 — 测试策略

## 测试框架

- **pytest** (asyncio_mode=auto)
- testpaths: `tests/`
- 各插件自带: `plugins/<name>/tests/`

## 测试层级

| 层级 | 位置 | 职责 |
|---|---|---|
| 单元测试 | `tests/unit/` | 纯函数/类的隔离测试 |
| 插件冒烟 | `plugins/<name>/tests/` | 单插件 init → process → 验证输出 |
| 集成测试 | `tests/integration/` | 插件间交互、平台加载 |
| SDK 测试 | `tests/sdk/` | darkbreaker_sdk 接口 |
| 全插件扫描 | `plugins/test_all_plugins.py` | 批量加载检查 |
| 验收测试 | `tests/test_acceptance_closure.py` | 五条线验收 |

## 运行命令

```bash
# 全部测试
pytest tests/ -v

# 单个插件
pytest plugins/transformer_inspection/tests/ -v

# 仅单元测试
pytest tests/unit/ -v

# 覆盖率
pytest tests/ --cov=platform_core --cov=plugins --cov-report=html
```

## 测试规范

1. 每个 L3 插件必须有 `tests/test_plugin.py`
2. 测试文件命名: `test_*.py`
3. 异步测试直接用 `async def test_xxx()` (asyncio_mode=auto)
4. mock 外部设备/网络调用，不依赖物理硬件
5. 测试数据放在 `tests/replay_data/` 或插件的 `tests/fixtures/`

## CI 卡点

- `black --check .` 通过
- `ruff check .` 通过
- `pytest tests/` 全绿
- 新插件 PR 必须附带冒烟测试
