# 03 — 技术栈与依赖

## 运行时

| 层 | 技术 | 版本约束 |
|---|---|---|
| 语言 | Python | >=3.10, <3.12 |
| Web 框架 | FastAPI | >=0.104 |
| 数据校验 | Pydantic v2 | >=2.5 |
| ASGI | Uvicorn | >=0.24 |
| 模板 | Jinja2 | >=3.1 |
| HTTP 客户端 | httpx | >=0.25 |

## 视觉 / AI

| 库 | 用途 | 必选? |
|---|---|---|
| OpenCV (headless) | 图像处理 | ✓ |
| NumPy | 数组运算 | ✓ |
| Pillow | 图像 I/O | ✓ |
| PyTorch | DL 推理/训练 | 可选 (gpu extra) |
| Ultralytics | YOLOv8 | 可选 |
| librosa | 声学特征 | 可选 (advanced extra) |
| open3d | 点云/SLAM | 可选 (advanced extra) |
| spectral | 高光谱 | 可选 (advanced extra) |

## 开发工具链

| 工具 | 用途 | 配置位置 |
|---|---|---|
| black | 格式化 (line-length=100) | pyproject.toml |
| ruff | lint | pyproject.toml |
| mypy | 类型检查 | pyproject.toml |
| pytest | 测试 (asyncio_mode=auto) | pyproject.toml |
| pre-commit | Git 钩子 | — |

## 构建 / 打包

| 工具 | 用途 |
|---|---|
| hatchling | PEP 517 构建后端 |
| PyInstaller | 单文件打包 |
| Nuitka | AOT 编译 (可选) |

## 安装命令

```bash
pip install -e .            # 基础
pip install -e ".[gpu]"     # + PyTorch
pip install -e ".[advanced]"# + 声学/高光谱/SLAM
pip install -e ".[dev]"     # + 开发工具
```
