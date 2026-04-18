# PROJECT_CARD — DarkBreaker (破夜绘明)

> 单文件项目身份证，供人类与 AI 快速定位项目边界。

| 字段 | 值 |
|---|---|
| 项目名 | powerstation-monitor / DarkBreaker / 破夜绘明 |
| 一句话定位 | 输变电站 AI 自主巡视与监测平台 |
| 版本 | 3.0.0 |
| Python | >=3.10, <3.12 |
| 许可证 | Proprietary |
| 入口 | `python run.py` / `apps.main:main` |
| 包管理 | `pyproject.toml` (hatchling) |
| 核心依赖 | FastAPI · Pydantic v2 · OpenCV · PyTorch · Ultralytics |
| 架构风格 | 分层 + 插件化 + 微服务 |

## 目录骨架 (一级)

```
DarkBreaker/
├── apps/              # 应用入口 (API/UI/Dashboard Server)
├── platform_core/     # 平台内核 (调度/证据链/设备适配/插件管理)
├── plugins/           # 19 个功能插件 (详见 .agent_skills/02)
├── ai_models/         # 深度学习模型 + 训练管道
├── configs/           # 全局与插件配置
├── tests/             # 集成/单元/SDK 测试
├── training/          # 训练脚本与数据管道
├── darkbreaker_sdk/   # SDK 封装
├── data/              # 数据资产
├── docs/              # 文档
├── evidence/          # 证据链存储
├── models/            # 模型权重/配置
├── mlops/             # MLOps 管道
├── scripts/           # 运维脚本
└── logs/ / output/ / exports/  # 运行时产物
```

## 插件验收五条线 + DL

1. **可运行** — 按平台接口接入即跑
2. **可回放** — 给定回放数据, 结果可复现
3. **可解释** — 输出 bbox/关键点/置信度/失败原因码
4. **可追溯** — 输出 model_version + code_hash
5. **可维护** — README + 配置样例 + 最小单测
6. **深度学习** — 支持 DL 模型集成 (V3.0)

## 治理文件索引

| 文件 | 用途 |
|---|---|
| `PROJECT_CARD.md` | 本文件 — 项目身份证 |
| `CLAUDE.md` | AI Agent 行为契约与路由指南 |
| `.agent_skills/` | 分片知识库 (00-08) |
| `README.md` | 面向人类的完整项目文档 |
| `pyproject.toml` | 构建/依赖/工具链配置 |
