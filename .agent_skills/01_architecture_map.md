# 01 — 架构与目录映射

## 分层架构

```
┌─────────────────────────────────────────────┐
│  apps/          应用层 (API Server / UI / Dashboard)  │
├─────────────────────────────────────────────┤
│  plugins/       插件层 (19 个功能插件)                  │
├─────────────────────────────────────────────┤
│  platform_core/ 平台内核                               │
│  ├── plugin_manager/   插件生命周期                     │
│  ├── scheduler/        任务调度引擎                     │
│  ├── schema/           统一数据模型                     │
│  ├── evidence/         证据链管理                       │
│  ├── replay/           确定性回放                       │
│  ├── device_adapter/   设备适配层                       │
│  └── logging/          统一日志                         │
├─────────────────────────────────────────────┤
│  ai_models/     AI 模型层 (DL 模型 + 训练管道)          │
├─────────────────────────────────────────────┤
│  darkbreaker_sdk/  SDK 封装层                           │
└─────────────────────────────────────────────┘
```

## 关键目录职责

| 目录 | 职责 | 关键文件 |
|---|---|---|
| `apps/` | HTTP 入口, 路由注册 | `main.py`, `api_server.py` |
| `platform_core/plugin_manager/` | 插件发现/加载/生命周期 | `base.py`, `enhanced_base.py`, `dl_integration.py` |
| `platform_core/schema/` | Pydantic 数据模型 | UnifiedResult 等 |
| `platform_core/scheduler/` | 定时/事件驱动任务调度 | — |
| `platform_core/evidence/` | 证据链 CRUD | — |
| `configs/` | YAML 配置文件 | `platform.yaml`, `plugins_config.yaml` |
| `ai_models/deep_learning/` | DL 模型封装 | `yolov8_vit.py`, `gl_translstm.py` 等 |
| `ai_models/training/` | 训练管道 | `data_pipeline.py` |
| `tests/` | 测试套件 | `integration/`, `unit/`, `sdk/` |

## 数据流

```
设备/传感器 → device_adapter → scheduler(任务) → plugin.process()
    → UnifiedResult → evidence(存证) → alarm_manager → dashboard(展示)
```
