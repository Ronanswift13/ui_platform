# 输变电激光星芒破夜绘明监测平台

高智能化输变电站自主巡视与监测系统

## 项目概述

本平台采用**分层 + 插件化**架构，将复杂系统拆解为5个高内聚、低耦合的独立功能模块，支持各博士生算法以"插件"方式直接导入集成。

### 核心特性

- **模块化**: 五大业务模块独立开发、独立部署
- **标准化**: 统一输入输出接口（JSON）、统一文件结构
- **可调度**: 基于任务引擎的算法动态调用
- **可追溯**: 完整证据链（原图/ROI/结果/置信度/时间戳/模型版本）
- **可回放**: 确定性回放，结果可复现

## 快速开始

### 环境要求

- Python 3.10 或 3.11
- 支持 Windows / macOS / Linux

### 安装

```bash
# 克隆项目
git clone <repository_url>
cd 破夜绘明激光监测平台

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动平台

```bash
# 启动完整平台 (UI + API)
python run.py

# 仅启动API服务
python run.py --api --port 8000

# 调试模式
python run.py --debug --reload
```

访问地址: http://127.0.0.1:8080

## 项目结构

```
破夜绘明激光监测平台/
├── apps/                   # 应用入口
├── platform_core/          # 平台核心
│   ├── schema/             # 数据模型
│   ├── plugin_manager/     # 插件管理
│   ├── scheduler/          # 任务调度
│   ├── evidence/           # 证据链
│   ├── replay/             # 回放功能
│   ├── device_adapter/     # 设备适配
│   └── logging/            # 统一日志
├── plugins/                # 插件目录
│   ├── transformer_inspection/  # A组: 主变巡视
│   ├── switch_inspection/       # B组: 开关间隔
│   ├── busbar_inspection/       # C组: 母线巡视
│   ├── capacitor_inspection/    # D组: 电容器
│   └── meter_reading/           # E组: 表计读数
├── configs/                # 配置文件
├── ui/                     # Web UI
├── evidence/               # 证据存储
├── logs/                   # 日志
├── tests/                  # 测试
└── docs/                   # 文档
```

## 五大功能模块

| 模块 | 负责组 | 状态 | 功能描述 |
|------|--------|------|----------|
| 主变自主巡视 | A组 | 待集成 | 外观缺陷、状态识别、热成像分析 |
| 开关间隔巡视 | B组 | 待集成 | 分合位识别、逻辑校验、清晰度评价 |
| 母线自主巡视 | C组 | 待集成 | 远距小目标检测、多目标并发处理 |
| 电容器巡视 | D组 | 待集成 | 结构完整性检测、区域入侵检测 |
| 表计读数 | E组 | 待集成 | 任意角度读数、自动量程识别 |

## 插件开发指南

### 插件目录结构

```
plugins/your_plugin/
├── manifest.json       # 插件清单 (必须)
├── plugin.py           # 入口文件 (必须)
├── models/             # 模型文件
├── configs/            # 配置文件
│   └── default.yaml
├── tests/              # 测试用例
└── README.md           # 说明文档
```

### 必须实现的接口

```python
from platform_core.plugin_manager.base import BasePlugin

class YourPlugin(BasePlugin):

    def init(self, config: dict) -> bool:
        """初始化插件"""
        pass

    def infer(self, frame, rois, context) -> list[RecognitionResult]:
        """执行推理"""
        pass

    def postprocess(self, results, rules) -> list[Alarm]:
        """后处理和告警生成"""
        pass

    def healthcheck(self) -> HealthStatus:
        """健康检查"""
        pass
```

### 标准输出格式

```json
{
    "task_id": "xxx",
    "plugin_id": "your_plugin",
    "plugin_version": "1.0.0",
    "code_hash": "abc123",
    "success": true,
    "results": [
        {
            "roi_id": "xxx",
            "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            "label": "正常",
            "confidence": 0.95,
            "model_version": "v1.0"
        }
    ],
    "alarms": []
}
```

## 打包发布

```bash
# 打包为可执行文件
python build.py --onefile --name PowerStationMonitor

# 输出目录: dist/
```

## API 文档

启动平台后访问: http://127.0.0.1:8080/api/docs

### 主要接口

| 接口 | 方法 | 描述 |
|------|------|------|
| /api/health | GET | 健康检查 |
| /api/plugins | GET | 获取插件列表 |
| /api/tasks/run | POST | 运行任务 |
| /api/evidence/runs | GET | 获取证据记录 |

## 验收标准

每个插件必须通过:

1. **可运行**: 按平台接口接入即跑
2. **可回放**: 给定回放数据,结果可复现
3. **可解释**: 输出 bbox/关键点/置信度/失败原因码
4. **可追溯**: 输出 model_version + code_hash
5. **可维护**: README + 配置样例 + 最小单测

## 技术栈

- **后端**: Python 3.10+, FastAPI, Pydantic
- **前端**: Bootstrap 5, Jinja2
- **视觉**: OpenCV, NumPy
- **日志**: Loguru
- **打包**: PyInstaller

## 文档

- [架构文档](docs/ARCHITECTURE.md)
- [插件开发指南](docs/plugins/)
- [API文档](docs/api/)

## License

Proprietary - All Rights Reserved
