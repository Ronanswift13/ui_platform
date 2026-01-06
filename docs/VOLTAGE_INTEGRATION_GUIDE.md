# 变电站电压等级管理系统 - 集成指南

## 目录

1. [系统概述](#系统概述)
2. [电压等级分类](#电压等级分类)
3. [快速开始](#快速开始)
4. [API 参考](#api-参考)
5. [配置说明](#配置说明)
6. [插件功能](#插件功能)
7. [模型库结构](#模型库结构)
8. [集成步骤](#集成步骤)
9. [常见问题](#常见问题)

---

## 系统概述

变电站电压等级管理系统是一个完整的解决方案，用于管理不同电压等级变电站的AI模型、设备配置和检测参数。

### 主要功能

- ✅ 支持全电压等级分类（特高压、超高压、高压、中压、低压）
- ✅ 自动匹配对应的AI模型库
- ✅ 差异化的设备配置参数
- ✅ 电压等级专属插件功能
- ✅ RESTful API 接口
- ✅ Web 管理界面

---

## 电压等级分类

### 分类标准

| 分类 | 代码 | 交流电压 | 直流电压 |
|------|------|----------|----------|
| 特高压 (UHV) | UHV | 1000kV及以上 | ±800kV及以上 |
| 超高压 (EHV) | EHV | 330kV、500kV、750kV | ±500kV、±660kV |
| 高压 (HV) | HV | 110kV、220kV | - |
| 中压 (MV) | MV | 35kV、66kV | - |
| 低压 (LV) | LV | 10kV及以下 | - |

### 支持的电压等级

```
特高压:
  - 1000kV_AC  (交流特高压)
  - ±800kV_DC  (直流特高压)
  - ±1100kV_DC (直流特高压)

超高压:
  - 500kV_AC   (交流超高压)
  - 330kV_AC   (交流超高压，西北电网)
  - 750kV_AC   (交流超高压)
  - ±500kV_DC  (直流超高压)

高压:
  - 220kV
  - 110kV

中压:
  - 35kV
  - 66kV

低压:
  - 10kV
  - 6kV
  - 380V
```

---

## 快速开始

### 安装依赖

```bash
pip install fastapi uvicorn pyyaml
```

### 基本使用

```python
from platform_core import VoltageAdapterManager

# 创建管理器
manager = VoltageAdapterManager()

# 设置电压等级
manager.set_voltage_level("500kV_AC")

# 获取当前电压等级
print(manager.get_current_level())  # 输出: 500kV_AC
print(manager.get_current_category())  # 输出: 超高压

# 获取设备配置
transformer_config = manager.get_equipment_config("transformer")
print(transformer_config["thermal_thresholds"])
# 输出: {'normal': 65, 'warning': 80, 'alarm': 95}

# 获取支持的插件
plugins = manager.get_supported_plugins()
for plugin in plugins:
    print(f"- {plugin['name']}: {plugin['description']}")
```

### 启动示例应用

```bash
cd apps
python example_app.py
```

访问:
- http://localhost:8000 - 主页
- http://localhost:8000/docs - API文档
- http://localhost:8000/demo - 功能演示

---

## API 参考

### 基础端点

#### 获取当前电压等级
```http
GET /api/voltage/current
```
响应:
```json
{
  "success": true,
  "voltage_level": "500kV_AC",
  "category": "超高压",
  "message": "获取成功"
}
```

#### 设置电压等级
```http
POST /api/voltage/set
Content-Type: application/json

{
  "level": "220kV"
}
```

#### 获取所有电压分类
```http
GET /api/voltage/categories
```

#### 获取可用电压等级
```http
GET /api/voltage/available
```

### 配置端点

#### 获取设备配置
```http
GET /api/voltage/config/{equipment_type}
```
equipment_type: transformer, switch, busbar, capacitor, meter, dc_system, gis

#### 获取所有配置
```http
GET /api/voltage/config
```

### 模型端点

#### 获取模型列表
```http
GET /api/voltage/models
```

#### 检查模型状态
```http
GET /api/voltage/model-status
```

### 检测类别端点

#### 获取检测类别
```http
GET /api/voltage/detection-classes/{equipment_type}
```

#### 获取所有检测类别
```http
GET /api/voltage/detection-classes
```

### 阈值端点

#### 获取热成像阈值
```http
GET /api/voltage/thermal-thresholds
```

#### 获取角度参考值
```http
GET /api/voltage/angle-reference/{switch_type}
```
switch_type: breaker, isolator, grounding

### 插件端点

#### 获取支持的插件
```http
GET /api/voltage/plugins
```

#### 获取所有插件
```http
GET /api/voltage/plugins/all
```

#### 获取插件详情
```http
GET /api/voltage/plugins/{plugin_id}
```

### 综合端点

#### 获取完整信息
```http
GET /api/voltage/info
```

#### 比较电压等级
```http
GET /api/voltage/compare?level1=500kV_AC&level2=220kV
```

---

## 配置说明

### 热成像温度阈值

不同电压等级的变压器热成像温度阈值:

| 电压等级 | 正常 (°C) | 警告 (°C) | 告警 (°C) |
|----------|-----------|-----------|-----------|
| 1000kV 特高压 | 70 | 85 | 100 |
| ±800kV 直流 | 70 | 85 | 100 |
| 500kV 超高压 | 65 | 80 | 95 |
| 330kV 超高压 | 63 | 78 | 92 |
| 220kV 高压 | 60 | 75 | 85 |
| 110kV 高压 | 55 | 70 | 80 |
| 35kV 中压 | 50 | 65 | 75 |
| 10kV 低压 | 45 | 60 | 70 |

### 开关角度参考值 (220kV示例)

| 开关类型 | 分闸角度 | 合闸角度 |
|----------|----------|----------|
| 断路器 (breaker) | -55° | 35° |
| 隔离开关 (isolator) | -65° | 25° |
| 接地刀闸 (grounding) | -75° | 15° |

### 母线参数

| 电压等级 | 母线高度 (m) | 相间距 (m) | 导线型号 |
|----------|--------------|------------|----------|
| 1000kV | 20 | 15.0 | LGJ-800/55, LGJQ-1000/70 |
| 500kV | 15 | 9.0 | LGJ-630/45, LGJQ-800/55 |
| 220kV | 8 | 4.5 | LGJ-400/35 |
| 110kV | 6 | 3.0 | LGJ-240/30, LGJ-300/35 |
| 35kV | 4 | 1.5 | LGJ-120/20, 矩形母线 |
| 10kV | 2.5 | 0.3 | 矩形母线, 管形母线 |

---

## 插件功能

### 特高压专有插件

| 插件ID | 名称 | 支持电压 | 功能说明 |
|--------|------|----------|----------|
| uhv_bushing_monitor | 特高压套管监测 | 1000kV, ±800kV | 套管裂纹、污损、局放检测 |
| uhv_corona_detection | 特高压电晕检测 | 1000kV, ±800kV, 500kV | 电晕放电检测 |
| converter_valve_monitor | 换流阀监测 | ±800kV, ±1100kV, ±500kV | 换流阀温度、冷却监测 |

### 通用插件

| 插件ID | 名称 | 功能说明 |
|--------|------|----------|
| transformer_monitor | 变压器监测 | 油位、油温、渗漏、外观缺陷 |
| switch_state_detection | 开关状态检测 | 断路器、隔离开关、接地刀闸状态 |
| busbar_inspection | 母线巡检 | 绝缘子、金具、导线缺陷 |
| meter_reading | 表计读数识别 | 指针式、数字式表计 |
| thermal_imaging | 红外热成像分析 | 设备温度异常检测 |
| gis_monitoring | GIS设备监测 | GIS位置、SF6密度、局放 |

### 中低压专有插件

| 插件ID | 名称 | 支持电压 | 功能说明 |
|--------|------|----------|----------|
| cabinet_inspection | 开关柜巡检 | 35kV, 10kV, 6kV | 柜门、指示灯、开关位置 |
| environment_monitor | 环境监测 | 35kV, 10kV | SF6浓度、温湿度、烟雾 |
| smart_meter | 智能仪表读取 | 35kV, 10kV, 6kV, 380V | 数字显示屏、多功能电力仪表 |

---

## 模型库结构

```
models/
├── uhv/                          # 特高压模型
│   ├── 1000kV_AC/
│   │   ├── transformer/
│   │   │   ├── transformer_defect_uhv.onnx
│   │   │   ├── bushing_uhv.onnx
│   │   │   └── pd_uhv.onnx
│   │   ├── switch/
│   │   │   ├── switch_state_uhv.onnx
│   │   │   └── gis_position_uhv.onnx
│   │   └── busbar/
│   │       ├── busbar_defect_uhv.onnx
│   │       └── corona_uhv.onnx
│   └── 800kV_DC/
│       └── converter/
│           └── converter_valve_uhv.onnx
├── ehv/                          # 超高压模型
│   ├── 500kV/
│   └── 330kV/
├── hv/                           # 高压模型
│   ├── 220kV/
│   └── 110kV/
├── mv/                           # 中压模型
│   └── 35kV/
└── lv/                           # 低压模型
    └── 10kV/
```

---

## 集成步骤

### 1. 复制文件到项目

```bash
# 复制核心模块
cp -r platform_core/ your_project/platform_core/

# 复制配置文件
cp -r configs/ your_project/configs/

# 复制UI文件 (可选)
cp -r ui/ your_project/ui/
```

### 2. 修改现有 api_server.py

```python
# apps/api_server.py

from fastapi import FastAPI
from platform_core.voltage_api_extended import integrate_voltage_routes

app = FastAPI()

# 集成电压等级管理路由
integrate_voltage_routes(app)

# ... 其他路由
```

### 3. 修改现有 ui_server.py

```python
# apps/ui_server.py

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from platform_core import VoltageAdapterManager

app = FastAPI()
templates = Jinja2Templates(directory="ui/templates")
voltage_manager = VoltageAdapterManager()

@app.get("/settings/voltage")
async def voltage_settings(request: Request):
    return templates.TemplateResponse(
        "pages/settings_voltage_extended.html",
        {
            "request": request,
            "current_voltage_level": voltage_manager.get_current_level()
        }
    )
```

### 4. 在插件中使用

```python
# plugins/transformer_plugin.py

from platform_core import VoltageAdapterManager

class TransformerPlugin:
    def __init__(self):
        self.voltage_manager = VoltageAdapterManager()
    
    def get_detection_config(self):
        # 根据当前电压等级获取配置
        config = self.voltage_manager.get_equipment_config("transformer")
        return {
            "thermal_thresholds": config.get("thermal_thresholds"),
            "detection_classes": config.get("detection_classes"),
            "special_features": config.get("special_features")
        }
    
    def get_model_path(self):
        return self.voltage_manager.get_model_path("transformer", "defect_detection")
```

---

## 常见问题

### Q1: 如何添加新的电压等级?

在 `voltage_adapter_extended.py` 中:

1. 添加 `VoltageLevel` 枚举值
2. 创建新的 `EquipmentConfig` 实例
3. 添加到 `VOLTAGE_CONFIGS` 映射
4. 添加到 `MODEL_LIBRARIES` (如需要)

### Q2: 如何自定义插件功能?

在 `PLUGIN_CAPABILITIES` 字典中添加新的 `PluginCapability`:

```python
PLUGIN_CAPABILITIES["my_custom_plugin"] = PluginCapability(
    name="自定义插件",
    description="描述",
    supported_voltage_levels=["220kV", "110kV"],
    detection_types=["custom_detection"],
    requires_models=["custom_model.onnx"]
)
```

### Q3: 配置保存在哪里?

默认保存在 `configs/voltage_config.yaml`，可通过构造函数参数修改:

```python
manager = VoltageAdapterManager(config_path="custom/path/config.yaml")
```

### Q4: 如何在不同环境使用不同配置?

```python
import os

config_path = os.environ.get("VOLTAGE_CONFIG_PATH", "configs/voltage_config.yaml")
manager = VoltageAdapterManager(config_path=config_path)
```

---

## 更新日志

### v2.0.0 (2025-01)
- 新增特高压、中压、低压电压等级支持
- 新增直流特高压 (±800kV, ±1100kV) 支持
- 新增换流站设备配置
- 新增GIS组合电器配置
- 新增电压等级比较功能
- 完善API接口
- 优化前端界面

### v1.0.0 (2024)
- 初始版本
- 支持 220kV 和 500kV 电压等级
