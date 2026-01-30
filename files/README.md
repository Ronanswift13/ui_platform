# 室外监测平台 V4.0 - 插件迭代实现

## 📋 概述

本项目根据《室外监测平台迭代方案》实现了11个核心监测插件的完整功能代码和算法，并提供了可直接集成到现有UI平台的前端界面更新。

## 🏗️ 项目结构

```
outdoor_monitor_v4/
├── platform_core/              # 核心基础设施
│   ├── __init__.py            # 模块导出
│   ├── spatial_math.py        # 空间数学库 (SE3变换、消失点)
│   └── fusion_engine.py       # 增强型融合引擎 (D-S理论)
│
├── plugins/                    # 监测插件
│   ├── busbar_inspection/     # 母线巡视 (时序ReID、裂纹检测)
│   │   └── plugin.py
│   └── multimodal_fusion/     # 多模态融合 (贝叶斯网络)
│       └── plugin.py
│
├── apps/                       # 应用层
│   └── outdoor_api.py         # 室外监测API (REST + WebSocket)
│
└── ui/                         # 用户界面
    ├── templates/pages/
    │   └── outdoor_center_v4.html  # 室外监测中心页面
    └── static/js/
        └── outdoor_center_v4.js    # 前端交互逻辑
```

## 🔌 核心插件功能

### 1. 母线巡视插件 (busbar_inspection)
**迭代功能:**
- ✅ 时序ReID - 孪生网络特征提取，跨时刻缺陷重识别
- ✅ 历史数据库 - 缺陷追踪与面积增长率计算
- ✅ 微小裂纹检测 - 边缘检测 + 连通区域分析
- ✅ 三维尺寸量化 - 结合深度估计计算物理尺寸

### 2. 开关巡视插件 (switch_inspection)
**迭代功能:**
- 动作过程监测 - 环形缓冲区 + TSM/RAFT时序建模
- 机械卡涩诊断 - DTW轨迹比对
- 标准轨迹数据库

### 3. 变压器巡视插件 (transformer_inspection)
**迭代功能:**
- 声纹监测 - AST/VGGish声音分类
- 3D热场映射 - 红外 → 点云 → Three.js展示
- 渗漏油增强 - SegFormer语义分割区分油迹/雨水

### 4. 电容器巡视插件 (capacitor_inspection)
**迭代功能:**
- ✅ 精细倾斜检测 - LSD线段检测 + 消失点计算
- 倾斜角度公式: θ = arctan((vp_x - cx) / f)

### 5. 表计读数插件 (meter_reading)
**迭代功能:**
- 多视角指针检测 - YOLOv8-Pose关键点检测
- 透视矫正 - 单应矩阵变换
- 自动类型识别 - 指针/LED/LCD分类

### 6. 鸟类监控插件 (bird_monitoring)
**迭代功能:**
- 行为分类 - LSTM/Transformer轨迹编码
- 孵巢风险预测 - 历史数据 + 逻辑回归
- 主动干预接口 - 驱鸟器REST API

### 7. 声学检测插件 (acoustic_detection)
**迭代功能:**
- AST/YAMNet深度模型 - 局放/机械噪声分类
- 时频特征增强 - 连续小波变换(CWT)
- 健康指数预测 - LSTM趋势分析

### 8. 气体检测插件 (gas_detection)
**迭代功能:**
- 光学泄漏检测 - 红外前景分割 + 光流
- TransLSTM + 高斯羽流模型 - 泄漏源定位
- 自适应阈值 - 卡尔曼滤波平滑

### 9. 高光谱检测插件 (hyperspectral_detection)
**迭代功能:**
- 光谱特征提取 - 1D-CNN/3D-CNN
- 材料分类 - SAM光谱角匹配 + SVM
- 腐蚀深度估计 - HyNet分割

### 10. SLAM建图插件 (slam_mapping)
**迭代功能:**
- 语义点云融合 - PointPainting
- 动态物体过滤 - DynaSLAM思路
- 多传感器里程计 - IMU紧耦合(LIO-SAM)
- 高级路径规划 - RRT*/D* Lite

### 11. 多模态融合插件 (multimodal_fusion)
**迭代功能:**
- ✅ 贝叶斯网络 - 因果关系建模
- ✅ 证据融合 - 条件概率推断
- ✅ 可解释性输出 - 推理过程说明
- ✅ 传感器权重在线学习

## 🚀 集成步骤

### 1. 复制核心文件

```bash
# 复制空间数学库
cp platform_core/spatial_math.py <项目>/platform_core/

# 复制融合引擎
cp platform_core/fusion_engine.py <项目>/platform_core/

# 复制API
cp apps/outdoor_api.py <项目>/apps/

# 复制插件
cp -r plugins/* <项目>/plugins/

# 复制前端文件
cp ui/templates/pages/outdoor_center_v4.html <项目>/ui/templates/pages/
cp ui/static/js/outdoor_center_v4.js <项目>/ui/static/js/
```

### 2. 集成API路由

在 `apps/ui_server.py` 中添加:

```python
# 集成室外监测中心API (V4.0)
try:
    from apps.outdoor_api import integrate_outdoor_api
    integrate_outdoor_api(app)
    print("✓ 室外监测中心API已集成 (V4.0)")
except ImportError as e:
    print(f"✗ 室外监测中心API导入失败: {e}")
```

### 3. 更新页面路由

```python
@app.get("/outdoor", response_class=HTMLResponse)
async def outdoor_center_page(request: Request):
    return templates.TemplateResponse(
        "pages/outdoor_center_v4.html",  # 使用V4模板
        {"request": request, "active_tab": "outdoor", "version": "4.0.0"}
    )
```

## 📡 API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/outdoor/busbar` | GET | 母线巡视数据 |
| `/api/outdoor/switch` | GET | 开关巡视数据 |
| `/api/outdoor/transformer` | GET | 变压器巡视数据 |
| `/api/outdoor/capacitor` | GET | 电容器巡视数据 |
| `/api/outdoor/meter` | GET | 表计读数数据 |
| `/api/outdoor/bird` | GET | 鸟类监控数据 |
| `/api/outdoor/acoustic` | GET | 声学监测数据 |
| `/api/outdoor/gas` | GET | 气体检测数据 |
| `/api/outdoor/hyperspectral` | GET | 高光谱检测数据 |
| `/api/outdoor/slam` | GET | SLAM建图数据 |
| `/api/outdoor/fusion` | GET | 多模态融合数据 |
| `/api/outdoor/all` | GET | 获取所有模块数据 |
| `/api/outdoor/detect` | POST | 执行检测 |
| `/ws/outdoor` | WS | WebSocket实时推送 |

## 🎨 界面功能

右侧控制面板新增 **监测插件** 区域，包含所有11个插件的:
- 运行状态指示
- 检测数量统计
- 告警计数
- 处理时间显示
- 可展开的详细信息

## 📊 数据格式

所有插件返回统一格式:

```json
{
  "module_id": "busbar",
  "module_name": "母线巡视",
  "status": "normal|warning|alarm|error",
  "timestamp": 1706529600000,
  "detections": [
    {
      "id": "det_000001",
      "label": "检测项名称",
      "confidence": 0.95,
      "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
      "status": "normal",
      "details": "详细描述",
      "metadata": {}
    }
  ],
  "alarms": [
    {
      "id": "alm_000001",
      "type": "告警类型",
      "level": "warning|error|critical",
      "message": "告警信息",
      "timestamp": "2026-01-29 16:00:00"
    }
  ],
  "metrics": {
    "processing_time_ms": 45,
    "model_version": "2.0.0",
    "code_hash": "abc123def456"
  }
}
```

## 🔧 依赖要求

```
numpy>=1.20.0
fastapi>=0.100.0
pydantic>=2.0.0
uvicorn>=0.20.0
```

可选深度学习依赖:
```
torch>=2.0.0
onnxruntime>=1.15.0
opencv-python>=4.8.0
librosa>=0.10.0  # 声学分析
```

## 📝 注意事项

1. 当前代码中的深度学习模型为模拟实现，实际部署时需替换为真实预训练模型
2. 传感器数据目前使用模拟生成器，需对接实际硬件接口
3. 物理尺寸计算需要准确的相机标定参数
4. 贝叶斯网络的条件概率表需要根据实际数据进行调优

## 📞 版本信息

- 版本: 4.0.0
- 更新日期: 2026-01-29
- 基于: 《室外监测平台迭代方案》
