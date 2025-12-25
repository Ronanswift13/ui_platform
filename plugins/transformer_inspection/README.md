# 主变自主巡视插件 (A组)

## 功能概述

主变压器本体及附属设施（套管、散热器、油位计、呼吸器、端子箱等）的智能巡视检测。

### 核心功能

1. **外观缺陷识别**
   - ✅ 破损检测 (基于边缘检测)
   - ✅ 锈蚀识别 (基于颜色检测) 
   - ✅ 渗漏油检测 (基于深色区域检测)
   - ✅ 异物悬挂识别 (基于轮廓检测)

2. **状态识别**
   - ✅ 呼吸器硅胶变色识别 (基于颜色分析)
   - ✅ 阀门开闭状态识别 (基于轮廓方向)

3. **热成像集成** (可选)
   - ⏳ 红外图像温度提取 (待扩展)

## 快速开始

### 安装依赖

```bash
pip install numpy opencv-python
```

### 基本使用

```python
from transformer_inspection import TransformerInspectionPlugin

# 创建插件实例
plugin = TransformerInspectionPlugin(manifest, plugin_dir)

# 初始化
config = {
    "inference": {
        "confidence_threshold": 0.5
    },
    "recognition": {
        "defect_types": ["damage", "rust", "oil_leak", "foreign_object"],
        "state_types": ["silica_gel_normal", "valve_open"]
    }
}
plugin.init(config)

# 执行推理
results = plugin.infer(frame, rois, context)

# 后处理
alarms = plugin.postprocess(results, rules)
```

## ROI类型说明

| ROI类型 | 说明 | 检测目标 |
|---------|------|----------|
| bushing | 套管 | 破损、污损、渗漏油 |
| radiator | 散热器 | 渗漏油、锈蚀 |
| oil_level | 油位计 | 刻度读数、破损 |
| breather | 呼吸器 | 硅胶颜色变化 |
| terminal_box | 端子箱 | 外观异常、破损 |

## 输出格式

### 识别结果

```json
{
    "task_id": "xxx",
    "roi_id": "xxx",
    "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
    "label": "oil_leak",
    "confidence": 0.85,
    "model_version": "1.0.0",
    "code_version": "abc123"
}
```

### 告警输出

```json
{
    "level": "error",
    "title": "检测到渗漏油",
    "message": "在 radiator_01 区域检测到渗漏油，置信度: 0.85"
}
```

## 检测算法说明

### 1. 油漏检测

- **方法**: 基于深色区域检测
- **原理**: 油漏通常表现为深色的斑点或痕迹
- **步骤**:
  1. 灰度化
  2. 阈值化（检测深色区域）
  3. 形态学去噪
  4. 轮廓提取
  5. 面积过滤

### 2. 锈蚀检测

- **方法**: 基于HSV颜色空间检测
- **原理**: 铁锈呈现橙红色
- **步骤**:
  1. BGR转HSV
  2. 定义铁锈颜色范围
  3. 颜色掩码
  4. 轮廓提取
  5. 面积过滤

### 3. 破损检测

- **方法**: 基于边缘密度分析
- **原理**: 破损区域边缘密集
- **步骤**:
  1. Canny边缘检测
  2. 计算边缘密度
  3. 密度阈值判断
  4. 边缘区域定位

### 4. 异物检测

- **方法**: 基于轮廓形状分析
- **原理**: 异物通常为不规则小物体
- **步骤**:
  1. 边缘检测
  2. 轮廓提取
  3. 圆形度计算
  4. 面积和形状过滤

### 5. 硅胶变色检测

- **方法**: 基于颜色比例分析
- **原理**: 正常硅胶为蓝色，变色后为粉红色
- **步骤**:
  1. BGR转HSV
  2. 蓝色/粉红色掩码
  3. 计算颜色占比
  4. 占比阈值判断

### 6. 阀门状态检测

- **方法**: 基于霍夫直线检测
- **原理**: 阀门开闭对应不同角度
- **步骤**:
  1. 边缘检测
  2. 霍夫直线检测
  3. 角度计算
  4. 状态判断

## 配置参数

### 推理配置

```yaml
inference:
  confidence_threshold: 0.5  # 置信度阈值
  nms_threshold: 0.4        # NMS阈值
  max_detections: 100       # 最大检测数量
```

### 识别类型

```yaml
recognition:
  defect_types:
    - damage          # 破损
    - rust            # 锈蚀
    - oil_leak        # 渗漏油
    - foreign_object  # 异物
  state_types:
    - silica_gel_normal    # 硅胶正常
    - silica_gel_abnormal  # 硅胶变色
    - valve_open           # 阀门开启
    - valve_closed         # 阀门关闭
```

## 性能指标

- **单帧处理时间**: < 500ms (CPU)
- **置信度范围**: 0.4 - 0.95
- **适用光照**: 室外自然光/阴天/多云
- **分辨率**: 支持 1920x1080 及以上

## 局限性

1. **光照依赖**: 强逆光或极暗环境下性能下降
2. **遮挡处理**: 严重遮挡可能导致漏检
3. **模型依赖**: 当前使用OpenCV规则检测，深度学习模型可提升性能
4. **热成像**: 需要额外的热成像摄像头支持

## 未来改进

- [ ] 集成深度学习模型（YOLO/Faster R-CNN）
- [ ] 添加时序分析（多帧融合）
- [ ] 支持夜间模式检测
- [ ] 添加3D点云分析
- [ ] 完善热成像温度提取

## 技术支持

如有问题，请联系A组团队或提交Issue。