# 输变电激光监测平台 - 五组插件代码包

## 📦 代码包内容

本代码包包含五组完整的插件实现代码，按照性能指标参数标定，可直接部署到项目中。

```
outputs/
├── A组_主变巡视/              # transformer_inspection
│   ├── manifest.json         # 插件清单
│   ├── plugin.py             # 插件主文件
│   ├── detector.py           # 检测器实现
│   ├── __init__.py           # 模块导出
│   └── configs/
│       └── default.yaml      # 配置文件(含性能参数)
│
├── B组_开关间隔/              # switch_inspection
│   ├── manifest.json
│   ├── plugin.py
│   ├── detector.py
│   ├── __init__.py
│   └── configs/
│       └── default.yaml
│
├── C组_母线巡视/              # busbar_inspection
│   ├── manifest.json
│   ├── plugin.py
│   ├── detector.py
│   ├── __init__.py
│   └── configs/
│       └── default.yaml
│
├── D组_电容器/                # capacitor_inspection
│   ├── manifest.json
│   ├── plugin.py
│   ├── detector.py
│   ├── __init__.py
│   └── configs/
│       └── default.yaml
│
├── E组_表计读数/              # meter_reading
│   ├── manifest.json
│   ├── plugin.py
│   ├── detector.py
│   ├── __init__.py
│   └── configs/
│       └── default.yaml
│
├── deploy.sh                  # 自动部署脚本
└── README.md                  # 本文档
```

---

## 🚀 快速部署

### 方法一：使用部署脚本（推荐）

```bash
# 1. 进入项目根目录
cd /path/to/your/project

# 2. 运行部署脚本
bash /path/to/outputs/deploy.sh

# 3. 重启平台
python run.py
```

### 方法二：手动部署

```bash
# 进入项目根目录
cd /path/to/your/project

# 复制各组代码到对应插件目录
cp -r /path/to/outputs/A组_主变巡视/* plugins/transformer_inspection/
cp -r /path/to/outputs/B组_开关间隔/* plugins/switch_inspection/
cp -r /path/to/outputs/C组_母线巡视/* plugins/busbar_inspection/
cp -r /path/to/outputs/D组_电容器/* plugins/capacitor_inspection/
cp -r /path/to/outputs/E组_表计读数/* plugins/meter_reading/

# 重启平台
python run.py
```

---

## 📊 性能指标参数汇总

### A组 - 主变自主巡视

| 参数项 | 值 | 说明 |
|--------|------|------|
| confidence_threshold | 0.5 | 置信度阈值 |
| nms_threshold | 0.4 | NMS阈值 |
| max_detections | 100 | 最大检测数 |
| thermal_threshold | 80°C | 热成像温度阈值 |
| oil_leak.gray_threshold | 60 | 渗漏油灰度阈值 |
| rust.min_area | 300px | 锈蚀最小面积 |
| damage.edge_density | 0.15 | 破损边缘密度阈值 |
| silica_gel.ratio_threshold | 0.1 | 硅胶变色比例阈值 |
| valve.angle_threshold | 30° | 阀门角度判定阈值 |

### B组 - 开关间隔巡视

| 参数项 | 值 | 说明 |
|--------|------|------|
| confidence_threshold | 0.6 | 置信度阈值 |
| min_state_score | 0.55 | 最小状态评分 |
| fusion_weights.text | 0.5 | OCR文字权重 |
| fusion_weights.color | 0.3 | 颜色提示权重 |
| fusion_weights.angle | 0.2 | 角度检测权重 |
| min_clarity_score | 0.70 | 最小清晰度分数 |
| stable_iou_threshold | 0.70 | 稳定IoU阈值 |
| stable_frames | 3 | 稳定帧数 |
| max_inference_time | 300ms | 单帧处理时间 |
| state_accuracy | ≥95% | 状态识别准确率 |

### C组 - 母线巡视

| 参数项 | 值 | 说明 |
|--------|------|------|
| pin_missing.recall | ≥0.85 | 销钉缺失召回率 |
| pin_missing.precision | ≥0.85 | 销钉缺失精确率 |
| crack.recall | ≥0.70 | 裂纹召回率 |
| crack.precision | ≥0.80 | 裂纹精确率 |
| gpu_p95_ms | ≤800ms | GPU P95延迟 |
| cpu_p95_ms | ≤5000ms | CPU P95延迟 |
| slice_size | 640×640 | 切片尺寸 |
| slice_overlap | 0.2 | 切片重叠率 |

### D组 - 电容器巡视

| 参数项 | 值 | 说明 |
|--------|------|------|
| confidence_threshold | 0.55 | 置信度阈值 |
| max_tilt_angle | 5.0° | 最大允许倾斜角 |
| warning_angle | 3.0° | 警告倾斜角 |
| alert_delay | 2.0s | 入侵告警延迟 |
| bank_rows | 3 | 电容器组行数 |
| bank_columns | 4 | 电容器组列数 |

### E组 - 表计读数

| 参数项 | 值 | 说明 |
|--------|------|------|
| confidence_threshold | 0.6 | 置信度阈值 |
| max_rotation | 45° | 最大旋转角度 |
| retry_count | 3 | 重试次数 |
| manual_review_threshold | 0.5 | 人工复核阈值 |
| decimal_places | 2 | 输出精度 |
| contrast_enhancement | 1.2 | 对比度增强系数 |

---

## 🔧 验收标准

所有插件均需满足以下统一验收标准：

### 五大要求

1. **可运行**: 插件能正常加载和执行
2. **可回放**: 输入确定则输出确定
3. **可解释**: 输出包含置信度和reason_code
4. **可追溯**: 结果包含model_version和code_version
5. **可维护**: 代码结构清晰，配置分离

### 输出格式

```json
{
    "task_id": "xxx",
    "roi_id": "xxx",
    "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
    "label": "defect_type",
    "confidence": 0.85,
    "model_version": "v1.0",
    "code_version": "abc123",
    "reason_code": null
}
```

### 失败原因码

| 代码 | 含义 | 适用模块 |
|------|------|----------|
| 101 | 逆光/过曝/低对比 | C组 |
| 102 | 遮挡/不可见 | C组 |
| 103 | 模糊/失焦 | B/C组 |
| 201 | 目标过小,需要变焦 | C组 |
| 1001 | 清晰度过低 | B组 |
| 1003 | 未检测到有效角度 | B组 |
| 2001 | 互锁逻辑异常 | B组 |

---

## 📝 注意事项

1. **依赖**: 确保已安装 `numpy>=1.24.0` 和 `opencv-python>=4.8.0`

2. **模型文件**: 各插件的 `models/` 目录需要放置对应的ONNX模型文件

3. **配置覆盖**: 可通过平台配置或环境变量覆盖默认参数

4. **日志**: 插件运行日志输出到 `logs/platform.log`

5. **测试**: 部署后建议运行各插件的测试脚本验证功能

---

## 🔗 相关文档

- 架构文档: `docs/ARCHITECTURE.md`
- 插件开发指南: `docs/plugins/DEVELOPMENT.md`
- API文档: `docs/api/README.md`
