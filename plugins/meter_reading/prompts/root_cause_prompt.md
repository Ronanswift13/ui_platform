# Root Cause Analysis Prompt — 表计读数

## 用途
当表计读数出现异常 (精度下降/推理失败/置信度偏低) 时，使用此模板进行根因分析。

## 分析模板

### 1. 现象描述
- 什么表计类型出问题: [pressure_gauge / temperature_gauge / digital_display / led_indicator / ...]
- 错误表现: [读数偏差 / 完全失败 / 低置信度 / 超时]
- 发生频率: [偶发 / 必现 / 特定条件下]
- 影响范围: [单个设备 / 某类表计 / 全部]

### 2. 五问定位法
1. **输入问题?** — 图像质量差 (光照/遮挡/模糊/倾斜超限)?
2. **预处理问题?** — CLAHE/去眩光是否适得其反?
3. **检测问题?** — Keypoint/Hough 检测失败? 哪一级 fallback 触发了?
4. **计算问题?** — 角度计算/量程映射是否异常? 边界值处理?
5. **配置问题?** — 阈值/参数是否不合理? 新表计类型缺配置?

### 3. 数据收集
```python
# 启用 DEBUG 日志
import logging
logging.getLogger('meter_reading').setLevel(logging.DEBUG)

# 查看关键中间结果
result = detector.read_meter(img, meter_type)
print(f"Keypoints: {result.keypoints}")
print(f"Angle: {result.metadata.get('pointer_angle')}")
print(f"Fallback used: {result.metadata.get('fallback_method')}")
```

### 4. 修复决策
- 输入问题 → 增强预处理 / 调整 ROI
- 检测问题 → 调 Hough 参数 / 检查模型
- 计算问题 → 修正映射公式 / 边界处理
- 配置问题 → 更新 default.yaml

### 5. 验证
- 修复后该用例通过
- 回归测试无劣化
- 添加防复发的回归测试
