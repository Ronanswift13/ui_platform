# Cross-Project Reuse Prompt — 表计读数

## 用途
当其他插件或项目需要复用表计读数的算法模块时使用。

## 可复用模块

| 模块 | 接口 | 适用场景 |
|------|------|----------|
| 透视校正 | `perspective_correct(image, corners)` | 任何需要矫正倾斜拍摄的场景 |
| 指针角度检测 | `detect_pointer_angle(image, center)` | 模拟仪表通用 |
| HSV 色彩分类 | `classify_color_hsv(image)` | LED/信号灯状态检测 |
| OCR 数字提取 | `extract_digits(image)` | 数字显示屏读取 |
| Hough 圆检测 | `detect_circles(image)` | 圆形目标定位 |

## 复用规则
1. **不直接 import**: 应通过 SDK 插件接口调用，不跨插件 import
2. **可提取为公共库**: 如需复用，将模块提取到 `darkbreaker_sdk.cv_utils`
3. **参数独立**: 复用模块的参数不应与 meter_reading 耦合
4. **测试独立**: 复用模块需有独立的单元测试

## 复用流程
1. 确认需要复用的模块
2. 从 detector_enhanced.py 提取为独立函数
3. 添加独立的类型注解和文档
4. 编写独立的单元测试
5. 移入公共库或创建共享模块
6. 原插件改为调用公共库版本
