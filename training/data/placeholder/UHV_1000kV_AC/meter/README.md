# 占位符数据目录 - UHV_1000kV_AC / meter

## 说明
此目录用于存放现场采集的训练数据。

## 电压等级信息
- 名称: 1000kV交流特高压
- 类别: UHV
- 热成像阈值: 正常<70°C, 预警<85°C, 报警>100°C

## 检测类别 (7个)
- 0: sf6_pressure_gauge
- 1: sf6_density_relay
- 2: oil_temp_gauge
- 3: oil_level_gauge
- 4: gas_relay
- 5: digital_display
- 6: pointer_gauge

## 数据要求
1. 图像格式: JPG/PNG, 建议分辨率 1920x1080 或更高
2. 标注格式: YOLO格式 (class_id x_center y_center width height)
3. 建议样本数: 每类至少100张图像

## 采集建议
- 多角度拍摄设备
- 包含正常和异常状态
- 不同光照条件 (白天/夜间)
- 不同天气条件 (晴天/阴天/雨天)

## 目录结构
```
meter/
├── images/        # 放置图像文件
└── labels/        # 放置YOLO格式标注文件
```

## 生成时间
2026-01-02 22:24:12
