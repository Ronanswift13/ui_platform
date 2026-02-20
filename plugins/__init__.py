"""
插件包 - 16个独立微应用
========================

每个插件均可独立运行（Standalone Mode）或由主平台加载（Integrated Mode）。

独立启动: python -m plugins.<plugin_id>
主平台加载: PluginManager.load_plugin("<plugin_id>")

室内监控阵营 (Indoor) - 6个:
    - indoor_fence:          室内电子围栏（多人跟踪/越线检测/授权校验）
    - fire_detection:        消防监测（火焰烟雾检测/热力融合/抑制控制）
    - animal_detection:      动物入侵检测（小动物识别/热力融合/驱离）
    - meter_reading:         表计读数（压力/温度/油位/数字显示屏）
    - temperature_monitoring: 温度监测（测温传感/热点检测/趋势分析）
    - device_monitoring:     设备状态监测（健康指数/故障预测/异常检测）

室外监控阵营 (Outdoor) - 10个:
    - transformer_inspection: 主变自主巡视（缺陷/状态/热力分析）
    - switch_inspection:      开关间隔巡视（断路器/刀闸位置识别）
    - busbar_inspection:      母线自主巡视（绝缘子/金具/线夹检测）
    - capacitor_inspection:   电容器巡视（结构完整性/入侵检测）
    - bird_monitoring:        鸟类监控（检测/物种识别/风险评估/驱鸟）
    - acoustic_monitoring:    声学监测（声信号监测/异常检测）
    - gas_detection:          气体泄漏检测（浓度监测/预警）
    - hyperspectral_detection: 高光谱缺陷检测（光谱分析）
    - slam_mapping:           SLAM三维建图（点云建图/路径规划）
    - multimodal_fusion:      多模态融合诊断（多传感器融合）
"""
