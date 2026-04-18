# 05 — 跨插件任务路由

## 插件依赖图

```
                    multimodal_fusion
                   /    |    |    \    \
            visual  acoustic gas  hyper thermal
              |        |      |     |      |
    ┌────────┤    acoustic  gas   hyper  thermal(L0)
    │        │   monitoring detect detect
    │        │
    ├── transformer_inspection
    ├── switch_inspection
    ├── busbar_inspection
    ├── capacitor_inspection
    ├── meter_reading
    ├── bird_monitoring
    ├── animal_detection
    ├── fire_detection
    ├── indoor_fence
    ├── device_monitoring
    ├── temperature_monitoring
    └── action_event_monitoring

独立:
    ├── radar (L0, 未接入)
    └── slam_mapping (无下游依赖)
```

## 加载顺序 (来自 plugins_config.yaml)

```
1. acoustic_monitoring
2. gas_detection
3. hyperspectral_detection
4. slam_mapping
5. multimodal_fusion     ← 依赖 1-4
```

## 跨插件修改决策树

```
要修改一个插件?
│
├── 该插件被 multimodal_fusion 依赖?
│   ├── 是 → 同时检查 multimodal_fusion 的输入契约
│   └── 否 → 可独立修改
│
├── 修改涉及 UnifiedResult 输出格式?
│   └── 是 → 必须检查 platform_core/schema/ 和所有下游消费者
│
├── 修改涉及配置 key?
│   └── 是 → 同步更新 configs/plugins_config.yaml
│
└── 修改涉及设备适配接口?
    └── 是 → 检查 platform_core/device_adapter/
```

## 路由规则

| 任务类型 | 路由到 |
|---|---|
| "检测 XX 设备缺陷" | 对应巡视插件 (transformer/switch/busbar 等) |
| "分析气体数据" | gas_detection |
| "分析声音" | acoustic_monitoring |
| "融合多传感器" | multimodal_fusion |
| "室内入侵" | indoor_fence |
| "毫米波数据" | radar (L0 — 暂不可用) |
| "热成像分析" | thermal (L0 — 暂不可用) |
