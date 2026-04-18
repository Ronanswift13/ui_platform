# Device Monitoring Upload Examples

本目录提供 `device_monitoring` 的上传样例，覆盖：

- 单设备时序
- 多设备批量数据
- 告警标签
- 故障发生时间标签

目录结构：

```text
device_monitoring_upload/
├── single_device/
├── multi_device/
└── shared_labels/
```

使用建议：

1. 复制任一目录作为上传包模板。
2. 按实际数据替换 `timeseries/*.csv`。
3. 保持 `manifest.json` 中的 `sensor_columns / target_schema / temporal_schema` 与数据列一致。
4. 若训练 `health_index_calibration`，请确保 `labels/health_targets.jsonl` 存在。
