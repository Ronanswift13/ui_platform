"""
视觉缺陷数据集统一存储

目录结构:
  visual_defect/
  ├── {plugin_id}/
  │   ├── detection/          # 目标检测数据
  │   │   ├── {voltage_level}/
  │   │   │   ├── train/images/  train/labels/
  │   │   │   ├── val/images/    val/labels/
  │   │   │   └── manifest.json
  │   │   └── manifest.json
  │   ├── classification/     # 分类数据
  │   ├── ocr/               # OCR 数据 (meter_reading)
  │   ├── thermal/           # 热像数据 (temperature_monitoring)
  │   └── hyperspectral/     # 高光谱数据 (hyperspectral_detection)
  └── __init__.py

数据通过 pipelines/ingestion/ 摄入，自动按 plugin_id + task_type 归档。
"""
