# 02 Algorithm Contract

## Platform Input Contract

- `init(config)` reads:
  - `inference.confidence_threshold`
  - `inference.nms_threshold`
  - `inference.max_detections`
  - `thermal.enabled`
  - `thermal.temperature_threshold`
- `infer(frame, rois, context)` expects:
  - `frame`: BGR `numpy.ndarray`
  - `rois`: iterable of SDK `ROI`
  - `context`: real `PluginContext` in platform mode; `None` is tolerated only for local smoke paths

## ROI Routing Contract

- all ROIs -> `detect_defects()`
- `roi.name` / `roi.id` / `roi.metadata` containing `breather|silica` -> `recognize_silica_gel()`
- `roi.name` / `roi.id` / `roi.metadata` containing `oil_level|meter|gauge` -> `detect_oil_level()`
- thermal alarm path -> only when `thermal.enabled=true` and `context.metadata["thermal_frame"]` is a `numpy.ndarray`
- valve labels remain legacy config vocabulary and are not part of the verified runtime path

## Detector Output Contract

- `detect_defects()` -> `list[Detection]`
  - `defect_type`
  - `bbox`
  - `confidence`
  - `class_name`
  - `metadata`
- `detect_oil_level()` -> `OilLevelResult(level_ratio, level_status, confidence, metadata)`
- `recognize_silica_gel()` -> `SilicaGelResult(state, confidence, color_rgb, metadata)`
- `analyze_thermal()` -> `ThermalResult(max_temperature, avg_temperature, hotspot_count, level, hotspots, metadata)`

## Plugin Output Contract

- defect labels are normalized to:
  - `oil_leak`
  - `rust`
  - `damage`
  - `foreign_object`
- silica states emit:
  - `silica_gel_normal`
  - `silica_gel_abnormal`
  - `silica_gel_unknown`
- oil-level state emits:
  - `oil_level_reading`
  - `value=level_ratio`
  - `metadata.level_status=<status string>`
- thermal over-threshold emits:
  - `overtemp`
  - `value=max_temp`

## Degradation Rules

1. If optional deep-learning stacks are unavailable, detector falls back to traditional OpenCV/HSV/contour logic.
2. If a state ROI has no recognized semantic hint, plugin emits no state result for that ROI.
3. If no thermal frame is provided, plugin emits no thermal result even if thermal is enabled.
4. If a single ROI fails, plugin skips that ROI and keeps the frame-level call alive.

## External Dependencies

- Required locally: `numpy`, `opencv-python`, `pyyaml`, `darkbreaker-sdk`
- Optional:
  - `ai_models.deep_learning.yolov8_vit`
  - `ai_models.deep_learning.segformer`
  - `ai_models.deep_learning.gabor_texture`
  - external `model_registry`

## Security Boundary Inside The Contract

- Outputs are for inspection and alarming only.
- No result from this plugin may be interpreted as a control command.
