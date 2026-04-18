# 02 Algorithm Contract

## Platform Input Contract

- `init(config)` reads:
  - `inference.confidence_threshold`
  - `inference.nms_threshold`
  - `structural_integrity.tilt_detection.*`
  - `intrusion_detection.alert_delay`
  - `capacitor_bank.rows|columns`
- `infer(frame, rois, context)` expects:
  - `frame`: BGR `numpy.ndarray`
  - `rois`: iterable of SDK `ROI`
  - `context`: real `PluginContext` in platform mode; `None` is tolerated only for local smoke paths

## ROI Routing Contract

- `ROIType.DEFECT` or name containing `capacitor_bank|capacitor_unit|fuse|connecting_bar|insulator`
  -> `detect_structural_defects()`
- `ROIType.INTRUSION` or name containing `fence|warning_zone|restricted_zone`
  -> `detect_intrusion()`

## Detector Output Contract

- `detect_structural_defects()` -> `list[CapacitorDetection]`
  - `defect_type`
  - `bbox`
  - `confidence`
  - `class_name`
  - `tilt_angle`
  - `metadata`
- `detect_intrusion()` -> `list[IntrusionDetection]`
  - `intrusion_type`
  - `bbox`
  - `confidence`
  - `zone`
  - `track_id`
  - `duration_sec`
  - `confirmed`
  - `metadata`

## Plugin Output Contract

- structural labels are normalized directly from detector enum values:
  - `tilt_warning`
  - `tilt_error`
  - `collapse`
  - `missing_unit`
- intrusion labels are normalized to:
  - `intrusion_person`
  - `intrusion_vehicle`
  - `intrusion_animal`
  - `intrusion_unknown`

## Degradation Rules

1. If optional deep-learning stacks are unavailable, structural detection falls back to traditional CV.
2. If detector returns no intrusion targets, plugin emits no intrusion result.
3. If a single ROI fails, plugin skips that ROI and keeps the frame-level call alive.
4. Intrusion positive behavior is currently contract-tested, not replay-validated.

## External Dependencies

- Required locally: `numpy`, `opencv-python`, `pyyaml`, `darkbreaker-sdk`
- Optional:
  - `ai_models.deep_learning.yolov8_vit`
  - `ai_models.deep_learning.thermal_visible_registration`
  - `ai_models.deep_learning.yolov8_obb`
  - external `model_registry`

## Security Boundary Inside The Contract

- Outputs are for inspection and alarming only.
- No result from this plugin may be interpreted as a control command.
