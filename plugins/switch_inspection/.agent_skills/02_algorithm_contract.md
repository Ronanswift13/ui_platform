# 02 Algorithm Contract

## Platform Input Contract

- `init(config)`:
  - plugin reads `inference.*` and `image_quality.*`
  - detector reads `state_recognition.*`, `logic_validation.*`, `gauge_reading.*`, and `image_quality.*`
- `infer(frame, rois, context)`:
  - `frame`: BGR `numpy.ndarray`
  - `rois`: iterable of objects exposing `id`, `bbox`, and `name` or `roi_type`
  - `context`: real `PluginContext` in platform mode; `None` is tolerated only for local smoke paths

## ROI Routing Contract

- `breaker_indicator`, `isolator_indicator`, `grounding_indicator` -> indicator recognition
- `breaker_linkage`, `isolator_linkage`, `grounding_handle` -> linkage recognition
- `clarity_anchor` -> clarity-only result
- `gauge_pressure`, `gauge_density` -> gauge path when `gauge_reading.enabled=true`

## Detector Output Contract

- `evaluate_clarity()` -> `ClarityResult(score, level, method, metrics, reshoot_suggestion)`
- `recognize_indicator_state()` / `recognize_linkage_state()` -> `SwitchRecognitionResult`
  - `state`: `open|closed|unknown|intermediate`
  - `confidence`: `[0, 1]`
  - `reason_code`: optional integer
  - `evidence`: normalized summary for plugin metadata
- `validate_interlock()` -> list of alarm dicts with:
  - `severity`
  - `rule_name`
  - `rule_id`
  - `states`
  - `message`
  - `reason_code`
- `read_gauge()` -> `GaugeReadingResult`

## Plugin Output Contract

- low clarity -> `RecognitionResult(label="clarity_low")`
- valid state -> `RecognitionResult(label="<device>_<state>")`
- interlock violation -> `RecognitionResult(label="logic_error|logic_warning")`
- gauge disabled -> no gauge result
- any ROI failure -> skip the ROI and keep the frame-level call alive

## Degradation Rules

1. If optional deep-learning stack is unavailable, detector falls back to OCR/color/angle chain.
2. If no evidence reaches `confidence_threshold`, plugin emits no state result for that ROI.
3. If `image_quality.min_clarity_score` is not met, plugin emits `clarity_low` and stops deeper processing for that ROI.
4. `switch_consistency.py` is not part of the runtime degradation chain until explicitly integrated.

## External Dependencies

- Required locally: `numpy`, `opencv-python`, `pyyaml`, `darkbreaker-sdk`
- Optional:
  - `ai_models.deep_learning.yolov8_vit`
  - external `model_registry`
  - external `fusion_engine`

## Security Boundary Inside The Contract

- Outputs are for inspection and alarming only.
- No result from this plugin may be interpreted as a remote control command.
