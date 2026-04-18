# 01 Architecture Rules

## Fixed Boundaries

1. `plugin.py` only does SDK adaptation, ROI routing, and result/alarm packaging.
2. `detector_enhanced.py` owns defect/state/thermal logic and must not depend on `darkbreaker_sdk` or standalone UI code.
3. `standalone/`, `run_standalone.py`, and `demo/` are wrappers, not places for business judgments.
4. `manifest.json` fields `id` / `entrypoint` / `plugin_class` are treated as frozen unless the user explicitly asks.

## Real Dependency Flow

```text
standalone/app.py ----------> plugin.py
run_standalone.py ----------> plugin.py
demo/run_demo.py -----------> plugin.py

plugin.py ------------------> detector_enhanced.py
plugin.py ------------------> darkbreaker_sdk.*

detector_enhanced.py -------> numpy / cv2 / optional ai_models.*
```

## Transformer-Specific Red Lines

1. Do not document `defect_detector.py` or `thermal_analyzer.py` as active runtime modules unless code changes prove it.
2. Do not document valve-state recognition as verified; the current main chain only wires silica gel and oil-level state paths.
3. Keep `plugin.py` and `detector_enhanced.py` contract-aligned:
   - defects: dataclass -> `RecognitionResult`
   - thermal: `ThermalResult` -> `RecognitionResult`
   - config: both read nested `inference.*`
4. If config shape changes, re-check `manifest.json`, `configs/default.yaml`, `plugin.py`, and `detector_enhanced.py` together.

## Architecture Checks

```bash
rg -n "darkbreaker_sdk|standalone" detector_enhanced.py
rg -n "defect_detector|thermal_analyzer" plugin.py README.md .agent_skills
rg -n "\bprint\(" plugin.py detector_enhanced.py
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py
```
