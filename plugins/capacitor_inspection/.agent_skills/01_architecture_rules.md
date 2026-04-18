# 01 Architecture Rules

## Fixed Boundaries

1. `plugin.py` only does SDK adaptation, ROI routing, and result/alarm packaging.
2. `detector_enhanced.py` owns structural-defect and intrusion logic and must not depend on `darkbreaker_sdk` or standalone code.
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

## Capacitor-Specific Red Lines

1. Do not merge structural-defect and intrusion routing into one undifferentiated path.
2. Keep detector outputs dataclass-based; SDK conversion stays in `plugin.py`.
3. Do not document replay coverage or stable intrusion field accuracy that does not exist.
4. If config shape changes, re-check `manifest.json`, `configs/default.yaml`, `plugin.py`, and `detector_enhanced.py` together.

## Architecture Checks

```bash
rg -n "darkbreaker_sdk|standalone" detector_enhanced.py
rg -n "\bprint\(" plugin.py detector_enhanced.py
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py
```
