# bird_monitoring docs

Engineering notes for future algorithm upgrades.

## Current status

- Production runtime: `plugin.py -> detector.py::BirdDetector`
- Standalone UI: `standalone/app.py`
- Isolated simulator: `standalone/bird_simulator.py` via `/api/simulator/*`
- Experimental detector ideas: `experimental/advanced_bird_detector.py`

Keep long-lived architecture decisions here instead of scattering them through demo scripts.
