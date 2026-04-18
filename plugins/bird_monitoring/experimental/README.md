# bird_monitoring experimental

This directory contains code that is not part of the production inference chain.

## Current contents

- `advanced_bird_detector.py`: PyTorch concept/demo detector with trajectory and deterrence ideas. It may use synthetic or random detections for demonstration and must not be imported by `plugin.py` or `detector.py` in production.

## Boundary

- Production chain remains `plugin.py -> detector.py::BirdDetector`.
- Any promotion from this directory requires explicit opt-in configuration, runtime truth fields, tests, and README / PROJECT_CARD updates.
