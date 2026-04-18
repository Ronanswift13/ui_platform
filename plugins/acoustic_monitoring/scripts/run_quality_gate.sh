#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$(dirname "$PLUGIN_DIR")")"
PYTHON_CMD="${PYTHON_BIN:-python3}"
RG_CMD="${RG_BIN:-rg}"

cd "$PLUGIN_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:urllib3 v2 only supports OpenSSL"

if "$RG_CMD" -n -U 'except\s+Exception:\s*\n\s*pass|except:\s*\n\s*pass' plugin.py detector.py analyzer.py standalone tests; then
  echo "quality gate failed: swallowed exceptions found"
  exit 1
fi

if "$RG_CMD" -n '\bprint\(' plugin.py detector.py analyzer.py standalone; then
  echo "quality gate failed: print() found in production modules"
  exit 1
fi

if "$RG_CMD" -n 'np\.(float|int|bool|complex)([^0-9A-Za-z_]|$)|datetime\.utcnow|np\.fromstring|pkg_resources|imp\.' plugin.py detector.py analyzer.py standalone tests scripts/benchmark.py run_standalone.py demo; then
  echo "quality gate failed: deprecated API usage found"
  exit 1
fi

"$PYTHON_CMD" -m py_compile plugin.py detector.py analyzer.py standalone/app.py standalone/audio_manager.py
"$PYTHON_CMD" -m pytest -c "$PLUGIN_DIR/pytest.ini" tests -q
