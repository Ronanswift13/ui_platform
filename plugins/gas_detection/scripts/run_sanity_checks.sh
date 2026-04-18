#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$(dirname "$PLUGIN_DIR")")"

PYTHON_CMD="${PYTHON_BIN:-python3}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_CMD" -m pytest \
  plugins/gas_detection/tests/test_config_contract.py \
  plugins/gas_detection/tests/test_process_contract.py \
  plugins/gas_detection/tests/test_trend_contract.py \
  plugins/gas_detection/tests/test_standalone.py
