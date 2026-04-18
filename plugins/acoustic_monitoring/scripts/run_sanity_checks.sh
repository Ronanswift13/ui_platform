#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$(dirname "$PLUGIN_DIR")")"

PYTHON_CMD="${PYTHON_BIN:-python3}"
cd "$PLUGIN_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:urllib3 v2 only supports OpenSSL"

"$PYTHON_CMD" -m pytest \
  -c "$PLUGIN_DIR/pytest.ini" \
  tests/test_config_contract.py \
  tests/test_process_contract.py \
  tests/test_standalone.py
