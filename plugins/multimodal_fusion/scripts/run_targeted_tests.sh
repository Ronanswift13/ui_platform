#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PLUGIN_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

python3 -m pytest \
  plugins/multimodal_fusion/tests/test_config_contract.py \
  plugins/multimodal_fusion/tests/test_fusion_input_contract.py \
  plugins/multimodal_fusion/tests/test_fusion_output_contract.py \
  plugins/multimodal_fusion/tests/test_missing_modality_degradation.py \
  plugins/multimodal_fusion/tests/test_standalone.py \
  -q
