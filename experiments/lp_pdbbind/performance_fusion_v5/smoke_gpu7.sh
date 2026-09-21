#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-anchored_v5_smoke}"
GPU_ID="${GPU_ID:-7}"
SMOKE_ROOT="/tmp/LP_PDBBind_AnchoredExpertFusion_v5_smoke_${RUN_ID}"
source "${SCRIPT_DIR}/common.sh"

preflight_validation_inputs
mkdir -p "${SMOKE_ROOT}"

"${PYTHON}" -m dta.anchored_expert_fusion_v5.run_validation \
  --train-cache-root "${V4_CACHE_ROOT}" \
  --valid-cache-root "${V4_CACHE_ROOT}" \
  --train-count 11037 \
  --valid-count 2340 \
  --output "${SMOKE_ROOT}/run" \
  --device "${DEVICE}" \
  --maximum-epochs 1 \
  --patience 1 \
  --batch-size 64 \
  --seeds 1 7 \
  --limit 64

printf 'smoke_root=%s\n' "${SMOKE_ROOT}"
printf 'validation_package=%s\n' "${SMOKE_ROOT}/run/validation_package"
