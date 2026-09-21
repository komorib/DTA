#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RESUME="${RESUME:-0}"
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  printf 'RESUME must be 0 or 1\n' >&2
  exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Model/LP_PDBBind_RepeatedRankFusion_v8/run}"
preflight_v8

command=(
  "${PYTHON}" -m dta.repeated_rank_fusion_v8.run_pipeline
  --cache-root "${V4_CACHE_ROOT}"
  --metadata-csv "${METADATA_CSV}"
  --v5-validation-root "${V5_VALIDATION_ROOT}"
  --v5-package "${V5_PACKAGE}"
  --v7-root "${V7_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --source-revision "${SOURCE_REVISION}"
  --device "${DEVICE}"
  --maximum-epochs 160
  --patience 20
  --batch-size 256
  --model-width 256
  --transformer-layers 3
  --attention-heads 8
  --maximum-pairs 200000
)
if [[ "${RESUME}" == "1" ]]; then
  command+=(--resume)
fi
"${command[@]}"
printf 'v8_output=%s\n' "${OUTPUT_ROOT}"
printf 'v8_result=%s\n' "${OUTPUT_ROOT}/result.json"
