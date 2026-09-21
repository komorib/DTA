#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RESUME="${RESUME:-0}"
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  printf 'RESUME must be 0 or 1\n' >&2
  exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Model/LP_PDBBind_RankUnlockedFusion_v7/run}"
preflight_v7

command=(
  "${PYTHON}" -m dta.rank_unlocked_fusion_v7.run_pipeline
  --cache-root "${V4_CACHE_ROOT}"
  --metadata-csv "${METADATA_CSV}"
  --v5-validation-root "${V5_VALIDATION_ROOT}"
  --v5-package "${V5_PACKAGE}"
  --v5-fallback-result "${V5_FALLBACK_RESULT}"
  --output-root "${OUTPUT_ROOT}"
  --source-revision "${SOURCE_REVISION}"
  --device "${DEVICE}"
  --maximum-epochs 160
  --patience 20
  --batch-size 256
  --model-width 256
  --transformer-layers 3
  --attention-heads 8
)
if [[ "${RESUME}" == "1" ]]; then
  command+=(--resume)
fi
"${command[@]}"
printf 'v7_output=%s\n' "${OUTPUT_ROOT}"
printf 'v7_result=%s\n' "${OUTPUT_ROOT}/result.json"
