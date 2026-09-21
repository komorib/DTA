#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_ID="${RUN_ID:-rank_unlocked_v7_smoke_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/LP_PDBBind_RankUnlockedFusion_v7_${RUN_ID}}"
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
  --limit 64
  --smoke
  --maximum-epochs 2
  --patience 2
  --batch-size 32
  --model-width 64
  --transformer-layers 1
  --attention-heads 4
)
"${command[@]}"
printf 'smoke_root=%s\n' "${OUTPUT_ROOT}"
printf 'smoke_result=%s\n' "${OUTPUT_ROOT}/result.json"
