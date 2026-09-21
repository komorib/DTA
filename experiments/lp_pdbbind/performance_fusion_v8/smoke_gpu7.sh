#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_ID="${RUN_ID:-repeated_rank_v8_smoke_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/LP_PDBBind_RepeatedRankFusion_v8_${RUN_ID}}"
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
  --limit 64
  --smoke
  --maximum-epochs 2
  --patience 2
  --batch-size 32
  --model-width 64
  --transformer-layers 1
  --attention-heads 4
  --maximum-pairs 4096
)
"${command[@]}"
printf 'v8_smoke_output=%s\n' "${OUTPUT_ROOT}"
printf 'v8_smoke_result=%s\n' "${OUTPUT_ROOT}/result.json"
