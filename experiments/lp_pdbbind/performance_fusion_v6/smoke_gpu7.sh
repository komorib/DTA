#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-rank_locked_v6_smoke}"
GPU_ID="${GPU_ID:-7}"
SMOKE_ROOT="/tmp/LP_PDBBind_RankLockedCalibration_v6_smoke_${RUN_ID}"
source "${SCRIPT_DIR}/common.sh"

preflight_validation_inputs
if [[ -e "${SMOKE_ROOT}" ]]; then
  printf 'smoke_output_already_exists: %s\n' "${SMOKE_ROOT}" >&2
  exit 1
fi
mkdir -p "${SMOKE_ROOT}"

"${PYTHON}" -m dta.rank_locked_calibration_v6.run_validation \
  --cache-root "${V4_CACHE_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --v5-validation-root "${V5_VALIDATION_ROOT}" \
  --v5-package "${V5_PACKAGE}" \
  --v5-fallback-result "${V5_FALLBACK_RESULT}" \
  --output-root "${SMOKE_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --device "${DEVICE}" \
  --limit 64 \
  --smoke-epochs 1

printf 'smoke_root=%s\n' "${SMOKE_ROOT}"
printf 'smoke_result=%s\n' "${SMOKE_ROOT}/smoke_result.json"
