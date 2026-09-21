#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-7}"
RESUME="${RESUME:-0}"
source "${SCRIPT_DIR}/common.sh"

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  printf 'RESUME must be 0 or 1\n' >&2
  exit 2
fi
if [[ -e "${OFFICIAL_TEST_ROOT}" ]]; then
  printf 'official_test_output_already_exists: %s\n' \
    "${OFFICIAL_TEST_ROOT}" >&2
  exit 1
fi
preflight_official_inputs

mkdir -p "${MODEL_ROOT}"
validation_command=(
  "${PYTHON}" -m dta.rank_locked_calibration_v6.run_validation
  --cache-root "${V4_CACHE_ROOT}"
  --metadata-csv "${METADATA_CSV}"
  --v5-validation-root "${V5_VALIDATION_ROOT}"
  --v5-package "${V5_PACKAGE}"
  --v5-fallback-result "${V5_FALLBACK_RESULT}"
  --output-root "${VALIDATION_ROOT}"
  --source-revision "${SOURCE_REVISION}"
  --device "${DEVICE}"
)
if [[ "${RESUME}" == "1" ]]; then
  validation_command+=(--resume)
fi

"${validation_command[@]}"

if [[ ! -f "${PACKAGE_ROOT}/selection_seal.json" ]]; then
  printf 'validation_did_not_produce_a_passing_seal: %s\n' \
    "${VALIDATION_ROOT}" >&2
  exit 1
fi
"${PYTHON}" -c \
  'import sys; from dta.rank_locked_calibration_v6.selection import verify_validation_package; verify_validation_package(sys.argv[1]); print("validation_seal_verified=" + sys.argv[1], flush=True)' \
  "${PACKAGE_ROOT}"

"${PYTHON}" -m dta.rank_locked_calibration_v6.evaluate_test \
  --package "${PACKAGE_ROOT}" \
  --test-cache "${TEST_CACHE}" \
  --output "${OFFICIAL_TEST_ROOT}" \
  --device "${DEVICE}"

printf 'validation_package=%s\n' "${PACKAGE_ROOT}"
printf 'validation_result=%s\n' "${VALIDATION_ROOT}/validation_result.json"
printf 'official_test_result=%s\n' "${OFFICIAL_TEST_ROOT}/result.json"
