#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-7}"
RESUME="${RESUME:-0}"
MAXIMUM_EPOCHS="${MAXIMUM_EPOCHS:-60}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
source "${SCRIPT_DIR}/common.sh"

preflight_validation_inputs
preflight_cache "${TEST_CACHE}"

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  printf 'RESUME must be 0 or 1\n' >&2
  exit 2
fi
if [[ -e "${OFFICIAL_TEST_ROOT}" ]]; then
  printf 'official_test_output_already_exists: %s\n' \
    "${OFFICIAL_TEST_ROOT}" >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}"
validation_command=(
  "${PYTHON}" -m dta.anchored_expert_fusion_v5.run_validation
  --train-cache-root "${V4_CACHE_ROOT}"
  --valid-cache-root "${V4_CACHE_ROOT}"
  --train-count 11037
  --valid-count 2340
  --output "${VALIDATION_ROOT}"
  --device "${DEVICE}"
  --maximum-epochs "${MAXIMUM_EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --seeds 1 7 21 42 87
)
if [[ "${RESUME}" == "1" ]]; then
  validation_command+=(--resume)
fi

"${validation_command[@]}"

if [[ ! -f "${PACKAGE_ROOT}/selection_seal.json" ]]; then
  printf 'validation_seal_missing: %s\n' "${PACKAGE_ROOT}" >&2
  exit 1
fi
"${PYTHON}" -c \
  'import sys; from dta.anchored_expert_fusion_v5.selection import verify_validation_seal; verify_validation_seal(sys.argv[1], artifact_root=sys.argv[2]); print("validation_seal_verified=" + sys.argv[1], flush=True)' \
  "${PACKAGE_ROOT}" "${VALIDATION_ROOT}"

"${PYTHON}" -m dta.anchored_expert_fusion_v5.evaluate_test \
  --package "${PACKAGE_ROOT}" \
  --test-cache "${TEST_CACHE}" \
  --output "${OFFICIAL_TEST_ROOT}" \
  --device "${DEVICE}"

printf 'validation_package=%s\n' "${PACKAGE_ROOT}"
printf 'official_test_result=%s\n' "${OFFICIAL_TEST_ROOT}/result.json"
