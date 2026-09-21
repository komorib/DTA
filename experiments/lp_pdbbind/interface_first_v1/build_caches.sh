#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1}"
LOG_DIR="${ROOT}/experiments/lp_pdbbind/interface_first_v1/logs"
TRAIN_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
VALID_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/valid_manifest.json"
TEST_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/test_manifest.json"
DIAGNOSTIC_KEYS="${CACHE_ROOT}/diagnostic_keys.json"

mkdir -p "${CACHE_ROOT}" "${LOG_DIR}"
cd "${ROOT}"

LOG_FILE="${LOG_DIR}/build_caches_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "cache_root: ${CACHE_ROOT}"
echo "log_file: ${LOG_FILE}"

"${PY}" scripts/build_local_interface_diagnostic_split.py \
  --manifest "${TRAIN_MANIFEST}" \
  --output "${DIAGNOSTIC_KEYS}" \
  --fraction 0.10 \
  --seed 42

build_native_cache() {
  local manifest="$1"
  local output="$2"
  local map_size_gb="$3"
  if [[ -f "${output}/metadata.json" ]]; then
    set -- --resume
  else
    set --
  fi
  "${PY}" scripts/build_local_interface_cache.py \
    --manifest "${manifest}" \
    --output-lmdb "${output}" \
    --core-cutoff 6.0 \
    --outer-cutoff 10.0 \
    --shell-budget 128 \
    --map-size-gb "${map_size_gb}" \
    --progress-every 100 \
    "$@"
}

build_native_cache \
  "${TRAIN_MANIFEST}" "${CACHE_ROOT}/native_train" 16
build_native_cache \
  "${VALID_MANIFEST}" "${CACHE_ROOT}/native_valid" 8
build_native_cache \
  "${TEST_MANIFEST}" "${CACHE_ROOT}/native_test" 8

if [[ -f "${CACHE_ROOT}/decoys_train/decoy_report.json" ]]; then
  set -- --resume
else
  set --
fi

"${PY}" scripts/build_local_interface_decoys.py \
  --manifest "${TRAIN_MANIFEST}" \
  --native-lmdb "${CACHE_ROOT}/native_train" \
  --output-lmdb "${CACHE_ROOT}/decoys_train" \
  --diagnostic-keys "${DIAGNOSTIC_KEYS}" \
  --seed 42 \
  --map-size-gb 16 \
  --progress-every 25 \
  "$@"

echo "cache build complete"
echo "diagnostic_keys: ${DIAGNOSTIC_KEYS}"
echo "native_train: ${CACHE_ROOT}/native_train"
echo "native_valid: ${CACHE_ROOT}/native_valid"
echo "native_test: ${CACHE_ROOT}/native_test"
echo "decoys_train: ${CACHE_ROOT}/decoys_train"
