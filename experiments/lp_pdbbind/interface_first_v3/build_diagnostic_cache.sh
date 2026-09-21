#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v3}"
NATIVE_CACHE_ROOT="${NATIVE_CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1}"
SEED="${SEED:-42}"
TRAIN_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
DIAGNOSTIC_KEYS="${DIAGNOSTIC_KEYS:-${NATIVE_CACHE_ROOT}/diagnostic_keys.json}"
DIAGNOSTIC_DECOYS="${CACHE_ROOT}/decoys_diagnostic_v3"
DIAGNOSTIC_OUTPUT="${CACHE_ROOT}/diagnostic"
LOG_DIR="${ROOT}/experiments/lp_pdbbind/interface_first_v3/logs"

for path in "${TRAIN_MANIFEST}" "${NATIVE_CACHE_ROOT}/native_train" "${DIAGNOSTIC_KEYS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required diagnostic input is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${CACHE_ROOT}" "${LOG_DIR}"
cd "${ROOT}"
LOG_FILE="${LOG_DIR}/build_diagnostic_cache_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -f "${DIAGNOSTIC_DECOYS}/decoy_report.json" ]]; then
  set -- --resume
else
  set --
fi

"${PY}" -u scripts/build_local_interface_decoys_v3.py \
  --manifest "${TRAIN_MANIFEST}" \
  --native-lmdb "${NATIVE_CACHE_ROOT}/native_train" \
  --diagnostic-keys "${DIAGNOSTIC_KEYS}" \
  --cohort diagnostic \
  --output-lmdb "${DIAGNOSTIC_DECOYS}" \
  --seed "${SEED}" \
  --map-size-gb 16 \
  --progress-every 25 \
  "$@"

"${PY}" -u scripts/evaluate_local_interface_decoys_v3.py \
  --manifest "${TRAIN_MANIFEST}" \
  --native-lmdb "${NATIVE_CACHE_ROOT}/native_train" \
  --diagnostic-decoy-lmdb "${DIAGNOSTIC_DECOYS}" \
  --diagnostic-keys "${DIAGNOSTIC_KEYS}" \
  --output-dir "${DIAGNOSTIC_OUTPUT}" \
  --seed "${SEED}" \
  --require-pass

echo "diagnostic_decoys: ${DIAGNOSTIC_DECOYS}"
echo "data_diagnostic_report: ${DIAGNOSTIC_OUTPUT}/decoy_v3_data_diagnostic.json"
