#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
SEED="${SEED:-42}"
MAX_ELIGIBLE_SAMPLES="${MAX_ELIGIBLE_SAMPLES:-}"
ROTATION_LIMIT="${ROTATION_LIMIT:-}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v4b}"
NATIVE_CACHE_ROOT="${NATIVE_CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1}"
TRAIN_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
V3_SMOKE_REPORT="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v3/decoys_generator_smoke_v3/decoy_report.json"
V4_REPORT="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v4/translation_probe_high_seed${SEED}/translation_probe_report.json"
LOG_DIR="${ROOT}/experiments/lp_pdbbind/interface_first_v4b/logs"

if [[ -n "${MAX_ELIGIBLE_SAMPLES}" || -n "${ROTATION_LIMIT}" ]]; then
  OUTPUT_LMDB="${CACHE_ROOT}/rotation_probe_high_seed${SEED}_debug_e${MAX_ELIGIBLE_SAMPLES:-all}_r${ROTATION_LIMIT:-all}"
else
  OUTPUT_LMDB="${CACHE_ROOT}/rotation_probe_high_seed${SEED}"
fi

for path in "${TRAIN_MANIFEST}" "${NATIVE_CACHE_ROOT}/native_train" "${V3_SMOKE_REPORT}" "${V4_REPORT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required V4b probe input is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${CACHE_ROOT}" "${LOG_DIR}"
cd "${ROOT}"
LOG_FILE="${LOG_DIR}/rotation_probe_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

set --
if [[ -f "${OUTPUT_LMDB}/rotation_probe_report.json" ]]; then
  set -- "$@" --resume
fi
if [[ -n "${MAX_ELIGIBLE_SAMPLES}" ]]; then
  set -- "$@" --max-eligible-samples "${MAX_ELIGIBLE_SAMPLES}"
fi
if [[ -n "${ROTATION_LIMIT}" ]]; then
  set -- "$@" --rotation-limit "${ROTATION_LIMIT}"
fi

"${PY}" -u scripts/probe_local_interface_rotation_v4b.py \
  --manifest "${TRAIN_MANIFEST}" \
  --native-lmdb "${NATIVE_CACHE_ROOT}/native_train" \
  --v3-smoke-report "${V3_SMOKE_REPORT}" \
  --v4-report "${V4_REPORT}" \
  --output-lmdb "${OUTPUT_LMDB}" \
  --seed "${SEED}" \
  --map-size-gb 16 \
  --progress-every 1 \
  "$@"

echo "rotation_probe_report: ${OUTPUT_LMDB}/rotation_probe_report.json"
