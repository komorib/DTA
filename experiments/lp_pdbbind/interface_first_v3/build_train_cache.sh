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
DIAGNOSTIC_REPORT="${CACHE_ROOT}/diagnostic/decoy_v3_data_diagnostic.json"
TRAIN_DECOYS="${CACHE_ROOT}/decoys_train_v3"
LOG_DIR="${ROOT}/experiments/lp_pdbbind/interface_first_v3/logs"

for path in \
  "${TRAIN_MANIFEST}" \
  "${NATIVE_CACHE_ROOT}/native_train" \
  "${DIAGNOSTIC_KEYS}" \
  "${DIAGNOSTIC_DECOYS}" \
  "${DIAGNOSTIC_REPORT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required train-cache input is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${CACHE_ROOT}" "${LOG_DIR}"
cd "${ROOT}"
LOG_FILE="${LOG_DIR}/build_train_cache_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

"${PY}" -c '
import sys
from types import SimpleNamespace
from scripts.evaluate_local_interface_decoys_v3 import verify_published_v3_diagnostic

args = SimpleNamespace(
    train_manifest=sys.argv[1],
    native_lmdb=sys.argv[2],
    diagnostic_keys=sys.argv[3],
    diagnostic_decoy_lmdb=sys.argv[4],
    data_diagnostic_report=sys.argv[5],
    seed=int(sys.argv[6]),
)
verify_published_v3_diagnostic(args.train_manifest, args.native_lmdb, args.diagnostic_keys, args.diagnostic_decoy_lmdb, args.data_diagnostic_report, args.seed)
print("v3 diagnostic report provenance and gate verified")
' \
  "${TRAIN_MANIFEST}" \
  "${NATIVE_CACHE_ROOT}/native_train" \
  "${DIAGNOSTIC_KEYS}" \
  "${DIAGNOSTIC_DECOYS}" \
  "${DIAGNOSTIC_REPORT}" \
  "${SEED}"

if [[ -f "${TRAIN_DECOYS}/decoy_report.json" ]]; then
  set -- --resume
else
  set --
fi

"${PY}" -u scripts/build_local_interface_decoys_v3.py \
  --manifest "${TRAIN_MANIFEST}" \
  --native-lmdb "${NATIVE_CACHE_ROOT}/native_train" \
  --diagnostic-keys "${DIAGNOSTIC_KEYS}" \
  --cohort train \
  --output-lmdb "${TRAIN_DECOYS}" \
  --seed "${SEED}" \
  --map-size-gb 16 \
  --progress-every 25 \
  "$@"

echo "train_decoys: ${TRAIN_DECOYS}"
