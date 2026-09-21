#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index for the paired smoke run}"
SEED="${SEED:-42}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v2}"
NATIVE_CACHE_ROOT="${NATIVE_CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1}"
TRAIN_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
DIAGNOSTIC_KEYS="${DIAGNOSTIC_KEYS:-${NATIVE_CACHE_ROOT}/diagnostic_keys.json}"
DIAGNOSTIC_DECOYS="${CACHE_ROOT}/decoys_diagnostic_v2"
DIAGNOSTIC_REPORT="${CACHE_ROOT}/diagnostic/decoy_v2_data_diagnostic.json"
TRAIN_DECOYS="${CACHE_ROOT}/decoys_train_v2"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${ROOT}/Model/LP_PDBBind_InterfaceFirstGerNA_v2_smoke_seed${SEED}_${RUN_ID}}"
LOG_DIR="${ROOT}/experiments/lp_pdbbind/interface_first_v2/logs"

for path in \
  "${TRAIN_MANIFEST}" \
  "${NATIVE_CACHE_ROOT}/native_train" \
  "${DIAGNOSTIC_KEYS}" \
  "${DIAGNOSTIC_DECOYS}" \
  "${DIAGNOSTIC_REPORT}" \
  "${TRAIN_DECOYS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required paired-smoke input is missing: ${path}" >&2
    exit 1
  fi
done

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "paired-smoke output already exists; choose a new SMOKE_OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"
echo "gpu_id: ${GPU_ID}"
echo "output_dir: ${OUTPUT_DIR}"

run_smoke() {
  local bank="$1"
  shift
  local bank_output="${OUTPUT_DIR}/${bank}"
  local log_file="${LOG_DIR}/smoke_${bank}_seed${SEED}_${RUN_ID}.log"
  mkdir -p "${bank_output}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  "${PY}" -u dta/pretrain_local_interface.py \
    --train-manifest "${TRAIN_MANIFEST}" \
    --native-lmdb "${NATIVE_CACHE_ROOT}/native_train" \
    --decoy-lmdb "${TRAIN_DECOYS}" \
    --diagnostic-keys "${DIAGNOSTIC_KEYS}" \
    --diagnostic-decoy-lmdb "${DIAGNOSTIC_DECOYS}" \
    --data-diagnostic-report "${DIAGNOSTIC_REPORT}" \
    --decoy-schema-version 2 \
    "$@" \
    --require-paired-bank-intersection \
    --output-dir "${bank_output}" \
    --epochs 1 \
    --seed "${SEED}" \
    --device cuda:0 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --num-workers 0 \
    --grad-clip 1.0 \
    --temperature 0.2 \
    --consistency-weight 0.1 \
    --ligand-gnn-depth 3 \
    --ligand-dropout 0.1 \
    --log-every 25 2>&1 | tee "${log_file}"
}

run_smoke random_physical --decoy-bank random_physical
run_smoke matched --decoy-bank matched

"${PY}" scripts/compare_interface_v2_smokes.py \
  --random-output-dir "${OUTPUT_DIR}/random_physical" \
  --matched-output-dir "${OUTPUT_DIR}/matched" \
  --output-dir "${OUTPUT_DIR}"

echo "paired_smoke_report: ${OUTPUT_DIR}/paired_smoke_report.json"
