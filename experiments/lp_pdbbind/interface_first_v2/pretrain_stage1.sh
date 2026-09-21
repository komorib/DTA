#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index for Stage 1}"
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
OUTPUT_DIR="${ROOT}/Model/LP_PDBBind_InterfaceFirstGerNA_v2_stage1_seed${SEED}"
LOG_DIR="${ROOT}/experiments/lp_pdbbind/interface_first_v2/logs"
RESUME_STAGE1="${RESUME_STAGE1:-0}"

case "${SEED}" in
  42|43|44|45|46) ;;
  *) echo "SEED must be one of 42,43,44,45,46" >&2; exit 1 ;;
esac

for path in \
  "${TRAIN_MANIFEST}" \
  "${NATIVE_CACHE_ROOT}/native_train" \
  "${DIAGNOSTIC_KEYS}" \
  "${DIAGNOSTIC_DECOYS}" \
  "${DIAGNOSTIC_REPORT}" \
  "${TRAIN_DECOYS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required Stage-1 input is missing: ${path}" >&2
    exit 1
  fi
done

if [[ -f "${OUTPUT_DIR}/stage1_final.pth" \
   && -f "${OUTPUT_DIR}/model_diagnostic_report.json" \
   && -f "${OUTPUT_DIR}/stage2_ready.json" ]]; then
  echo "Stage 1 is already complete: ${OUTPUT_DIR}"
  exit 0
fi

if [[ "${RESUME_STAGE1}" == "1" ]]; then
  set -- --resume
elif [[ -e "${OUTPUT_DIR}/training_checkpoint.pth" || -e "${OUTPUT_DIR}/stage1_final.pth" ]]; then
  echo "Stage-1 output already exists; set RESUME_STAGE1=1 to resume" >&2
  exit 1
else
  set --
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"
LOG_FILE="${LOG_DIR}/pretrain_stage1_v2_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
"${PY}" -u dta/pretrain_local_interface.py \
  --train-manifest "${TRAIN_MANIFEST}" \
  --native-lmdb "${NATIVE_CACHE_ROOT}/native_train" \
  --decoy-lmdb "${TRAIN_DECOYS}" \
  --diagnostic-keys "${DIAGNOSTIC_KEYS}" \
  --diagnostic-decoy-lmdb "${DIAGNOSTIC_DECOYS}" \
  --data-diagnostic-report "${DIAGNOSTIC_REPORT}" \
  --decoy-schema-version 2 \
  --decoy-bank matched \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 30 \
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
  --log-every 100 \
  "$@" 2>&1 | tee "${LOG_FILE}"

echo "stage1_checkpoint: ${OUTPUT_DIR}/stage1_final.pth"
echo "model_diagnostic_report: ${OUTPUT_DIR}/model_diagnostic_report.json"
echo "stage2_ready: ${OUTPUT_DIR}/stage2_ready.json"
