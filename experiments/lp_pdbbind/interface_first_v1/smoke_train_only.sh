#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index for the train-only smoke run}"
SEED="${SEED:-42}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1}"
TRAIN_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/tmp/LP_PDBBind_InterfaceFirstGerNA_v1_smoke_seed${SEED}_${RUN_ID}}"
LOG_DIR="${ROOT}/experiments/lp_pdbbind/interface_first_v1/logs"

for path in \
  "${TRAIN_MANIFEST}" \
  "${CACHE_ROOT}/native_train" \
  "${CACHE_ROOT}/decoys_train" \
  "${CACHE_ROOT}/diagnostic_keys.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "required train-only input is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

LOG_FILE="${LOG_DIR}/smoke_train_only_seed${SEED}_${RUN_ID}.log"
echo "gpu_id: ${GPU_ID}"
echo "output_dir: ${OUTPUT_DIR}"
echo "log_file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
"${PY}" -u dta/pretrain_local_interface.py \
  --train-manifest "${TRAIN_MANIFEST}" \
  --native-lmdb "${CACHE_ROOT}/native_train" \
  --decoy-lmdb "${CACHE_ROOT}/decoys_train" \
  --diagnostic-keys "${CACHE_ROOT}/diagnostic_keys.json" \
  --output-dir "${OUTPUT_DIR}" \
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
  --log-every 25 2>&1 | tee "${LOG_FILE}"
