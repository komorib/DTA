#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V12 training on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_pair_pretrained_v12_full_$(date +%Y%m%d_%H%M%S)}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
COMPLEX_CACHE_ROOT="${COMPLEX_CACHE_ROOT:-${V9_DATA_ROOT}/complex_cache_full}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${ROOT}/Model/LP_PDBBind_MSAPairFusion_v11/msa_pair_v11_full_20260825_165726}"
PRETRAIN_OUTPUT="${PRETRAIN_OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAPairPretraining_v12/${RUN_ID}}"
OUTPUT="${OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAPairExpert_v12/${RUN_ID}}"
FUSION_OUTPUT="${FUSION_OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAPairFusion_v12/${RUN_ID}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=40"
echo "row_cache=${ROW_CACHE}"
echo "anchor_root=${ANCHOR_ROOT}"
echo "pretrain_output=${PRETRAIN_OUTPUT}"
echo "output=${OUTPUT}"
echo "fusion_output=${FUSION_OUTPUT}"
cd "${ROOT}"

if [[ ! -f "${ROW_CACHE}/row_cache_report.json" ]]; then
  echo "missing V11 row cache: ${ROW_CACHE}" >&2
  exit 2
fi
echo "reusing_complete_row_cache=${ROW_CACHE}"

"${PYTHON}" -m dta.msa_pair_pretrained_v12.run_pretraining \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --train-cache "${COMPLEX_CACHE_ROOT}/train" \
  --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
  --row-cache "${ROW_CACHE}" \
  --output "${PRETRAIN_OUTPUT}" \
  --device "${DEVICE}" \
  --seed 42 \
  --batch-size 4 \
  --evaluation-batch-size 6 \
  --num-workers 4 \
  --amp-dtype bfloat16 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --maximum-epochs 12 \
  --patience 4 \
  --msa-width 192 \
  --msa-layers 4 \
  --msa-heads 6 \
  --pair-factor 8 \
  --pair-width 96 \
  --dropout 0.10 \
  --row-depth 64 \
  --mask-probability 0.15 \
  --distogram-weight 1.0 \
  --contact-weight 0.20

"${PYTHON}" -m dta.msa_pair_pretrained_v12.run_training \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --train-cache "${COMPLEX_CACHE_ROOT}/train" \
  --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
  --test-cache "${COMPLEX_CACHE_ROOT}/test" \
  --row-cache "${ROW_CACHE}" \
  --pretrained-checkpoint "${PRETRAIN_OUTPUT}/best_pretrain.pt" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --seed 42 \
  --batch-size 4 \
  --evaluation-batch-size 6 \
  --num-workers 4 \
  --amp-dtype bfloat16 \
  --learning-rate 1.5e-4 \
  --encoder-learning-rate-scale 0.30 \
  --weight-decay 1e-4 \
  --maximum-epochs 80 \
  --patience 14 \
  --model-width 512 \
  --layers 8 \
  --heads 8 \
  --dropout 0.10 \
  --latent-width 1024 \
  --msa-width 192 \
  --msa-layers 4 \
  --msa-heads 6 \
  --pair-factor 8 \
  --pair-width 96 \
  --row-depth 64 \
  --inference-subsets 2 \
  --rank-weight 0.40 \
  --mse-weight 0.15 \
  --contact-weight 0.05 \
  --residue-contact-weight 0.03 \
  --affinity-class-weight 0.05 \
  --distogram-weight 0.05

"${PYTHON}" -m dta.msa_pair_pretrained_v12.run_fusion \
  --anchor-root "${ANCHOR_ROOT}" \
  --v12-root "${OUTPUT}" \
  --output "${FUSION_OUTPUT}" \
  --pca-components 64 \
  --oof-estimators 300 \
  --final-estimators 700 \
  --n-jobs 8 \
  --seed 20260826

printf 'v12_output=%s\n' "${OUTPUT}"
printf 'v12_result=%s\n' "${OUTPUT}/result.json"
printf 'v12_fusion_output=%s\n' "${FUSION_OUTPUT}"
printf 'v12_fusion_result=%s\n' "${FUSION_OUTPUT}/result.json"
