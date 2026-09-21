#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V12 smoke training on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_pair_pretrained_v12_smoke_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-24}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
COMPLEX_CACHE_ROOT="${COMPLEX_CACHE_ROOT:-${V9_DATA_ROOT}/complex_cache_full}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${ROOT}/Model/LP_PDBBind_MSAPairFusion_v11/msa_pair_v11_full_20260825_165726}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/LP_PDBBind_MSAPairPretrained_v12_${RUN_ID}}"
PRETRAIN_OUTPUT="${PRETRAIN_OUTPUT:-${SMOKE_ROOT}/pretrain}"
OUTPUT="${OUTPUT:-${SMOKE_ROOT}/run}"
FUSION_OUTPUT="${FUSION_OUTPUT:-${SMOKE_ROOT}/fusion}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=40"
echo "smoke_root=${SMOKE_ROOT}"
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
  --limit-train "${LIMIT}" \
  --limit-valid "${LIMIT}" \
  --device "${DEVICE}" \
  --seed 42 \
  --batch-size 2 \
  --evaluation-batch-size 4 \
  --num-workers 2 \
  --amp-dtype bfloat16 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --maximum-epochs 2 \
  --patience 2 \
  --msa-width 64 \
  --msa-layers 2 \
  --msa-heads 4 \
  --pair-factor 4 \
  --pair-width 32 \
  --dropout 0.10 \
  --row-depth 16 \
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
  --limit-train "${LIMIT}" \
  --limit-valid "${LIMIT}" \
  --limit-test "${LIMIT}" \
  --device "${DEVICE}" \
  --seed 42 \
  --batch-size 2 \
  --evaluation-batch-size 4 \
  --num-workers 2 \
  --amp-dtype bfloat16 \
  --learning-rate 1.5e-4 \
  --encoder-learning-rate-scale 0.30 \
  --weight-decay 1e-4 \
  --maximum-epochs 3 \
  --patience 3 \
  --model-width 128 \
  --layers 2 \
  --heads 4 \
  --dropout 0.10 \
  --latent-width 256 \
  --msa-width 64 \
  --msa-layers 2 \
  --msa-heads 4 \
  --pair-factor 4 \
  --pair-width 32 \
  --row-depth 16 \
  --inference-subsets 1 \
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
  --limit "${LIMIT}" \
  --pca-components 8 \
  --oof-estimators 8 \
  --final-estimators 12 \
  --n-jobs 1 \
  --seed 20260826

printf 'v12_output=%s\n' "${OUTPUT}"
printf 'v12_result=%s\n' "${OUTPUT}/result.json"
printf 'v12_fusion_output=%s\n' "${FUSION_OUTPUT}"
printf 'v12_fusion_result=%s\n' "${FUSION_OUTPUT}/result.json"
