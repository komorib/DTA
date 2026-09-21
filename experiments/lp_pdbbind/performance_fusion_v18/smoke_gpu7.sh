#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V18 smoke training on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_interface_pair_pooling_v18_smoke_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-24}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
COMPLEX_CACHE_ROOT="${COMPLEX_CACHE_ROOT:-${V9_DATA_ROOT}/complex_cache_full}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/msa_interface_spatial_stability_fusion_v16_full_20260830_112824}"
V12_PRETRAIN="${V12_PRETRAIN:-${ROOT}/Model/LP_PDBBind_MSAPairPretraining_v12/msa_pair_pretrained_v12_full_20260826_144811/best_pretrain.pt}"
V12_AFFINITY="${V12_AFFINITY:-${ROOT}/Model/LP_PDBBind_MSAPairExpert_v12/msa_pair_pretrained_v12_full_20260826_144811/best.pt}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/LP_PDBBind_MSAInterfacePairPooling_v18_${RUN_ID}}"
PRETRAIN_OUTPUT="${PRETRAIN_OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAInterfacePretraining_v13/msa_interface_pretrained_v13_full_20260827_132414}"
OUTPUT="${OUTPUT:-${SMOKE_ROOT}/run}"
FUSION_OUTPUT="${FUSION_OUTPUT:-${SMOKE_ROOT}/fusion}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${FUSION_OUTPUT}_validation_selection}"
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
for required_file in \
  "${V12_PRETRAIN}" \
  "${V12_AFFINITY}" \
  "${ANCHOR_ROOT}/valid_predictions.npz" \
  "${ANCHOR_ROOT}/test_predictions.npz"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing required V12 artifact: ${required_file}" >&2
    exit 2
  fi
done

if [[ -f "${PRETRAIN_OUTPUT}/best_pretrain.pt" ]]; then
  echo "reusing_v18_smoke_pretraining=${PRETRAIN_OUTPUT}/best_pretrain.pt"
else
  "${PYTHON}" -m dta.msa_interface_pretrained_v13.run_pretraining \
    --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
    --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
    --train-cache "${COMPLEX_CACHE_ROOT}/train" \
    --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
    --row-cache "${ROW_CACHE}" \
    --initial-msa-checkpoint "${V12_PRETRAIN}" \
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
    --msa-width 192 \
    --msa-layers 4 \
    --msa-heads 6 \
    --pair-factor 8 \
    --pair-width 96 \
    --interface-blocks 2 \
    --dropout 0.10 \
    --row-depth 16 \
    --mask-probability 0.15 \
    --interface-mask-probability 0.35 \
    --distogram-weight 0.35 \
    --contact-weight 0.10 \
    --interface-distogram-weight 1.0 \
    --interface-contact-weight 0.30 \
    --match-weight 0.10
fi

if [[ -f "${OUTPUT}/best.pt" && -f "${OUTPUT}/valid_predictions.npz" && -f "${OUTPUT}/result.json" ]]; then
  echo "reusing_v18_smoke_validation=${OUTPUT}/valid_predictions.npz"
else
  "${PYTHON}" -m dta.msa_interface_pair_pooling_v18.run_training \
    --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
    --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
    --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
    --train-cache "${COMPLEX_CACHE_ROOT}/train" \
    --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
    --test-cache "${COMPLEX_CACHE_ROOT}/test" \
    --row-cache "${ROW_CACHE}" \
    --pretrained-checkpoint "${PRETRAIN_OUTPUT}/best_pretrain.pt" \
    --affinity-initial-checkpoint "${V12_AFFINITY}" \
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
    --encoder-learning-rate-scale 0.20 \
    --weight-decay 1e-4 \
    --maximum-epochs 3 \
    --patience 3 \
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
    --interface-blocks 2 \
    --pair-token-width 64 \
    --pair-token-heads 4 \
    --pair-token-layers 1 \
    --pair-token-count 32 \
    --row-depth 16 \
    --inference-subsets 1 \
    --rank-weight 0.40 \
    --mse-weight 0.15 \
    --contact-weight 0.05 \
    --residue-contact-weight 0.03 \
    --affinity-class-weight 0.05 \
    --distogram-weight 0.02 \
    --interface-distogram-weight 0.08 \
    --interface-contact-weight 0.05 \
    --match-weight 0.02 \
    --interface-mask-probability 0.15 \
    --skip-test
fi

if [[ -f "${SELECTION_OUTPUT}/result.json" ]]; then
  echo "reusing_v18_smoke_selection=${SELECTION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_pair_pooling_v18.run_fusion \
    --anchor-root "${ANCHOR_ROOT}" \
    --v18-root "${OUTPUT}" \
    --output "${SELECTION_OUTPUT}" \
    --limit "${LIMIT}" \
    --minimum-validation-ci-gain -1.0 \
    --minimum-spearman-gain -1.0 \
    --maximum-rmse-increase 999.0 \
    --minimum-positive-folds 0 \
    --allow-zero-contribution \
    --pca-components 8 \
    --oof-estimators 8 \
    --final-estimators 12 \
    --n-jobs 1 \
    --seed 20260901
fi

if ! grep -q "\"status\": \"validation_passed\"" "${SELECTION_OUTPUT}/result.json"; then
  echo "V18 smoke validation selection did not pass" >&2
  exit 2
fi

if [[ -f "${OUTPUT}/test_predictions.npz" ]]; then
  echo "reusing_v18_smoke_test=${OUTPUT}/test_predictions.npz"
else
  "${PYTHON}" -m dta.msa_interface_pair_pooling_v18.run_training \
    --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
    --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
    --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
    --train-cache "${COMPLEX_CACHE_ROOT}/train" \
    --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
    --test-cache "${COMPLEX_CACHE_ROOT}/test" \
    --row-cache "${ROW_CACHE}" \
    --pretrained-checkpoint "${PRETRAIN_OUTPUT}/best_pretrain.pt" \
    --affinity-initial-checkpoint "${V12_AFFINITY}" \
    --output "${OUTPUT}" \
    --limit-test "${LIMIT}" \
    --device "${DEVICE}" \
    --evaluation-batch-size 4 \
    --num-workers 2 \
    --test-only \
    --selection-result "${SELECTION_OUTPUT}/result.json"
fi

if [[ -f "${FUSION_OUTPUT}/result.json" ]]; then
  echo "reusing_v18_smoke_final_fusion=${FUSION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_pair_pooling_v18.run_fusion \
    --anchor-root "${ANCHOR_ROOT}" \
    --v18-root "${OUTPUT}" \
    --output "${FUSION_OUTPUT}" \
    --limit "${LIMIT}" \
    --minimum-validation-ci-gain -1.0 \
    --minimum-spearman-gain -1.0 \
    --maximum-rmse-increase 999.0 \
    --minimum-positive-folds 0 \
    --allow-zero-contribution \
    --pca-components 8 \
    --oof-estimators 8 \
    --final-estimators 12 \
    --n-jobs 1 \
    --seed 20260901
fi

echo "v18_smoke_root=${SMOKE_ROOT}"
echo "v18_pretrain_output=${PRETRAIN_OUTPUT}"
echo "v18_output=${OUTPUT}"
echo "v18_result=${OUTPUT}/result.json"
echo "v18_selection_result=${SELECTION_OUTPUT}/result.json"
echo "v18_fusion_output=${FUSION_OUTPUT}"
echo "v18_fusion_result=${FUSION_OUTPUT}/result.json"
