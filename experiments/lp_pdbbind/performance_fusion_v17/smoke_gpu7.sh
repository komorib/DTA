#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V17 smoke on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_interface_train_supervised_fusion_v17_smoke_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind"
CACHE_ROOT="${ROOT}/data/LP_PDBBind_MSA_v9/complex_cache_full"
ROW_CACHE="${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128"
PRETRAIN="${ROOT}/Model/LP_PDBBind_MSAInterfacePretraining_v13/msa_interface_pretrained_v13_full_20260827_132414/best_pretrain.pt"
AFFINITY="${ROOT}/Model/LP_PDBBind_MSAInterfaceExpert_v13/msa_interface_pretrained_v13_full_20260827_132414/best.pt"
ANCHOR="${ROOT}/Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/msa_interface_spatial_stability_fusion_v16_full_20260830_112824"
FEATURES="${ROOT}/Model/LP_PDBBind_MSAInterfaceTrainFeatures_v17/${RUN_ID}"
SELECTION="${ROOT}/Model/LP_PDBBind_MSAInterfaceTrainSupervisedFusion_v17/${RUN_ID}_validation_selection"
FINAL="${ROOT}/Model/LP_PDBBind_MSAInterfaceTrainSupervisedFusion_v17/${RUN_ID}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
cd "${ROOT}"

"${PYTHON}" -m dta.msa_interface_train_supervised_fusion_v17.run_features \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --train-cache "${CACHE_ROOT}/train" \
  --valid-cache "${CACHE_ROOT}/valid" \
  --test-cache "${CACHE_ROOT}/test" \
  --row-cache "${ROW_CACHE}" \
  --pretrain-checkpoint "${PRETRAIN}" \
  --affinity-checkpoint "${AFFINITY}" \
  --output "${FEATURES}" \
  --device "${DEVICE}" \
  --evaluation-batch-size 1 \
  --num-workers 2 \
  --amp-dtype bfloat16 \
  --row-depth 8 \
  --inference-subsets 1 \
  --limit-train 64 \
  --limit-valid 40 \
  --limit-test 24

"${PYTHON}" -m dta.msa_interface_train_supervised_fusion_v17.run_fusion \
  --feature-root "${FEATURES}" \
  --anchor-root "${ANCHOR}" \
  --output "${SELECTION}" \
  --validation-only \
  --limit-train 64 \
  --limit-valid 40 \
  --limit-test 24 \
  --pca-components 8 16 \
  --minimum-validation-ci-gain -1.0 \
  --minimum-spearman-gain -1.0 \
  --maximum-rmse-increase 100.0 \
  --minimum-positive-folds 0 \
  --allow-zero-head-weight \
  --selection-estimators 5 \
  --final-estimators 7 \
  --hist-max-iter 12 \
  --n-jobs 1 \
  --seed 1701

"${PYTHON}" -m dta.msa_interface_train_supervised_fusion_v17.run_features \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --train-cache "${CACHE_ROOT}/train" \
  --valid-cache "${CACHE_ROOT}/valid" \
  --test-cache "${CACHE_ROOT}/test" \
  --row-cache "${ROW_CACHE}" \
  --pretrain-checkpoint "${PRETRAIN}" \
  --affinity-checkpoint "${AFFINITY}" \
  --output "${FEATURES}" \
  --device "${DEVICE}" \
  --evaluation-batch-size 1 \
  --num-workers 2 \
  --amp-dtype bfloat16 \
  --row-depth 8 \
  --inference-subsets 1 \
  --limit-train 64 \
  --limit-valid 40 \
  --limit-test 24 \
  --test-only \
  --selection-result "${SELECTION}/result.json"

"${PYTHON}" -m dta.msa_interface_train_supervised_fusion_v17.run_fusion \
  --feature-root "${FEATURES}" \
  --anchor-root "${ANCHOR}" \
  --output "${FINAL}" \
  --selection-result "${SELECTION}/result.json" \
  --limit-train 64 \
  --limit-valid 40 \
  --limit-test 24 \
  --pca-components 8 16 \
  --minimum-validation-ci-gain -1.0 \
  --minimum-spearman-gain -1.0 \
  --maximum-rmse-increase 100.0 \
  --minimum-positive-folds 0 \
  --allow-zero-head-weight \
  --selection-estimators 5 \
  --final-estimators 7 \
  --hist-max-iter 12 \
  --n-jobs 1 \
  --seed 1701

echo "v17_smoke_feature_result=${FEATURES}/result.json"
echo "v17_smoke_selection_result=${SELECTION}/result.json"
echo "v17_smoke_final_result=${FINAL}/result.json"

