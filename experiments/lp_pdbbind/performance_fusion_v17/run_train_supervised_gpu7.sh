#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V17 extraction on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_interface_train_supervised_fusion_v17_full_$(date +%Y%m%d_%H%M%S)}"
EVALUATION_BATCH_SIZE="${EVALUATION_BATCH_SIZE:-2}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
COMPLEX_CACHE_ROOT="${COMPLEX_CACHE_ROOT:-${V9_DATA_ROOT}/complex_cache_full}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
V13_PRETRAIN="${V13_PRETRAIN:-${ROOT}/Model/LP_PDBBind_MSAInterfacePretraining_v13/msa_interface_pretrained_v13_full_20260827_132414/best_pretrain.pt}"
V13_AFFINITY="${V13_AFFINITY:-${ROOT}/Model/LP_PDBBind_MSAInterfaceExpert_v13/msa_interface_pretrained_v13_full_20260827_132414/best.pt}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/msa_interface_spatial_stability_fusion_v16_full_20260830_112824}"
FEATURE_OUTPUT="${FEATURE_OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceTrainFeatures_v17/${RUN_ID}}"
FUSION_ROOT="${FUSION_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceTrainSupervisedFusion_v17}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${FUSION_ROOT}/${RUN_ID}_validation_selection}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${FUSION_ROOT}/${RUN_ID}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

cd "${ROOT}"
echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} batch_size=${EVALUATION_BATCH_SIZE} memory_budget_gb=40"
echo "v16_anchor=${ANCHOR_ROOT}"
echo "feature_output=${FEATURE_OUTPUT}"
echo "selection_output=${SELECTION_OUTPUT}"
echo "final_output=${FINAL_OUTPUT}"

for required_valid_file in \
  "${ROW_CACHE}/row_cache_report.json" \
  "${V13_PRETRAIN}" \
  "${V13_AFFINITY}" \
  "${MANIFEST_ROOT}/train_manifest.json" \
  "${MANIFEST_ROOT}/valid_manifest.json" \
  "${ANCHOR_ROOT}/valid_predictions.npz"; do
  if [[ ! -f "${required_valid_file}" ]]; then
    echo "missing required V17 train/validation input: ${required_valid_file}" >&2
    exit 2
  fi
done

if [[ -f "${FEATURE_OUTPUT}/train_features.npz" && -f "${FEATURE_OUTPUT}/valid_features.npz" && -f "${FEATURE_OUTPUT}/result.json" ]]; then
  echo "reusing_v17_train_validation_features=${FEATURE_OUTPUT}"
else
  "${PYTHON}" -m dta.msa_interface_train_supervised_fusion_v17.run_features \
    --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
    --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
    --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
    --train-cache "${COMPLEX_CACHE_ROOT}/train" \
    --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
    --test-cache "${COMPLEX_CACHE_ROOT}/test" \
    --row-cache "${ROW_CACHE}" \
    --pretrain-checkpoint "${V13_PRETRAIN}" \
    --affinity-checkpoint "${V13_AFFINITY}" \
    --output "${FEATURE_OUTPUT}" \
    --device "${DEVICE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --num-workers 4 \
    --amp-dtype bfloat16 \
    --row-depth 64 \
    --inference-subsets 2
fi

if [[ -f "${SELECTION_OUTPUT}/result.json" ]]; then
  echo "reusing_v17_validation_selection=${SELECTION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_train_supervised_fusion_v17.run_fusion \
    --feature-root "${FEATURE_OUTPUT}" \
    --anchor-root "${ANCHOR_ROOT}" \
    --output "${SELECTION_OUTPUT}" \
    --validation-only \
    --pca-components 96 192 \
    --minimum-validation-ci-gain 0.001 \
    --minimum-spearman-gain 0.0 \
    --maximum-rmse-increase 0.003 \
    --minimum-positive-folds 4 \
    --selection-estimators 400 \
    --final-estimators 1200 \
    --hist-max-iter 400 \
    --n-jobs 8 \
    --seed 20260901
fi

if grep -q '"status": "validation_failed"' "${SELECTION_OUTPUT}/result.json"; then
  echo "v17_validation_gate=failed; test inputs were not read"
  echo "v17_feature_output=${FEATURE_OUTPUT}"
  echo "v17_selection_result=${SELECTION_OUTPUT}/result.json"
  exit 0
fi
if ! grep -q '"status": "validation_passed"' "${SELECTION_OUTPUT}/result.json"; then
  echo "unexpected V17 validation selection status" >&2
  exit 2
fi

for required_test_file in \
  "${MANIFEST_ROOT}/test_manifest.json" \
  "${ANCHOR_ROOT}/test_predictions.npz"; do
  if [[ ! -f "${required_test_file}" ]]; then
    echo "missing required V17 test input after validation passed: ${required_test_file}" >&2
    exit 2
  fi
done

if [[ -f "${FEATURE_OUTPUT}/test_features.npz" ]]; then
  echo "reusing_v17_test_features=${FEATURE_OUTPUT}/test_features.npz"
else
  "${PYTHON}" -m dta.msa_interface_train_supervised_fusion_v17.run_features \
    --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
    --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
    --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
    --train-cache "${COMPLEX_CACHE_ROOT}/train" \
    --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
    --test-cache "${COMPLEX_CACHE_ROOT}/test" \
    --row-cache "${ROW_CACHE}" \
    --pretrain-checkpoint "${V13_PRETRAIN}" \
    --affinity-checkpoint "${V13_AFFINITY}" \
    --output "${FEATURE_OUTPUT}" \
    --device "${DEVICE}" \
    --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
    --num-workers 4 \
    --amp-dtype bfloat16 \
    --row-depth 64 \
    --inference-subsets 2 \
    --test-only \
    --selection-result "${SELECTION_OUTPUT}/result.json"
fi

if [[ -f "${FINAL_OUTPUT}/result.json" ]]; then
  echo "reusing_v17_final=${FINAL_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_train_supervised_fusion_v17.run_fusion \
    --feature-root "${FEATURE_OUTPUT}" \
    --anchor-root "${ANCHOR_ROOT}" \
    --output "${FINAL_OUTPUT}" \
    --selection-result "${SELECTION_OUTPUT}/result.json" \
    --pca-components 96 192 \
    --minimum-validation-ci-gain 0.001 \
    --minimum-spearman-gain 0.0 \
    --maximum-rmse-increase 0.003 \
    --minimum-positive-folds 4 \
    --selection-estimators 400 \
    --final-estimators 1200 \
    --hist-max-iter 400 \
    --n-jobs 8 \
    --seed 20260901
fi

echo "v17_feature_output=${FEATURE_OUTPUT}"
echo "v17_feature_result=${FEATURE_OUTPUT}/result.json"
echo "v17_selection_result=${SELECTION_OUTPUT}/result.json"
echo "v17_final_output=${FINAL_OUTPUT}"
echo "v17_final_result=${FINAL_OUTPUT}/result.json"

