#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V19 test inference on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_interface_pair_latent_fusion_v19_$(date +%Y%m%d_%H%M%S)}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
COMPLEX_CACHE_ROOT="${COMPLEX_CACHE_ROOT:-${V9_DATA_ROOT}/complex_cache_full}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/msa_interface_spatial_stability_fusion_v16_full_20260830_112824}"
V18_ROOT="${V18_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfacePairPooling_v18/msa_interface_pair_pooling_v18_full_20260901_182434}"
FEATURE_OUTPUT="${FEATURE_OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAInterfacePairLatent_v19/${RUN_ID}}"
FUSION_OUTPUT="${FUSION_OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAInterfacePairLatentFusion_v19/${RUN_ID}}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${FUSION_OUTPUT}_validation_selection}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

cd "${ROOT}"
echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=40"
echo "v18_root=${V18_ROOT}"
echo "feature_output=${FEATURE_OUTPUT}"
echo "fusion_output=${FUSION_OUTPUT}"

for required_file in \
  "${ROW_CACHE}/row_cache_report.json" \
  "${V18_ROOT}/best.pt" \
  "${V18_ROOT}/result.json" \
  "${V18_ROOT}/valid_predictions.npz" \
  "${ANCHOR_ROOT}/valid_predictions.npz" \
  "${ANCHOR_ROOT}/test_predictions.npz"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing V19 input: ${required_file}" >&2
    exit 2
  fi
done

if [[ -f "${FEATURE_OUTPUT}/result.json" && -f "${FEATURE_OUTPUT}/valid_features.npz" ]]; then
  echo "reusing_v19_validation_features=${FEATURE_OUTPUT}/valid_features.npz"
else
  "${PYTHON}" -m dta.msa_interface_pair_latent_fusion_v19.run_features \
    --v18-root "${V18_ROOT}" \
    --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
    --test-cache "${COMPLEX_CACHE_ROOT}/test" \
    --row-cache "${ROW_CACHE}" \
    --output "${FEATURE_OUTPUT}" \
    --device "${DEVICE}" \
    --evaluation-batch-size 4 \
    --num-workers 4
fi

if [[ -f "${SELECTION_OUTPUT}/result.json" ]]; then
  echo "reusing_v19_validation_selection=${SELECTION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_pair_latent_fusion_v19.run_fusion \
    --anchor-root "${ANCHOR_ROOT}" \
    --v19-root "${FEATURE_OUTPUT}" \
    --output "${SELECTION_OUTPUT}" \
    --minimum-validation-ci-gain 0.0005 \
    --minimum-spearman-gain 0.0 \
    --maximum-rmse-increase 0.003 \
    --minimum-positive-folds 4 \
    --pca-components 128 \
    --oof-estimators 400 \
    --final-estimators 1200 \
    --n-jobs 8 \
    --seed 20260901
fi

if grep -q "\"status\": \"validation_failed\"" "${SELECTION_OUTPUT}/result.json"; then
  echo "v19_validation_gate=failed; test inference was not run"
  echo "v19_feature_output=${FEATURE_OUTPUT}"
  echo "v19_selection_result=${SELECTION_OUTPUT}/result.json"
  exit 0
fi
if ! grep -q "\"status\": \"validation_passed\"" "${SELECTION_OUTPUT}/result.json"; then
  echo "unexpected V19 validation selection status" >&2
  exit 2
fi

if [[ -f "${FEATURE_OUTPUT}/test_features.npz" ]]; then
  echo "reusing_v19_test_features=${FEATURE_OUTPUT}/test_features.npz"
else
  "${PYTHON}" -m dta.msa_interface_pair_latent_fusion_v19.run_features \
    --v18-root "${V18_ROOT}" \
    --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
    --test-cache "${COMPLEX_CACHE_ROOT}/test" \
    --row-cache "${ROW_CACHE}" \
    --output "${FEATURE_OUTPUT}" \
    --device "${DEVICE}" \
    --evaluation-batch-size 4 \
    --num-workers 4 \
    --test-only \
    --selection-result "${SELECTION_OUTPUT}/result.json"
fi

if [[ -f "${FUSION_OUTPUT}/result.json" ]]; then
  echo "reusing_v19_final_fusion=${FUSION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_pair_latent_fusion_v19.run_fusion \
    --anchor-root "${ANCHOR_ROOT}" \
    --v19-root "${FEATURE_OUTPUT}" \
    --output "${FUSION_OUTPUT}" \
    --selection-result "${SELECTION_OUTPUT}/result.json" \
    --minimum-validation-ci-gain 0.0005 \
    --minimum-spearman-gain 0.0 \
    --maximum-rmse-increase 0.003 \
    --minimum-positive-folds 4 \
    --pca-components 128 \
    --oof-estimators 400 \
    --final-estimators 1200 \
    --n-jobs 8 \
    --seed 20260901
fi

echo "v19_feature_output=${FEATURE_OUTPUT}"
echo "v19_feature_result=${FEATURE_OUTPUT}/result.json"
echo "v19_selection_result=${SELECTION_OUTPUT}/result.json"
echo "v19_fusion_output=${FUSION_OUTPUT}"
echo "v19_fusion_result=${FUSION_OUTPUT}/result.json"
