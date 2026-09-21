#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V15 smoke on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_interface_spatial_fusion_v15_smoke_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-24}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
COMPLEX_CACHE_ROOT="${COMPLEX_CACHE_ROOT:-${V9_DATA_ROOT}/complex_cache_full}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
BASE_ROOT="${BASE_ROOT:-${ROOT}/Model/LP_PDBBind_MSAPairFusion_v12/msa_pair_pretrained_v12_full_20260826_144811}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceFusion_v14/msa_interface_feature_fusion_v14_full_20260829_145451}"
V13_PRETRAIN="${V13_PRETRAIN:-${ROOT}/Model/LP_PDBBind_MSAInterfacePretraining_v13/msa_interface_pretrained_v13_full_20260827_132414/best_pretrain.pt}"
V13_AFFINITY="${V13_AFFINITY:-${ROOT}/Model/LP_PDBBind_MSAInterfaceExpert_v13/msa_interface_pretrained_v13_full_20260827_132414/best.pt}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/LP_PDBBind_MSAInterfaceSpatial_v15_${RUN_ID}}"
FEATURE_OUTPUT="${FEATURE_OUTPUT:-${SMOKE_ROOT}/features}"
FUSION_OUTPUT="${FUSION_OUTPUT:-${SMOKE_ROOT}/fusion}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${SMOKE_ROOT}/validation_selection}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

cd "${ROOT}"
echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} limit=${LIMIT} batch_size=1"
echo "smoke_root=${SMOKE_ROOT}"

for required_file in \
  "${ROW_CACHE}/row_cache_report.json" \
  "${V13_PRETRAIN}" \
  "${V13_AFFINITY}" \
  "${BASE_ROOT}/valid_predictions.npz" \
  "${BASE_ROOT}/test_predictions.npz" \
  "${BENCHMARK_ROOT}/valid_predictions.npz" \
  "${BENCHMARK_ROOT}/test_predictions.npz"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing required V15 smoke input: ${required_file}" >&2
    exit 2
  fi
done

"${PYTHON}" -m dta.msa_interface_spatial_fusion_v15.run_features \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
  --test-cache "${COMPLEX_CACHE_ROOT}/test" \
  --row-cache "${ROW_CACHE}" \
  --pretrain-checkpoint "${V13_PRETRAIN}" \
  --affinity-checkpoint "${V13_AFFINITY}" \
  --output "${FEATURE_OUTPUT}" \
  --limit-valid "${LIMIT}" \
  --limit-test "${LIMIT}" \
  --device "${DEVICE}" \
  --evaluation-batch-size 1 \
  --num-workers 2 \
  --amp-dtype bfloat16 \
  --row-depth 16 \
  --inference-subsets 1

"${PYTHON}" -m dta.msa_interface_spatial_fusion_v15.run_fusion \
  --base-root "${BASE_ROOT}" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --feature-root "${FEATURE_OUTPUT}" \
  --output "${SELECTION_OUTPUT}" \
  --limit "${LIMIT}" \
  --pca-components 8 \
  --minimum-validation-ci-gain -1.0 \
  --minimum-spearman-gain -1.0 \
  --maximum-rmse-increase 100.0 \
  --minimum-positive-folds 0 \
  --allow-zero-budget \
  --oof-estimators 8 \
  --final-estimators 12 \
  --n-jobs 1 \
  --seed 20260830

if ! grep -q "\"status\": \"validation_passed\"" "${SELECTION_OUTPUT}/result.json"; then
  echo "V15 smoke validation selection did not pass" >&2
  exit 2
fi

"${PYTHON}" -m dta.msa_interface_spatial_fusion_v15.run_features \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
  --test-cache "${COMPLEX_CACHE_ROOT}/test" \
  --row-cache "${ROW_CACHE}" \
  --pretrain-checkpoint "${V13_PRETRAIN}" \
  --affinity-checkpoint "${V13_AFFINITY}" \
  --output "${FEATURE_OUTPUT}" \
  --limit-valid "${LIMIT}" \
  --limit-test "${LIMIT}" \
  --device "${DEVICE}" \
  --evaluation-batch-size 1 \
  --num-workers 2 \
  --amp-dtype bfloat16 \
  --row-depth 16 \
  --inference-subsets 1 \
  --test-only \
  --selection-result "${SELECTION_OUTPUT}/result.json"

"${PYTHON}" -m dta.msa_interface_spatial_fusion_v15.run_fusion \
  --base-root "${BASE_ROOT}" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --feature-root "${FEATURE_OUTPUT}" \
  --output "${FUSION_OUTPUT}" \
  --limit "${LIMIT}" \
  --pca-components 8 \
  --minimum-validation-ci-gain -1.0 \
  --minimum-spearman-gain -1.0 \
  --maximum-rmse-increase 100.0 \
  --minimum-positive-folds 0 \
  --allow-zero-budget \
  --oof-estimators 8 \
  --final-estimators 12 \
  --n-jobs 1 \
  --seed 20260830

echo "v15_smoke_root=${SMOKE_ROOT}"
echo "v15_feature_result=${FEATURE_OUTPUT}/result.json"
echo "v15_selection_result=${SELECTION_OUTPUT}/result.json"
echo "v15_fusion_result=${FUSION_OUTPUT}/result.json"
