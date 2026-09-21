#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
RUN_ID="${RUN_ID:-msa_interface_pair_latent_stability_fusion_v20_full_$(date +%Y%m%d_%H%M%S)}"
BASE_ROOT="${BASE_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/msa_interface_spatial_stability_fusion_v16_full_20260830_112824}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfacePairLatentFusion_v19/msa_interface_pair_latent_fusion_v19_full_20260902_114500}"
FEATURE_ROOT="${FEATURE_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfacePairLatent_v19/msa_interface_pair_latent_fusion_v19_full_20260902_114500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfacePairLatentStabilityFusion_v20}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${OUTPUT_ROOT}/${RUN_ID}_validation_selection}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${OUTPUT_ROOT}/${RUN_ID}}"

export PYTHONUNBUFFERED=1
cd "${ROOT}"

echo "V20 is CPU-only and reuses sealed V16/V19 artifacts; no GPU is started."
echo "v16_base=${BASE_ROOT}"
echo "v19_benchmark=${BENCHMARK_ROOT}"
echo "v19_features=${FEATURE_ROOT}"
echo "selection_output=${SELECTION_OUTPUT}"
echo "final_output=${FINAL_OUTPUT}"

for required_valid_file in \
  "${BASE_ROOT}/valid_predictions.npz" \
  "${BENCHMARK_ROOT}/valid_predictions.npz" \
  "${FEATURE_ROOT}/valid_features.npz"; do
  if [[ ! -f "${required_valid_file}" ]]; then
    echo "missing required V20 validation input: ${required_valid_file}" >&2
    exit 2
  fi
done

if [[ -f "${SELECTION_OUTPUT}/result.json" ]]; then
  echo "reusing_v20_validation_selection=${SELECTION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_pair_latent_stability_fusion_v20.run_fusion \
    --base-root "${BASE_ROOT}" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --feature-root "${FEATURE_ROOT}" \
    --output "${SELECTION_OUTPUT}" \
    --validation-only \
    --pca-components 64 \
    --minimum-validation-ci-gain 0.0005 \
    --minimum-spearman-gain 0.0 \
    --maximum-rmse-increase 0.003 \
    --minimum-positive-folds 4 \
    --oof-estimators 300 \
    --final-estimators 1200 \
    --n-jobs 8 \
    --seed 20260902
fi

if grep -q '"status": "validation_failed"' "${SELECTION_OUTPUT}/result.json"; then
  echo "v20_validation_gate=failed; test inputs were not read"
  echo "v20_selection_result=${SELECTION_OUTPUT}/result.json"
  exit 0
fi
if ! grep -q '"status": "validation_passed"' "${SELECTION_OUTPUT}/result.json"; then
  echo "unexpected V20 validation selection status" >&2
  exit 2
fi

for required_test_file in \
  "${BASE_ROOT}/test_predictions.npz" \
  "${BENCHMARK_ROOT}/test_predictions.npz" \
  "${FEATURE_ROOT}/test_features.npz"; do
  if [[ ! -f "${required_test_file}" ]]; then
    echo "missing required V20 test input after validation passed: ${required_test_file}" >&2
    exit 2
  fi
done

if [[ -f "${FINAL_OUTPUT}/result.json" ]]; then
  echo "reusing_v20_final=${FINAL_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_pair_latent_stability_fusion_v20.run_fusion \
    --base-root "${BASE_ROOT}" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --feature-root "${FEATURE_ROOT}" \
    --output "${FINAL_OUTPUT}" \
    --selection-result "${SELECTION_OUTPUT}/result.json" \
    --pca-components 64 \
    --minimum-validation-ci-gain 0.0005 \
    --minimum-spearman-gain 0.0 \
    --maximum-rmse-increase 0.003 \
    --minimum-positive-folds 4 \
    --oof-estimators 300 \
    --final-estimators 1200 \
    --n-jobs 8 \
    --seed 20260902
fi

echo "v20_selection_result=${SELECTION_OUTPUT}/result.json"
echo "v20_final_output=${FINAL_OUTPUT}"
echo "v20_final_result=${FINAL_OUTPUT}/result.json"
