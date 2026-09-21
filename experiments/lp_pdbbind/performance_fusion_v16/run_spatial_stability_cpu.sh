#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
RUN_ID="${RUN_ID:-msa_interface_spatial_stability_fusion_v16_full_$(date +%Y%m%d_%H%M%S)}"
BASE_ROOT="${BASE_ROOT:-${ROOT}/Model/LP_PDBBind_MSAPairFusion_v12/msa_pair_pretrained_v12_full_20260826_144811}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceSpatialFusion_v15/msa_interface_spatial_fusion_v15_full_20260829_160048}"
FEATURE_ROOT="${FEATURE_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceSpatialFeatures_v15/msa_interface_spatial_fusion_v15_full_20260829_160048}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${OUTPUT_ROOT}/${RUN_ID}_validation_selection}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${OUTPUT_ROOT}/${RUN_ID}}"

export PYTHONUNBUFFERED=1
cd "${ROOT}"

echo "v12_base=${BASE_ROOT}"
echo "v15_benchmark=${BENCHMARK_ROOT}"
echo "v15_features=${FEATURE_ROOT}"
echo "selection_output=${SELECTION_OUTPUT}"
echo "final_output=${FINAL_OUTPUT}"

for required_valid_file in \
  "${BASE_ROOT}/valid_predictions.npz" \
  "${BENCHMARK_ROOT}/valid_predictions.npz" \
  "${FEATURE_ROOT}/valid_features.npz"; do
  if [[ ! -f "${required_valid_file}" ]]; then
    echo "missing required V16 validation input: ${required_valid_file}" >&2
    exit 2
  fi
done

if [[ -f "${SELECTION_OUTPUT}/result.json" ]]; then
  echo "reusing_v16_validation_selection=${SELECTION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_spatial_stability_fusion_v16.run_fusion \
    --base-root "${BASE_ROOT}" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --feature-root "${FEATURE_ROOT}" \
    --output "${SELECTION_OUTPUT}" \
    --validation-only \
    --pca-components 96 \
    --minimum-validation-ci-gain 0.0005 \
    --minimum-spearman-gain 0.0 \
    --maximum-rmse-increase 0.003 \
    --minimum-positive-folds 4 \
    --oof-estimators 300 \
    --final-estimators 1200 \
    --n-jobs 8 \
    --seed 20260831
fi

if grep -q "\"status\": \"validation_failed\"" "${SELECTION_OUTPUT}/result.json"; then
  echo "v16_validation_gate=failed; test inputs were not read"
  echo "v16_selection_result=${SELECTION_OUTPUT}/result.json"
  exit 0
fi
if ! grep -q "\"status\": \"validation_passed\"" "${SELECTION_OUTPUT}/result.json"; then
  echo "unexpected V16 validation selection status" >&2
  exit 2
fi

for required_test_file in \
  "${BASE_ROOT}/test_predictions.npz" \
  "${BENCHMARK_ROOT}/test_predictions.npz" \
  "${FEATURE_ROOT}/test_features.npz"; do
  if [[ ! -f "${required_test_file}" ]]; then
    echo "missing required V16 test input after validation passed: ${required_test_file}" >&2
    exit 2
  fi
done

if [[ -f "${FINAL_OUTPUT}/result.json" ]]; then
  echo "reusing_v16_final=${FINAL_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_spatial_stability_fusion_v16.run_fusion \
    --base-root "${BASE_ROOT}" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --feature-root "${FEATURE_ROOT}" \
    --output "${FINAL_OUTPUT}" \
    --selection-result "${SELECTION_OUTPUT}/result.json" \
    --pca-components 96 \
    --minimum-validation-ci-gain 0.0005 \
    --minimum-spearman-gain 0.0 \
    --maximum-rmse-increase 0.003 \
    --minimum-positive-folds 4 \
    --oof-estimators 300 \
    --final-estimators 1200 \
    --n-jobs 8 \
    --seed 20260831
fi

echo "v16_selection_result=${SELECTION_OUTPUT}/result.json"
echo "v16_final_output=${FINAL_OUTPUT}"
echo "v16_final_result=${FINAL_OUTPUT}/result.json"
