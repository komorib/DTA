#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:?set RUN_ID to the validation-passed V22 run}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/Model/OptionalStructureStaged_v22/${RUN_ID}}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${OUTPUT_ROOT}/structure}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  -m dta.optional_structure_staged_v22.run_test \
  --selection-output "${SELECTION_OUTPUT}" \
  --test-manifest "${MANIFEST_ROOT}/test.jsonl" \
  --device cuda:0 \
  --num-workers 4 \
  --evaluation-batch-size 4

echo "v22_result=${SELECTION_OUTPUT}/result.json"
