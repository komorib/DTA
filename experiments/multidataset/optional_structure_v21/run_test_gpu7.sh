#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:?set RUN_ID to the sealed V21 training run}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${ROOT}/Model/OptionalStructureMultitask_v21/${RUN_ID}}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  -m dta.optional_structure_multitask_v21.run_test \
  --selection-output "${SELECTION_OUTPUT}" \
  --test-manifest "${MANIFEST_ROOT}/test.jsonl" \
  --device cuda:0 \
  --num-workers 4 \
  --evaluation-batch-size 4

echo "v21_result=${SELECTION_OUTPUT}/result.json"
