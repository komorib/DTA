#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:?set RUN_ID to a validation-passed V30 run}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
ANCHOR_TEST="${ANCHOR_TEST:-${ROOT}/Model/OptionalStructureSimplexRouter_v26/optional_structure_simplex_router_v26_20260910_161346/test_predictions.npz}"
V22_TEST="${V22_TEST:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/test_predictions.npz}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${ROOT}/Model/OptionalStructurePrivilegedMSA_v30/${RUN_ID}}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m dta.optional_structure_privileged_msa_v30.run_test \
  --selection-output "${SELECTION_OUTPUT}" \
  --test-manifest "${MANIFEST_ROOT}/test.jsonl" \
  --anchor-test "${ANCHOR_TEST}" \
  --v22-test "${V22_TEST}" \
  --device cuda:0 \
  --num-workers 4 \
  --evaluation-batch-size 4

echo "v30_result=${SELECTION_OUTPUT}/result.json"
