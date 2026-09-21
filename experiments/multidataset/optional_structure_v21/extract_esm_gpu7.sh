#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  -m dta.optional_structure_multitask_v21.extract_esm \
  --manifests \
    "${MANIFEST_ROOT}/train.jsonl" \
    "${MANIFEST_ROOT}/valid.jsonl" \
    "${MANIFEST_ROOT}/test.jsonl" \
  --output "${PROTEIN_CACHE}" \
  --backend transformers \
  --model facebook/esm2_t33_650M_UR50D \
  --maximum-tokens 512 \
  --window-size 1022 \
  --overlap 128 \
  --window-batch-size 4 \
  --precision bf16 \
  --device cuda:0 \
  --resume

echo "protein_cache=${PROTEIN_CACHE}"
