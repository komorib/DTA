#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_staged_v22_smoke_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/Model/OptionalStructureStaged_v22/${RUN_ID}}"
SEQUENCE_OUTPUT="${OUTPUT_ROOT}/sequence"
STRUCTURE_OUTPUT="${OUTPUT_ROOT}/structure"

MODEL_ARGS=(
  --width 96
  --heads 4
  --protein-layers 1
  --ligand-layers 1
  --cross-layers 1
  --latent-width 192
  --pair-width 64
  --structure-self-layers 1
  --structure-pair-layers 1
  --structure-token-layers 1
  --structure-top-pairs 16
  --maximum-pocket-tokens 48
  --adapter-width 48
)

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  -m dta.optional_structure_staged_v22.run_sequence \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl" \
  --protein-cache "${PROTEIN_CACHE}" \
  --ligand-cache "${LIGAND_CACHE}" \
  --output "${SEQUENCE_OUTPUT}" \
  --device cuda:0 \
  --limit-train 128 \
  --limit-valid 64 \
  --batch-size 3 \
  --evaluation-batch-size 3 \
  --gradient-accumulation 1 \
  --num-workers 0 \
  --maximum-epochs 1 \
  --steps-per-epoch 6 \
  --patience 1 \
  "${MODEL_ARGS[@]}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  -m dta.optional_structure_staged_v22.run_structure \
  --sequence-output "${SEQUENCE_OUTPUT}" \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl" \
  --protein-cache "${PROTEIN_CACHE}" \
  --ligand-cache "${LIGAND_CACHE}" \
  --output "${STRUCTURE_OUTPUT}" \
  --device cuda:0 \
  --limit-train 128 \
  --limit-valid 64 \
  --batch-size 3 \
  --evaluation-batch-size 3 \
  --gradient-accumulation 1 \
  --num-workers 0 \
  --maximum-epochs 1 \
  --steps-per-epoch 6 \
  --patience 1 \
  --minimum-structure-ci-gain -1.0 \
  --maximum-structure-rmse-increase 10.0 \
  --minimum-structure-spearman-gain -1.0 \
  "${MODEL_ARGS[@]}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  -m dta.optional_structure_staged_v22.run_test \
  --selection-output "${STRUCTURE_OUTPUT}" \
  --test-manifest "${MANIFEST_ROOT}/test.jsonl" \
  --device cuda:0 \
  --num-workers 0 \
  --evaluation-batch-size 3 \
  --limit-test 8

echo "v22_smoke_root=${OUTPUT_ROOT}"
echo "v22_sequence_selection=${SEQUENCE_OUTPUT}/selection.json"
echo "v22_structure_selection=${STRUCTURE_OUTPUT}/selection.json"
echo "v22_smoke_result=${STRUCTURE_OUTPUT}/result.json"
