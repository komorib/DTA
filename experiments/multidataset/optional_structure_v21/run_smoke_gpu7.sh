#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_v21_smoke_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
OUTPUT="${OUTPUT:-${ROOT}/Model/OptionalStructureMultitask_v21/${RUN_ID}}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  -m dta.optional_structure_multitask_v21.run_training \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl" \
  --protein-cache "${PROTEIN_CACHE}" \
  --ligand-cache "${LIGAND_CACHE}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --limit-train 64 \
  --limit-valid 8 \
  --batch-size 3 \
  --evaluation-batch-size 3 \
  --gradient-accumulation 1 \
  --num-workers 0 \
  --maximum-epochs 1 \
  --steps-per-epoch 6 \
  --patience 1 \
  --width 96 \
  --heads 4 \
  --protein-layers 1 \
  --ligand-layers 1 \
  --cross-layers 1 \
  --latent-width 192 \
  --pair-width 64 \
  --structure-self-layers 1 \
  --structure-pair-layers 1 \
  --structure-token-layers 1 \
  --structure-top-pairs 16 \
  --maximum-pocket-tokens 48

echo "v21_smoke_output=${OUTPUT}"
