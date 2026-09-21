#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
EXPORT_ID="${EXPORT_ID:-optional_structure_privileged_msa_pair_v31_v22_train_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
SEALED_V22_CHECKPOINT="${SEALED_V22_CHECKPOINT:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt}"
OUTPUT="${OUTPUT:-${ROOT}/data/OptionalStructurePrivilegedMSAPair_v31/${EXPORT_ID}.npz}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m dta.optional_structure_privileged_msa_pair_v31.export_v22_train \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --protein-cache "${PROTEIN_CACHE}" \
  --ligand-cache "${LIGAND_CACHE}" \
  --sealed-v22-checkpoint "${SEALED_V22_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --evaluation-batch-size 4 \
  --num-workers 4 \
  --amp-dtype bfloat16

echo "v31_v22_train_predictions=${OUTPUT}"
