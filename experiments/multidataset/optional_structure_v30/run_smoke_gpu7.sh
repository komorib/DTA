#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_privileged_msa_v30_smoke_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
SEALED_V22_CHECKPOINT="${SEALED_V22_CHECKPOINT:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt}"
V22_VALID="${V22_VALID:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/valid_predictions.npz}"
ANCHOR_VALID="${ANCHOR_VALID:-${ROOT}/Model/OptionalStructureSimplexRouter_v26/optional_structure_simplex_router_v26_20260910_161346/valid_predictions.npz}"
TEACHER_CACHE="${TEACHER_CACHE:?set TEACHER_CACHE to a completed train-only V30 cache directory}"
OUTPUT="${OUTPUT:-${ROOT}/Model/OptionalStructurePrivilegedMSA_v30/${RUN_ID}}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m dta.optional_structure_privileged_msa_v30.run_training \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl" \
  --protein-cache "${PROTEIN_CACHE}" \
  --ligand-cache "${LIGAND_CACHE}" \
  --sealed-v22-checkpoint "${SEALED_V22_CHECKPOINT}" \
  --v22-valid "${V22_VALID}" \
  --anchor-valid "${ANCHOR_VALID}" \
  --teacher-cache "${TEACHER_CACHE}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --limit-train 128 \
  --limit-valid 64 \
  --batch-size 4 \
  --evaluation-batch-size 4 \
  --gradient-accumulation 1 \
  --num-workers 0 \
  --amp-dtype bfloat16 \
  --learning-rate 0.00015 \
  --maximum-epochs 3 \
  --steps-per-epoch 4 \
  --patience 3 \
  --teacher-warmup-epochs 1 \
  --residual-width 384 \
  --maximum-residual 1.50 \
  --minimum-structure-ci-gain -1.0 \
  --maximum-structure-rmse-increase 10.0 \
  --minimum-structure-spearman-gain -1.0 \
  --minimum-student-ci-gain -1.0 \
  --maximum-student-rmse-increase 10.0 \
  --minimum-student-spearman-gain -1.0 \
  --minimum-anchor-ci-gain -1.0 \
  --maximum-anchor-rmse-increase 10.0 \
  --minimum-anchor-spearman-gain -1.0 \
  --minimum-positive-folds 0

echo "v30_smoke_output=${OUTPUT}"
echo "v30_smoke_selection=${OUTPUT}/selection.json"
