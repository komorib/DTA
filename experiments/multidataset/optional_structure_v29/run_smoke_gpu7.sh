#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_sealed_residual_v29_smoke_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
SEALED_V22_CHECKPOINT="${SEALED_V22_CHECKPOINT:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt}"
V22_VALID="${V22_VALID:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/valid_predictions.npz}"
ANCHOR_VALID="${ANCHOR_VALID:-${ROOT}/Model/OptionalStructureSimplexRouter_v26/optional_structure_simplex_router_v26_20260910_161346/valid_predictions.npz}"
OUTPUT="${OUTPUT:-${ROOT}/Model/OptionalStructureSealedResidual_v29/${RUN_ID}}"
RESUME="${RESUME:-0}"

case "${RESUME}" in
  0|false|FALSE|no|NO) ;;
  1|true|TRUE|yes|YES) ;;
  *) echo "RESUME must be one of: 0, 1, false, true, no, yes" >&2; exit 2 ;;
esac

ARGS=(
  -m dta.optional_structure_sealed_residual_v29.run_training
  --train-manifest "${MANIFEST_ROOT}/train.jsonl"
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl"
  --protein-cache "${PROTEIN_CACHE}"
  --ligand-cache "${LIGAND_CACHE}"
  --sealed-v22-checkpoint "${SEALED_V22_CHECKPOINT}"
  --v22-valid "${V22_VALID}"
  --anchor-valid "${ANCHOR_VALID}"
  --output "${OUTPUT}"
  --device cuda:0
  --limit-train 128
  --limit-valid 64
  --batch-size 4
  --evaluation-batch-size 4
  --gradient-accumulation 1
  --num-workers 0
  --amp-dtype bfloat16
  --learning-rate 0.00020
  --maximum-epochs 3
  --steps-per-epoch 4
  --patience 3
  --pair-batch-fraction 0.70
  --residual-width 192
  --maximum-residual 0.75
  --minimum-structure-ci-gain -1.0
  --maximum-structure-rmse-increase 10.0
  --minimum-structure-spearman-gain -1.0
  --minimum-residual-ci-gain -1.0
  --maximum-residual-rmse-increase 10.0
  --minimum-residual-spearman-gain -1.0
  --minimum-anchor-ci-gain -1.0
  --maximum-anchor-rmse-increase 10.0
  --minimum-anchor-spearman-gain -1.0
  --minimum-positive-folds 0
)
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "TRUE" || "${RESUME}" == "yes" || "${RESUME}" == "YES" ]]; then
  ARGS+=(--resume)
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ARGS[@]}"

echo "v29_smoke_output=${OUTPUT}"
echo "v29_smoke_selection=${OUTPUT}/selection.json"
