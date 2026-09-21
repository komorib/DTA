#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_privileged_msa_pair_v31_full_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
SEALED_V22_CHECKPOINT="${SEALED_V22_CHECKPOINT:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt}"
V22_VALID="${V22_VALID:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/valid_predictions.npz}"
ANCHOR_VALID="${ANCHOR_VALID:-${ROOT}/Model/OptionalStructureSimplexRouter_v26/optional_structure_simplex_router_v26_20260910_161346/valid_predictions.npz}"
TEACHER_CACHE="${TEACHER_CACHE:?set TEACHER_CACHE to a completed train-only V31 cache directory}"
OUTPUT="${OUTPUT:-${ROOT}/Model/OptionalStructurePrivilegedMSAPair_v31/${RUN_ID}}"
RESUME="${RESUME:-0}"

case "${RESUME}" in
  0|false|FALSE|no|NO) ;;
  1|true|TRUE|yes|YES) ;;
  *) echo "RESUME must be one of: 0, 1, false, true, no, yes" >&2; exit 2 ;;
esac

ARGS=(
  -m dta.optional_structure_privileged_msa_pair_v31.run_training
  --train-manifest "${MANIFEST_ROOT}/train.jsonl"
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl"
  --protein-cache "${PROTEIN_CACHE}"
  --ligand-cache "${LIGAND_CACHE}"
  --sealed-v22-checkpoint "${SEALED_V22_CHECKPOINT}"
  --v22-valid "${V22_VALID}"
  --anchor-valid "${ANCHOR_VALID}"
  --teacher-cache "${TEACHER_CACHE}"
  --output "${OUTPUT}"
  --device cuda:0
  --batch-size 4
  --evaluation-batch-size 4
  --gradient-accumulation 4
  --num-workers 4
  --amp-dtype bfloat16
  --learning-rate 0.00015
  --maximum-epochs 35
  --steps-per-epoch 7000
  --patience 10
  --teacher-warmup-epochs 3
  --pair-batch-fraction 0.70
  --residual-width 384
  --maximum-residual 1.50
  --smooth-weight 1.0
  --mse-weight 0.25
  --rank-weight 0.20
  --teacher-value-weight 0.25
  --teacher-rank-weight 0.10
  --backbone-anchor-weight 0.04
  --delta-l2-weight 0.01
  --gate-center-weight 0.002
  --mix-center-weight 0.002
  --minimum-structure-ci-gain 0.005
  --maximum-structure-rmse-increase 0.01
  --minimum-structure-spearman-gain 0.0
  --minimum-student-ci-gain -0.001
  --maximum-student-rmse-increase 0.003
  --minimum-student-spearman-gain -0.001
  --minimum-anchor-ci-gain 0.0005
  --maximum-anchor-rmse-increase 0.003
  --minimum-anchor-spearman-gain 0.0
  --minimum-positive-folds 3
)
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "TRUE" || "${RESUME}" == "yes" || "${RESUME}" == "YES" ]]; then
  ARGS+=(--resume)
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ARGS[@]}"

echo "v31_run_id=${RUN_ID}"
echo "v31_output=${OUTPUT}"
echo "v31_selection=${OUTPUT}/selection.json"
