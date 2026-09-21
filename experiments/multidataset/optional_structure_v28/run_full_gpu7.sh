#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_stable_pair_v28_full_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
WARM_CHECKPOINT="${WARM_CHECKPOINT:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt}"
ANCHOR_VALID="${ANCHOR_VALID:-${ROOT}/Model/OptionalStructureSimplexRouter_v26/optional_structure_simplex_router_v26_20260910_161346/valid_predictions.npz}"
OUTPUT="${OUTPUT:-${ROOT}/Model/OptionalStructureStablePair_v28/${RUN_ID}}"
RESUME="${RESUME:-0}"

case "${RESUME}" in
  0|false|FALSE|no|NO) ;;
  1|true|TRUE|yes|YES) ;;
  *) echo "RESUME must be one of: 0, 1, false, true, no, yes" >&2; exit 2 ;;
esac

ARGS=(
  -m dta.optional_structure_stable_pair_v28.run_training
  --train-manifest "${MANIFEST_ROOT}/train.jsonl"
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl"
  --protein-cache "${PROTEIN_CACHE}"
  --ligand-cache "${LIGAND_CACHE}"
  --anchor-valid "${ANCHOR_VALID}"
  --warm-checkpoint "${WARM_CHECKPOINT}"
  --output "${OUTPUT}"
  --device cuda:0
  --batch-size 4
  --evaluation-batch-size 4
  --gradient-accumulation 4
  --num-workers 4
  --amp-dtype bfloat16
  --learning-rate 0.00010
  --pretrained-learning-rate-scale 0.05
  --structure-learning-rate-scale 0.35
  --maximum-epochs 45
  --steps-per-epoch 7000
  --patience 10
  --lp-sampling 0.30
  --davis-sampling 0.25
  --kiba-sampling 0.45
  --pair-batch-fraction 0.70
  --geometry-mask-fraction 0.25
  --structure-dropout 0.15
  --pretrained-freeze-epochs 2
  --pair-distillation-warmup-epochs 5
  --maximum-pair-distillation 4.0
  --base-anchor-distillation-weight 0.50
  --privileged-affinity-distillation-weight 0.02
  --pair-distillation-weight 0.03
  --minimum-structure-ci-gain 0.005
  --maximum-structure-rmse-increase 0.01
  --minimum-structure-spearman-gain 0.0
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

echo "v28_run_id=${RUN_ID}"
echo "v28_output=${OUTPUT}"
echo "v28_selection=${OUTPUT}/selection.json"
