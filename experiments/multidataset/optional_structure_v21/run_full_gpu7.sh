#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_v21_full_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
OUTPUT="${OUTPUT:-${ROOT}/Model/OptionalStructureMultitask_v21/${RUN_ID}}"
RESUME="${RESUME:-0}"

case "${RESUME}" in
  0|false|FALSE|no|NO)
    ;;
  1|true|TRUE|yes|YES)
    ;;
  *)
    echo "RESUME must be one of: 0, 1, false, true, no, yes" >&2
    exit 2
    ;;
esac

TRAIN_ARGS=(
  -m dta.optional_structure_multitask_v21.run_training
  --train-manifest "${MANIFEST_ROOT}/train.jsonl"
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl"
  --protein-cache "${PROTEIN_CACHE}"
  --ligand-cache "${LIGAND_CACHE}"
  --output "${OUTPUT}"
  --device cuda:0
  --batch-size 4
  --evaluation-batch-size 4
  --gradient-accumulation 4
  --num-workers 4
  --amp-dtype bfloat16
  --maximum-epochs 60
  --steps-per-epoch 3000
  --patience 12
  --structure-dropout 0.30
  --lp-sampling 0.40
  --davis-sampling 0.25
  --kiba-sampling 0.35
)
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "TRUE" || "${RESUME}" == "yes" || "${RESUME}" == "YES" ]]; then
  TRAIN_ARGS+=(--resume)
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"

echo "v21_output=${OUTPUT}"
echo "v21_selection=${OUTPUT}/selection.json"
