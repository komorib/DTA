#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_staged_v22_full_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${ROOT}/Model/OptionalStructureMultitask_v21/optional_structure_v21_full_20260905_143425/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/Model/OptionalStructureStaged_v22/${RUN_ID}}"
OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/sequence}"
RESUME="${RESUME:-0}"

case "${RESUME}" in
  0|false|FALSE|no|NO) ;;
  1|true|TRUE|yes|YES) ;;
  *) echo "RESUME must be one of: 0, 1, false, true, no, yes" >&2; exit 2 ;;
esac

ARGS=(
  -m dta.optional_structure_staged_v22.run_sequence
  --train-manifest "${MANIFEST_ROOT}/train.jsonl"
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl"
  --protein-cache "${PROTEIN_CACHE}"
  --ligand-cache "${LIGAND_CACHE}"
  --initial-checkpoint "${INITIAL_CHECKPOINT}"
  --output "${OUTPUT}"
  --device cuda:0
  --batch-size 4
  --evaluation-batch-size 4
  --gradient-accumulation 4
  --num-workers 4
  --amp-dtype bfloat16
  --learning-rate 0.00015
  --maximum-epochs 30
  --steps-per-epoch 7000
  --patience 8
  --lp-sampling 0.15
  --davis-sampling 0.25
  --kiba-sampling 0.60
  --sequence-mse-weight 0.50
  --sequence-rank-weight 0.25
  --sequence-binary-weight 0.20
)
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "TRUE" || "${RESUME}" == "yes" || "${RESUME}" == "YES" ]]; then
  ARGS+=(--resume)
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ARGS[@]}"

echo "v22_run_id=${RUN_ID}"
echo "v22_sequence_output=${OUTPUT}"
echo "v22_sequence_selection=${OUTPUT}/selection.json"
