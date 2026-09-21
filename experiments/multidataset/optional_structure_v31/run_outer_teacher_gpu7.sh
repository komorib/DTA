#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
FOLD_ROOT="${FOLD_ROOT:?set FOLD_ROOT to the completed V31 fold root}"
FOLD="${FOLD:?set FOLD to 0, 1, 2, 3, or 4}"
TEACHER_ID="${TEACHER_ID:?set TEACHER_ID; use one shared ID for all five folds}"
TEACHER_ROOT="${TEACHER_ROOT:-${ROOT}/Model/OptionalStructurePrivilegedMSAPair_v31/${TEACHER_ID}}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
TRAIN_CACHE="${TRAIN_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v9/complex_cache_full/train}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:?set PRETRAINED_CHECKPOINT to a V13-style self-supervised encoder checkpoint}"
INITIALIZATION_POLICY="${INITIALIZATION_POLICY:-historical_self_supervised}"
REFIT_OUTER="${REFIT_OUTER:-1}"
OUTPUT="${OUTPUT:-${TEACHER_ROOT}/outer_fold_${FOLD}}"

case "${REFIT_OUTER}" in
  0|false|FALSE|no|NO) REFIT_ARGS=() ;;
  1|true|TRUE|yes|YES) REFIT_ARGS=(--refit-outer) ;;
  *) echo "REFIT_OUTER must be boolean" >&2; exit 2 ;;
esac

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m dta.optional_structure_privileged_msa_pair_v31.teacher_folds \
  --fold-root "${FOLD_ROOT}" \
  --source-train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --fold "${FOLD}" \
  --train-cache "${TRAIN_CACHE}" \
  --row-cache "${ROW_CACHE}" \
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
  --initialization-policy "${INITIALIZATION_POLICY}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --batch-size 4 \
  --evaluation-batch-size 4 \
  --num-workers 4 \
  --amp-dtype bfloat16 \
  --learning-rate 0.00015 \
  --maximum-epochs 24 \
  --patience 6 \
  --require-full-folds \
  "${REFIT_ARGS[@]}"

echo "v31_outer_teacher_fold=${FOLD}"
echo "v31_outer_teacher_output=${OUTPUT}"
