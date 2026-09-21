#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
CACHE_ID="${CACHE_ID:-optional_structure_privileged_msa_pair_v31_teacher_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
FOLD_ROOT="${FOLD_ROOT:?set FOLD_ROOT to the completed V31 fold directory}"
TEACHER_ROOT="${TEACHER_ROOT:?set TEACHER_ROOT to the completed five-fold V31 teacher directory}"
V22_TRAIN_PREDICTIONS="${V22_TRAIN_PREDICTIONS:?set V22_TRAIN_PREDICTIONS to the frozen V22 LP-train export}"
OUTPUT="${OUTPUT:-${ROOT}/data/OptionalStructurePrivilegedMSAPair_v31/${CACHE_ID}}"

cd "${ROOT}"
"${PYTHON_BIN}" -m dta.optional_structure_privileged_msa_pair_v31.teacher_cache \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --fold-root "${FOLD_ROOT}" \
  --teacher-root "${TEACHER_ROOT}" \
  --v22-train-predictions "${V22_TRAIN_PREDICTIONS}" \
  --output "${OUTPUT}" \
  --components "${COMPONENTS:-128}" \
  --estimators "${ESTIMATORS:-800}" \
  --minimum-leaf "${MINIMUM_LEAF:-5}" \
  --max-features "${MAX_FEATURES:-1.0}" \
  --n-jobs "${N_JOBS:-8}" \
  --seed "${SEED:-20260918}" \
  --require-full-folds

echo "v31_teacher_cache=${OUTPUT}"
echo "v31_teacher_archive=${OUTPUT}/train_teacher_oof.npz"
