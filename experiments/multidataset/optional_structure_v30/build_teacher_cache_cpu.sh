#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
CACHE_ID="${CACHE_ID:-optional_structure_privileged_msa_v30_teacher_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
MSA_TRAIN_FEATURES="${MSA_TRAIN_FEATURES:-${ROOT}/Model/LP_PDBBind_MSAInterfaceTrainFeatures_v17/msa_interface_train_supervised_fusion_v17_full_20260831_170712/train_features.npz}"
OUTPUT="${OUTPUT:-${ROOT}/data/OptionalStructurePrivilegedMSA_v30/${CACHE_ID}}"

cd "${ROOT}"
"${PYTHON_BIN}" -m dta.optional_structure_privileged_msa_v30.teacher_cache \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --msa-train-features "${MSA_TRAIN_FEATURES}" \
  --output "${OUTPUT}" \
  --folds "${FOLDS:-5}" \
  --components "${COMPONENTS:-96}" \
  --estimators "${ESTIMATORS:-400}" \
  --minimum-leaf "${MINIMUM_LEAF:-3}" \
  --max-features "${MAX_FEATURES:-0.85}" \
  --n-jobs "${N_JOBS:-8}" \
  --seed "${SEED:-20260917}"

echo "v30_teacher_cache=${OUTPUT}"
echo "v30_teacher_archive=${OUTPUT}/train_teacher_oof.npz"
