#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
FOLD_ID="${FOLD_ID:-optional_structure_privileged_msa_pair_v31_folds_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
MSA_TRAIN_MANIFEST="${MSA_TRAIN_MANIFEST:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json}"
OUTPUT="${OUTPUT:-${ROOT}/data/OptionalStructurePrivilegedMSAPair_v31/${FOLD_ID}}"

cd "${ROOT}"
"${PYTHON_BIN}" -m dta.optional_structure_privileged_msa_pair_v31.folds \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --msa-train-manifest "${MSA_TRAIN_MANIFEST}" \
  --output "${OUTPUT}" \
  --folds "${FOLDS:-5}" \
  --inner-valid-fraction "${INNER_VALID_FRACTION:-0.10}" \
  --seed "${SEED:-20260918}"

echo "v31_fold_root=${OUTPUT}"
