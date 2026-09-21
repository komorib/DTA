#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
RUN_ID="${RUN_ID:-sequence_smiles_binary_probe_v1_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT}/Model/DavisKIBA_SequenceSmilesBinaryProbe_v1/${RUN_ID}}"

cd "${ROOT}"
for DATASET in DAVIS KIBA; do
  OUTPUT="${MODEL_ROOT}/${DATASET}"
  "${PYTHON_BIN}" -m dta.sequence_smiles_binary_probe_v1.run_probe select \
    --dataset-dir "${ROOT}/data/${DATASET}" \
    --dataset-name "${DATASET}" \
    --output "${OUTPUT}"
  "${PYTHON_BIN}" -m dta.sequence_smiles_binary_probe_v1.run_probe test \
    --output "${OUTPUT}"
done

echo "run_id=${RUN_ID}"
echo "model_root=${MODEL_ROOT}"
