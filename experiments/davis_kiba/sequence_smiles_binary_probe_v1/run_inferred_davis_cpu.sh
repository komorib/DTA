#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
RUN_ID="${RUN_ID:-inferred_prevalence_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${MODEL_ROOT:-${ROOT}/Model/DavisKIBA_SequenceSmilesBinaryProtocolDiagnostic_v1/${RUN_ID}}/DAVIS"

cd "${ROOT}"
"${PYTHON_BIN}" -m dta.sequence_smiles_binary_probe_v1.run_probe select \
  --dataset-dir "${ROOT}/data/DAVIS" \
  --dataset-name DAVIS \
  --affinity-threshold 5.292429823902063 \
  --decision-objective f1 \
  --output "${OUTPUT}"
"${PYTHON_BIN}" -m dta.sequence_smiles_binary_probe_v1.run_probe test \
  --output "${OUTPUT}"

echo "output=${OUTPUT}"
