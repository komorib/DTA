#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
V8_ROOT="${V8_ROOT:-${ROOT}/Model/LP_PDBBind_RepeatedRankFusion_v8/run}"
V9_ROOT="${V9_ROOT:-${ROOT}/Model/LP_PDBBind_MSAPocketExpert_v9/msa_pocket_v9_full_retry_20260824_154027}"
METADATA_CSV="${METADATA_CSV:-${ROOT}/data/LP_PDBBind.csv}"
OUTPUT="${OUTPUT:-${ROOT}/Model/LP_PDBBind_MSARankFusion_v10/run}"

cd "${ROOT}"
"${PYTHON}" -m dta.msa_rank_fusion_v10.run_pipeline \
  --v8-root "${V8_ROOT}" \
  --v9-root "${V9_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --output "${OUTPUT}" \
  --pca-components 64 \
  --oof-estimators 200 \
  --final-estimators 500 \
  --n-jobs 8 \
  --fold-seed 42 \
  --seed 20260824

printf 'v10_output=%s\n' "${OUTPUT}"
printf 'v10_result=%s\n' "${OUTPUT}/result.json"
