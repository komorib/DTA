#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
A3M_DB="${A3M_DB:-${V9_DATA_ROOT}/uniref30_full/work/uniref.a3m}"
SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST:-${V9_DATA_ROOT}/sequences/sequence_manifest.json}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
MAXIMUM_ROWS="${MAXIMUM_ROWS:-128}"
MAP_SIZE_GB="${MAP_SIZE_GB:-8}"

cd "${ROOT}"
"${PYTHON}" -m dta.msa_pair_expert_v11.row_cache \
  --a3m-db "${A3M_DB}" \
  --sequence-manifest "${SEQUENCE_MANIFEST}" \
  --output "${ROW_CACHE}" \
  --maximum-rows "${MAXIMUM_ROWS}" \
  --map-size-gb "${MAP_SIZE_GB}"

printf 'v11_row_cache=%s\n' "${ROW_CACHE}"
printf 'v11_row_cache_report=%s\n' "${ROW_CACHE}/row_cache_report.json"
