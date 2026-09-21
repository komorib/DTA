#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run a full UniRef30 scan on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
MMSEQS="${MMSEQS:-/home/user/wangzhiwen/miniconda3/envs/msa_v9/bin/mmseqs}"
DATABASE_PREFIX="${DATABASE_PREFIX:-/home/user/wangzhiwen/databases/colabfold_v9_uniref30/uniref30_2302_db}"
QUERY_FASTA="${QUERY_FASTA:-${ROOT}/data/LP_PDBBind_MSA_v9/smoke_queries/queries.fasta}"
OUTPUT="${OUTPUT:-${ROOT}/data/LP_PDBBind_MSA_v9/uniref30_smoke}"
SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST:-${ROOT}/data/LP_PDBBind_MSA_v9/smoke_queries/sequence_manifest.json}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-${ROOT}/data/LP_PDBBind_MSA_v9/uniref30_smoke_profiles}"
THREADS="${THREADS:-32}"
SENSITIVITY="${SENSITIVITY:-8.0}"

echo "host=${HOST} threads=${THREADS}"
echo "database_prefix=${DATABASE_PREFIX}"
echo "query_fasta=${QUERY_FASTA}"
echo "output=${OUTPUT}"

cd "${ROOT}"
"${PYTHON}" -m dta.msa_pocket_expert_v9.mmseqs_search \
  --query-fasta "${QUERY_FASTA}" \
  --database-prefix "${DATABASE_PREFIX}" \
  --output "${OUTPUT}" \
  --mmseqs "${MMSEQS}" \
  --threads "${THREADS}" \
  --sensitivity "${SENSITIVITY}"

echo "search_report=${OUTPUT}/search_report.json"
echo "search_log=${OUTPUT}/mmseqs.log"

"${PYTHON}" -m dta.msa_pocket_expert_v9.msa_profile \
  --a3m-dir "${OUTPUT}/a3m" \
  --sequence-manifest "${SEQUENCE_MANIFEST}" \
  --output "${PROFILE_OUTPUT}" \
  --map-size-gb 1
echo "profile_report=${PROFILE_OUTPUT}/profile_report.json"
