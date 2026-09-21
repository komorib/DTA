#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
WORKERS="${WORKERS:-8}"
SHARD_SIZE="${SHARD_SIZE:-64}"

cd "${ROOT_DIR}"
"${PYTHON}" -m dta.rank_fusion_v4.three_dprot_prepare \
  --lp-csv "${ROOT_DIR}/data/LP_PDBBind.csv" \
  --manifest-root "${ROOT_DIR}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind" \
  --pdbbind-root "${ROOT_DIR}/PDBbind" \
  --output "${ROOT_DIR}/external_benchmarks/3d-prot-dta/data/lp_pdbbind" \
  --shard-root "${ROOT_DIR}/data/LP_PDBBind_RankFusionCache_v4/three_dprot_prepare_shards" \
  --workers "${WORKERS}" \
  --shard-size "${SHARD_SIZE}" \
  --protein-cutoff 10.0
