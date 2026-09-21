#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ID="${RUN_ID:-rank_fusion_v4_smoke}"
GPU_ID="${GPU_ID:-7}"
export CACHE_ROOT="/tmp/LP_PDBBind_RankFusionCache_v4_${RUN_ID}"
export MODEL_ROOT="/tmp/LP_PDBBind_RankFusion_v4_${RUN_ID}"
export THREED_DATA_DIR="${ROOT_DIR}/external_benchmarks/3d-prot-dta/data/lp_pdbbind"
source "${SCRIPT_DIR}/common.sh"

LIMIT="64"
EPOCHS="1"
PATIENCE="1"
SMOKE_SEEDS=(42 87)

preflight_paths
export_split train "${LIMIT}"
build_join train "${LIMIT}"
export_split valid "${LIMIT}"
build_join valid "${LIMIT}"
train_seeds "${EPOCHS}" "${PATIENCE}" "${SMOKE_SEEDS[@]}"
select_validation "${LIMIT}" "${SMOKE_SEEDS[@]}" # --expected-count 64 --expected-seed 42 --expected-seed 87
export_split test "${LIMIT}"
build_join test "${LIMIT}"
evaluate_test "${LIMIT}"
write_pipeline_summary

printf 'smoke_cache_root=%s\n' "${CACHE_ROOT}"
printf 'smoke_model_root=%s\n' "${MODEL_ROOT}"
