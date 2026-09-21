#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-7}"
source "${SCRIPT_DIR}/common.sh"

preflight_paths
mkdir -p "${CACHE_ROOT}" "${MODEL_ROOT}"
export_split train
build_join train
export_split valid
build_join valid
train_seeds 100 12 "${SEEDS[@]}"
select_validation 2340 "${SEEDS[@]}"
export_split test
build_join test
evaluate_test 4751
write_pipeline_summary

printf 'pipeline_summary=%s\n' "${MODEL_ROOT}/pipeline_summary.json"
