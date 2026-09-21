#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
NATIVE_CACHE_ROOT="${NATIVE_CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1}"
BASELINE_PREFIX="${BASELINE_PREFIX:?Set BASELINE_PREFIX to the matched GerNA output prefix ending in seed}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${ROOT}/Model/LP_PDBBind_InterfaceFirstGerNA_v2_summary}"
MMSEQS_BINARY="${MMSEQS_BINARY:-mmseqs}"
MMSEQS_THREADS="${MMSEQS_THREADS:-8}"
SEEDS=(42 43 44 45 46)

mkdir -p "${SUMMARY_ROOT}"
cd "${ROOT}"

prediction_args=()
for seed in "${SEEDS[@]}"; do
  baseline="${BASELINE_PREFIX}${seed}/test_predictions.npz"
  interface="${ROOT}/Model/LP_PDBBind_InterfaceFirstGerNA_v2_seed${seed}/test_predictions.npz"
  if [[ ! -f "${baseline}" ]]; then
    echo "matched baseline prediction is missing: ${baseline}" >&2
    exit 1
  fi
  if [[ ! -f "${interface}" ]]; then
    echo "interface-first prediction is missing: ${interface}" >&2
    exit 1
  fi
  prediction_args+=(
    --baseline-prediction "${seed}=${baseline}"
    --interface-prediction "${seed}=${interface}"
  )
done

"${PY}" scripts/summarize_interface_first_seeds.py \
  "${prediction_args[@]}" \
  --target-native-lmdb "${NATIVE_CACHE_ROOT}/native_test" \
  --output-json "${SUMMARY_ROOT}/matched_seed_clustered_bootstrap.json" \
  --mmseqs-output-dir "${SUMMARY_ROOT}/mmseqs_related_targets" \
  --mmseqs-binary "${MMSEQS_BINARY}" \
  --mmseqs-threads "${MMSEQS_THREADS}" \
  --bootstrap-draws 2000 \
  --bootstrap-seed 42

echo "summary: ${SUMMARY_ROOT}/matched_seed_clustered_bootstrap.json"
