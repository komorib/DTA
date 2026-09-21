#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT_DIR/data/ExternalBenchmarkPSICHIC_KIBA_v1/psichic_sequence_edge_features_v1}"
RUN_ID="${RUN_ID:-psichic_sequence_edge_kiba_binary_v1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/Model/ExternalBenchmarkPSICHIC_KIBA_v1/$RUN_ID}"

if [[ ! -f "$FEATURE_ROOT/metadata.json" ]]; then
  echo "missing PSICHIC feature cache: $FEATURE_ROOT" >&2
  echo "run experiments/external_benchmarks/psichic_kiba_binary_v1/build_features_gpu7.sh first" >&2
  exit 2
fi

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -u -m dta.external_benchmark_psichic_kiba_v1.run_training \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/ExternalBenchmarkKIBA_v1}" \
  --feature-root "$FEATURE_ROOT" \
  --output "$OUTPUT_DIR" \
  --device "cuda:${GPU_ID}" \
  --epochs "${EPOCHS:-50}" \
  --patience "${PATIENCE:-10}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --seed "${SEED:-20260920}"

echo "psichic_output=$OUTPUT_DIR"
echo "psichic_result=$OUTPUT_DIR/result.json"

