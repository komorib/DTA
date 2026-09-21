#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-graphdta_kiba_binary_v1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/Model/ExternalBenchmarkGraphDTA_KIBA_v1/$RUN_ID}"

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -u -m dta.external_benchmark_graphdta_kiba_v1.run_training \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/ExternalBenchmarkKIBA_v1}" \
  --output "$OUTPUT_DIR" \
  --device "cuda:${GPU_ID}" \
  --epochs "${EPOCHS:-80}" \
  --patience "${PATIENCE:-12}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --learning-rate "${LEARNING_RATE:-5e-4}" \
  --seed "${SEED:-20260919}"

echo "graphdta_output=$OUTPUT_DIR"
echo "graphdta_result=$OUTPUT_DIR/result.json"
