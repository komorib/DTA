#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT_DIR/data/ExternalBenchmarkPSICHIC_KIBA_v1/psichic_sequence_edge_features_v1}"

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -u -m dta.external_benchmark_psichic_kiba_v1.build_features \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/ExternalBenchmarkKIBA_v1}" \
  --output "$FEATURE_ROOT" \
  --model "${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}" \
  --device "cuda:${GPU_ID}" \
  --precision "${PRECISION:-fp16}"

echo "psichic_feature_root=$FEATURE_ROOT"

