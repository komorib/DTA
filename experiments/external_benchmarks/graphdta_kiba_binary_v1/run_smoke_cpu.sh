#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT_DIR" \
  "$PYTHON_BIN" -u -m dta.external_benchmark_graphdta_kiba_v1.smoke \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/ExternalBenchmarkKIBA_v1}" \
  --limit "${SMOKE_LIMIT:-1}"

