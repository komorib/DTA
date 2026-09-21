#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/Model/ExternalBenchmarkPSICHIC_KIBA_v1/preflight}"
mkdir -p "$OUTPUT_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -u -m dta.external_benchmark_psichic_kiba_v1.audit \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/ExternalBenchmarkKIBA_v1}" \
  --protein-cache "${PROTEIN_CACHE:-$ROOT_DIR/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}" \
  --output "$OUTPUT_DIR/result.json"
echo "psichic_audit=$OUTPUT_DIR/result.json"

