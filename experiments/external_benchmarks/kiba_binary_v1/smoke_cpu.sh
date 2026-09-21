#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/ExternalBenchmarkKIBA_v1}"

cd "${ROOT_DIR}"
exec /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m dta.external_benchmark_kiba_v1.smoke \
  --root "${OUTPUT_DIR}"

