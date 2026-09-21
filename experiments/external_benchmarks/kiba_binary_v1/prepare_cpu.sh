#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
KIBA_ROOT="${KIBA_ROOT:-${ROOT_DIR}/data/KIBA}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/ExternalBenchmarkKIBA_v1}"
cd "${ROOT_DIR}"
if [[ -n "${LIMIT:-}" ]]; then
  exec /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m dta.external_benchmark_kiba_v1.prepare \
    --kiba-root "${KIBA_ROOT}" \
    --output "${OUTPUT_DIR}" \
    --limit "${LIMIT}"
fi
exec /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m dta.external_benchmark_kiba_v1.prepare \
  --kiba-root "${KIBA_ROOT}" \
  --output "${OUTPUT_DIR}"
