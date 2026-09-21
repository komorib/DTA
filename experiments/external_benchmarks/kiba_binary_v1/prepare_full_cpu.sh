#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
KIBA_ROOT="${KIBA_ROOT:-${ROOT_DIR}/data/KIBA}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/ExternalBenchmarkKIBA_v1}"
PARTS_DIR="${PARTS_DIR:-${OUTPUT_DIR}.parts}"

cd "${ROOT_DIR}"
rm -rf "${PARTS_DIR}" "${OUTPUT_DIR}"
mkdir -p "${PARTS_DIR}"
for split in train valid test; do
  /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m dta.external_benchmark_kiba_v1.prepare \
    --kiba-root "${KIBA_ROOT}" \
    --output "${PARTS_DIR}/${split}" \
    --split "${split}"
done
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m dta.external_benchmark_kiba_v1.merge_parts \
  --parts-root "${PARTS_DIR}" \
  --output "${OUTPUT_DIR}"

