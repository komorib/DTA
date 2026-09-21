#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
DEVICE="cuda:0"
MEMORY_BUDGET_GB="40"

V4_CACHE_ROOT="${V4_CACHE_ROOT:-${REPO_ROOT}/data/LP_PDBBind_RankFusionCache_v4}"
METADATA_CSV="${METADATA_CSV:-${REPO_ROOT}/data/LP_PDBBind.csv}"
V5_MODEL_ROOT="${V5_MODEL_ROOT:-${REPO_ROOT}/Model/LP_PDBBind_AnchoredExpertFusion_v5}"
V5_VALIDATION_ROOT="${V5_MODEL_ROOT}/validation"
V5_PACKAGE="${V5_VALIDATION_ROOT}/validation_package"
V5_FALLBACK_RESULT="${V5_MODEL_ROOT}/official_test/result.json"

SOURCE_REVISION="${SOURCE_REVISION:-}"
if [[ -z "${SOURCE_REVISION}" ]]; then
  SOURCE_REVISION="$(cd "${REPO_ROOT}" && git rev-parse HEAD)"
fi

preflight_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'preflight_missing_file: %s\n' "${path}" >&2
    return 1
  fi
}

preflight_cache() {
  local split="$1"
  local directory="${V4_CACHE_ROOT}/joined/${split}"
  local name
  if [[ ! -d "${directory}" ]]; then
    printf 'preflight_missing_cache: %s\n' "${directory}" >&2
    return 1
  fi
  for name in features.npz metadata.json coverage.json normalization.json; do
    preflight_file "${directory}/${name}"
  done
}

preflight_v7() {
  if [[ ! -x "${PYTHON}" ]]; then
    printf 'preflight_missing_python: %s\n' "${PYTHON}" >&2
    return 1
  fi
  preflight_cache train
  preflight_cache valid
  preflight_cache test
  preflight_file "${METADATA_CSV}"
  preflight_file "${V5_PACKAGE}/selection_seal.json"
  preflight_file "${V5_FALLBACK_RESULT}"
  printf 'physical_gpu=%s logical_device=%s memory_budget_gb=%s\n' \
    "${GPU_ID}" "${DEVICE}" "${MEMORY_BUDGET_GB}"
  printf 'v4_cache_root=%s\n' "${V4_CACHE_ROOT}"
  printf 'v5_package=%s\n' "${V5_PACKAGE}"
}
