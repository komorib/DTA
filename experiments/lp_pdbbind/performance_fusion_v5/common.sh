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
TRAIN_CACHE="${V4_CACHE_ROOT}/joined/train"
VALID_CACHE="${V4_CACHE_ROOT}/joined/valid"
TEST_CACHE="${V4_CACHE_ROOT}/joined/test"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/Model/LP_PDBBind_AnchoredExpertFusion_v5}"
VALIDATION_ROOT="${VALIDATION_ROOT:-${MODEL_ROOT}/validation}"
PACKAGE_ROOT="${VALIDATION_ROOT}/validation_package"
OFFICIAL_TEST_ROOT="${OFFICIAL_TEST_ROOT:-${MODEL_ROOT}/official_test}"

preflight_cache() {
  local cache="$1"
  local name
  if [[ ! -d "${cache}" ]]; then
    printf 'preflight_missing_cache: %s\n' "${cache}" >&2
    return 1
  fi
  for name in features.npz metadata.json coverage.json normalization.json; do
    if [[ ! -f "${cache}/${name}" ]]; then
      printf 'preflight_missing_file: %s\n' "${cache}/${name}" >&2
      return 1
    fi
  done
}

preflight_validation_inputs() {
  if [[ ! -x "${PYTHON}" ]]; then
    printf 'preflight_missing_python: %s\n' "${PYTHON}" >&2
    return 1
  fi
  preflight_cache "${TRAIN_CACHE}"
  preflight_cache "${VALID_CACHE}"
  printf 'physical_gpu=%s logical_device=%s memory_budget_gb=%s\n' \
    "${GPU_ID}" "${DEVICE}" "${MEMORY_BUDGET_GB}"
  printf 'v4_cache_root=%s\n' "${V4_CACHE_ROOT}"
}
