#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V11 smoke training on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_pair_v11_smoke_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-24}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
A3M_DB="${A3M_DB:-${V9_DATA_ROOT}/uniref30_full/work/uniref.a3m}"
SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST:-${V9_DATA_ROOT}/sequences/sequence_manifest.json}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
COMPLEX_CACHE_ROOT="${COMPLEX_CACHE_ROOT:-${V9_DATA_ROOT}/complex_cache_full}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/LP_PDBBind_MSAPairExpert_v11_${RUN_ID}}"
ROW_CACHE="${ROW_CACHE:-${SMOKE_ROOT}/rows}"
OUTPUT="${OUTPUT:-${SMOKE_ROOT}/run}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=40"
echo "smoke_root=${SMOKE_ROOT}"
cd "${ROOT}"

"${PYTHON}" -m dta.msa_pair_expert_v11.row_cache \
  --a3m-db "${A3M_DB}" \
  --sequence-manifest "${SEQUENCE_MANIFEST}" \
  --output "${ROW_CACHE}" \
  --maximum-rows 32 \
  --map-size-gb 1 \
  --manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --limit-per-split "${LIMIT}"

"${PYTHON}" -m dta.msa_pair_expert_v11.run_training \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --train-cache "${COMPLEX_CACHE_ROOT}/train" \
  --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
  --test-cache "${COMPLEX_CACHE_ROOT}/test" \
  --row-cache "${ROW_CACHE}" \
  --output "${OUTPUT}" \
  --limit-train "${LIMIT}" \
  --limit-valid "${LIMIT}" \
  --limit-test "${LIMIT}" \
  --device "${DEVICE}" \
  --seed 42 \
  --batch-size 2 \
  --evaluation-batch-size 4 \
  --num-workers 2 \
  --maximum-epochs 3 \
  --patience 3 \
  --model-width 128 \
  --layers 2 \
  --heads 4 \
  --latent-width 256 \
  --msa-width 64 \
  --msa-layers 2 \
  --msa-heads 4 \
  --pair-factor 4 \
  --pair-width 32 \
  --row-depth 16 \
  --inference-subsets 1

printf 'v11_smoke_root=%s\n' "${SMOKE_ROOT}"
printf 'v11_smoke_result=%s\n' "${OUTPUT}/result.json"
