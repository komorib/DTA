#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V9 GPU smoke on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-msa_pocket_v9_smoke_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-64}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
PROFILE_LMDB="${PROFILE_LMDB:-${DATA_ROOT}/uniref30_profiles/profiles.lmdb}"
SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST:-${DATA_ROOT}/sequences/sequence_manifest.json}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/complex_cache_smoke}"
OUTPUT="${OUTPUT:-/tmp/LP_PDBBind_MSAPocketExpert_v9_${RUN_ID}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=40"
echo "profile_lmdb=${PROFILE_LMDB}"
echo "cache_root=${CACHE_ROOT}"
echo "output=${OUTPUT}"

cd "${ROOT}"
for split in train valid test; do
  "${PYTHON}" -m dta.msa_pocket_expert_v9.complex_cache \
    --manifest "${MANIFEST_ROOT}/${split}_manifest.json" \
    --sequence-manifest "${SEQUENCE_MANIFEST}" \
    --profile-lmdb "${PROFILE_LMDB}" \
    --output "${CACHE_ROOT}/${split}" \
    --map-size-gb 2 \
    --limit "${LIMIT}"
done

"${PYTHON}" -m dta.msa_pocket_expert_v9.run_training \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --train-cache "${CACHE_ROOT}/train" \
  --valid-cache "${CACHE_ROOT}/valid" \
  --test-cache "${CACHE_ROOT}/test" \
  --limit-train "${LIMIT}" \
  --limit-valid "${LIMIT}" \
  --limit-test "${LIMIT}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --seed 42 \
  --batch-size 4 \
  --evaluation-batch-size 8 \
  --num-workers 4 \
  --amp-dtype bfloat16 \
  --maximum-epochs 3 \
  --patience 3 \
  --model-width 128 \
  --layers 2 \
  --heads 4 \
  --latent-width 1024

printf 'v9_smoke_output=%s\n' "${OUTPUT}"
printf 'v9_smoke_result=%s\n' "${OUTPUT}/result.json"
