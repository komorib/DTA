#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V9 training on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-msa_pocket_v9_full_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
PROFILE_LMDB="${PROFILE_LMDB:-${DATA_ROOT}/uniref30_profiles/profiles.lmdb}"
SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST:-${DATA_ROOT}/sequences/sequence_manifest.json}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/complex_cache_full}"
OUTPUT="${OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAPocketExpert_v9/${RUN_ID}}"
MAP_SIZE_GB="${MAP_SIZE_GB:-64}"
MAXIMUM_LIGAND_ATOMS="${MAXIMUM_LIGAND_ATOMS:-256}"

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
    --maximum-ligand-atoms "${MAXIMUM_LIGAND_ATOMS}" \
    --map-size-gb "${MAP_SIZE_GB}"
done

"${PYTHON}" -m dta.msa_pocket_expert_v9.run_training \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --train-cache "${CACHE_ROOT}/train" \
  --valid-cache "${CACHE_ROOT}/valid" \
  --test-cache "${CACHE_ROOT}/test" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --seed 42 \
  --batch-size 8 \
  --evaluation-batch-size 16 \
  --num-workers 4 \
  --amp-dtype bfloat16 \
  --learning-rate 1.5e-4 \
  --weight-decay 1e-4 \
  --maximum-epochs 90 \
  --patience 16 \
  --model-width 512 \
  --layers 8 \
  --heads 8 \
  --dropout 0.10 \
  --latent-width 1024 \
  --rank-weight 0.40 \
  --mse-weight 0.15 \
  --contact-weight 0.05

printf 'v9_output=%s\n' "${OUTPUT}"
printf 'v9_result=%s\n' "${OUTPUT}/result.json"
