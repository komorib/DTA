#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
GPU_ID=${GPU_ID:-7}
MEMORY_BUDGET_GB=40
DEVICE=cuda:0
RUN_ID=${RUN_ID:-upper_bound_v2_$(date +%Y%m%d_%H%M%S)}
SMOKE_ROOT="/tmp/LP_PDBBind_HybridUpperBound_v2_smoke_${RUN_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

TRAIN_MANIFEST="${ROOT_DIR}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
VALID_MANIFEST="${ROOT_DIR}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/valid_manifest.json"
TRAIN_NATIVE_CACHE="${ROOT_DIR}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_train"
VALID_NATIVE_CACHE="${ROOT_DIR}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_valid"
GERNA_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/Model_dta_best.pth"
INTERFACE_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/Model_dta_best.pth"

cd "${ROOT_DIR}"
mkdir -p "${SMOKE_ROOT}"
echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=${MEMORY_BUDGET_GB}"
"${PYTHON}" dta/build_hybrid_feature_cache_v2.py \
  --manifest "${TRAIN_MANIFEST}" --native-cache "${TRAIN_NATIVE_CACHE}" --split train \
  --gerna-checkpoint "${GERNA_CHECKPOINT}" --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
  --output "${SMOKE_ROOT}/cache_train" --device "${DEVICE}" \
  --map-size-gb "${MEMORY_BUDGET_GB}" --limit 16
"${PYTHON}" dta/build_hybrid_feature_cache_v2.py \
  --manifest "${VALID_MANIFEST}" --native-cache "${VALID_NATIVE_CACHE}" --split valid \
  --gerna-checkpoint "${GERNA_CHECKPOINT}" --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
  --output "${SMOKE_ROOT}/cache_valid" --device "${DEVICE}" \
  --map-size-gb "${MEMORY_BUDGET_GB}" --limit 16
"${PYTHON}" dta/train_hybrid_cached_v2.py \
  --train-cache "${SMOKE_ROOT}/cache_train" --valid-cache "${SMOKE_ROOT}/cache_valid" \
  --output "${SMOKE_ROOT}/cached" --seed 42 --device "${DEVICE}" \
  --batch-size 8 --epochs 1 --patience 1
"${PYTHON}" dta/train_hybrid_e2e_v2.py \
  --train-manifest "${TRAIN_MANIFEST}" --valid-manifest "${VALID_MANIFEST}" \
  --train-native-cache "${TRAIN_NATIVE_CACHE}" --valid-native-cache "${VALID_NATIVE_CACHE}" \
  --gerna-checkpoint "${GERNA_CHECKPOINT}" --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
  --cached-checkpoint "${SMOKE_ROOT}/cached/Model_cached_best.pth" \
  --output "${SMOKE_ROOT}/e2e" --seed 42 --device "${DEVICE}" \
  --epochs 1 --patience 1 --group-size 16 --limit 16
"${PYTHON}" scripts/select_hybrid_upper_bound_v2.py \
  --candidate "cached=${SMOKE_ROOT}/cached/valid_predictions.npz=${SMOKE_ROOT}/cached/candidate_cached.json=${SMOKE_ROOT}/cached/Model_cached_best.pth" \
  --candidate "e2e_raw=${SMOKE_ROOT}/e2e/valid_predictions_raw.npz=${SMOKE_ROOT}/e2e/candidate_raw.json=${SMOKE_ROOT}/e2e/Model_e2e_raw_best.pth" \
  --output "${SMOKE_ROOT}/selection" --seed 42 --random-starts 4
printf smoke_root=%sn "${SMOKE_ROOT}"
