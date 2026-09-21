#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
GPU_ID=${GPU_ID:-7}
MEMORY_BUDGET_GB=40
DEVICE=cuda:0
RUN_ID=${RUN_ID:-anchored_residual_v3_$(date +%Y%m%d_%H%M%S)}
SMOKE_ROOT="/tmp/LP_PDBBind_HybridResidual_v3_smoke_${RUN_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

MANIFEST_ROOT="${ROOT_DIR}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind"
NATIVE_ROOT="${ROOT_DIR}/data/LP_PDBBind_InterfaceFirstGerNA_v1"
GERNA_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/Model_dta_best.pth"
INTERFACE_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/Model_dta_best.pth"
CACHED_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_HybridUpperBound_v2/cached_seed87/Model_cached_best.pth"

cd "${ROOT_DIR}"
mkdir -p "${SMOKE_ROOT}"
echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=${MEMORY_BUDGET_GB}"
for split in train valid test; do
  "${PYTHON}" -m dta.anchored_residual_v3.build_cache \
    --manifest "${MANIFEST_ROOT}/${split}_manifest.json" \
    --native-cache "${NATIVE_ROOT}/native_${split}" --split "${split}" \
    --gerna-checkpoint "${GERNA_CHECKPOINT}" \
    --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
    --output "${SMOKE_ROOT}/cache_${split}" --device "${DEVICE}" \
    --map-size-gb "${MEMORY_BUDGET_GB}" --limit 16
  "${PYTHON}" -m dta.anchored_residual_v3.build_teacher \
    --cache "${SMOKE_ROOT}/cache_${split}" \
    --manifest "${MANIFEST_ROOT}/${split}_manifest.json" \
    --checkpoint "${CACHED_CHECKPOINT}" --split "${split}" \
    --output "${SMOKE_ROOT}/teacher_${split}" --device "${DEVICE}" \
    --batch-size 16 --limit 16
done

"${PYTHON}" -m dta.anchored_residual_v3.evaluate \
  --teacher "${SMOKE_ROOT}/teacher_test" \
  --manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --checkpoint "${CACHED_CHECKPOINT}" --output "${SMOKE_ROOT}/baseline" \
  --limit 16

"${PYTHON}" -m dta.anchored_residual_v3.train \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --train-native-cache "${NATIVE_ROOT}/native_train" \
  --valid-native-cache "${NATIVE_ROOT}/native_valid" \
  --test-native-cache "${NATIVE_ROOT}/native_test" \
  --teacher-train "${SMOKE_ROOT}/teacher_train" \
  --teacher-valid "${SMOKE_ROOT}/teacher_valid" \
  --teacher-test "${SMOKE_ROOT}/teacher_test" \
  --gerna-checkpoint "${GERNA_CHECKPOINT}" \
  --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
  --cached-checkpoint "${CACHED_CHECKPOINT}" \
  --output "${SMOKE_ROOT}/run" --seed 87 --device "${DEVICE}" \
  --group-size 16 --residual-epochs 1 --fusion-epochs 1 \
  --selective-epochs 1 --patience 1 --limit 16

printf 'smoke_root=%s\n' "${SMOKE_ROOT}"
