#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
GPU_ID=${GPU_ID:-7}
MEMORY_BUDGET_GB=40
DEVICE=cuda:0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

MANIFEST_ROOT="${ROOT_DIR}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind"
NATIVE_ROOT="${ROOT_DIR}/data/LP_PDBBind_InterfaceFirstGerNA_v1"
CACHE_ROOT="${ROOT_DIR}/data/LP_PDBBind_HybridFeatureCache_v3"
MODEL_ROOT="${ROOT_DIR}/Model/LP_PDBBind_HybridResidual_v3"
TEACHER_ROOT="${MODEL_ROOT}/teacher"
GERNA_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/Model_dta_best.pth"
INTERFACE_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/Model_dta_best.pth"
CACHED_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_HybridUpperBound_v2/cached_seed87/Model_cached_best.pth"

cache_complete() {
  "${PYTHON}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"metadata.json"; sys.exit(0 if p.is_file() and json.loads(p.read_text()).get("status") == "complete" else 1)' "$1"
}

build_cache() {
  local split="$1"
  local output="${CACHE_ROOT}/${split}"
  if cache_complete "${output}"; then
    echo "reusing complete v3 ${split} cache: ${output}"
    return
  fi
  local resume_arg=""
  if [[ -d "${output}" ]] && [[ -n "$(ls -A "${output}" 2>/dev/null)" ]]; then
    resume_arg="--resume"
  fi
  "${PYTHON}" -m dta.anchored_residual_v3.build_cache \
    --manifest "${MANIFEST_ROOT}/${split}_manifest.json" \
    --native-cache "${NATIVE_ROOT}/native_${split}" --split "${split}" \
    --gerna-checkpoint "${GERNA_CHECKPOINT}" \
    --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
    --output "${output}" --device "${DEVICE}" \
    --map-size-gb "${MEMORY_BUDGET_GB}" ${resume_arg:+"${resume_arg}"}
}

build_teacher() {
  local split="$1"
  local output="${TEACHER_ROOT}/${split}"
  if cache_complete "${output}"; then
    echo "reusing complete ${split} teacher: ${output}"
    return
  fi
  "${PYTHON}" -m dta.anchored_residual_v3.build_teacher \
    --cache "${CACHE_ROOT}/${split}" \
    --manifest "${MANIFEST_ROOT}/${split}_manifest.json" \
    --checkpoint "${CACHED_CHECKPOINT}" --split "${split}" \
    --output "${output}" --device "${DEVICE}" --batch-size 64
}

cd "${ROOT_DIR}"
mkdir -p "${CACHE_ROOT}" "${TEACHER_ROOT}" "${MODEL_ROOT}/baseline_test"
echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=${MEMORY_BUDGET_GB}"
for split in train valid test; do
  build_cache "${split}"
  build_teacher "${split}"
done

"${PYTHON}" -m dta.anchored_residual_v3.evaluate \
  --teacher "${TEACHER_ROOT}/test" \
  --manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --checkpoint "${CACHED_CHECKPOINT}" --output "${MODEL_ROOT}/baseline_test"

if [[ -f "${MODEL_ROOT}/run/summary.json" ]]; then
  echo "reusing complete anchored residual v3 run"
  cat "${MODEL_ROOT}/run/summary.json"
  exit 0
fi
resume_arg=""
if [[ -d "${MODEL_ROOT}/run" ]] && [[ -n "$(ls -A "${MODEL_ROOT}/run" 2>/dev/null)" ]]; then
  resume_arg="--resume"
fi
"${PYTHON}" -m dta.anchored_residual_v3.train \
  --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
  --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
  --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
  --train-native-cache "${NATIVE_ROOT}/native_train" \
  --valid-native-cache "${NATIVE_ROOT}/native_valid" \
  --test-native-cache "${NATIVE_ROOT}/native_test" \
  --teacher-train "${TEACHER_ROOT}/train" \
  --teacher-valid "${TEACHER_ROOT}/valid" \
  --teacher-test "${TEACHER_ROOT}/test" \
  --gerna-checkpoint "${GERNA_CHECKPOINT}" \
  --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
  --cached-checkpoint "${CACHED_CHECKPOINT}" \
  --output "${MODEL_ROOT}/run" --seed 87 --device "${DEVICE}" \
  --group-size 16 --residual-epochs 5 --fusion-epochs 5 \
  --selective-epochs 5 --patience 2 ${resume_arg:+"${resume_arg}"}
