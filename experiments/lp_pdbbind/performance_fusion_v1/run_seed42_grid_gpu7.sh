#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
TRAIN_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
VALID_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/valid_manifest.json"
TRAIN_NATIVE="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_train"
VALID_NATIVE="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_valid"
GERNA_CHECKPOINT="${ROOT}/Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/Model_dta_best.pth"
INTERFACE_CHECKPOINT="${ROOT}/Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/Model_dta_best.pth"
PREFLIGHT_ROOT="${ROOT}/data/LP_PDBBind_HybridAffinityPreflight"
DEV_ROOT="${ROOT}/Model/LP_PDBBind_HybridPerformance_v1/dev_seed42"

fixed_inputs=(
  --train-manifest "${TRAIN_MANIFEST}"
  --valid-manifest "${VALID_MANIFEST}"
  --train-native-cache "${TRAIN_NATIVE}"
  --valid-native-cache "${VALID_NATIVE}"
  --gerna-checkpoint "${GERNA_CHECKPOINT}"
  --interface-checkpoint "${INTERFACE_CHECKPOINT}"
  --train-preflight-report "${PREFLIGHT_ROOT}/train.json"
  --valid-preflight-report "${PREFLIGHT_ROOT}/valid.json"
  --device cuda:0
  --num-workers 0
  --memory-budget-gb 40
)

for path in \
  "${TRAIN_MANIFEST}" \
  "${VALID_MANIFEST}" \
  "${TRAIN_NATIVE}" \
  "${VALID_NATIVE}" \
  "${GERNA_CHECKPOINT}" \
  "${INTERFACE_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required grid input is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${PREFLIGHT_ROOT}" "${DEV_ROOT}"
cd "${ROOT}"
shopt -s nullglob dotglob

for fusion in mlp transformer; do
  for rank_weight in 0.1 0.2 0.4; do
    output="${DEV_ROOT}/${fusion}_rank${rank_weight}"
    if [[ -e "${output}" && ! -d "${output}" ]]; then
      echo "grid output is not a directory: ${output}" >&2
      exit 1
    fi
    if [[ -d "${output}" ]]; then
      existing=("${output}"/*)
      if (( ${#existing[@]} != 0 )); then
        echo "refusing nonempty grid output: ${output}" >&2
        exit 1
      fi
    fi

    echo "starting fusion=${fusion} rank_weight=${rank_weight}"
    echo "physical_gpu=${GPU_ID} logical_device=cuda:0"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    "${PY}" -u dta/train_hybrid_affinity.py \
      --fusion-type "${fusion}" \
      --rank-weight "${rank_weight}" \
      --seed 42 \
      --head-epochs 2 \
      --high-epochs 15 \
      --full-epochs 0 \
      --output-dir "${output}" \
      "${fixed_inputs[@]}"
  done
done
