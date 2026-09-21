#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index for formal training}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ROOT="${ROOT}/experiments/lp_pdbbind/interface_first_v1"
SEEDS=(42 43 44 45 46)
RESUME_STAGE1="${RESUME_STAGE1:-0}"
RESUME_STAGE2="${RESUME_STAGE2:-0}"

for seed in "${SEEDS[@]}"; do
  stage1_ready="${ROOT}/Model/LP_PDBBind_InterfaceFirstGerNA_v1_stage1_seed${seed}/stage2_ready.json"
  if [[ -f "${stage1_ready}" ]]; then
    echo "skipping Stage 1 seed ${seed}; stage2_ready.json exists"
  else
    echo "starting Stage 1 seed ${seed}"
    GPU_ID="${GPU_ID}" SEED="${seed}" \
      RESUME_STAGE1="${RESUME_STAGE1}" \
      bash "${RUN_ROOT}/pretrain_stage1.sh"
  fi

  echo "starting Stage 2 seed ${seed}"
  GPU_ID="${GPU_ID}" SEED="${seed}" \
    RESUME_STAGE2="${RESUME_STAGE2}" \
    bash "${RUN_ROOT}/finetune_seed.sh"
done

echo "all interface-first seeds complete"
echo "Run summarize.sh only after the five matched GerNA baseline"
echo "prediction archives are available on the identical ordered test keys."
